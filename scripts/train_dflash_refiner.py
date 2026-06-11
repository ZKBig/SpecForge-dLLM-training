#!/usr/bin/env python3
# coding=utf-8
"""Train the AR refiner on top of a FROZEN DFlash drafter.

Reuses train_dflash.py's scaffolding (data, target hidden_states, FSDP, BF16 optimizer,
training loop). The only differences:
  - the DFlash drafter is loaded from a trained checkpoint and FROZEN;
  - the model is OnlineDFlashRefiner (frozen drafter + trainable refiner);
  - only the refiner is FSDP-wrapped and optimized.
"""

import argparse
import logging
import math
import os
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import load_dataset
from specforge.args import TrackerArgs
from specforge.core.dflash_refiner import DFlashFeatureExtractor, OnlineDFlashRefiner
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import get_dflash_target_model
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint, print_on_rank0, print_with_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash AR Refiner")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--dflash-model-path",
        type=str,
        required=True,
        help="Path/HF-repo of the trained (frozen) DFlash drafter, e.g. z-lab/Qwen3-8B-DFlash-b16",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
    )
    model_group.add_argument("--num-anchors", type=int, default=512)
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument(
        "--embedding-key", type=str, default=None,
        help="Embedding weight key in the target model.",
    )
    model_group.add_argument(
        "--lm-head-key", type=str, default=None,
        help="LM head weight key in the target model.",
    )

    refiner_group = parser.add_argument_group("refiner")
    refiner_group.add_argument(
        "--window-size", type=int, default=0,
        help="Local context window the refiner cross-attends to (0 = pure v1).",
    )
    refiner_group.add_argument("--num-refiner-layers", type=int, default=1)
    refiner_group.add_argument(
        "--use-residual-gate",
        action="store_true",
        help="Refine via a gated residual on the drafter hidden "
        "(head_in = h[k] + gate*refined). Off = use refiner output directly.",
    )
    refiner_group.add_argument(
        "--residual-gate-init", type=float, default=0.0,
        help="Initial value of the residual gate (0 = starts == drafter / ReZero).",
    )
    refiner_group.add_argument(
        "--freeze-residual-gate", action="store_true",
        help="Do NOT train the residual gate; fix it at --residual-gate-init "
        "(e.g. 1.0 for a plain full residual).",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc", type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--save-interval", type=int, default=1000)
    output_group.add_argument("--eval-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    return parser.parse_args()


def build_dataloader(args, tokenizer, block_size) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build the train (+ optional eval) dataloader."""
    import hashlib

    cache_params_string = (
        f"{args.train_data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )

    min_loss_tokens = 2 * block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3 = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_eagle3 = eval_eagle3.filter(lambda x: x["loss_mask"].sum() >= min_loss_tokens)
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )
    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, refiner_fsdp, optimizer):
    """Save ONLY the refiner weights (+ training/optimizer state)."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(refiner_fsdp, StateDictType.FULL_STATE_DICT):
        refiner_state_dict = refiner_fsdp.state_dict()
        if dist.get_rank() == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "args": args,
                    "window_size": args.window_size,
                    "num_refiner_layers": args.num_refiner_layers,
                    "refiner_state_dict": refiner_state_dict,
                    **optimizer.state_dict(),
                },
                os.path.join(save_dir, "refiner.pt"),
            )
            print_on_rank0(f"Saved refiner checkpoint to {save_dir}")
    dist.barrier()


def record_metrics(args, loss, accuracy, global_step, tracker, optimizer, train_dataloader):
    logdict = {"train/lr": optimizer.get_learning_rate(), "train/loss": loss, "train/accuracy": accuracy}
    print_on_rank0(
        f"Train - Step {global_step} "
        f"[{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?], "
        f"Loss: {loss:.4f}, Acc: {accuracy:.4f}"
    )
    tracker.log(logdict, step=global_step)


@torch.no_grad()
def run_eval(refiner_model, eval_dataloader, target_model, max_batches=50):
    """Mean free-running accept length: (refiner, drafter baseline) over a few eval batches."""
    r_tot, d_tot, cnt = 0.0, 0.0, 0
    for i, data in enumerate(eval_dataloader):
        if i >= max_batches:
            break
        input_ids = data["input_ids"].cuda()
        attention_mask = data["attention_mask"].cuda()
        loss_mask = data["loss_mask"].cuda()
        out = target_model.generate_dflash_data(input_ids, attention_mask, loss_mask)
        r, d = refiner_model.accept_lengths(input_ids, out.hidden_states.cuda(), loss_mask)
        r_tot += r.item()
        d_tot += d.item()
        cnt += 1
    if cnt == 0:
        return 0.0, 0.0
    vals = torch.tensor([r_tot / cnt, d_tot / cnt], device="cuda")
    dist.all_reduce(vals)
    vals = vals / dist.get_world_size()
    return vals[0].item(), vals[1].item()


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    # --- frozen DFlash drafter ---
    print_on_rank0(f"Loading frozen DFlash drafter from {args.dflash_model_path}")
    draft_model = (
        DFlashDraftModel.from_pretrained(args.dflash_model_path, torch_dtype=torch.bfloat16)
        .cuda()
        .to(torch.bfloat16)
    )
    draft_model.config._attn_implementation = args.attention_backend
    block_size = draft_model.block_size
    mask_token_id = (draft_model.config.dflash_config or {}).get("mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = draft_model.mask_token_id
    print_on_rank0(
        f"block_size={block_size} target_layer_ids={draft_model.target_layer_ids} "
        f"mask_token_id={mask_token_id}"
    )

    # --- target model (produces hidden_states online) ---
    print_on_rank0(f"Loading target model from {args.target_model_path}")
    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend="hf",
        torch_dtype=torch.bfloat16,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )
    target_model.set_capture_layers(draft_model.target_layer_ids)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # --- data ---
    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer, block_size)
    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    # --- target embed / lm_head (reused, frozen) ---
    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )

    # --- feature extractor (frozen) + refiner (trainable) ---
    feature_extractor = DFlashFeatureExtractor(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head.cuda().to(torch.bfloat16),
        target_embed_tokens=target_components.embed_tokens.cuda().to(torch.bfloat16),
        mask_token_id=mask_token_id,
        block_size=block_size,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=None,
    ).cuda()
    feature_extractor.eval()

    refiner_model = OnlineDFlashRefiner(
        feature_extractor,
        window_size=args.window_size,
        num_refiner_layers=args.num_refiner_layers,
        use_residual_gate=args.use_residual_gate,
        residual_gate_init=args.residual_gate_init,
        freeze_residual_gate=args.freeze_residual_gate,
    ).cuda()
    refiner_model.refiner = refiner_model.refiner.to(torch.bfloat16)
    print_on_rank0(
        f"Refiner params: {sum(p.numel() for p in refiner_model.refiner.parameters()):,} "
        f"(window_size={args.window_size}, layers={args.num_refiner_layers})"
    )

    # FSDP only the trainable refiner; the frozen feature extractor stays replicated.
    refiner_model.refiner = FSDP(
        refiner_model.refiner,
        use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    print_with_rank("Initialized FSDP (refiner)")

    optimizer = BF16Optimizer(
        refiner_model.refiner,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    # --- resume ---
    start_epoch, global_step = 0, 0
    if args.resume and os.path.isdir(args.output_dir):
        last_ckpt, ckpt_info = get_last_checkpoint(args.output_dir)
        if last_ckpt and os.path.exists(os.path.join(last_ckpt, "refiner.pt")):
            state = torch.load(os.path.join(last_ckpt, "refiner.pt"), map_location="cpu", weights_only=False)
            with FSDP.state_dict_type(refiner_model.refiner, StateDictType.FULL_STATE_DICT):
                refiner_model.refiner.load_state_dict(state["refiner_state_dict"])
            optimizer.scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch = state["epoch"]
            global_step = state["global_step"]
            print_on_rank0(f"Resumed from epoch {start_epoch}, step {global_step}")
    skip_steps = global_step - start_epoch * len(train_dataloader)

    tracker = create_tracker(args, args.output_dir)
    last_time = time.time()
    print_on_rank0(f"Starting refiner training from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        refiner_model.refiner.train()
        feature_extractor.eval()

        progress_bar = (
            tqdm(train_dataloader, desc=f"Refiner Epoch {epoch}", leave=True)
            if dist.get_rank() == 0
            else train_dataloader
        )

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()

            target_output = target_model.generate_dflash_data(input_ids, attention_mask, loss_mask)
            hidden_states = target_output.hidden_states.cuda()

            loss, accuracy = refiner_model(
                input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask
            )
            (loss / args.accumulation_steps).backward()
            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log, acc_log = loss.clone(), accuracy.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                loss_log /= dist.get_world_size()
                acc_log /= dist.get_world_size()
                record_metrics(args, loss_log.item(), acc_log.item(), global_step, tracker, optimizer, train_dataloader)

            if eval_dataloader is not None and global_step % args.eval_interval == 0:
                r_acc, d_acc = run_eval(refiner_model, eval_dataloader, target_model)
                tracker.log(
                    {"eval/refiner_accept_len": r_acc, "eval/drafter_accept_len": d_acc},
                    step=global_step,
                )
                print_on_rank0(
                    f"Eval - Step {global_step}: refiner_accept={r_acc:.3f} drafter_accept={d_acc:.3f}"
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{accuracy.item():.4f}", "iter_time": f"{elapsed:.2f}s"}
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(args, epoch, global_step, refiner_model.refiner, optimizer)

    save_checkpoint(args, args.num_epochs, global_step, refiner_model.refiner, optimizer)
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
