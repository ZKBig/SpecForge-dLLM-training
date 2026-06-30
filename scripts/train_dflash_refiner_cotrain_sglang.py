#!/usr/bin/env python3
# coding=utf-8
"""CO-TRAIN the AR refiner together with the DFlash drafter backbone -- SGLANG-BACKEND variant.

This is a SEPARATE copy of `train_dflash_refiner_cotrain.py` that ADDS a selectable target-hidden
backend (`--target-model-backend hf|sglang`, wired exactly like train_domino.py) plus an `--eval-only`
mode. Use this to train the SGU refiner on SGLANG-produced target hidden states (so it matches an sglang
inference deployment -- removes the HF-train/sglang-infer shift) without touching the original HF trainer.
The original `train_dflash_refiner_cotrain.py` is left UNCHANGED so uploading this does not overwrite it.

This is the co-train ABLATION counterpart of `train_dflash_refiner.py` (frozen drafter).
It is a SEPARATE script and shares no code path with the frozen trainer beyond imports.

Differences vs the frozen trainer:
  - feature extractor is `CoTrainFeatureExtractor` (drafter forward keeps autograd);
  - model is `OnlineDFlashRefinerCoTrain` (drafter backbone un-frozen; target lm_head/embed frozen);
  - BOTH the refiner head AND the drafter backbone are FSDP-wrapped and optimized;
  - checkpoints save refiner_state_dict + draft_state_dict (+ optimizer/scheduler).

The drafter is still initialized from --dflash-model-path (e.g. z-lab/Qwen3-8B-DFlash-b16),
so frozen vs co-train differ only in whether that backbone keeps training -> a clean ablation.
"""

import argparse
import logging
import math
import os
import sys
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dflash_refiner_cotrain import (
    CoTrainBF16Optimizer,
    CoTrainFeatureExtractor,
    OnlineDFlashRefinerCoTrain,
)
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import get_dflash_target_model
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint, print_on_rank0, print_with_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Co-train DFlash AR Refiner + drafter")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--dflash-model-path",
        type=str,
        required=True,
        help="Path/HF-repo of the DFlash drafter to INITIALIZE from (then co-trained), "
        "e.g. z-lab/Qwen3-8B-DFlash-b16",
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
    model_group.add_argument(
        "--target-model-backend", type=str, default="hf", choices=["hf", "sglang"],
        help="Backend that produces the target HIDDEN STATES. cotrain was trained with 'hf'; "
        "use 'sglang' to align/measure against an sglang inference deployment (the hidden "
        "numerics differ between HF and sglang -> drafter/refiner see a different distribution).",
    )

    refiner_group = parser.add_argument_group("refiner")
    refiner_group.add_argument(
        "--window-size", type=int, default=0,
        help="Local context window the refiner cross-attends to (0 = pure v1).",
    )
    refiner_group.add_argument("--num-refiner-layers", type=int, default=1)
    refiner_group.add_argument(
        "--mlp-intermediate", type=int, default=None,
        help="Refiner MLP intermediate size. None = Qwen3 default (full); >0 = shrink to this; "
        "<=0 = drop the MLP entirely. For MLP-size ablation / cost reduction.",
    )
    refiner_group.add_argument(
        "--lowrank-lmhead-rank", type=int, default=0,
        help="If >0, read out as base lm_head(h) + low-rank(refined-h) with this rank (Domino-style "
        "base+correction; only the small delta is low-ranked so argmax is preserved). 0 = full lm_head.",
    )
    refiner_group.add_argument(
        "--lowrank-lmhead-init", type=str, default="svd", choices=["svd", "zero"],
        help="Low-rank head init: 'svd' = warm-start so up@down ~= lm_head (starts ~equivalent to "
        "full); 'zero' = correction starts at 0 (readout == drafter at step 0).",
    )
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
        help="Do NOT train the residual gate; fix it at --residual-gate-init.",
    )
    refiner_group.add_argument(
        "--loss-decay-gamma", type=float, default=None,
        help="If set, weight block-position k loss by exp(-(k-1)/gamma).",
    )
    refiner_group.add_argument(
        "--drafter-lr-scale", type=float, default=1.0,
        help="Scale the drafter LR relative to the head LR (e.g. 0.1 = drafter learns 10x "
        "slower). !=1.0 -> a 2-param-group optimizer; =1.0 -> single shared LR.",
    )
    refiner_group.add_argument(
        "--lambda-base-start", type=float, default=1.0,
        help="Initial weight of base_loss = CE(lm_head(drafter_hidden)) = the drafter's own "
        "prediction. Anchors the co-trained drafter (Domino's trick) so training doesn't collapse. "
        "loss=(1-lambda)*refined+lambda*base. Set 0 to disable (pure refiner loss, may diverge).",
    )
    refiner_group.add_argument(
        "--lambda-base-decay-ratio", type=float, default=1.0,
        help="Fraction of total steps over which lambda_base decays linearly to 0 "
        "(1.0 = decay across the whole run, Domino default).",
    )
    # --- refiner architecture ablations (all optional, default == current) ---
    refiner_group.add_argument(
        "--mixer-type", type=str, default="attention", choices=["attention", "sgu"],
        help="causal mixer: 'attention' (Qwen3 self-attn) | 'sgu' (channel-wise lower-triangular "
        "causal mix — cheap, fixed, block-internal; pair with --gate-type perpos).",
    )
    refiner_group.add_argument(
        "--pool-type", type=str, default="mean", choices=["mean", "xattn"],
        help="global pool token: 'mean' (mean-pool) | 'xattn' (cross-attention over the dflash output set).",
    )
    refiner_group.add_argument(
        "--gate-type", type=str, default="scalar", choices=["scalar", "perpos"],
        help="residual gate: 'scalar' (one global ReZero) | 'perpos' (input-dependent sigmoid(w.h) "
        "per position — learns WHERE to refine; needs --use-residual-gate).",
    )
    refiner_group.add_argument(
        "--zero-init-oproj", action="store_true",
        help="Stability route 1 (entry fix): zero-init the o_proj of EVERY randomly-init "
        "data-dependent attention module — the attention mixer AND the xattn pool — so each "
        "starts as a no-op and grows in from 0 instead of injecting harmful random scramble at "
        "step 0 (which slams the perpos gate shut). No-op for sgu mixer / mean pool.",
    )
    refiner_group.add_argument(
        "--gate-floor", type=float, default=0.0,
        help="Stability route 2 (trap fix): perpos gate floor eps in g = eps + (1-eps)*sigmoid(w.h). "
        "Keeps g>=eps so the mixer's gradient (~g) is never throttled to 0 (breaks gate-collapse). "
        "0.0 = off (current behavior); try 0.1.",
    )
    refiner_group.add_argument(
        "--gate-bias-init", type=float, default=-5.0,
        help="Stability route 3 (soft start): perpos gate bias init. -5.0 => g~0 (starts == drafter, "
        "but mutes the mixer's early gradient); 0.0 => g~0.5 (strong early gradient so the mixer can "
        "learn before the gate shuts; accept an initial score drop). No floor — gate may still close.",
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

    eval_group = parser.add_argument_group("eval-only")
    eval_group.add_argument(
        "--eval-only", action="store_true",
        help="Skip training: load --eval-checkpoint, run run_eval (free-running accept), print, exit. "
        "Compare HF vs sglang target-hidden accept by re-running with --target-model-backend hf | sglang. "
        "Reuse the SAME refiner args (mixer/gate/layers/...) as training so the checkpoint loads cleanly.",
    )
    eval_group.add_argument("--eval-checkpoint", type=str, default=None,
                            help="refiner_cotrain.pt to load for --eval-only.")
    eval_group.add_argument("--eval-max-batches", type=int, default=50)

    # sglang target-backend knobs (--sglang-attention-backend, --sglang-mem-fraction-static, ...);
    # only used when --target-model-backend sglang.
    SGLangBackendArgs.add_args(parser)
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


def save_checkpoint(args, epoch, step, refiner_fsdp, draft_fsdp, optimizer):
    """Save the refiner AND the co-trained drafter (+ optimizer/scheduler state).

    Both state_dict() calls are collectives -> every rank must enter them; only rank 0
    writes. The drafter is saved as a FULL_STATE_DICT so it can be reloaded into a fresh
    DFlashDraftModel(config) for benchmarking.
    """
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(refiner_fsdp, StateDictType.FULL_STATE_DICT):
        refiner_state_dict = refiner_fsdp.state_dict()
    with FSDP.state_dict_type(draft_fsdp, StateDictType.FULL_STATE_DICT):
        draft_state_dict = draft_fsdp.state_dict()

    if dist.get_rank() == 0:
        torch.save(
            {
                "epoch": epoch,
                "global_step": step,
                "args": args,
                "window_size": args.window_size,
                "num_refiner_layers": args.num_refiner_layers,
                "mixer_type": args.mixer_type,
                "pool_type": args.pool_type,
                "gate_type": args.gate_type,
                "zero_init_oproj": args.zero_init_oproj,
                "gate_floor": args.gate_floor,
                "gate_bias_init": args.gate_bias_init,
                "mlp_intermediate": args.mlp_intermediate,
                "lowrank_lmhead_rank": args.lowrank_lmhead_rank,
                "lowrank_lmhead_init": args.lowrank_lmhead_init,
                "refiner_state_dict": refiner_state_dict,
                "draft_state_dict": draft_state_dict,
                # Scheduler is REPLICATED across ranks -> store it in the rank-0 file.
                # The optimizer (Adam) state is FSDP-SHARDED (per-rank, different sizes)
                # because fp32_params clone the LOCAL FSDP shards -> saved per-rank below.
                "scheduler_state_dict": optimizer.scheduler.state_dict(),
            },
            os.path.join(save_dir, "refiner_cotrain.pt"),
        )
        print_on_rank0(f"Saved co-train checkpoint (refiner+drafter) to {save_dir}")

    # Every rank writes its OWN optimizer (Adam) shard. save_dir already exists for all
    # ranks (rank-0 mkdir + barrier above). On resume each rank reloads its matching shard.
    torch.save(
        {"optimizer_state_dict": optimizer.optimizer.state_dict()},
        os.path.join(save_dir, f"optim_rank{dist.get_rank()}.pt"),
    )
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
    vals = torch.tensor([r_tot, d_tot, float(cnt)], device="cuda")
    dist.all_reduce(vals)
    total = vals[2].item()
    if total == 0:
        return 0.0, 0.0
    return (vals[0] / total).item(), (vals[1] / total).item()


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

    # eval-only: pull the refiner ARCH args FROM the checkpoint so the build always matches it
    # (otherwise a mismatched --mixer-type/--num-refiner-layers/... silently loads garbage).
    # Loaded once here (2.5GB) and reused by the eval-only branch below.
    _eval_ck = None
    if args.eval_only:
        if not args.eval_checkpoint:
            raise ValueError("--eval-only requires --eval-checkpoint")
        _eval_ck = torch.load(args.eval_checkpoint, map_location="cpu", weights_only=False)
        _saved = _eval_ck.get("args")
        for _k in ("mixer_type", "pool_type", "gate_type", "num_refiner_layers", "window_size",
                   "use_residual_gate", "residual_gate_init", "freeze_residual_gate",
                   "zero_init_oproj", "gate_floor", "gate_bias_init", "mlp_intermediate",
                   "lowrank_lmhead_rank", "lowrank_lmhead_init"):
            if _saved is not None and hasattr(_saved, _k):
                setattr(args, _k, getattr(_saved, _k))
            elif _k in _eval_ck:
                setattr(args, _k, _eval_ck[_k])
        print_on_rank0(
            f"[eval-only] synced refiner arch from checkpoint: mixer={args.mixer_type} "
            f"gate={args.gate_type} layers={args.num_refiner_layers} window={args.window_size} "
            f"mlp_intermediate={args.mlp_intermediate} lowrank={args.lowrank_lmhead_rank}"
        )

    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        _logf = open(os.path.join(args.output_dir, "train.log"), "a", buffering=1)

        class _Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, msg):
                for s in self.streams:
                    s.write(msg)
                    s.flush()

            def flush(self):
                for s in self.streams:
                    s.flush()

            def isatty(self):
                # reflect the REAL underlying stream: wandb shows its progress bar when this
                # runs in a terminal, and auto-suppresses it when output is piped to a file
                # (where a dynamic \r/ANSI bar would just be log garbage). Without this method
                # at all, wandb.init crashes with AttributeError on the _Tee.
                return self.streams[0].isatty()

            def __getattr__(self, name):
                # delegate anything else (fileno, encoding, buffer, ...) to the real stream
                # so libraries probing the stream don't hit AttributeError.
                return getattr(self.streams[0], name)

        sys.stdout = _Tee(sys.stdout, _logf)
        sys.stderr = _Tee(sys.stderr, _logf)
        print(f"[tee] logging to {os.path.join(args.output_dir, 'train.log')}")

    # --- DFlash drafter (initialized from checkpoint, then CO-TRAINED) ---
    print_on_rank0(f"Loading DFlash drafter (to co-train) from {args.dflash_model_path}")
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

    # --- target model (produces hidden_states online; HF by default, sglang to match deployment) ---
    print_on_rank0(
        f"Loading target model from {args.target_model_path} "
        f"using {args.target_model_backend} backend"
    )
    _tm_kwargs = {}
    if args.target_model_backend == "sglang":
        _tm_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **_tm_kwargs,
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

    # --- co-train feature extractor (grad flows to drafter) + refiner ---
    feature_extractor = CoTrainFeatureExtractor(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head.cuda().to(torch.bfloat16),
        target_embed_tokens=target_components.embed_tokens.cuda().to(torch.bfloat16),
        mask_token_id=mask_token_id,
        block_size=block_size,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=None,
    ).cuda()

    refiner_model = OnlineDFlashRefinerCoTrain(
        feature_extractor,
        window_size=args.window_size,
        num_refiner_layers=args.num_refiner_layers,
        use_residual_gate=args.use_residual_gate,
        residual_gate_init=args.residual_gate_init,
        freeze_residual_gate=args.freeze_residual_gate,
        loss_decay_gamma=args.loss_decay_gamma,
        mixer_type=args.mixer_type,
        pool_type=args.pool_type,
        gate_type=args.gate_type,
        zero_init_oproj=args.zero_init_oproj,
        gate_floor=args.gate_floor,
        gate_bias_init=args.gate_bias_init,
        mlp_intermediate=args.mlp_intermediate,
        lowrank_rank=args.lowrank_lmhead_rank,
        lowrank_init=args.lowrank_lmhead_init,
    ).cuda()
    refiner_model.refiner = refiner_model.refiner.to(torch.bfloat16)
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "\n==================== REFINER ARCH ====================\n"
            f"  mixer_type   = {args.mixer_type}        (attention | sgu)\n"
            f"  pool_type    = {args.pool_type}         (mean | xattn)\n"
            f"  gate_type    = {args.gate_type}         (scalar | perpos)  use_residual_gate={args.use_residual_gate}\n"
            f"  zero_init_oproj = {args.zero_init_oproj}   gate_floor = {args.gate_floor}   "
            f"gate_bias_init = {args.gate_bias_init}   (stability knobs)\n"
            f"  window_size  = {args.window_size}   num_refiner_layers = {args.num_refiner_layers}   "
            f"mlp_intermediate = {getattr(args, 'mlp_intermediate', None)}\n"
            f"  lambda_base  = {args.lambda_base_start}->0 (ratio {args.lambda_base_decay_ratio})   "
            f"drafter_lr_scale = {args.drafter_lr_scale}\n"
            f"  head params  = {sum(p.numel() for p in refiner_model.refiner.parameters()):,} | "
            f"co-trained drafter params = {refiner_model._cotrain_drafter_params:,}\n"
            "======================================================\n",
            flush=True,
        )

    # --- eval-only: load checkpoint, run free-running accept, exit (no FSDP / no training) ---
    # Isolates the target-hidden BACKEND: same checkpoint + same eval data + same metric, only the
    # source of the hidden states changes (--target-model-backend hf vs sglang). A drop hf->sglang
    # is the train(HF)/infer(sglang) distribution shift, NOT a code bug.
    if args.eval_only:
        if not args.eval_checkpoint:
            raise ValueError("--eval-only requires --eval-checkpoint")
        print_on_rank0(
            f"[eval-only] loading {args.eval_checkpoint} "
            f"(target hidden backend = {args.target_model_backend})"
        )
        ck = _eval_ck if _eval_ck is not None else torch.load(
            args.eval_checkpoint, map_location="cpu", weights_only=False
        )
        rm = refiner_model.refiner.load_state_dict(ck["refiner_state_dict"], strict=False)
        dm = refiner_model.feature_extractor.draft_model.load_state_dict(
            ck["draft_state_dict"], strict=False
        )
        print_on_rank0(
            f"[eval-only] refiner load: missing={list(rm.missing_keys)[:4]} "
            f"unexpected={list(rm.unexpected_keys)[:4]} "
            f"(if these are non-empty, your refiner args don't match the checkpoint)"
        )
        print_on_rank0(
            f"[eval-only] drafter load: missing={list(dm.missing_keys)[:4]} "
            f"unexpected={list(dm.unexpected_keys)[:4]}"
        )
        refiner_model.eval()
        r_acc, d_acc = run_eval(
            refiner_model, eval_dataloader, target_model, max_batches=args.eval_max_batches
        )
        print_on_rank0(
            f"\n============== EVAL-ONLY accept ({args.target_model_backend} hidden) ==============\n"
            f"  refiner accept_len = {r_acc:.4f}\n"
            f"  drafter accept_len = {d_acc:.4f}\n"
            f"  checkpoint   = {args.eval_checkpoint}\n"
            f"  eval_batches = {args.eval_max_batches}\n"
            "=================================================================\n"
        )
        destroy_distributed()
        return

    # FSDP-wrap BOTH trainable modules: the refiner head AND the drafter backbone.
    refiner_model.refiner = FSDP(
        refiner_model.refiner,
        use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    refiner_model.feature_extractor.draft_model = FSDP(
        refiner_model.feature_extractor.draft_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    print_with_rank("Initialized FSDP (refiner head + drafter)")

    # optimizer over both FSDP units. drafter-lr-scale != 1.0 -> 2 param groups
    # (head at lr, drafter at lr*scale); == 1.0 -> single shared LR.
    if args.drafter_lr_scale != 1.0:
        optimizer = CoTrainBF16Optimizer(
            refiner_model.refiner,
            refiner_model.feature_extractor.draft_model,
            lr=args.learning_rate,
            drafter_lr_scale=args.drafter_lr_scale,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=total_steps,
        )
        print_on_rank0(f"Optimizer: 2 groups (head lr={args.learning_rate}, "
                       f"drafter lr={args.learning_rate * args.drafter_lr_scale})")
    else:
        trainable = nn.ModuleList(
            [refiner_model.refiner, refiner_model.feature_extractor.draft_model]
        )
        optimizer = BF16Optimizer(
            trainable,
            lr=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=total_steps,
        )

    # --- resume (refiner + drafter + optimizer) ---
    start_epoch, global_step = 0, 0
    if args.resume and os.path.isdir(args.output_dir):
        last_ckpt, ckpt_info = get_last_checkpoint(args.output_dir)
        ckpt_file = os.path.join(last_ckpt, "refiner_cotrain.pt") if last_ckpt else None
        if ckpt_file and os.path.exists(ckpt_file):
            state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            with FSDP.state_dict_type(refiner_model.refiner, StateDictType.FULL_STATE_DICT):
                refiner_model.refiner.load_state_dict(state["refiner_state_dict"])
            if "draft_state_dict" in state:
                with FSDP.state_dict_type(
                    refiner_model.feature_extractor.draft_model, StateDictType.FULL_STATE_DICT
                ):
                    refiner_model.feature_extractor.draft_model.load_state_dict(
                        state["draft_state_dict"]
                    )
            # CRITICAL: the BF16 optimizer holds fp32 MASTER copies cloned at construction
            # (pre-resume init weights). The load_state_dict calls above updated the model's
            # p.data but NOT these masters. On the first step(), p.data.copy_(master) would
            # overwrite the freshly-loaded checkpoint weights with the stale init masters and
            # silently revert the model (train loss looks fine for one step, then eval craters).
            # Re-sync the masters from the just-loaded model params.
            with torch.no_grad():
                for p, mp in zip(optimizer.model_params, optimizer.fp32_params):
                    mp.data.copy_(p.data.to(torch.float32))
            # Scheduler is replicated -> always restore (continues LR schedule exactly).
            if "scheduler_state_dict" in state:
                optimizer.scheduler.load_state_dict(state["scheduler_state_dict"])
            # Optimizer (Adam) state is FSDP-SHARDED -> each rank loads ITS OWN shard.
            # Loading rank-0's state into every rank crashes (size mismatch). Old
            # checkpoints without per-rank shards just reset Adam moments (small dip).
            optim_file = os.path.join(last_ckpt, f"optim_rank{dist.get_rank()}.pt")
            if os.path.exists(optim_file):
                opt_state = torch.load(optim_file, map_location="cpu", weights_only=False)
                optimizer.optimizer.load_state_dict(opt_state["optimizer_state_dict"])
                print_on_rank0("Restored per-rank optimizer (Adam) shards + scheduler.")
            else:
                print_on_rank0("[warn] no per-rank optim shards (old checkpoint); "
                               "Adam moments reset (small dip), scheduler restored.")
            start_epoch = state["epoch"]
            global_step = state["global_step"]
            print_on_rank0(f"Resumed from epoch {start_epoch}, step {global_step}")
    skip_steps = global_step - start_epoch * len(train_dataloader)

    tracker = create_tracker(args, args.output_dir)
    last_time = time.time()
    print_on_rank0(f"Starting co-train from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        refiner_model.refiner.train()
        # everything frozen-eval first, then put ONLY the drafter back in train mode
        feature_extractor.eval()
        refiner_model.feature_extractor.draft_model.train()

        progress_bar = (
            tqdm(train_dataloader, desc=f"Co-train Epoch {epoch}", leave=True)
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

            # base_loss curriculum: lambda_base decays start -> 0 over decay_ratio*total_steps
            decay_steps = max(1, int(total_steps * args.lambda_base_decay_ratio))
            lambda_base = max(0.0, args.lambda_base_start * (1.0 - min(global_step / decay_steps, 1.0)))

            loss, accuracy = refiner_model(
                input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask,
                lambda_base=lambda_base,
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
                tracker.log({"train/lambda_base": lambda_base}, step=global_step)

            if eval_dataloader is not None and global_step % args.eval_interval == 0:
                refiner_model.feature_extractor.draft_model.eval()
                r_acc, d_acc = run_eval(refiner_model, eval_dataloader, target_model)
                refiner_model.feature_extractor.draft_model.train()
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
                save_checkpoint(
                    args, epoch, global_step,
                    refiner_model.refiner, refiner_model.feature_extractor.draft_model,
                    optimizer,
                )

    save_checkpoint(
        args, args.num_epochs, global_step,
        refiner_model.refiner, refiner_model.feature_extractor.draft_model,
        optimizer,
    )
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
