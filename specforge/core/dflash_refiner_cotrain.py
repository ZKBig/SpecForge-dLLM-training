#!/usr/bin/env python3
# coding=utf-8
"""Co-train variant of the AR refiner: trains the DFlash drafter backbone TOGETHER
with the refiner head (instead of freezing the drafter).

This file is fully SEPARATE from and does NOT modify the frozen-drafter code in
`dflash_refiner.py`; it only subclasses it. Use it for the ablation:

    frozen drafter + attention head   (dflash_refiner.OnlineDFlashRefiner)   -- proposed
    co-train drafter + attention head (this file)                            -- ceiling

Two things differ from the frozen path:
  1. The drafter forward must run WITH autograd so gradients reach draft_model.
     `DFlashFeatureExtractor.compute_block_features` is `@torch.no_grad()`-decorated;
     `CoTrainFeatureExtractor` re-runs the SAME body in the ambient grad mode (its
     undecorated function via `__wrapped__`), so training builds a graph while eval
     (called under `@torch.no_grad` in `accept_lengths`) stays grad-free.
  2. The drafter backbone params are un-frozen (target lm_head / embed stay frozen).
"""

import torch
import torch.nn.functional as F

from specforge.core.dflash_refiner import (
    DFlashFeatureExtractor,
    OnlineDFlashRefiner,
)
from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.optimizer import BF16Optimizer


class CoTrainFeatureExtractor(DFlashFeatureExtractor):
    """DFlashFeatureExtractor whose `compute_block_features` follows the AMBIENT grad
    mode instead of forcing `torch.no_grad()`, so gradients flow into the drafter
    during training but eval (under an outer no_grad) stays cheap."""

    # the parent method is `@torch.no_grad()`-decorated; functools.wraps (used inside
    # torch's decorator) exposes the original undecorated function as `__wrapped__`,
    # so we reuse the exact same body without duplicating it (no divergence risk).
    _UNDECORATED = DFlashFeatureExtractor.compute_block_features.__wrapped__

    def compute_block_features(self, input_ids, hidden_states, loss_mask):
        # NO forced no_grad here: inherit whatever grad mode the caller is in.
        #   * training forward (grad enabled)  -> graph built -> drafter trains
        #   * accept_lengths (@torch.no_grad)  -> grad disabled -> eval stays cheap
        return CoTrainFeatureExtractor._UNDECORATED(
            self, input_ids, hidden_states, loss_mask
        )


class OnlineDFlashRefinerCoTrain(OnlineDFlashRefiner):
    """OnlineDFlashRefiner that also trains the DFlash drafter backbone.

    `OnlineDFlashRefiner.__init__` freezes the whole feature extractor; here we
    re-enable grad on the drafter backbone (`draft_model`) only. The target lm_head
    and embedding stay frozen. For gradients to actually reach the drafter, pass a
    `CoTrainFeatureExtractor` (not the plain `DFlashFeatureExtractor`).
    """

    def __init__(self, *args, **kwargs):
        # 2-point consistency: weight on the EXTRA loss term that conditions the refiner on the
        # drafter's SEED prev (what the first Jacobi pass actually sees at inference) instead of the
        # ground-truth prev. Closes the teacher-forcing vs inference gap -> Jacobi converges in
        # fewer passes. 0 = off (default; existing recipes unchanged).
        self.consistency_weight = float(kwargs.pop("consistency_weight", 0.0))
        super().__init__(*args, **kwargs)
        n = 0
        for p in self.feature_extractor.draft_model.parameters():
            p.requires_grad = True
            n += p.numel()
        self._cotrain_drafter_params = n

    def forward(self, input_ids, hidden_states, loss_mask, lambda_base=0.0):
        """Mirror of OnlineDFlashRefiner.forward, but ALSO computes the drafter's own
        base_loss (CE on lm_head(h), i.e. the gate=0 / no-refinement prediction) and blends:

            loss = (1 - lambda_base) * refined_loss + lambda_base * base_loss

        base_loss anchors the (now trainable) drafter so it stays a valid standalone drafter
        while it co-adapts to the head -> co-training does NOT collapse. This is exactly the
        role of Domino's lambda_base. With lambda_base=0 this reduces to the plain refiner loss
        (and the drafter trains only through refined_loss -> can be unstable).
        """
        fe = self.feature_extractor
        f = fe.compute_block_features(input_ids, hidden_states, loss_mask)

        B = f.output_hidden.size(0)
        H = f.output_hidden.size(-1)
        block = self.block_size
        n = f.output_hidden.size(1) // block
        BN = B * n

        h = f.output_hidden.view(B, n, block, H).reshape(BN, block, H)
        g = h.mean(dim=1, keepdim=True).expand(-1, block, -1)

        tgt = f.target_ids.reshape(BN, block)
        prev_tok = tgt.roll(shifts=1, dims=1)
        am1_idx = (f.anchor_positions - 1).clamp(min=0)
        prev_tok[:, 0] = torch.gather(input_ids, 1, am1_idx).reshape(BN)
        prev_emb = fe.embed_tokens(prev_tok)

        window_hidden = window_mask = None
        if self.refiner.window_size > 0:
            window_hidden, window_mask = self._gather_window(
                hidden_states, f.anchor_positions, f.seq_len, B, n
            )

        V = fe.lm_head.weight.size(0)
        base_3d = fe.lm_head(h)  # (BN, block, V); gate=0 drafter prediction
        if self.refiner.lowrank_head is not None:
            # correction computed INSIDE refiner.forward (FSDP-safe); reuse base_3d (1 full lm_head).
            refined, corr = self.refiner(
                h, g, prev_emb, window_hidden, window_mask, return_correction=True
            )
            refined_logits = (base_3d + corr).reshape(-1, V)
        else:
            refined = self.refiner(h, g, prev_emb, window_hidden, window_mask)
            refined_logits = fe.lm_head(refined).reshape(-1, V)
        base_logits = base_3d.reshape(-1, V)

        flat_tgt = tgt.reshape(-1)
        valid = f.binary_mask  # (B, n, block) 0/1
        w = valid
        if self.loss_decay_gamma:
            kk = torch.arange(self.block_size, device=valid.device)
            decay = torch.exp(-(kk - 1).clamp(min=0).float() / self.loss_decay_gamma)
            w = w * decay.view(1, 1, -1)
        flat_w = w.reshape(-1)
        flat_valid = valid.reshape(-1)
        denom = flat_w.sum() + 1e-6

        refined_loss = (F.cross_entropy(refined_logits, flat_tgt, reduction="none") * flat_w).sum() / denom
        base_loss = (F.cross_entropy(base_logits, flat_tgt, reduction="none") * flat_w).sum() / denom
        loss = (1.0 - lambda_base) * refined_loss + lambda_base * base_loss

        # --- 2-point consistency: refine ALSO from the drafter's SEED prev (= the first Jacobi pass's
        #     real input at inference), targeting the SAME gt. Trains the refiner to correct an
        #     imperfect prev in ONE pass -> fewer inference passes. Costs one extra refiner+lm_head. ---
        if self.consistency_weight > 0:
            seed = base_3d.argmax(dim=-1).detach()             # (BN, block) drafter argmax per position
            prev_seed = prev_tok.clone()
            # slots 0,1 (tok_am1, anchor) stay correct; slots 2..block-1 <- drafter's predicted prev
            prev_seed[:, 2:] = seed[:, 1:block - 1]
            prev_seed_emb = fe.embed_tokens(prev_seed)
            if self.refiner.lowrank_head is not None:
                refined_s, corr_s = self.refiner(
                    h, g, prev_seed_emb, window_hidden, window_mask, return_correction=True
                )
                seed_logits = (base_3d + corr_s).reshape(-1, V)
            else:
                refined_s = self.refiner(h, g, prev_seed_emb, window_hidden, window_mask)
                seed_logits = fe.lm_head(refined_s).reshape(-1, V)
            seed_loss = (F.cross_entropy(seed_logits, flat_tgt, reduction="none") * flat_w).sum() / denom
            loss = loss + self.consistency_weight * seed_loss

        with torch.no_grad():
            pred = refined_logits.argmax(dim=-1)
            correct = (pred == flat_tgt) & (flat_valid > 0.5)
            accuracy = correct.sum().float() / (flat_valid.sum() + 1e-6)

        return loss, accuracy


class CoTrainBF16Optimizer(BF16Optimizer):
    """BF16Optimizer with TWO param groups: refiner head at `lr`, drafter at `lr * drafter_lr_scale`.

    Lets the co-trained ~1B drafter learn more gently than the small head (a lower drafter LR
    further stabilizes co-training, on top of the base_loss anchor). Reuses BF16Optimizer.step /
    state_dict / load_state_dict (they operate on the parallel model_params/fp32_params lists).
    """

    def __init__(self, refiner_module, draft_module, lr, drafter_lr_scale=1.0,
                 weight_decay=0.0, max_grad_norm=0.5, total_steps=800_000, warmup_ratio=0.015):
        self.model = None
        ref_params = [p for p in refiner_module.parameters() if p.requires_grad]
        drf_params = [p for p in draft_module.parameters() if p.requires_grad]
        self.model_params = ref_params + drf_params  # order preserved -> step() zips correctly
        self.max_grad_norm = max_grad_norm
        self.fp32_params = [p.detach().clone().to(torch.float32) for p in self.model_params]
        for mp in self.fp32_params:
            mp.requires_grad = True
        n_ref = len(ref_params)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.fp32_params[:n_ref], "lr": lr},
                {"params": self.fp32_params[n_ref:], "lr": lr * drafter_lr_scale},
            ],
            weight_decay=weight_decay,
        )
        self.last_grad_norm = None
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )
