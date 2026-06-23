#!/usr/bin/env python3
# coding=utf-8
"""Latent (hidden-space) Jacobi variant of the AR refiner.

SEPARATE file: does NOT modify dflash_refiner.py / dflash_refiner_cotrain.py (only subclasses /
reuses them). The idea: avoid the per-pass lm_head round-trip (hidden -> lm_head -> argmax ->
token -> embed -> hidden) by iterating in HIDDEN space.

Differences from the token-space co-train refiner:
  1. The refiner's `prev` input is the PREVIOUS-position HIDDEN (continuous), not embed(prev_token).
     -> at inference the K-pass Jacobi feeds each pass's refined hidden as the next pass's prev, and
        lm_head runs ONCE at the end (K+1 lm_heads -> 1).
  2. Training UNROLLS K passes, feeds each pass's own (DETACHED) output hidden as the next prev, and
     supervises EVERY pass to predict the target (consistency loss). This trains the latent iteration
     to converge stably and matches inference (no exposure bias). detach -> no backprop-through-
     iteration (DEQ-style 1-step gradient): the drafter/refiner still get gradient every pass via
     h/g and the current forward; only the iteration coupling is cut.
  3. base_loss + lambda_base still anchors the co-trained drafter (unchanged).

Position-0 (anchor) predecessor: the block anchor is GIVEN, so its predecessor hidden is a stand-in
-- we use the anchor's own drafter hidden h[:,0]. Inference must use the same convention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from specforge.core.dflash_refiner import RefinerDecoder  # noqa: F401  (reused via from_decoder)
from specforge.core.dflash_refiner_cotrain import (
    CoTrainFeatureExtractor,   # noqa: F401  (re-exported for the train script)
    CoTrainBF16Optimizer,      # noqa: F401  (re-exported for the train script)
    OnlineDFlashRefinerCoTrain,
)


class LatentRefinerHead(nn.Module):
    """Wraps a RefinerDecoder whose `prev` slot consumes a HIDDEN (continuous) instead of a token
    embedding. Adds an RMSNorm on the prev hidden (it is the refiner's own post-gate output, a
    different scale than an input embedding). in_proj stays 3H->H -> no other arch change."""

    def __init__(self, config, ctx_dim, prev_norm=True, **decoder_kwargs):
        super().__init__()
        self.decoder = RefinerDecoder(config, ctx_dim, **decoder_kwargs)
        self.window_size = self.decoder.window_size
        self.prev_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if prev_norm else None

    @classmethod
    def from_decoder(cls, decoder, config, prev_norm=True):
        """Wrap an ALREADY-built RefinerDecoder (reuse its exact config) + add the prev_norm."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.decoder = decoder
        self.window_size = decoder.window_size
        self.prev_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if prev_norm else None
        return self

    def forward(self, h, g, prev_hidden, window_hidden=None, window_mask=None):
        p = self.prev_norm(prev_hidden) if self.prev_norm is not None else prev_hidden
        return self.decoder(h, g, p, window_hidden, window_mask)


class OnlineDFlashRefinerLatentCoTrain(OnlineDFlashRefinerCoTrain):
    """Co-train (drafter + LATENT refiner). Reuses OnlineDFlashRefinerCoTrain's __init__ (RefinerDecoder
    build + drafter un-freeze) and _gather_window, then swaps the head to LatentRefinerHead and the
    forward to the unrolled latent-Jacobi consistency loss. Pass a CoTrainFeatureExtractor."""

    def __init__(self, *args, latent_k=3, latent_prev_norm=True, latent_loss="uniform", **kwargs):
        super().__init__(*args, **kwargs)  # builds RefinerDecoder + unfreezes drafter
        config = self.feature_extractor.draft_model.config
        self.refiner = LatentRefinerHead.from_decoder(self.refiner, config, latent_prev_norm)
        self.latent_k = int(latent_k)               # fixed, user-set: number of unrolled Jacobi passes
        self.latent_loss = latent_loss              # "uniform" | "final_heavy"

    def _loss_weight(self, t, K):
        return float(t + 1) if self.latent_loss == "final_heavy" else 1.0

    def _weight_norm(self, K):
        return float(K * (K + 1) // 2) if self.latent_loss == "final_heavy" else float(K)

    def forward(self, input_ids, hidden_states, loss_mask, lambda_base=0.0):
        fe = self.feature_extractor
        f = fe.compute_block_features(input_ids, hidden_states, loss_mask)

        B = f.output_hidden.size(0)
        H = f.output_hidden.size(-1)
        block = self.block_size
        n = f.output_hidden.size(1) // block
        BN = B * n
        V = fe.lm_head.weight.size(0)

        h = f.output_hidden.view(B, n, block, H).reshape(BN, block, H)
        g = h.mean(dim=1, keepdim=True).expand(-1, block, -1)
        tgt = f.target_ids.reshape(BN, block)

        window_hidden = window_mask = None
        if self.refiner.window_size > 0:
            window_hidden, window_mask = self._gather_window(
                hidden_states, f.anchor_positions, f.seq_len, B, n
            )

        # loss weights: binary mask (+ optional per-position decay) -- same as the token co-train
        flat_tgt = tgt.reshape(-1)
        valid = f.binary_mask
        w = valid
        if self.loss_decay_gamma:
            kk = torch.arange(block, device=valid.device)
            decay = torch.exp(-(kk - 1).clamp(min=0).float() / self.loss_decay_gamma)
            w = w * decay.view(1, 1, -1)
        flat_w = w.reshape(-1)
        flat_valid = valid.reshape(-1)
        denom = flat_w.sum() + 1e-6

        def wce(logits):
            return (F.cross_entropy(logits, flat_tgt, reduction="none") * flat_w).sum() / denom

        # ---- unrolled latent Jacobi: feed own (detached) prev hidden each pass, supervise every pass ----
        anchor_prev = h[:, 0:1, :]                       # (BN,1,H) position-0 predecessor stand-in
        K = self.latent_k                                # fixed, user-set
        prev = h                                         # seed = drafter block hidden (matches inference)
        refined_loss = h.new_zeros(())
        last_logits = None
        for t in range(K):
            prev_shift = torch.cat([anchor_prev, prev[:, :-1, :]], dim=1)   # shift right by 1, pos0=anchor
            refined = self.refiner(h, g, prev_shift, window_hidden, window_mask)
            logits = fe.lm_head(refined).reshape(-1, V)
            refined_loss = refined_loss + self._loss_weight(t, K) * wce(logits)
            last_logits = logits
            prev = refined.detach()                      # no backprop-through-iteration (DEQ 1-step)
        refined_loss = refined_loss / self._weight_norm(K)

        base_loss = wce(fe.lm_head(h).reshape(-1, V))    # gate=0 drafter anchor
        loss = (1.0 - lambda_base) * refined_loss + lambda_base * base_loss

        with torch.no_grad():
            pred = last_logits.argmax(dim=-1)
            accuracy = ((pred == flat_tgt) & (flat_valid > 0.5)).sum().float() / (flat_valid.sum() + 1e-6)
        return loss, accuracy

    @torch.no_grad()
    def accept_lengths(self, input_ids, hidden_states, loss_mask):
        """LATENT-Jacobi eval (overrides the token-space accept_lengths). Runs the SAME par-K
        hidden-space iteration as inference -- prev = previous refined hidden, lm_head ONLY at the
        end -- so eval accept_len reflects what the latent model actually does. (The inherited
        method feeds embed(prev_token) + sequential AR, which is meaningless for a latent refiner.)
        """
        fe = self.feature_extractor
        f = fe.compute_block_features(input_ids, hidden_states, loss_mask)
        B = f.output_hidden.size(0)
        H = f.output_hidden.size(-1)
        block = self.block_size
        n = f.output_hidden.size(1) // block
        BN = B * n
        device = input_ids.device

        h = f.output_hidden.view(B, n, block, H).reshape(BN, block, H)
        g = h.mean(dim=1, keepdim=True).expand(-1, block, -1)
        tgt = f.target_ids.reshape(BN, block)
        anchors = f.anchor_positions.reshape(BN, 1)
        pos_abs = anchors + torch.arange(block, device=device).view(1, block)
        valid = pos_abs < f.seq_len
        keep = f.block_keep_mask.reshape(BN).float()

        def _accept(pred):
            match = (pred[:, 1:] == tgt[:, 1:]) & valid[:, 1:]
            accept = match.float().cumprod(dim=1).sum(dim=1)
            return (accept * keep).sum() / keep.sum().clamp(min=1.0)

        drafter_pred = fe.lm_head(h).argmax(dim=-1)
        drafter_accept = _accept(drafter_pred)

        window_hidden = window_mask = None
        if self.refiner.window_size > 0:
            window_hidden, window_mask = self._gather_window(
                hidden_states, f.anchor_positions, f.seq_len, B, n
            )

        # par-K latent Jacobi: iterate K passes in hidden space, lm_head once at the end
        anchor_prev = h[:, 0:1, :]
        prev = h
        refined = h
        for _ in range(self.latent_k):
            prev_shift = torch.cat([anchor_prev, prev[:, :-1, :]], dim=1)
            refined = self.refiner(h, g, prev_shift, window_hidden, window_mask)
            prev = refined
        pred = tgt.clone()
        pred[:, 1:] = fe.lm_head(refined[:, 1:, :]).argmax(dim=-1)
        refiner_accept = _accept(pred)

        return refiner_accept, drafter_accept
