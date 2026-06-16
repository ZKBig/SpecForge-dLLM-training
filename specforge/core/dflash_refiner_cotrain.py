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

from specforge.core.dflash_refiner import (
    DFlashFeatureExtractor,
    OnlineDFlashRefiner,
)


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
        super().__init__(*args, **kwargs)
        n = 0
        for p in self.feature_extractor.draft_model.parameters():
            p.requires_grad = True
            n += p.numel()
        self._cotrain_drafter_params = n
