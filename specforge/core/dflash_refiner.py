# coding=utf-8
"""AR refiner on top of a frozen DFlash drafter.

This module is fully SEPARATE from the DFlash training code (`dflash.py`) — it only
*imports* and *reuses* it, never modifies it. The refiner consumes the frozen drafter's
per-block hidden states and autoregressively re-predicts the block, conditioned on the
previously (teacher-forced / generated) token.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from specforge.core.dflash import (
    OnlineDFlashModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft.dflash import apply_rotary_pos_emb


@dataclass
class BlockFeatures:
    """Intermediate products of the DFlash parallel forward, exposed for the refiner.

    B = batch, n = blocks per seq, block = block_size, H = hidden, V = vocab.
    """

    output_hidden: torch.Tensor  # (B, n*block, H)  drafter per-position hidden = h[k]
    target_ids: torch.Tensor  # (B, n, block)    real token at anchor+k ([:,:,0]=anchor)
    binary_mask: torch.Tensor  # (B, n, block)   0/1 validity (excl. anchor/OOB/non-asst/pad)
    anchor_positions: torch.Tensor  # (B, n)     absolute anchor index per block
    block_keep_mask: torch.Tensor  # (B, n)      which blocks are real (not padding)
    seq_len: int


class DFlashFeatureExtractor(OnlineDFlashModel):
    """A thin subclass of OnlineDFlashModel that exposes the parallel-pass intermediates.

    Inherits all the DFlash helpers (_sample_anchor_positions, _create_noise_embed,
    _create_position_ids, draft_model, lm_head, embed_tokens). `compute_block_features`
    is the body of OnlineDFlashModel.forward up to (but excluding) the loss reduction —
    numerically identical, just returns the tensors instead of a scalar loss.
    """

    @torch.no_grad()
    def compute_block_features(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> BlockFeatures:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n_blocks = anchor_positions.size(1)

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )

        # drafter per-position hidden states (the refiner's h[k])
        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )

        # --- same-position labels: position k -> token at anchor+k ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )

        # --- 0/1 validity mask: real block * in-bounds * exclude anchor(pos 0) * assistant ---
        binary_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        binary_mask = binary_mask * valid_label_mask.float()
        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        binary_mask = binary_mask * (pos_in_block > 0).float()
        loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        binary_mask = binary_mask * loss_mask_gathered

        return BlockFeatures(
            output_hidden=output_hidden,
            target_ids=target_ids,
            binary_mask=binary_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
        )


class ChannelWiseCausalMix(nn.Module):
    """Per-channel learned lower-triangular token mixing (causal gMLP / MLP-Mixer style).

    Replaces attention's q/k/v/o + softmax with a FIXED (learned) per-channel position mixing:
        u[b,k,c] = sum_{j<=k} L[c,k,j] * x[b,j,c]
    ~H*block*block params (tiny), one bmm, no softmax, no projections -> ~40x less weight to load
    than attention. NOT input-adaptive (the gate supplies that). Init to identity (starts as no-op).
    """

    def __init__(self, hidden_size, block_size):
        super().__init__()
        self.L = nn.Parameter(torch.eye(block_size).unsqueeze(0).repeat(hidden_size, 1, 1))
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):                                  # x: (BN, block, H)
        Lm = self.L * self.tril                            # (H, block, block) causal
        return torch.bmm(Lm, x.permute(2, 1, 0)).permute(2, 1, 0)   # (BN, block, H)


class CrossAttentionPool(nn.Module):
    """Learned global pool: each position cross-attends (NON-causal) to the SET of dflash
    outputs {h[j]} -> a per-position learned summary, replacing the crude mean-pool token."""

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", H // self.n_heads)
        self.q_proj = nn.Linear(H, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(H, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(H, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, H, bias=False)

    def forward(self, h):                                  # (BN, block, H)
        bn, block, _ = h.shape
        q = self.q_proj(h).view(bn, block, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bn, block, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bn, block, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)      # no mask -> non-causal over the whole set
        out = out.transpose(1, 2).reshape(bn, block, -1)
        return self.o_proj(out)


class RefinerLayer(nn.Module):
    """One refiner layer. mixer_type='attention' = Qwen3-style causal self-attn (+ optional window);
    mixer_type='sgu' = channel-wise lower-triangular causal mix (cheap, fixed, block-internal only)."""

    def __init__(self, config, mlp_intermediate=None, mixer_type="attention", block_size=16,
                 zero_init_oproj=False):
        super().__init__()
        H = config.hidden_size
        self.mixer_type = mixer_type
        self.input_layernorm = Qwen3RMSNorm(H, eps=config.rms_norm_eps)
        if mixer_type == "sgu":
            self.mix = ChannelWiseCausalMix(H, block_size)
            self.mix_out = nn.Linear(H, H, bias=False)
        else:
            self.head_dim = getattr(config, "head_dim", H // config.num_attention_heads)
            self.n_heads = config.num_attention_heads
            self.n_kv = config.num_key_value_heads
            self.q_proj = nn.Linear(H, self.n_heads * self.head_dim, bias=config.attention_bias)
            self.k_proj = nn.Linear(H, self.n_kv * self.head_dim, bias=config.attention_bias)
            self.v_proj = nn.Linear(H, self.n_kv * self.head_dim, bias=config.attention_bias)
            self.o_proj = nn.Linear(self.n_heads * self.head_dim, H, bias=config.attention_bias)
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if zero_init_oproj:
                # Start the attention sublayer as a no-op (out = residual + o_proj(attn) = residual).
                # The cross-position scramble grows in from 0 instead of injecting harmful random
                # mixing at step 0 -> avoids the per-position gate slamming shut (entry-side fix).
                nn.init.zeros_(self.o_proj.weight)
                if self.o_proj.bias is not None:
                    nn.init.zeros_(self.o_proj.bias)
        # MLP cost = 3 * H * intermediate; mlp_intermediate=None -> default; >0 -> shrink; <=0 -> drop.
        if mlp_intermediate is not None and mlp_intermediate <= 0:
            self.mlp = None
        else:
            self.post_attention_layernorm = Qwen3RMSNorm(H, eps=config.rms_norm_eps)
            if mlp_intermediate is None:
                self.mlp = Qwen3MLP(config)
            else:
                import copy
                cfg = copy.copy(config)
                cfg.intermediate_size = int(mlp_intermediate)
                self.mlp = Qwen3MLP(cfg)

    def forward(self, x, win_h, cos, sin, attn_mask):
        bn, q_len, _ = x.shape
        residual = x
        xn = self.input_layernorm(x)

        if self.mixer_type == "sgu":
            # channel-wise causal mix; ignores window/rope/mask (block-internal causal only)
            x = residual + self.mix_out(self.mix(xn))
        else:
            q = self.q_proj(xn).view(bn, q_len, -1, self.head_dim)
            q = self.q_norm(q).transpose(1, 2)
            kv_in = xn if win_h is None else torch.cat([win_h, xn], dim=1)
            kv_len = kv_in.size(1)
            k = self.k_proj(kv_in).view(bn, kv_len, -1, self.head_dim)
            k = self.k_norm(k).transpose(1, 2)
            v = self.v_proj(kv_in).view(bn, kv_len, -1, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if self.n_kv != self.n_heads:
                rep = self.n_heads // self.n_kv
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            attn = attn.transpose(1, 2).reshape(bn, q_len, -1)
            x = residual + self.o_proj(attn)

        if self.mlp is not None:
            residual = x
            x = self.post_attention_layernorm(x)
            x = self.mlp(x)
            x = residual + x
        return x


class RefinerDecoder(nn.Module):
    """AR refiner head. Consumes the drafter's per-block hiddens and produces refined
    hidden states for each block position.

    Inputs (all per (B*n) flattened blocks):
        h        : (BN, block, H)   drafter hidden h[k]
        g        : (BN, block, H)   block summary (mean-pooled, broadcast)
        prev_emb : (BN, block, H)   embedding of t_{k-1} (teacher-forced / generated)
        window_hidden : (BN, W, ctx_dim) | None   target hiddens of the last W context tokens
        window_mask   : (BN, W)      | None        1 = valid window token

    window_size = 0 disables the window (pure v1: only h+g+prev + block-causal attn).
    """

    def __init__(
        self,
        config,
        ctx_dim,
        window_size: int = 0,
        num_layers: int = 1,
        use_residual_gate: bool = False,
        residual_gate_init: float = 0.0,
        freeze_residual_gate: bool = False,
        mlp_intermediate=None,
        mixer_type: str = "attention",   # "attention" | "sgu" (channel-wise causal mix)
        pool_type: str = "mean",         # "mean" | "xattn" (cross-attention pool over dflash set)
        gate_type: str = "scalar",       # "scalar" (global ReZero) | "perpos" (input-dependent sigma(w.h))
        block_size: int = 16,
        zero_init_oproj: bool = False,   # zero-init attention o_proj (attention sublayer starts as no-op)
        gate_floor: float = 0.0,         # perpos gate floor: g = eps + (1-eps)*sigmoid(w.h), keeps g>=eps
    ):
        super().__init__()
        H = config.hidden_size
        self.window_size = window_size
        self.use_residual_gate = use_residual_gate
        self.mixer_type = mixer_type
        self.pool_type = pool_type
        self.gate_type = gate_type
        self.gate_floor = float(gate_floor)
        self.in_proj = nn.Linear(3 * H, H, bias=False)
        if window_size > 0:
            self.window_in_proj = nn.Linear(ctx_dim, H, bias=False)
            self.window_norm = Qwen3RMSNorm(H, eps=config.rms_norm_eps)
        if pool_type == "xattn":
            self.pool = CrossAttentionPool(config)         # learned global pool (replaces mean)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [RefinerLayer(config, mlp_intermediate=mlp_intermediate,
                          mixer_type=mixer_type, block_size=block_size,
                          zero_init_oproj=zero_init_oproj) for _ in range(num_layers)]
        )
        self.norm = Qwen3RMSNorm(H, eps=config.rms_norm_eps)
        # Residual gate: out = h + gate * correction.
        #   scalar: one ReZero scalar (global); perpos: sigma(w.h) per-position, input-dependent.
        if use_residual_gate:
            if gate_type == "perpos":
                self.gate_proj = nn.Linear(H, 1, bias=True)   # g[k] = sigmoid(w . h[k])
                nn.init.zeros_(self.gate_proj.weight)
                nn.init.constant_(self.gate_proj.bias, -5.0)  # sigmoid(-5) ~ 0 -> starts == drafter
            else:
                gate0 = torch.full((1,), float(residual_gate_init))
                if freeze_residual_gate:
                    self.register_buffer("residual_gate", gate0)
                else:
                    self.residual_gate = nn.Parameter(gate0)

    def forward(
        self,
        h: torch.Tensor,
        g: torch.Tensor,
        prev_emb: torch.Tensor,
        window_hidden: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bn, block, _ = h.shape
        device = h.device

        if self.pool_type == "xattn":
            g = self.pool(h)                                   # learned pool (ignores the passed mean g)

        x = self.in_proj(torch.cat([h, g, prev_emb], dim=-1))  # (BN, block, H)

        win_h, cos, sin, attn_mask = None, None, None, None
        if self.mixer_type == "attention":
            W = 0
            if self.window_size > 0 and window_hidden is not None:
                win_h = self.window_norm(self.window_in_proj(window_hidden))
                W = win_h.size(1)
            kv_len = W + block
            position_ids = torch.arange(kv_len, device=device).unsqueeze(0).expand(bn, -1)
            cos, sin = self.rotary_emb(x, position_ids)
            causal = torch.tril(torch.ones(block, block, dtype=torch.bool, device=device))
            if W > 0:
                win_vis = window_mask.bool()[:, None, :].expand(bn, block, W)
                mask = torch.cat([win_vis, causal[None].expand(bn, block, block)], dim=-1)
            else:
                mask = causal[None].expand(bn, block, block)
            attn_mask = mask[:, None, :, :]

        for layer in self.layers:
            x = layer(x, win_h, cos, sin, attn_mask)
        out = self.norm(x)
        if self.use_residual_gate:
            # out = h + gate * correction; gate starts ~0 => == drafter
            if self.gate_type == "perpos":
                gate = torch.sigmoid(self.gate_proj(h))        # (BN, block, 1) input-dependent
                if self.gate_floor > 0.0:
                    # g = eps + (1-eps)*sigma; g never reaches 0, so the mixer's gradient
                    # (d loss / d Mix ~ g) is never throttled -> breaks the gate-collapse trap.
                    gate = self.gate_floor + (1.0 - self.gate_floor) * gate
            else:
                gate = self.residual_gate
            out = h + gate * out
        return out


class OnlineDFlashRefiner(nn.Module):
    """Training wrapper: frozen DFlash drafter + trainable AR refiner.

    forward(input_ids, hidden_states, loss_mask) -> (loss, accuracy).
    Only `self.refiner` has trainable params; the feature extractor (drafter + target
    lm_head + embed) is frozen.
    """

    def __init__(
        self,
        feature_extractor: DFlashFeatureExtractor,
        window_size: int = 0,
        num_refiner_layers: int = 1,
        use_residual_gate: bool = False,
        residual_gate_init: float = 0.0,
        freeze_residual_gate: bool = False,
        loss_decay_gamma: float = None,
        mlp_intermediate=None,
        mixer_type: str = "attention",
        pool_type: str = "mean",
        gate_type: str = "scalar",
        zero_init_oproj: bool = False,
        gate_floor: float = 0.0,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.block_size = feature_extractor.block_size
        self.loss_decay_gamma = loss_decay_gamma

        config = feature_extractor.draft_model.config
        ctx_dim = feature_extractor.draft_model.fc.in_features  # len(target_layer_ids)*H
        self.refiner = RefinerDecoder(
            config,
            ctx_dim,
            window_size=window_size,
            num_layers=num_refiner_layers,
            use_residual_gate=use_residual_gate,
            residual_gate_init=residual_gate_init,
            freeze_residual_gate=freeze_residual_gate,
            mlp_intermediate=mlp_intermediate,
            mixer_type=mixer_type,
            pool_type=pool_type,
            gate_type=gate_type,
            block_size=self.block_size,
            zero_init_oproj=zero_init_oproj,
            gate_floor=gate_floor,
        )

        # freeze everything except the refiner
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def _gather_window(self, hidden_states, anchor_positions, seq_len, B, n):
        """Gather the W target hiddens before each anchor: [a-W, ..., a-1]."""
        W = self.refiner.window_size
        device = hidden_states.device
        ctx_dim = hidden_states.size(-1)
        offs = torch.arange(W, device=device).view(1, 1, W)
        widx = anchor_positions.unsqueeze(-1) - W + offs  # (B, n, W) = a-W .. a-1
        wvalid = widx >= 0  # out-of-bounds (near sequence start) -> masked
        safe = widx.clamp(0, seq_len - 1)
        gather_idx = safe.reshape(B, n * W, 1).expand(B, n * W, ctx_dim)
        window_hidden = torch.gather(hidden_states, 1, gather_idx).view(
            B, n, W, ctx_dim
        )
        return window_hidden.reshape(B * n, W, ctx_dim), wvalid.reshape(B * n, W)

    def forward(self, input_ids, hidden_states, loss_mask):
        fe = self.feature_extractor
        f = fe.compute_block_features(input_ids, hidden_states, loss_mask)  # frozen, no_grad

        B = f.output_hidden.size(0)
        H = f.output_hidden.size(-1)
        block = self.block_size
        n = f.output_hidden.size(1) // block
        BN = B * n

        # h[k], g (mean over the block), prev-token embedding (teacher forcing)
        h = f.output_hidden.view(B, n, block, H).reshape(BN, block, H)
        g = h.mean(dim=1, keepdim=True).expand(-1, block, -1)

        tgt = f.target_ids.reshape(BN, block)
        prev_tok = tgt.roll(shifts=1, dims=1)
        # slot 0 input = real token at anchor-1 (standard AR shift; no anchor duplication)
        am1_idx = (f.anchor_positions - 1).clamp(min=0)
        prev_tok[:, 0] = torch.gather(input_ids, 1, am1_idx).reshape(BN)
        prev_emb = fe.embed_tokens(prev_tok)

        window_hidden = window_mask = None
        if self.refiner.window_size > 0:
            window_hidden, window_mask = self._gather_window(
                hidden_states, f.anchor_positions, f.seq_len, B, n
            )

        refined = self.refiner(h, g, prev_emb, window_hidden, window_mask)
        logits = fe.lm_head(refined)  # (BN, block, V)

        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_tgt = tgt.reshape(-1)
        valid = f.binary_mask  # (B, n, block) 0/1
        w = valid
        if self.loss_decay_gamma:
            # weight earlier block positions more (prefix acceptance matters most):
            # weight_k = exp(-(k-1)/gamma)
            kk = torch.arange(self.block_size, device=valid.device)
            decay = torch.exp(-(kk - 1).clamp(min=0).float() / self.loss_decay_gamma)
            w = w * decay.view(1, 1, -1)
        flat_w = w.reshape(-1)
        flat_valid = valid.reshape(-1)

        loss_per = F.cross_entropy(flat_logits, flat_tgt, reduction="none")
        loss = (loss_per * flat_w).sum() / (flat_w.sum() + 1e-6)

        with torch.no_grad():
            # accuracy stays over the 0/1 valid mask (not the decayed weight)
            pred = flat_logits.argmax(dim=-1)
            correct = (pred == flat_tgt) & (flat_valid > 0.5)
            accuracy = correct.sum().float() / (flat_valid.sum() + 1e-6)

        return loss, accuracy

    @torch.no_grad()
    def accept_lengths(self, input_ids, hidden_states, loss_mask):
        """Mean accepted draft length for (refiner, drafter) on the SAME sampled blocks.

        - drafter: frozen DFlash parallel argmax (the baseline the refiner must beat).
        - refiner: free-running greedy AR (feeds its OWN predictions forward) -> the honest
          inference proxy, unlike the optimistic teacher-forced training accuracy.
        Accept length = leading run of block slots (1..L-1) whose argmax matches the
        ground-truth token (cumprod prefix), averaged over real blocks.
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
        tgt = f.target_ids.reshape(BN, block)  # [:,0] = real anchor

        anchors = f.anchor_positions.reshape(BN, 1)
        pos_abs = anchors + torch.arange(block, device=device).view(1, block)
        valid = pos_abs < f.seq_len  # [BN, block] in-bounds
        keep = f.block_keep_mask.reshape(BN).float()

        def _accept(pred):
            match = (pred[:, 1:] == tgt[:, 1:]) & valid[:, 1:]
            accept = match.float().cumprod(dim=1).sum(dim=1)
            return (accept * keep).sum() / keep.sum().clamp(min=1.0)

        # drafter baseline: parallel argmax, no AR
        drafter_pred = fe.lm_head(h).argmax(dim=-1)  # [BN, block]
        drafter_accept = _accept(drafter_pred)

        # refiner: free-running greedy AR (slot 0 stays the real anchor)
        window_hidden = window_mask = None
        if self.refiner.window_size > 0:
            window_hidden, window_mask = self._gather_window(
                hidden_states, f.anchor_positions, f.seq_len, B, n
            )
        # slot 0 input = real token at anchor-1 (matches forward / version2)
        am1_idx = (f.anchor_positions - 1).clamp(min=0)
        tok_am1 = torch.gather(input_ids, 1, am1_idx).reshape(BN)
        pred = tgt.clone()
        for k in range(1, block):
            prev_tok = pred.roll(shifts=1, dims=1)
            prev_tok[:, 0] = tok_am1
            prev_emb = fe.embed_tokens(prev_tok)
            refined = self.refiner(h, g, prev_emb, window_hidden, window_mask)
            pred[:, k] = fe.lm_head(refined[:, k, :]).argmax(dim=-1)
        refiner_accept = _accept(pred)

        return refiner_accept, drafter_accept
