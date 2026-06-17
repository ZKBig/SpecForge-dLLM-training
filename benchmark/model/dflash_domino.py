"""Self-contained Domino-capable DFlash drafter for dflash_old2.

dflash_old2's inference `DFlashDraftModel` (cached forward + spec_generate) is byte-identical
to specforge-version2's backbone (same layers/fc/norm/hidden_norm). The ONLY thing it lacks is
the Domino head (a GRU + a logit-correction MLP). This subclass adds exactly that head, matching
version2's `DFlashDraftModel(projector_type="domino")` construction, so a Domino checkpoint trained
there loads cleanly with `DominoDraftModel.from_pretrained(ckpt_dir)` — no specforge dependency.

Head (active only when dflash_config.projector_type == "domino"):
    prefix_gru : nn.GRU(hidden_size -> gru_hidden_dim)        # causal AR state over prev tokens
    embed_proj : Linear(hidden+gru_hidden -> emb_dim) -> SiLU -> Linear(emb_dim -> vocab)  # logit correction
"""
from torch import nn

from .dflash import DFlashDraftModel


class DominoDraftModel(DFlashDraftModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        dfc = getattr(config, "dflash_config", None) or {}
        self.projector_type = dfc.get("projector_type", None)
        self.pure_draft_prefix_len = dfc.get("pure_draft_prefix_len", 0)
        self.shift_label = dfc.get("shift_label", False)

        if self.projector_type == "domino":
            self.emb_dim = dfc["emb_dim"]
            self.gru_hidden_dim = dfc["gru_hidden_dim"]
            self.prefix_gru = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=self.gru_hidden_dim,
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            in_dim = config.hidden_size + self.gru_hidden_dim
            self.embed_proj = nn.Sequential(
                nn.Linear(in_dim, self.emb_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.emb_dim, config.vocab_size, bias=False),
            )
        elif self.projector_type is not None:
            raise ValueError(f"Unknown draft projector_type: {self.projector_type}")
