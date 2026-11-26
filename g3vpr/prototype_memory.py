from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeInteractionLayer(nn.Module):
    """
    Single per-image calibration block with DINOv2-style LayerScale gates:
      1) Intra-image Self-Attention across the (K + 1) Gaussian splat tokens (strictly within each image!).
      2) Cross-Attention from image splat tokens (Queries) to M Learnable Geo-Context Prototypes (Keys/Values).
      3) Lightweight GELU FFN with gated residual preservation.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.05):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.sa_gamma = nn.Parameter(torch.full((dim,), 0.05))

        self.norm_ca_q = nn.LayerNorm(dim)
        self.norm_ca_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ca_gamma = nn.Parameter(torch.full((dim,), 0.05))

        hidden = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.ffn_gamma = nn.Parameter(torch.full((dim,), 0.05))

    def forward(self, tokens: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, D], prototypes: [B, M, D] — strictly batch_first=True (zero cross-batch leakage!)
        q_sa = self.norm_sa(tokens)
        sa_out, _ = self.self_attn(q_sa, q_sa, q_sa, need_weights=False)
        tokens = tokens + self.sa_gamma * sa_out

        q_ca = self.norm_ca_q(tokens)
        kv_ca = self.norm_ca_kv(prototypes)
        ca_out, _ = self.cross_attn(q_ca, kv_ca, kv_ca, need_weights=False)
        tokens = tokens + self.ca_gamma * ca_out

        tokens = tokens + self.ffn_gamma * self.ffn(self.norm_ffn(tokens))
        return tokens


class PrototypeMemoryCrossInteraction(nn.Module):
    """
    Prototype Memory Cross-Interaction (PMCI v2).
    Replaces D2VPR's 11.03M-parameter batch-dependent cross_image_encoder with a compact
    Learnable Geo-Prototype Memory Bank that operates identically at batch_size = 1, 8, or 128,
    while preserving unit-sphere regional discriminability from Step 0.
    """

    def __init__(
        self,
        dim: int = 512,
        num_prototypes: int = 128,
        num_layers: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.dim = dim
        self.num_prototypes = num_prototypes

        proto = torch.empty(num_prototypes, dim)
        nn.init.orthogonal_(proto)
        self.prototypes = nn.Parameter(proto)

        self.layers = nn.ModuleList(
            [
                PrototypeInteractionLayer(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

    def get_normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, p=2, dim=-1)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = tokens.shape[0]
        proto_expanded = self.prototypes.unsqueeze(0).expand(B, -1, -1)
        out = tokens
        for layer in self.layers:
            out = layer(out, proto_expanded)
        out = F.normalize(out.float(), p=2, dim=-1).to(tokens.dtype)
        return {
            "tokens": out,
            "prototypes_norm": self.get_normalized_prototypes(),
        }
