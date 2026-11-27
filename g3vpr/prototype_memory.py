from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class NystromPrototypeLayer(nn.Module):
    """
    Nystrom-PMCI (v3): Batch-Invariant Nystrom Manifold Whitening Layer.
      1) Intra-image Self-Attention across the (K + 1) Gaussian splat tokens (strictly per-image).
      2) Nystrom Landmark Kernel Projection onto M learnable geo-context landmarks U (nuisance basis)
         and invariant value bank V (structural basis):
           A = softmax(Q(x) K(U)^T / sqrt(d))
           x' = LN(x - alpha * W_nuis(A U) + beta * W_inv(A V))
         Explicitly subtracts domain/weather/color nuisance components and reinforces
         structural landmarks with zero cross-batch dependency (100% BS=1 invariant).
      3) Lightweight GELU FFN with gated residual preservation.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.05):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.sa_gamma = nn.Parameter(torch.full((dim,), 0.05))

        # Nystrom nuisance-subtraction & invariant-reinforcement projections
        self.norm_q = nn.LayerNorm(dim)
        self.norm_u = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.w_nuis = nn.Linear(dim, dim, bias=False)
        self.w_inv = nn.Linear(dim, dim, bias=False)
        self.nuis_alpha = nn.Parameter(torch.full((dim,), 0.05))
        self.inv_beta = nn.Parameter(torch.full((dim,), 0.05))
        self.post_nystrom_norm = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(dropout)

        hidden = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.ffn_gamma = nn.Parameter(torch.full((dim,), 0.05))

    def forward(
        self,
        tokens: torch.Tensor,
        prototypes_u: torch.Tensor,
        prototypes_v: torch.Tensor,
    ):
        # tokens: [B, T, D], prototypes_u/v: [B, M, D]
        B, T, D = tokens.shape
        M = prototypes_u.shape[1]

        # 1. Intra-image Self-Attention (strictly batch_first=True, zero cross-batch leakage)
        q_sa = self.norm_sa(tokens)
        sa_out, _ = self.self_attn(q_sa, q_sa, q_sa, need_weights=False)
        tokens = tokens + self.sa_gamma * sa_out

        # 2. Nystrom Kernel Attention A(x, U): [B, H, T, M] -> averaged assignment [B, T, M]
        q = self.q_proj(self.norm_q(tokens)).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.norm_u(prototypes_u)).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, M]
        attn_probs = F.softmax(attn_logits.float(), dim=-1).to(tokens.dtype)
        proto_assign = attn_probs.mean(dim=1)  # [B, T, M] for SwAV Sinkhorn equipartition loss

        attn_d = self.attn_drop(proto_assign)
        nuis_proj = self.w_nuis(torch.matmul(attn_d, prototypes_u))
        inv_proj = self.w_inv(torch.matmul(attn_d, prototypes_v))

        tokens = self.post_nystrom_norm(tokens - self.nuis_alpha * nuis_proj + self.inv_beta * inv_proj)

        # 3. Gated FFN
        tokens = tokens + self.ffn_gamma * self.ffn(self.norm_ffn(tokens))
        return tokens, proto_assign


# Backwards-compatible alias
PrototypeInteractionLayer = NystromPrototypeLayer


class PrototypeMemoryCrossInteraction(nn.Module):
    """
    Nystrom-PMCI (v3): Batch-Invariant Nystrom Manifold Whitening.
    Replaces D2VPR's 11.03M-parameter batch-dependent cross_image_encoder with a compact
    Nystrom landmark operator (U: nuisance manifold, V: invariant structural bank) that
    operates identically at batch_size = 1, 8, or 128, with SwAV equipartition support.
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

        proto_u = torch.empty(num_prototypes, dim)
        nn.init.orthogonal_(proto_u)
        self.prototypes = nn.Parameter(proto_u)

        proto_v = torch.empty(num_prototypes, dim)
        nn.init.orthogonal_(proto_v)
        self.prototypes_inv = nn.Parameter(proto_v)

        self.layers = nn.ModuleList(
            [
                NystromPrototypeLayer(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

    def get_normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, p=2, dim=-1)

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = tokens.shape[0]
        u_exp = self.prototypes.unsqueeze(0).expand(B, -1, -1)
        v_exp = self.prototypes_inv.unsqueeze(0).expand(B, -1, -1)
        out = tokens
        last_assign = None
        for layer in self.layers:
            out, last_assign = layer(out, u_exp, v_exp)
        out = F.normalize(out.float(), p=2, dim=-1).to(tokens.dtype)
        return {
            "tokens": out,
            "prototypes_norm": self.get_normalized_prototypes(),
            "proto_assign": last_assign,
        }
