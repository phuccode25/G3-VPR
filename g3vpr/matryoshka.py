from typing import Dict, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


MATRYOSHKA_DIMS: Tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)


class MatryoshkaProjectionHead(nn.Module):
    """
    All-Token-Aware Nested Matryoshka Projection Head.
    Compresses (K + 1) calibrated Gaussian splat tokens [B, T, D] into an ordered 4096-D descriptor
    such that every prefix z[:, :d] for d in {128, 256, 512, 1024, 2048, 4096} forms a complete,
    L2-normalized global VPR descriptor without requiring any offline PCA matrix.
    """

    def __init__(
        self,
        in_dim: int = 512,
        num_tokens: int = 17,
        max_dim: int = 4096,
        nested_dims: Sequence[int] = MATRYOSHKA_DIMS,
    ):
        super().__init__()
        assert max_dim % 256 == 0, "max_dim must be a multiple of 256."
        self.in_dim = in_dim
        self.num_tokens = num_tokens
        self.max_dim = max_dim
        self.num_bands = max_dim // 256  # 16 spectral bands x 256 = 4096
        self.nested_dims = tuple(sorted(d for d in nested_dims if d <= max_dim))

        # Feature projection per token
        self.feat_proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # Mix all (K + 1) spatial/global tokens into 16 ordered coarse-to-fine spectral bands
        self.token_mixer = nn.Sequential(
            nn.Linear(num_tokens, num_tokens * 2),
            nn.GELU(),
            nn.Linear(num_tokens * 2, self.num_bands),
        )

        # Causal coarse-to-fine band refinement (band b only depends on bands <= b so prefixes stay pure)
        self.band_ffn = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )
        # Dedicated ordering projection on Band 0 so [0:128] captures the highest-priority global signature
        self.tier1_head = nn.Linear(256, 256)

    def forward(
        self,
        tokens: torch.Tensor,
        output_dim: int = 4096,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        # tokens: [B, T, D]
        B, T, _ = tokens.shape
        x = self.feat_proj(tokens)                        # [B, T, 256]
        bands = self.token_mixer(x.transpose(1, 2)).transpose(1, 2).contiguous()  # [B, 16, 256]
        bands = bands + self.band_ffn(bands)

        band0 = self.tier1_head(bands[:, 0:1, :])
        ordered_bands = torch.cat([band0, bands[:, 1:, :]], dim=1)                # [B, 16, 256]
        full_raw = ordered_bands.reshape(B, self.max_dim)                         # [B, 4096]

        nested_dict: Dict[int, torch.Tensor] = {}
        for d in self.nested_dims:
            nested_dict[d] = F.normalize(full_raw[:, :d], p=2, dim=-1)

        out_desc = nested_dict.get(output_dim, F.normalize(full_raw[:, :output_dim], p=2, dim=-1))
        return out_desc, nested_dict
