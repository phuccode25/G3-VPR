import math
from typing import Dict, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


MATRYOSHKA_DIMS: Tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)


def _init_semi_orthogonal_linear(linear: nn.Linear) -> None:
    nn.init.orthogonal_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class MatryoshkaProjectionHead(nn.Module):
    """
    Hierarchical Multi-Scale Orthogonal Matryoshka Projection Head (v2 Compact, ~5.3M params).
    Keeps total G3-VPR v2 size (~35.8M) strictly smaller than D2VPR (38.24M) and 56.5% smaller
    than D2VPR + PCA4096 (82.28M), while preserving isometric spatial discriminability from Step 0!
    """

    def __init__(
        self,
        in_dim: int = 512,
        num_tokens: int = 17,
        max_dim: int = 4096,
        nested_dims: Sequence[int] = MATRYOSHKA_DIMS,
        region_dim: int = 192,
    ):
        super().__init__()
        assert max_dim == 4096, "MatryoshkaProjectionHead v2 is structured for max_dim=4096."
        assert region_dim % 4 == 0, "region_dim must be divisible by 4."
        self.in_dim = in_dim
        self.num_tokens = num_tokens
        self.max_dim = max_dim
        self.region_dim = region_dim
        self.nested_dims = tuple(sorted(d for d in nested_dims if d <= max_dim))

        # 1. Dedicated per-token orthogonal subspace [T, region_dim, in_dim]
        reg_w = torch.empty(num_tokens, region_dim, in_dim)
        for t in range(num_tokens):
            nn.init.orthogonal_(reg_w[t])
        self.region_weight = nn.Parameter(reg_w)
        self.region_bias = nn.Parameter(torch.zeros(num_tokens, region_dim))

        flat_reg_dim = num_tokens * region_dim  # 17 * 192 = 3264 (divisible by 4 -> 816 per group!)
        grp_in = flat_reg_dim // 4

        # Band 0 [0:128]: Global token (in_dim) + Mean spatial token (in_dim) -> 128
        self.band_128 = nn.Linear(in_dim * 2, 128)
        _init_semi_orthogonal_linear(self.band_128)

        # Band 1 [128:256]: 4 Medium Quadrant tokens -> 128
        n_quad = min(4, max(num_tokens - 1, 1))
        self.n_quad = n_quad
        self.band_256 = nn.Linear(n_quad * region_dim, 128)
        _init_semi_orthogonal_linear(self.band_256)

        # Band 2 [256:512]: Global + 3 Horizon/Facade tokens -> 256
        n_fac = min(4, num_tokens)
        self.n_fac = n_fac
        self.band_512 = nn.Linear(n_fac * region_dim, 256)
        _init_semi_orthogonal_linear(self.band_512)

        # Band 3 [512:1024]: 9 Fine 3x3 Grid tokens -> 512
        self.has_fine9 = num_tokens >= 14
        fine_in = 9 * region_dim if self.has_fine9 else flat_reg_dim
        self.band_1024 = nn.Linear(fine_in, 512)
        _init_semi_orthogonal_linear(self.band_1024)

        # Band 4 [1024:2048]: 4 interleaved spatial-channel groups -> 4 x 256 = 1024
        self.band_2048 = nn.ModuleList([nn.Linear(grp_in, 256) for _ in range(4)])
        for lin in self.band_2048:
            _init_semi_orthogonal_linear(lin)

        # Band 5 [2048:4096]: 4 cyclically-shifted spatial-channel groups -> 4 x 512 = 2048
        self.band_4096 = nn.ModuleList([nn.Linear(grp_in, 512) for _ in range(4)])
        for lin in self.band_4096:
            _init_semi_orthogonal_linear(lin)

        init_scales = torch.tensor([
            math.sqrt(128 / 4096),
            math.sqrt(128 / 4096),
            math.sqrt(256 / 4096),
            math.sqrt(512 / 4096),
            math.sqrt(1024 / 4096),
            math.sqrt(2048 / 4096),
        ], dtype=torch.float32)
        self.band_scales = nn.Parameter(init_scales)

    def forward(
        self,
        tokens: torch.Tensor,
        output_dim: int = 4096,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        # Compute multi-scale orthogonal projections in fp32 so AMP GradScaler never overflows F.normalize grads
        with torch.amp.autocast(device_type="cuda" if tokens.is_cuda else "cpu", enabled=False):
            B, T, _ = tokens.shape
            tokens_norm = F.normalize(tokens.float(), p=2, dim=-1)

            reg = torch.einsum("btd,trd->btr", tokens_norm, self.region_weight.float()) + self.region_bias.float().unsqueeze(0)
            reg = F.normalize(reg, p=2, dim=-1)

            cls_tok = tokens_norm[:, 0, :]
            mean_splat = F.normalize(tokens_norm[:, 1:, :].mean(dim=1), p=2, dim=-1) if T > 1 else cls_tok
            b0 = F.normalize(self.band_128(torch.cat([cls_tok, mean_splat], dim=-1)), p=2, dim=-1)

            quad_slice = reg[:, 1 : 1 + self.n_quad, :].reshape(B, -1) if T > 1 else reg.reshape(B, -1)
            b1 = F.normalize(self.band_256(quad_slice), p=2, dim=-1)

            if T >= 4:
                fac_tokens = torch.cat([reg[:, 0:1, :], reg[:, -3:, :]], dim=1).reshape(B, -1)
            else:
                fac_tokens = reg[:, : self.n_fac, :].reshape(B, -1)
            b2 = F.normalize(self.band_512(fac_tokens), p=2, dim=-1)

            if self.has_fine9:
                fine_tokens = reg[:, 5:14, :].reshape(B, -1)
            else:
                fine_tokens = reg.reshape(B, -1)
            b3 = F.normalize(self.band_1024(fine_tokens), p=2, dim=-1)

            grps = [reg[:, :, g::4].reshape(B, -1) for g in range(4)]
            b4 = F.normalize(torch.cat([self.band_2048[g](grps[g]) for g in range(4)], dim=-1), p=2, dim=-1)

            reg_rolled = torch.roll(reg, shifts=1, dims=-1)
            grps_r = [reg_rolled[:, :, g::4].reshape(B, -1) for g in range(4)]
            b5 = F.normalize(torch.cat([self.band_4096[g](grps_r[g]) for g in range(4)], dim=-1), p=2, dim=-1)

            scales = F.softplus(self.band_scales.float()) + 1e-3
            full_raw = torch.cat(
                [
                    scales[0] * b0,
                    scales[1] * b1,
                    scales[2] * b2,
                    scales[3] * b3,
                    scales[4] * b4,
                    scales[5] * b5,
                ],
                dim=-1,
            )

            nested_dict: Dict[int, torch.Tensor] = {}
            for d in self.nested_dims:
                nested_dict[d] = F.normalize(full_raw[:, :d], p=2, dim=-1)

            out_desc = nested_dict.get(output_dim, F.normalize(full_raw[:, :output_dim], p=2, dim=-1))
        return out_desc, nested_dict
