import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_default_anchors(num_splats: int = 16) -> torch.Tensor:
    """
    Returns [K, 4] anchor priors (cx, cy, sx, sy) in normalized [0, 1] image coordinates.
    Default K=16:
      - Indices 0..3:  4 medium quadrants (2x2)
      - Indices 4..12: 9 fine regions (3x3)
      - Indices 13..15: 3 perspective horizon/facade structural bands (left-mid-right street-canyon prior)
    """
    anchors: List[Tuple[float, float, float, float]] = []
    # 4 medium 2x2 quadrants
    for cy in (0.25, 0.75):
        for cx in (0.25, 0.75):
            anchors.append((cx, cy, 0.22, 0.22))
    # 9 fine 3x3 grid cells
    centers_3 = (1.0 / 6.0, 0.5, 5.0 / 6.0)
    for cy in centers_3:
        for cx in centers_3:
            anchors.append((cx, cy, 0.14, 0.14))
    # 3 upper-middle facade/skyline perspective bands (critical for urban & historical VPR)
    for cx in (0.25, 0.50, 0.75):
        anchors.append((cx, 0.38, 0.20, 0.12))

    if num_splats <= len(anchors):
        return torch.tensor(anchors[:num_splats], dtype=torch.float32)

    g = int(math.ceil(math.sqrt(num_splats)))
    extra = []
    for i in range(num_splats):
        r, c = divmod(i, g)
        extra.append(((c + 0.5) / g, (r + 0.5) / g, 0.5 / g, 0.5 / g))
    return torch.tensor(extra, dtype=torch.float32)


class GaussianPoseEncoder(nn.Module):
    """Encodes 2D Anisotropic Gaussian geometry + 3D depth/salience statistics into token space."""

    def __init__(self, feat_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        # 8 geometric descriptors: [cx, cy, sx, sy, cos(2*theta), sin(2*theta), mean_depth, mean_salience]
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, geom_vec: torch.Tensor) -> torch.Tensor:
        return self.mlp(geom_vec)


class Vectorized3DGaussianAggregator(nn.Module):
    """
    3D-Guided Deformable Anisotropic Gaussian Splatting Aggregator (3D-DGSA v2).

    Improvements in v2:
      1. Symmetric Signed GeM Pooling (pos_gem - neg_gem) eliminating positive DC bias on LayerNorm features.
      2. Learnable gated pose injection (pose_scale initialized small) preventing static anchor coordinates
         from overwhelming L2-normalized visual appearance features.
      3. Unit-norm regional token preservation across all (K + 1) splat tokens.
    """

    def __init__(
        self,
        feat_dim: int = 512,
        num_splats: int = 16,
        gem_p: float = 3.0,
        allow_rotation: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_splats = num_splats
        self.allow_rotation = allow_rotation

        anchors = _build_default_anchors(num_splats)  # [K, 4]
        self.register_buffer("anchors", anchors)

        # Single-pass spatial-global context encoder (computed ONCE per image, not 13 times!)
        self.spatial_ctx = nn.Sequential(
            nn.Conv2d(feat_dim + 2, feat_dim // 2, kernel_size=1, bias=False),
            nn.GroupNorm(16, feat_dim // 2),
            nn.GELU(),
            nn.Conv2d(feat_dim // 2, feat_dim // 2, kernel_size=3, padding=1, groups=feat_dim // 2, bias=False),
            nn.GroupNorm(16, feat_dim // 2),
            nn.GELU(),
        )
        # Predict dense 5-channel Gaussian deformation field: [dx, dy, d_log_sx, d_log_sy, raw_theta]
        self.param_field = nn.Conv2d(feat_dim // 2 + feat_dim, 5, kernel_size=1, bias=True)
        nn.init.zeros_(self.param_field.weight)
        nn.init.zeros_(self.param_field.bias)

        self.gem_p = nn.Parameter(torch.tensor(float(gem_p)))
        self.pose_encoder = GaussianPoseEncoder(feat_dim=feat_dim, hidden_dim=128)
        self.pose_scale = nn.Parameter(torch.tensor(0.05))
        self.global_gate = nn.Parameter(torch.tensor(0.10))

        # Hierarchical splat-to-global refinement
        self.global_refine = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(
        self,
        patch_map: torch.Tensor,
        cls_token: torch.Tensor,
        salience_gate: torch.Tensor,
        pseudo_depth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, D, H_p, W_p = patch_map.shape
        K = self.num_splats
        device = patch_map.device
        dtype = patch_map.dtype

        # 1. Single-pass deformation field prediction
        depth_norm = pseudo_depth / (pseudo_depth.amax(dim=(2, 3), keepdim=True).clamp(min=1e-4))
        local_ctx = self.spatial_ctx(torch.cat([patch_map, salience_gate, depth_norm], dim=1))
        cls_2d = cls_token.view(B, D, 1, 1).expand(-1, -1, H_p, W_p)
        dense_params = self.param_field(torch.cat([local_ctx, cls_2d], dim=1))  # [B, 5, H_p, W_p]

        # 2. Sample anchor centers in EXACT [-1, 1] normalized coordinates (fixing D2VPR [0,1] bug!)
        anchor_centers = self.anchors[:, :2].to(device=device, dtype=dtype)     # [K, 2] in [0, 1]
        anchor_scales = self.anchors[:, 2:].to(device=device, dtype=dtype)      # [K, 2] in [0, 1]
        sample_grid = (anchor_centers * 2.0 - 1.0).view(1, K, 1, 2).expand(B, -1, -1, -1)  # [B, K, 1, 2]

        sampled = F.grid_sample(
            dense_params,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1).permute(0, 2, 1)  # [B, K, 5]

        dx_dy = 0.22 * torch.tanh(sampled[..., 0:2])
        scale_mul = 0.5 + 1.5 * torch.sigmoid(sampled[..., 2:4])
        theta = (math.pi / 3.0) * torch.tanh(sampled[..., 4]) if self.allow_rotation else torch.zeros_like(sampled[..., 4])

        mu = (anchor_centers.unsqueeze(0) + dx_dy).clamp(0.05, 0.95)                # [B, K, 2]
        sigma = (anchor_scales.unsqueeze(0) * scale_mul).clamp(0.04, 0.48)          # [B, K, 2]

        # 3. Vectorized Anisotropic 2D Gaussian Splatting over the (H_p, W_p) patch grid
        gy = torch.linspace(0.5 / H_p, 1.0 - 0.5 / H_p, H_p, device=device, dtype=dtype)
        gx = torch.linspace(0.5 / W_p, 1.0 - 0.5 / W_p, W_p, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, H_p * W_p, 2)     # [1, 1, N_p, 2]

        diff = coords - mu.unsqueeze(2)                                              # [B, K, N_p, 2]
        cos_t = torch.cos(theta).unsqueeze(-1)                                       # [B, K, 1]
        sin_t = torch.sin(theta).unsqueeze(-1)                                       # [B, K, 1]

        dx_rot = cos_t * diff[..., 0] + sin_t * diff[..., 1]
        dy_rot = -sin_t * diff[..., 0] + cos_t * diff[..., 1]

        sx = sigma[..., 0].unsqueeze(-1)
        sy = sigma[..., 1].unsqueeze(-1)
        mahal_sq = (dx_rot / sx).square() + (dy_rot / sy).square()                   # [B, K, N_p]

        gauss_weights = torch.exp(-0.5 * mahal_sq)                                   # [B, K, N_p]
        salience_flat = salience_gate.flatten(2)                                     # [B, 1, N_p]
        combined_weights = gauss_weights * (0.15 + 0.85 * salience_flat)
        norm_weights = combined_weights / combined_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, K, N_p]

        # 4. Zero-DC Symmetric Signed Generalized Mean (GeM) Pooling across all K splats in fp32
        # Add eps BEFORE .pow(p_exp) so d/d(gem_p)[base^p = base^p * log(base)] never sees base=0 -> NaN!
        with torch.amp.autocast(device_type="cuda" if patch_map.is_cuda else "cpu", enabled=False):
            patch_f32 = F.normalize(patch_map.float().flatten(2), p=2, dim=1)        # [B, D, N_p]
            nw_f32 = norm_weights.float()
            p_exp = self.gem_p.float().clamp(min=1.0, max=5.0)
            eps = 1e-5

            pos_pow = (F.relu(patch_f32) + eps).pow(p_exp)
            neg_pow = (F.relu(-patch_f32) + eps).pow(p_exp)
            gem_pos = torch.einsum("bkn,bdn->bkd", nw_f32, pos_pow).clamp(min=eps ** 5).pow(1.0 / p_exp) - eps
            gem_neg = torch.einsum("bkn,bdn->bkd", nw_f32, neg_pow).clamp(min=eps ** 5).pow(1.0 / p_exp) - eps
            splat_feats_f32 = F.normalize(gem_pos - gem_neg, p=2, dim=-1)
        splat_feats = splat_feats_f32.to(dtype)                                      # [B, K, D]

        # 5. Gated Geometric Pose Injection (preserves 100% visual discriminability at init)
        depth_flat = depth_norm.flatten(2)
        mean_depth = (norm_weights * depth_flat).sum(dim=-1, keepdim=True)
        mean_sal = (norm_weights * salience_flat).sum(dim=-1, keepdim=True)

        geom_vec = torch.cat(
            [
                mu,
                sigma,
                torch.cos(2.0 * theta).unsqueeze(-1),
                torch.sin(2.0 * theta).unsqueeze(-1),
                mean_depth,
                mean_sal,
            ],
            dim=-1,
        )  # [B, K, 8]
        pose_emb = F.normalize(self.pose_encoder(geom_vec).float(), p=2, dim=-1)
        p_scale = torch.tanh(self.pose_scale.float())
        splat_feats_f32 = F.normalize(splat_feats_f32 + p_scale * pose_emb, p=2, dim=-1)

        # 6. Global token refinement from salience-weighted splats (fp32 unit-norm safe)
        ms_f32 = mean_sal.float()
        splat_summary = F.normalize(
            (splat_feats_f32 * ms_f32).sum(dim=1) / ms_f32.sum(dim=1).clamp(min=1e-6), p=2, dim=-1
        )
        cls_norm = F.normalize(cls_token.float(), p=2, dim=-1)
        refined_delta = F.normalize(
            self.global_refine(torch.cat([cls_norm, splat_summary], dim=-1).to(dtype)).float(), p=2, dim=-1
        )
        global_token_f32 = F.normalize(
            cls_norm + torch.tanh(self.global_gate.float()) * refined_delta, p=2, dim=-1
        ).unsqueeze(1)

        tokens = torch.cat([global_token_f32, splat_feats_f32], dim=1).to(dtype)     # [B, K + 1, D]

        return {
            "tokens": tokens,                    # [B, K + 1, D]
            "splat_feats": splat_feats_f32.to(dtype),  # [B, K, D]
            "mu": mu,                            # [B, K, 2]
            "sigma": sigma,                      # [B, K, 2]
            "theta": theta,                      # [B, K]
            "mean_depth": mean_depth,            # [B, K, 1]
            "mean_sal": mean_sal,                # [B, K, 1]
            "splat_weights": norm_weights,       # [B, K, N_p]
        }
