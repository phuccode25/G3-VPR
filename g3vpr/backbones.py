import math
from typing import Dict, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.vision_transformer_small import vit_small


def _patch_adapter_dynamic_hw(adapter: nn.Module) -> None:
    """Allow MulConvAdapter to handle arbitrary (H_p, W_p) patch grids, not just square images."""
    def dynamic_forward(x: torch.Tensor) -> torch.Tensor:
        x0 = adapter.D_fc1(x)
        B, P, D = x0.shape
        grid_hw: Optional[Tuple[int, int]] = getattr(adapter, "_grid_hw", None)
        if grid_hw is not None and grid_hw[0] * grid_hw[1] == P - 1:
            H_p, W_p = grid_hw
        else:
            H_p = W_p = int(math.sqrt(P - 1))

        x0 = F.relu(x0, inplace=True)
        xs = x0[:, 1:, :].reshape(B, H_p, W_p, D).permute(0, 3, 1, 2).contiguous()
        b1 = adapter.branch1(xs)
        b2 = adapter.branch2(xs)
        b3 = adapter.branch3(xs)
        outputs = torch.cat([b1, b2, b3], dim=1)
        outputs = outputs.reshape(B, D, H_p * W_p).permute(0, 2, 1).contiguous()
        clstoken = x0[:, 0:1, :]
        outputs = torch.cat([clstoken, outputs], dim=1) + x0
        outputs = adapter.D_fc2(outputs)
        if adapter.skip_connect:
            outputs = outputs + x
        return outputs

    adapter.forward = dynamic_forward


class DepthwiseSeparableHead(nn.Module):
    """Lightweight 3D geometric salience & scale-invariant pseudo-depth predictor."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, bias=True),
        )
        # Initialize salience bias slightly positive (~0.73) so early training preserves static cues
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([1.0, 0.0]))

    def forward(self, feat_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(feat_2d)
        salience_gate = torch.sigmoid(out[:, 0:1])  # (0, 1) static landmark confidence
        pseudo_depth = F.softplus(out[:, 1:2])      # > 0 scale-invariant depth proxy
        return salience_gate, pseudo_depth


class GeoSemanticGatedBackbone(nn.Module):
    """
    ViT-S/14 + MulConvAdapter with Geometry-Semantic Gated Distillation Recovery Module (Gated-DRM).
    Separates shallow structural layers [2, 4] (3D geometry & static salience) from deep semantic
    layers [8, 12] (illumination/season-invariant place semantics).
    """

    def __init__(
        self,
        out_dim: int = 512,
        shallow_layers: Sequence[int] = (2, 4),
        deep_layers: Sequence[int] = (8, 12),
        use_gated_drm: bool = True,
        pretrained_dinov2_path: Optional[str] = None,
        warmstart_d2vpr_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.vit = vit_small(patch_size=14, img_size=518, init_values=1, block_chunks=0)
        # Remove unused static linear projection from original D2VPR vision_transformer_small
        if hasattr(self.vit, "lin"):
            del self.vit.lin
        if hasattr(self.vit, "up_norm"):
            del self.vit.up_norm
        if hasattr(self.vit, "mask_token"):
            self.vit.mask_token.requires_grad_(False)

        for blk in self.vit.blocks:
            if hasattr(blk, "adapter"):
                _patch_adapter_dynamic_hw(blk.adapter)

        self.embed_dim = self.vit.embed_dim  # 384
        self.out_dim = out_dim
        self.shallow_layers = tuple(shallow_layers)
        self.deep_layers = tuple(deep_layers)
        self.use_gated_drm = use_gated_drm

        shallow_in = len(self.shallow_layers) * self.embed_dim  # 768
        deep_in = len(self.deep_layers) * self.embed_dim        # 768

        if use_gated_drm:
            self.proj_struct = nn.Linear(shallow_in, out_dim)
            self.proj_sem = nn.Linear(deep_in, out_dim)
            self.geo_head = DepthwiseSeparableHead(in_dim=shallow_in, hidden_dim=128)
            self.cls_gate = nn.Sequential(
                nn.Linear(shallow_in + deep_in, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
                nn.Sigmoid(),
            )
            self.fusion_norm = nn.LayerNorm(out_dim)
        else:
            # M1 fallback: static concatenation projection without 3D salience gating
            self.static_proj = nn.Linear(shallow_in + deep_in, out_dim)
            self.fusion_norm = nn.LayerNorm(out_dim)

        self._init_adapters()
        if pretrained_dinov2_path is not None:
            self.load_dinov2_pretrained(pretrained_dinov2_path)
        if warmstart_d2vpr_ckpt is not None:
            self.load_d2vpr_warmstart(warmstart_d2vpr_ckpt)

    def _init_adapters(self) -> None:
        for n, m in self.vit.named_modules():
            if "adapter" in n:
                for n2, m2 in m.named_modules():
                    if "D_fc2" in n2 and isinstance(m2, nn.Linear):
                        nn.init.zeros_(m2.weight)
                        nn.init.zeros_(m2.bias)
                    elif "conv" in n2 and isinstance(m2, nn.Conv2d):
                        nn.init.normal_(m2.weight, mean=0.0, std=1e-4)
                        if m2.bias is not None:
                            nn.init.zeros_(m2.bias)

    def load_dinov2_pretrained(self, path: str) -> Tuple[int, int]:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        missing, unexpected = self.vit.load_state_dict(sd, strict=False)
        return len(missing), len(unexpected)

    def load_d2vpr_warmstart(self, ckpt_path: str) -> int:
        """Transfers ViT-S + MulConvAdapter weights from a D2VPR checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        cleaned = {}
        for k, v in sd.items():
            k_clean = k.replace("module.", "", 1)
            if k_clean.startswith("backbone."):
                sub_k = k_clean[len("backbone."):]
                if not sub_k.startswith("lin.") and not sub_k.startswith("up_norm."):
                    cleaned[sub_k] = v
        vit_sd = self.vit.state_dict()
        matched = {k: v for k, v in cleaned.items() if k in vit_sd and vit_sd[k].shape == v.shape}
        vit_sd.update(matched)
        self.vit.load_state_dict(vit_sd, strict=True)
        return len(matched)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        H_p, W_p = H // self.vit.patch_size, W // self.vit.patch_size
        for blk in self.vit.blocks:
            if hasattr(blk, "adapter"):
                blk.adapter._grid_hw = (H_p, W_p)

        tokens = self.vit.prepare_tokens_with_masks(x, None)
        shallow_feats = []
        deep_feats = []

        for idx, blk in enumerate(self.vit.blocks, start=1):
            tokens = blk(tokens)
            if idx in self.shallow_layers:
                shallow_feats.append(tokens)
            if idx in self.deep_layers:
                deep_feats.append(tokens)

        f_struct = torch.cat(shallow_feats, dim=-1)  # [B, 1 + N_p, 768]
        f_sem = torch.cat(deep_feats, dim=-1)        # [B, 1 + N_p, 768]

        if self.use_gated_drm:
            f_struct_patches_2d = (
                f_struct[:, 1:, :].reshape(B, H_p, W_p, -1).permute(0, 3, 1, 2).contiguous()
            )
            salience_gate, pseudo_depth = self.geo_head(f_struct_patches_2d)  # [B, 1, H_p, W_p]

            sem_proj = self.proj_sem(f_sem)
            struct_proj = self.proj_struct(f_struct)

            cls_gate = self.cls_gate(torch.cat([f_struct[:, 0], f_sem[:, 0]], dim=-1))  # [B, D]
            cls_fused = sem_proj[:, 0] + cls_gate * struct_proj[:, 0]

            gate_flat = salience_gate.flatten(2).transpose(1, 2)  # [B, N_p, 1]
            patches_fused = sem_proj[:, 1:] + gate_flat * struct_proj[:, 1:]

            all_fused = self.fusion_norm(torch.cat([cls_fused.unsqueeze(1), patches_fused], dim=1))
            f_struct_patches = struct_proj[:, 1:].reshape(B, H_p, W_p, self.out_dim).permute(0, 3, 1, 2).contiguous()
        else:
            concat_all = torch.cat([f_struct, f_sem], dim=-1)
            all_fused = self.fusion_norm(self.static_proj(concat_all))
            salience_gate = torch.ones((B, 1, H_p, W_p), device=x.device, dtype=x.dtype)
            pseudo_depth = torch.ones((B, 1, H_p, W_p), device=x.device, dtype=x.dtype)
            f_struct_patches = all_fused[:, 1:].reshape(B, H_p, W_p, self.out_dim).permute(0, 3, 1, 2).contiguous()

        patch_map = all_fused[:, 1:].reshape(B, H_p, W_p, self.out_dim).permute(0, 3, 1, 2).contiguous()
        cls_token = all_fused[:, 0]

        return {
            "cls_token": cls_token,                # [B, D]
            "patch_map": patch_map,                # [B, D, H_p, W_p]
            "struct_map": f_struct_patches,        # [B, D, H_p, W_p]
            "salience_gate": salience_gate,        # [B, 1, H_p, W_p]
            "pseudo_depth": pseudo_depth,          # [B, 1, H_p, W_p]
        }
