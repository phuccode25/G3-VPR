from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import GeoSemanticGatedBackbone
from .gaussian_aggregator import Vectorized3DGaussianAggregator
from .prototype_memory import PrototypeMemoryCrossInteraction
from .matryoshka import MATRYOSHKA_DIMS, MatryoshkaProjectionHead


@dataclass
class G3VPRConfig:
    """
    Configuration supporting both Full G3-VPR (M5_full) and clean Table-4 Ablation variants:
      - M1: Static DRM + Axis-Aligned Gaussian/Grid (no rotation) + No PMCI + Fixed 4096D
      - M2: + Rotated Anisotropic 2D Gaussian Splatting (3D-DGSA) + Gated-DRM
      - M3: + Prototype Memory Cross-Interaction (PMCI, solving BS=1)
      - M4: + Dual-Foundation Relational + 3D Structural Distillation (handled in loss)
      - M5_full: Full G3-VPR with Matryoshka Elastic Nested Descriptors (128D..4096D)
    """
    ablation: str = "M5_full"
    feat_dim: int = 512
    num_splats: int = 16
    num_prototypes: int = 128
    pmci_layers: int = 1
    pmci_heads: int = 8
    max_desc_dim: int = 4096
    default_infer_dim: int = 4096
    use_gated_drm: bool = True
    allow_gaussian_rotation: bool = True
    use_pmci: bool = True
    pretrained_dinov2_path: Optional[str] = None
    warmstart_d2vpr_ckpt: Optional[str] = None

    @classmethod
    def from_ablation(
        cls,
        ablation: str = "M5_full",
        pretrained_dinov2_path: Optional[str] = None,
        warmstart_d2vpr_ckpt: Optional[str] = None,
    ) -> "G3VPRConfig":
        ablation = ablation.upper()
        if ablation == "M1":
            return cls(
                ablation="M1",
                use_gated_drm=False,
                allow_gaussian_rotation=False,
                use_pmci=False,
                pretrained_dinov2_path=pretrained_dinov2_path,
                warmstart_d2vpr_ckpt=warmstart_d2vpr_ckpt,
            )
        elif ablation == "M2":
            return cls(
                ablation="M2",
                use_gated_drm=True,
                allow_gaussian_rotation=True,
                use_pmci=False,
                pretrained_dinov2_path=pretrained_dinov2_path,
                warmstart_d2vpr_ckpt=warmstart_d2vpr_ckpt,
            )
        elif ablation in ("M3", "M4"):
            return cls(
                ablation=ablation,
                use_gated_drm=True,
                allow_gaussian_rotation=True,
                use_pmci=True,
                pretrained_dinov2_path=pretrained_dinov2_path,
                warmstart_d2vpr_ckpt=warmstart_d2vpr_ckpt,
            )
        else:
            return cls(
                ablation="M5_full",
                use_gated_drm=True,
                allow_gaussian_rotation=True,
                use_pmci=True,
                pretrained_dinov2_path=pretrained_dinov2_path,
                warmstart_d2vpr_ckpt=warmstart_d2vpr_ckpt,
            )


class G3VPR(nn.Module):
    """
    G3-VPR: Geometry-Grounded Gaussian Aggregation and Matryoshka Relational Distillation
    for Universal Single-Query Visual Place Recognition.
    """

    def __init__(self, cfg: Optional[G3VPRConfig] = None):
        super().__init__()
        self.cfg = cfg or G3VPRConfig()

        self.backbone = GeoSemanticGatedBackbone(
            out_dim=self.cfg.feat_dim,
            use_gated_drm=self.cfg.use_gated_drm,
            pretrained_dinov2_path=self.cfg.pretrained_dinov2_path,
            warmstart_d2vpr_ckpt=self.cfg.warmstart_d2vpr_ckpt,
        )
        self.aggregator = Vectorized3DGaussianAggregator(
            feat_dim=self.cfg.feat_dim,
            num_splats=self.cfg.num_splats,
            allow_rotation=self.cfg.allow_gaussian_rotation,
        )
        if self.cfg.use_pmci:
            self.pmci = PrototypeMemoryCrossInteraction(
                dim=self.cfg.feat_dim,
                num_prototypes=self.cfg.num_prototypes,
                num_layers=self.cfg.pmci_layers,
                num_heads=self.cfg.pmci_heads,
            )
        else:
            self.pmci = None

        self.head = MatryoshkaProjectionHead(
            in_dim=self.cfg.feat_dim,
            num_tokens=self.cfg.num_splats + 1,
            max_dim=self.cfg.max_desc_dim,
            nested_dims=MATRYOSHKA_DIMS,
        )

    def freeze_early_vit_blocks(self, unfreeze_last_n: int = 4) -> None:
        """Freeze early ViT blocks while keeping MulConvAdapters, last N blocks, and G3 heads trainable."""
        total_blocks = len(self.backbone.vit.blocks)
        trainable_start = max(0, total_blocks - unfreeze_last_n)
        for name, param in self.backbone.vit.named_parameters():
            if "adapter" in name:
                param.requires_grad = True
            elif name.startswith("blocks."):
                blk_idx = int(name.split(".")[1])
                param.requires_grad = blk_idx >= trainable_start
            else:
                param.requires_grad = False

    def parameter_breakdown(self) -> Dict[str, float]:
        counts = {
            "vit_base": 0,
            "vit_adapters": 0,
            "gated_drm_and_3d_head": 0,
            "gaussian_aggregator_3d_dgsa": 0,
            "prototype_memory_pmci": 0,
            "matryoshka_head": 0,
        }
        for name, p in self.named_parameters():
            n = p.numel()
            if name.startswith("backbone.vit."):
                if "adapter" in name:
                    counts["vit_adapters"] += n
                else:
                    counts["vit_base"] += n
            elif name.startswith("backbone."):
                counts["gated_drm_and_3d_head"] += n
            elif name.startswith("aggregator."):
                counts["gaussian_aggregator_3d_dgsa"] += n
            elif name.startswith("pmci."):
                counts["prototype_memory_pmci"] += n
            elif name.startswith("head."):
                counts["matryoshka_head"] += n
        total = sum(counts.values())
        res = {k: round(v / 1e6, 4) for k, v in counts.items()}
        res["total_M"] = round(total / 1e6, 4)
        res["trainable_M"] = round(sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6, 4)
        return res

    def forward(
        self,
        x: torch.Tensor,
        output_dim: Optional[int] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Dict[str, object]]:
        dim = output_dim or self.cfg.default_infer_dim
        bb_out = self.backbone(x)
        agg_out = self.aggregator(
            patch_map=bb_out["patch_map"],
            cls_token=bb_out["cls_token"],
            salience_gate=bb_out["salience_gate"],
            pseudo_depth=bb_out["pseudo_depth"],
        )

        tokens = agg_out["tokens"]
        prototypes_norm = None
        if self.pmci is not None:
            pmci_out = self.pmci(tokens)
            tokens = pmci_out["tokens"]
            prototypes_norm = pmci_out["prototypes_norm"]

        descriptor, nested_descriptors = self.head(tokens, output_dim=dim)

        if not return_aux:
            return descriptor

        return {
            "descriptor": descriptor,
            "nested_descriptors": nested_descriptors,
            "salience_gate": bb_out["salience_gate"],
            "pseudo_depth": bb_out["pseudo_depth"],
            "struct_map": bb_out["struct_map"],
            "mu": agg_out["mu"],
            "sigma": agg_out["sigma"],
            "theta": agg_out["theta"],
            "splat_weights": agg_out["splat_weights"],
            "prototypes_norm": prototypes_norm,
        }


def build_g3vpr(
    ablation: str = "M5_full",
    pretrained_dinov2_path: Optional[str] = None,
    warmstart_d2vpr_ckpt: Optional[str] = None,
) -> G3VPR:
    cfg = G3VPRConfig.from_ablation(
        ablation=ablation,
        pretrained_dinov2_path=pretrained_dinov2_path,
        warmstart_d2vpr_ckpt=warmstart_d2vpr_ckpt,
    )
    return G3VPR(cfg)
