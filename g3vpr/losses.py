from typing import Dict, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

from .matryoshka import MATRYOSHKA_DIMS


class UnifiedG3VPRLoss(nn.Module):
    """
    Unified Zero-Forgetting Objective for G3-VPR:
      1. L_MRL_MS: Nested Matryoshka Multi-Similarity Loss across {128, 256, 512, 1024, 2048, 4096}-D
      2. L_RGD:    Relational Geodesic Distillation Loss (KL + Angle Huber on pairwise similarity matrices
                   against 2D Universal Foundation Teacher — dimension-free & prevents domain forgetting!)
      3. L_3D:     Scale-Invariant Log-Depth + Pairwise 3D Structural Affinity + Dynamic Sky/Occluder Suppression
      4. L_proto:  Prototype Diversity Orthogonality Regularization
    """

    def __init__(
        self,
        nested_dims: Sequence[int] = MATRYOSHKA_DIMS,
        use_matryoshka: bool = True,
        lambda_rel_2d: float = 1.5,
        lambda_3d: float = 0.5,
        lambda_proto: float = 0.05,
        rel_temp: float = 0.07,
    ):
        super().__init__()
        self.nested_dims = tuple(nested_dims)
        self.use_matryoshka = use_matryoshka
        self.lambda_rel_2d = lambda_rel_2d
        self.lambda_3d = lambda_3d
        self.lambda_proto = lambda_proto
        self.rel_temp = rel_temp

        self.ms_loss = losses.MultiSimilarityLoss(
            alpha=1.0, beta=50.0, base=0.0, distance=DotProductSimilarity()
        )
        self.ms_miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())

        # Coarse prefixes get strong supervision so 128D & 256D perform competitively out-of-the-box
        self.dim_weights = {
            128: 0.20,
            256: 0.20,
            512: 0.15,
            1024: 0.15,
            2048: 0.15,
            4096: 0.15,
        }

    def compute_matryoshka_ms_loss(
        self,
        nested_descriptors: Dict[int, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_matryoshka:
            desc_full = nested_descriptors[max(nested_descriptors.keys())]
            mined = self.ms_miner(desc_full.float(), labels)
            return self.ms_loss(desc_full.float(), labels, mined)

        # Mine hard pairs once on the highest-capacity 4096D space + once on compact 256D space
        desc_max = nested_descriptors[max(nested_descriptors.keys())].float()
        mined_max = self.ms_miner(desc_max, labels)
        desc_256 = nested_descriptors.get(256, desc_max).float()
        mined_256 = self.ms_miner(desc_256, labels)

        total_ms = torch.tensor(0.0, device=desc_max.device)
        weight_sum = 0.0
        for d, desc in nested_descriptors.items():
            w = self.dim_weights.get(d, 1.0 / len(nested_descriptors))
            mined = mined_256 if d <= 256 else mined_max
            total_ms = total_ms + w * self.ms_loss(desc.float(), labels, mined)
            weight_sum += w
        return total_ms / max(weight_sum, 1e-6)

    def compute_relational_2d_loss(
        self,
        nested_descriptors: Dict[int, torch.Tensor],
        teacher_2d_desc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Matches relational geometry S_student (B x B) to S_teacher (B x B) without forcing student dim == teacher dim.
        """
        t_norm = F.normalize(teacher_2d_desc.float(), p=2, dim=-1)
        sim_t = torch.matmul(t_norm, t_norm.T)  # [B, B]
        B = sim_t.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=t_norm.device)

        mask = ~torch.eye(B, dtype=torch.bool, device=t_norm.device)
        sim_t_off = sim_t[mask].view(B, B - 1)
        prob_t = F.softmax(sim_t_off / self.rel_temp, dim=-1).detach()

        loss_rel = torch.tensor(0.0, device=t_norm.device)
        target_dims = [4096, 512, 256] if self.use_matryoshka else [max(nested_descriptors.keys())]
        valid_dims = [d for d in target_dims if d in nested_descriptors]

        for d in valid_dims:
            s_norm = F.normalize(nested_descriptors[d].float(), p=2, dim=-1)
            sim_s = torch.matmul(s_norm, s_norm.T)
            sim_s_off = sim_s[mask].view(B, B - 1)
            log_prob_s = F.log_softmax(sim_s_off / self.rel_temp, dim=-1)

            kl = F.kl_div(log_prob_s, prob_t, reduction="batchmean") * (self.rel_temp ** 2)
            huber = F.smooth_l1_loss(sim_s_off, sim_t_off.detach(), beta=0.1)
            loss_rel = loss_rel + (kl + 0.5 * huber)

        return loss_rel / max(len(valid_dims), 1)

    def compute_3d_geometry_loss(
        self,
        pseudo_depth: torch.Tensor,
        salience_gate: torch.Tensor,
        teacher_depth: torch.Tensor,
        teacher_salience_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Eigen Scale-Invariant Log-Depth Loss + 3D Structural Affinity + Sky/Occluder Gate Supervision.
        """
        B, _, H_p, W_p = pseudo_depth.shape
        if teacher_depth.shape[-2:] != (H_p, W_p):
            t_depth = F.interpolate(teacher_depth.float(), size=(H_p, W_p), mode="bilinear", align_corners=False)
        else:
            t_depth = teacher_depth.float()

        # Normalize teacher depth to positive range [0.05, 1.0]
        t_min = t_depth.amin(dim=(2, 3), keepdim=True)
        t_max = t_depth.amax(dim=(2, 3), keepdim=True)
        t_norm = ((t_depth - t_min) / (t_max - t_min).clamp(min=1e-5)).clamp(min=0.05, max=1.0)

        s_depth = pseudo_depth.float().clamp(min=1e-4)
        log_diff = torch.log(s_depth) - torch.log(t_norm.detach())
        # Eigen scale-invariant depth loss
        si_depth_loss = (log_diff.square().mean(dim=(1, 2, 3)) - 0.5 * log_diff.mean(dim=(1, 2, 3)).square()).mean()

        # Pairwise patch depth-structure affinity
        s_flat = F.normalize(s_depth.flatten(1), p=2, dim=-1)
        t_flat = F.normalize(t_norm.flatten(1).detach(), p=2, dim=-1)
        struct_aff_loss = F.mse_loss(s_flat, t_flat)

        # Supervise salience gate: suppress far-sky / extreme depth discontinuities or explicit prior
        if teacher_salience_prior is None:
            # High-gradient structural regions at mid-range depth get higher landmark salience than flat sky
            gx = (t_norm[:, :, :, 1:] - t_norm[:, :, :, :-1]).abs()
            gy = (t_norm[:, :, 1:, :] - t_norm[:, :, :-1, :]).abs()
            gx = F.pad(gx, (0, 1, 0, 0))
            gy = F.pad(gy, (0, 0, 0, 1))
            edge_mag = (gx + gy).clamp(0.0, 1.0)
            not_sky = (t_norm > 0.12).float()  # Depth-Anything inverse-depth: sky ~ 0
            target_sal = (0.35 + 0.65 * edge_mag) * not_sky + 0.10 * (1.0 - not_sky)
            target_sal = target_sal.clamp(0.05, 0.98).detach()
        else:
            target_sal = F.interpolate(
                teacher_salience_prior.float(), size=(H_p, W_p), mode="bilinear", align_corners=False
            ).detach()

        sal_loss = F.smooth_l1_loss(salience_gate.float(), target_sal, beta=0.1)
        return si_depth_loss + struct_aff_loss + 0.5 * sal_loss

    def compute_prototype_diversity_loss(self, prototypes_norm: Optional[torch.Tensor]) -> torch.Tensor:
        if prototypes_norm is None:
            return torch.tensor(0.0)
        M = prototypes_norm.shape[0]
        gram = torch.matmul(prototypes_norm.float(), prototypes_norm.float().T)
        eye = torch.eye(M, device=gram.device, dtype=gram.dtype)
        off_diag = (gram - eye)
        return (off_diag.square().sum()) / (M * (M - 1))

    def forward(
        self,
        model_out: Dict[str, object],
        labels: torch.Tensor,
        teacher_2d_desc: Optional[torch.Tensor] = None,
        teacher_depth: Optional[torch.Tensor] = None,
        teacher_salience_prior: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        nested: Dict[int, torch.Tensor] = model_out["nested_descriptors"]  # type: ignore
        device = labels.device

        loss_ms = self.compute_matryoshka_ms_loss(nested, labels)
        loss_rel = (
            self.compute_relational_2d_loss(nested, teacher_2d_desc)
            if teacher_2d_desc is not None and self.lambda_rel_2d > 0
            else torch.tensor(0.0, device=device)
        )
        loss_3d = (
            self.compute_3d_geometry_loss(
                model_out["pseudo_depth"],          # type: ignore
                model_out["salience_gate"],         # type: ignore
                teacher_depth,
                teacher_salience_prior,
            )
            if teacher_depth is not None and self.lambda_3d > 0
            else torch.tensor(0.0, device=device)
        )
        proto_tensor = model_out.get("prototypes_norm", None)
        loss_proto = (
            self.compute_prototype_diversity_loss(proto_tensor)  # type: ignore
            if proto_tensor is not None and self.lambda_proto > 0
            else torch.tensor(0.0, device=device)
        )

        total = (
            loss_ms
            + self.lambda_rel_2d * loss_rel
            + self.lambda_3d * loss_3d
            + self.lambda_proto * loss_proto
        )
        return {
            "loss": total,
            "loss_ms": loss_ms.detach(),
            "loss_rel_2d": loss_rel.detach(),
            "loss_3d": loss_3d.detach(),
            "loss_proto": loss_proto.detach(),
        }
