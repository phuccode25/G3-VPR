from typing import Dict, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity

from .matryoshka import MATRYOSHKA_DIMS


class MatryoshkaTeacherBridge(nn.Module):
    """
    Training-Only Factorized Subspace Alignment Bridge (0 inference parameters!).
    Projects nested student Matryoshka prefixes d in {256, 512, 1024, 4096} into the
    2D Foundation Teacher's high-dimensional space (e.g. 8448-D MegaLoc or 10752-D CricaVPR)
    so every single image receives dense per-dimension directional supervision.
    """

    def __init__(
        self,
        bridge_dims: Sequence[int] = (256, 512, 1024, 4096),
        latent_dim: int = 512,
        teacher_dim: int = 8448,
    ):
        super().__init__()
        self.bridge_dims = tuple(bridge_dims)
        self.teacher_dim = teacher_dim
        self.adapters = nn.ModuleDict()
        for d in self.bridge_dims:
            lin = nn.Linear(d, latent_dim, bias=False)
            nn.init.orthogonal_(lin.weight)
            self.adapters[str(d)] = lin

        self.teacher_proj = nn.Linear(latent_dim, teacher_dim, bias=False)
        nn.init.orthogonal_(self.teacher_proj.weight)

    def forward(self, desc_d: torch.Tensor, d: int, target_dim: int) -> torch.Tensor:
        key = str(d)
        if key not in self.adapters:
            key = str(max(self.bridge_dims))
        h = F.normalize(self.adapters[key](desc_d), p=2, dim=-1)
        out = self.teacher_proj(h)
        if out.shape[-1] > target_dim:
            out = out[:, :target_dim]
        return F.normalize(out, p=2, dim=-1)


class UnifiedG3VPRLoss(nn.Module):
    """
    Unified Zero-Forgetting Objective for G3-VPR v2:
      1. L_MRL_MS: Nested Matryoshka Multi-Similarity Loss (Cosine base=0.5) across {128..4096}-D
      2. L_2D:     Hybrid Direct Subspace Alignment (via training-only MatryoshkaTeacherBridge) +
                   Negative-Masked Geodesic Relational Distillation (NM-GRD, fixing v1's 0.07^2 bug!)
      3. L_3D:     Scale-Invariant Log-Depth + Pairwise 3D Structural Affinity + Sky/Occluder Suppression
      4. L_proto:  Prototype Diversity Orthogonality Regularization
    """

    def __init__(
        self,
        nested_dims: Sequence[int] = MATRYOSHKA_DIMS,
        use_matryoshka: bool = True,
        lambda_rel_2d: float = 2.0,
        lambda_3d: float = 0.5,
        lambda_proto: float = 0.05,
        rel_temp: float = 0.15,
        teacher_2d_dim: int = 8448,
    ):
        super().__init__()
        self.nested_dims = tuple(nested_dims)
        self.use_matryoshka = use_matryoshka
        self.lambda_rel_2d = lambda_rel_2d
        self.lambda_3d = lambda_3d
        self.lambda_proto = lambda_proto
        self.rel_temp = rel_temp

        # Standard gold-benchmark VPR Multi-Similarity setup (base=0.5 with CosineSimilarity)
        self.ms_loss = losses.MultiSimilarityLoss(
            alpha=2.0, beta=50.0, base=0.5, distance=CosineSimilarity()
        )
        self.ms_miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())

        self.bridge = MatryoshkaTeacherBridge(
            bridge_dims=(256, 512, 1024, 4096),
            latent_dim=512,
            teacher_dim=teacher_2d_dim,
        )

        self.dim_weights = {
            128: 0.15,
            256: 0.20,
            512: 0.15,
            1024: 0.15,
            2048: 0.15,
            4096: 0.20,
        }

    def compute_matryoshka_ms_loss(
        self,
        nested_descriptors: Dict[int, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_matryoshka:
            desc_full = nested_descriptors[max(nested_descriptors.keys())].float()
            mined = self.ms_miner(desc_full, labels)
            return self.ms_loss(desc_full, labels, mined)

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
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hybrid 2D Universal Foundation Distillation:
          (a) Direct Subspace Alignment: 1 - cos(Bridge_d(z^(d)), t_2D) across nested prefixes.
          (b) Pairwise Cosine Gram Matrix Alignment: 10 * MSE(S_student, S_teacher).
          (c) Negative-Masked Geodesic Relational KL: transfers cross-place dark knowledge without
              collapsing softmax mass onto same-place GSV positives (and without multiplying by T^2 < 1!).
        """
        t_raw = teacher_2d_desc.float()
        if t_raw.shape[-1] > self.bridge.teacher_dim:
            t_raw = t_raw[:, : self.bridge.teacher_dim]
        t_norm = F.normalize(t_raw, p=2, dim=-1).detach()
        target_dim = t_norm.shape[-1]

        sim_t = torch.matmul(t_norm, t_norm.T)  # [B, B]
        B = sim_t.shape[0]

        target_dims = [4096, 1024, 512, 256] if self.use_matryoshka else [max(nested_descriptors.keys())]
        valid_dims = [d for d in target_dims if d in nested_descriptors]

        loss_total_2d = torch.tensor(0.0, device=t_norm.device)

        off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=t_norm.device) if B > 1 else None
        # Mask out same-place positives so relational softmax captures pure inter-place similarity!
        if B > 4 and labels is not None and labels.shape[0] == B:
            neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
            row_neg_counts = neg_mask.sum(dim=1)
            use_neg_kl = bool((row_neg_counts >= 2).all().item())
        else:
            neg_mask = off_diag_mask
            use_neg_kl = B > 2

        if use_neg_kl and neg_mask is not None:
            # Pure float32 masked softmax over cross-place negatives (avoids float16 -inf * 0 = NaN!)
            sim_t_scaled = (sim_t / self.rel_temp).masked_fill(~neg_mask, -1e9)
            prob_t = F.softmax(sim_t_scaled, dim=-1).masked_fill(~neg_mask, 0.0).detach()
        else:
            prob_t = None

        for d in valid_dims:
            s_norm = F.normalize(nested_descriptors[d].float(), p=2, dim=-1)

            # (a) Direct Subspace Alignment (dense per-sample 8448-D supervision in [0, 1] scale)
            pred_t = self.bridge(s_norm, d=d, target_dim=target_dim)
            direct_cos = (1.0 - (pred_t * t_norm).sum(dim=-1)).mean()

            # (b) Pairwise Cosine Gram + Negative-Masked Relational KL (exact O(1) gradient scale: T * KL)
            rel_term = torch.tensor(0.0, device=t_norm.device)
            if B > 1 and off_diag_mask is not None:
                sim_s = torch.matmul(s_norm, s_norm.T)
                gram_mse = 4.0 * F.mse_loss(sim_s[off_diag_mask], sim_t[off_diag_mask])
                rel_term = rel_term + gram_mse

                if prob_t is not None and neg_mask is not None:
                    sim_s_scaled = (sim_s / self.rel_temp).masked_fill(~neg_mask, -1e9)
                    log_prob_s = F.log_softmax(sim_s_scaled, dim=-1)
                    p_valid = prob_t[neg_mask]
                    lp_s_valid = log_prob_s[neg_mask]
                    kl_neg = (p_valid * (p_valid.clamp(min=1e-8).log() - lp_s_valid)).sum() / float(B)
                    rel_term = rel_term + self.rel_temp * kl_neg

            loss_total_2d = loss_total_2d + (direct_cos + rel_term)

        return loss_total_2d / max(len(valid_dims), 1)

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

        t_min = t_depth.amin(dim=(2, 3), keepdim=True)
        t_max = t_depth.amax(dim=(2, 3), keepdim=True)
        t_norm = ((t_depth - t_min) / (t_max - t_min).clamp(min=1e-5)).clamp(min=0.05, max=1.0)

        s_depth = pseudo_depth.float().clamp(min=1e-4)
        log_diff = torch.log(s_depth) - torch.log(t_norm.detach())
        si_depth_loss = (log_diff.square().mean(dim=(1, 2, 3)) - 0.5 * log_diff.mean(dim=(1, 2, 3)).square()).mean()

        s_flat = F.normalize(s_depth.flatten(1), p=2, dim=-1)
        t_flat = F.normalize(t_norm.flatten(1).detach(), p=2, dim=-1)
        struct_aff_loss = F.mse_loss(s_flat, t_flat)

        if teacher_salience_prior is None:
            gx = (t_norm[:, :, :, 1:] - t_norm[:, :, :, :-1]).abs()
            gy = (t_norm[:, :, 1:, :] - t_norm[:, :, :-1, :]).abs()
            gx = F.pad(gx, (0, 1, 0, 0))
            gy = F.pad(gy, (0, 0, 0, 1))
            edge_mag = (gx + gy).clamp(0.0, 1.0)
            not_sky = (t_norm > 0.12).float()
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
        off_diag = gram - eye
        return (off_diag.square().sum()) / (M * (M - 1))

    def forward(
        self,
        model_out: Dict[str, object],
        labels: torch.Tensor,
        teacher_2d_desc: Optional[torch.Tensor] = None,
        teacher_depth: Optional[torch.Tensor] = None,
        teacher_salience_prior: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Run all metric, bridge, and KL loss math in strict float32 for numerical stability under AMP
        with torch.amp.autocast(device_type="cuda" if labels.is_cuda else "cpu", enabled=False):
            nested: Dict[int, torch.Tensor] = {
                d: v.float() for d, v in model_out["nested_descriptors"].items()  # type: ignore
            }
            device = labels.device

            loss_ms = self.compute_matryoshka_ms_loss(nested, labels)
            loss_rel = (
                self.compute_relational_2d_loss(nested, teacher_2d_desc.float(), labels=labels)
                if teacher_2d_desc is not None and self.lambda_rel_2d > 0
                else torch.tensor(0.0, device=device)
            )
            loss_3d = (
                self.compute_3d_geometry_loss(
                    model_out["pseudo_depth"].float(),          # type: ignore
                    model_out["salience_gate"].float(),         # type: ignore
                    teacher_depth.float(),
                    teacher_salience_prior.float() if teacher_salience_prior is not None else None,
                )
                if teacher_depth is not None and self.lambda_3d > 0
                else torch.tensor(0.0, device=device)
            )
            proto_tensor = model_out.get("prototypes_norm", None)
            loss_proto = (
                self.compute_prototype_diversity_loss(proto_tensor.float())  # type: ignore
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
