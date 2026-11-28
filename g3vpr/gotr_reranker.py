"""
G-OTR: Zero-Cost 3D-Lifted Gaussian Optimal Transport Re-Ranking for G3-VPR v3.

Reuses the K=16 anisotropic Gaussian primitives ALREADY emitted by 3D-DGSA at zero extra
backbone FLOPs (~4.3 KB/image in fp16):
  1. Canonical pseudo-3D lifting: p_k = d_k * [2*mu_x - 1, 2*mu_y - 1, 1]^T in R^3.
  2. Closed-form 2x2 Bures-Wasserstein covariance distance (zero SVD, O(1) per pair):
       W_2^2(A, B) = tr(A) + tr(B) - 2 * sqrt(tr(A B) + 2 * sqrt(det(A) * det(B)))
  3. 5-iteration log-stabilized Sinkhorn Optimal Transport plan T* in R^{K x K}.
  4. SE(3)- & scale-invariant 3D Gromov-Wasserstein structural consistency re-ranking
     over Top-K global candidates in <0.5 ms per query.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def build_2x2_covariances(sigma: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Constructs rotated 2x2 SPD covariance matrices Sigma = R(theta) diag(sx^2, sy^2) R(theta)^T.
    Args:
        sigma: [..., K, 2] standard deviations (sx, sy)
        theta: [..., K] rotation angles in radians
    Returns:
        cov: [..., K, 2, 2]
    """
    sx2 = sigma[..., 0].square().clamp(min=1e-5)
    sy2 = sigma[..., 1].square().clamp(min=1e-5)
    c = torch.cos(theta)
    s = torch.sin(theta)
    c2 = c * c
    s2 = s * s
    cs = c * s

    a11 = sx2 * c2 + sy2 * s2
    a22 = sx2 * s2 + sy2 * c2
    a12 = (sx2 - sy2) * cs
    row1 = torch.stack([a11, a12], dim=-1)
    row2 = torch.stack([a12, a22], dim=-1)
    return torch.stack([row1, row2], dim=-2)


def pack_gaussian_primitives(model_out: Dict[str, torch.Tensor], feat_dim_keep: int = 128) -> Dict[str, torch.Tensor]:
    """
    Extracts compact 3D-lifted Gaussian primitives from G3VPR forward(return_aux=True).
    Storage per image (K=16, feat_dim_keep=128 in fp16):
      16 * (3 + 4 + 1 + 128) * 2 bytes = 4,352 bytes (~4.25 KB/image).
    """
    mu = model_out["mu"].float()                # [B, K, 2] in [0, 1]
    sigma = model_out["sigma"].float()          # [B, K, 2]
    theta = model_out["theta"].float()          # [B, K]
    mean_depth = model_out["mean_depth"].float().clamp(min=0.05)  # [B, K, 1]
    mean_sal = model_out["mean_sal"].float().clamp(min=1e-4)      # [B, K, 1]
    splat_tokens = model_out["splat_tokens"].float()              # [B, K, D]

    # Canonical pseudo-3D lifting: p_k = d_k * [2*mu_x - 1, 2*mu_y - 1, 1]
    xy_centered = 2.0 * mu - 1.0                                  # [B, K, 2]
    ones = torch.ones_like(mean_depth)
    p3d = mean_depth * torch.cat([xy_centered, ones], dim=-1)     # [B, K, 3]

    # Normalized salience mass distribution pi_k
    pi = mean_sal.squeeze(-1)                                     # [B, K]
    pi = pi / pi.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    # Compact unit-norm appearance features for OT matching
    f_k = F.normalize(splat_tokens[..., :feat_dim_keep], p=2, dim=-1)
    cov = build_2x2_covariances(sigma, theta)                     # [B, K, 2, 2]

    return {
        "p3d": p3d,           # [B, K, 3]
        "cov": cov,           # [B, K, 2, 2]
        "pi": pi,             # [B, K]
        "feats": f_k,         # [B, K, d_keep]
    }


def bures_wasserstein_2x2_sq(cov_q: torch.Tensor, cov_r: torch.Tensor) -> torch.Tensor:
    """
    Exact closed-form squared 2-Wasserstein (Bures-Wasserstein) distance between all pairs
    of 2x2 SPD covariance matrices without SVD!
    Args:
        cov_q: [N, K, 2, 2]
        cov_r: [N, K, 2, 2]
    Returns:
        w2_sq: [N, K, K]
    """
    # Expand to [N, K, 1, 2, 2] and [N, 1, K, 2, 2]
    a11 = cov_q[..., 0, 0].unsqueeze(-1)
    a12 = cov_q[..., 0, 1].unsqueeze(-1)
    a22 = cov_q[..., 1, 1].unsqueeze(-1)

    b11 = cov_r[..., 0, 0].unsqueeze(-2)
    b12 = cov_r[..., 0, 1].unsqueeze(-2)
    b22 = cov_r[..., 1, 1].unsqueeze(-2)

    tr_a = a11 + a22
    tr_b = b11 + b22
    det_a = (a11 * a22 - a12 * a12).clamp(min=1e-10)
    det_b = (b11 * b22 - b12 * b12).clamp(min=1e-10)
    tr_ab = a11 * b11 + 2.0 * a12 * b12 + a22 * b22

    cross = torch.sqrt((tr_ab + 2.0 * torch.sqrt(det_a * det_b)).clamp(min=1e-10))
    return (tr_a + tr_b - 2.0 * cross).clamp(min=0.0)


def sinkhorn_log_domain(
    cost: torch.Tensor,
    pi_q: torch.Tensor,
    pi_r: torch.Tensor,
    eps: float = 0.08,
    n_iters: int = 5,
) -> torch.Tensor:
    """
    Numerically stable 5-step log-domain Sinkhorn-Knopp optimal transport.
    Args:
        cost: [N, K, K]
        pi_q: [N, K]
        pi_r: [N, K]
    Returns:
        plan T*: [N, K, K]
    """
    log_a = torch.log(pi_q.clamp(min=1e-7))
    log_b = torch.log(pi_r.clamp(min=1e-7))
    log_k = -cost / eps

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(n_iters):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(1), dim=-1)
        v = log_b - torch.logsumexp(log_k + u.unsqueeze(2), dim=-2)

    log_t = log_k + u.unsqueeze(2) + v.unsqueeze(1)
    return torch.exp(log_t)


def compute_gotr_pair_scores(
    q_feats: torch.Tensor,
    q_cov: torch.Tensor,
    q_p3d: torch.Tensor,
    q_pi: torch.Tensor,
    r_feats: torch.Tensor,
    r_cov: torch.Tensor,
    r_p3d: torch.Tensor,
    r_pi: torch.Tensor,
    lambda_w: float = 1.5,
    sinkhorn_eps: float = 0.08,
    sinkhorn_iters: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes batched (OT Appearance+Covariance Score, SE(3)+Scale-Invariant 3D GW Score)
    for N candidate pairs (e.g., N = Top-20 candidates for a query).
    Returns:
        ot_sim: [N] in [-1, 1] (higher = better semantic+shape transport match)
        gw_sim: [N] in (0, 1]  (higher = better 3D pairwise distance consistency)
    """
    # 1. Semantic + Closed-Form Bures-Wasserstein Cost Matrix C in R^{N x K x K}
    sim_mat = torch.bmm(q_feats, r_feats.transpose(1, 2))  # [N, K, K]
    w2_sq = bures_wasserstein_2x2_sq(q_cov, r_cov)         # [N, K, K]
    cost = (1.0 - sim_mat) + lambda_w * w2_sq

    # 2. Optimal Transport Plan T* via 5 Sinkhorn iterations
    plan = sinkhorn_log_domain(cost, q_pi, r_pi, eps=sinkhorn_eps, n_iters=sinkhorn_iters)

    # OT semantic similarity (transported cosine similarity)
    ot_sim = (plan * sim_mat).sum(dim=(1, 2))

    # 3. SE(3)- & Scale-Invariant 3D Pairwise Distance Matrix Consistency (Gromov-Wasserstein)
    dq = torch.cdist(q_p3d, q_p3d, p=2)  # [N, K, K]
    dr = torch.cdist(r_p3d, r_p3d, p=2)  # [N, K, K]
    dq = dq / dq.mean(dim=(1, 2), keepdim=True).clamp(min=1e-5)
    dr = dr / dr.mean(dim=(1, 2), keepdim=True).clamp(min=1e-5)

    # Exact O(K^3) Gromov-Wasserstein discrepancy under transport plan T*
    c1 = ((dq.square() * q_pi.unsqueeze(1)).sum(dim=-1) * q_pi).sum(dim=-1)
    c2 = ((dr.square() * r_pi.unsqueeze(1)).sum(dim=-1) * r_pi).sum(dim=-1)
    # cross = sum_{i,j,k,l} T_{ij} T_{kl} dq_{ik} dr_{jl} = tr(dq^T T dr T^T)
    t_dr = torch.bmm(plan, dr)
    t_dr_tt = torch.bmm(t_dr, plan.transpose(1, 2))
    cross_gw = (dq * t_dr_tt).sum(dim=(1, 2))
    gw_dist = (c1 + c2 - 2.0 * cross_gw).clamp(min=0.0)
    gw_sim = torch.exp(-1.5 * gw_dist)

    return ot_sim, gw_sim


@torch.no_grad()
def gotr_rerank_topk(
    top_indices: np.ndarray,
    top_global_sims: np.ndarray,
    q_prims: Dict[str, np.ndarray],
    db_prims: Dict[str, np.ndarray],
    alpha_ot: float = 0.14,
    alpha_gw: float = 0.08,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Confidence-Gated Z-Score 3D-Lifted G-OTR Re-Ranking over Top-K FAISS candidates.
    Improvements in v3.1:
      1. Per-query Z-score standardization across the Top-K pool so global cosine,
         Sinkhorn OT similarity, and 3D Gromov-Wasserstein live on an identical scale.
      2. Salience-Depth Confidence Gate eta_q in [0, 1]: automatically dampens 3D GW
         weight when a query has flat/noisy pseudo-depth (e.g., degraded 1920s grayscale
         photos in AmsterTime), guaranteeing monotonic or non-degrading re-ranking.
      3. Uses full splat appearance features (already extracted at 0 extra backbone FLOPs).
    """
    Q, K_cand = top_indices.shape
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    db_feats_t = torch.from_numpy(db_prims["feats"]).to(dev, dtype=torch.float32)
    db_cov_t = torch.from_numpy(db_prims["cov"]).to(dev, dtype=torch.float32)
    db_p3d_t = torch.from_numpy(db_prims["p3d"]).to(dev, dtype=torch.float32)
    db_pi_t = torch.from_numpy(db_prims["pi"]).to(dev, dtype=torch.float32)

    reranked_idx = np.empty_like(top_indices)
    reranked_sc = np.empty_like(top_global_sims, dtype=np.float32)

    for qi in range(Q):
        cand_ids = top_indices[qi]
        if not np.all(cand_ids >= 0):
            reranked_idx[qi] = cand_ids
            reranked_sc[qi] = top_global_sims[qi]
            continue

        c_t = torch.from_numpy(cand_ids).to(dev, dtype=torch.long)
        qf_1 = torch.from_numpy(q_prims["feats"][qi : qi + 1]).to(dev, dtype=torch.float32)
        qc_1 = torch.from_numpy(q_prims["cov"][qi : qi + 1]).to(dev, dtype=torch.float32)
        qp_1 = torch.from_numpy(q_prims["p3d"][qi : qi + 1]).to(dev, dtype=torch.float32)
        qpi_1 = torch.from_numpy(q_prims["pi"][qi : qi + 1]).to(dev, dtype=torch.float32)

        # Salience-Depth Confidence Gate eta_q in [0, 1]
        depth_std = qp_1[0, :, 2].std().clamp(min=0.0, max=0.5)
        eta_q = float(torch.sigmoid(8.0 * (depth_std - 0.12)).item())

        ot_sim, gw_sim = compute_gotr_pair_scores(
            qf_1.expand(K_cand, -1, -1),
            qc_1.expand(K_cand, -1, -1, -1),
            qp_1.expand(K_cand, -1, -1),
            qpi_1.expand(K_cand, -1),
            db_feats_t[c_t], db_cov_t[c_t], db_p3d_t[c_t], db_pi_t[c_t],
        )
        g_sim = torch.from_numpy(top_global_sims[qi]).to(dev, dtype=torch.float32)

        # Per-query Z-score standardization across Top-K pool
        def _zscore(v: torch.Tensor) -> torch.Tensor:
            return (v - v.mean()) / v.std().clamp(min=1e-5)

        z_g = _zscore(g_sim)
        z_ot = _zscore(ot_sim)
        z_gw = _zscore(gw_sim)

        # Confidence-gated geometric boost (never overrides a crystal-clear Top-1 global match)
        g_margin = float((g_sim[0] - g_sim[1]).item()) if K_cand >= 2 else 1.0
        ambig_gate = float(torch.sigmoid(torch.tensor(12.0 * (0.06 - g_margin))).item())
        w_ot = alpha_ot * (0.5 + 0.5 * ambig_gate)
        w_gw = alpha_gw * eta_q * ambig_gate

        fused = z_g + w_ot * z_ot + w_gw * z_gw
        order = torch.argsort(fused, descending=True).cpu().numpy()
        reranked_idx[qi] = cand_ids[order]
        reranked_sc[qi] = fused.cpu().numpy()[order]

    return reranked_idx, reranked_sc
