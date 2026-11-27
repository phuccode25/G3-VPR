"""
Signature Compact CVPR / NeurIPS Figure 2 for G3-VPR v3.
Features crisp Illustrator-style geometric polygon block arrows and 100% orthogonal
zero-crossing connectors, fusing 3DGS + Depth-Anything + Matryoshka + SALAD + SwAV motifs.
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Ellipse, Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\ADMIN\.gemini\antigravity\brain\59d5f9bb-49a4-4a21-9f40-e5671d200b1e")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "stix"


def box(ax, x, y, w, h, fc, ec="none", lw=0.0, ls="-", r=0.16, zorder=2, alpha=1.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder, alpha=alpha,
    )
    ax.add_patch(p)
    return p


def trap(ax, x, y, w, h, fc, shrink=0.18, zorder=3):
    dy = h * shrink
    poly = Polygon([(x, y), (x + w, y + dy), (x + w, y + h - dy), (x, y + h)],
                   closed=True, facecolor=fc, edgecolor="none", zorder=zorder)
    ax.add_patch(poly)
    return poly


def garrow(ax, x1, y1, x2, y2, shaft_h=0.072, head_h=0.21, head_l=0.13, color="#CBD5E1", zorder=3):
    """Crisp Illustrator-style flat geometric block arrow (sharp corners, balanced stem & head)."""
    dx = x2 - x1
    pad = min(0.035, dx * 0.14)
    sx1 = x1 + pad
    sx2 = x2 - pad
    avail = max(sx2 - sx1, 0.08)
    hl = min(head_l, avail * 0.56)
    neck = sx2 - hl
    sh = shaft_h / 2.0
    hh = head_h / 2.0
    pts = [
        (sx1, y1 - sh),
        (neck, y1 - sh),
        (neck, y1 - hh),
        (sx2, y1),
        (neck, y1 + hh),
        (neck, y1 + sh),
        (sx1, y1 + sh),
    ]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=zorder))


def tarrow(ax, x1, y1, x2, y2, color="#1F2937", lw=1.25, ls="-", rad=0.0, zorder=5):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=5.0,head_width=3.2",
        color=color, linewidth=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=zorder,
    ))


def snowflake(ax, cx, cy, s=0.13, color="#2B9AF3", zorder=6):
    for ang in (0, 60, 120):
        rad = np.deg2rad(ang)
        dx, dy = s * np.cos(rad), s * np.sin(rad)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, lw=1.5, solid_capstyle="round", zorder=zorder)
        for sgn in (-1, 1):
            bx, by = cx + sgn * 0.6 * dx, cy + sgn * 0.6 * dy
            for off in (-35, 35):
                ar = np.deg2rad(ang + off)
                ax.plot([bx, bx + sgn * 0.35 * s * np.cos(ar)],
                        [by, by + sgn * 0.35 * s * np.sin(ar)],
                        color=color, lw=1.0, solid_capstyle="round", zorder=zorder)


def flame(ax, cx, cy, s=0.13, zorder=6):
    t = np.linspace(0, 2 * np.pi, 36)
    ox = cx + s * 0.65 * np.sin(t) * (1.0 - 0.35 * np.cos(t))
    oy = cy + s * 0.92 * np.cos(t) - 0.02
    ax.fill(ox, oy, color="#F97316", zorder=zorder)
    ix = cx + s * 0.35 * np.sin(t) * (1.0 - 0.3 * np.cos(t))
    iy = cy - s * 0.22 + s * 0.50 * np.cos(t)
    ax.fill(ix, iy, color="#FDE047", zorder=zorder + 1)


def photo(ax, path, x, y, w, h, bc, overlay_ellipses=False, zorder=4):
    arr = np.asarray(Image.open(path).convert("RGB").resize((160, 120))) if os.path.isfile(path) else np.full((120, 160, 3), 210, np.uint8)
    ax.imshow(arr, extent=[x, x + w, y, y + h], aspect="auto", zorder=zorder)
    box(ax, x - 0.03, y - 0.03, w + 0.06, h + 0.06, fc="none", ec=bc, lw=2.2, r=0.05, zorder=zorder + 2)
    if overlay_ellipses:
        for rx, ry, rw, rh, ang, ec in [
            (0.48, 0.68, 0.42, 0.16, 18, "#10B981"),
            (0.28, 0.42, 0.24, 0.36, -12, "#34D399"),
            (0.70, 0.44, 0.22, 0.34, 10, "#10B981"),
            (0.52, 0.26, 0.46, 0.14, 0, "#059669"),
        ]:
            ax.add_patch(Ellipse(
                (x + rx * w, y + ry * h), rw * w, rh * h, angle=ang,
                fc=ec, alpha=0.28, ec="#ECFDF5", lw=0.8, zorder=zorder + 1,
            ))


def draw_mini_depth_map(ax, x, y, w, h, zorder=4):
    yy, xx = np.mgrid[0:24, 0:24]
    d = np.clip(0.25 + 0.65 * (yy / 23.0) - 0.22 * np.exp(-((xx - 12) ** 2 + (yy - 8) ** 2) / 36.0), 0, 1)
    ax.imshow(d, cmap="magma", extent=[x, x + w, y, y + h], aspect="auto", zorder=zorder)
    box(ax, x, y, w, h, fc="none", ec="#FFFFFF", lw=0.8, r=0.04, zorder=zorder + 1)


def draw_mini_sinkhorn_matrix(ax, x, y, w, h, zorder=4):
    mat = np.array([
        [0.08, 0.72, 0.06, 0.08, 0.06],
        [0.78, 0.07, 0.05, 0.05, 0.05],
        [0.05, 0.06, 0.12, 0.68, 0.09],
        [0.04, 0.05, 0.74, 0.11, 0.06],
        [0.05, 0.10, 0.03, 0.08, 0.74],
    ])
    ax.imshow(mat, cmap="YlGnBu", extent=[x, x + w, y, y + h], aspect="auto", interpolation="nearest", zorder=zorder)
    box(ax, x, y, w, h, fc="none", ec="#0F766E", lw=0.8, r=0.03, zorder=zorder + 1)


def draw_prototype_barcodes(ax, x, y, w, h, zorder=4):
    u_cols = ["#FCA5A5", "#F87171", "#EF4444", "#FCA5A5", "#DC2626", "#F87171"]
    v_cols = ["#6EE7B7", "#34D399", "#10B981", "#059669", "#34D399", "#10B981"]
    bw = w / 14.0
    for i, c in enumerate(u_cols):
        ax.add_patch(Rectangle((x + i * bw, y + h * 0.54), bw * 0.78, h * 0.42, fc=c, ec="none", zorder=zorder))
    for i, c in enumerate(v_cols):
        ax.add_patch(Rectangle((x + i * bw, y + h * 0.04), bw * 0.78, h * 0.42, fc=c, ec="none", zorder=zorder))
    ax.text(x + 6.5 * bw, y + h * 0.75, "−U", fontsize=6.6, fontweight="bold", color="#991B1B", va="center", ha="left", zorder=zorder)
    ax.text(x + 6.5 * bw, y + h * 0.25, "+V", fontsize=6.6, fontweight="bold", color="#065F46", va="center", ha="left", zorder=zorder)


def main():
    fig, ax = plt.subplots(figsize=(9.6, 7.6), dpi=300)
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 9.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    q_sped = str(ROOT / "datasets/sped/images/test/queries/@0@0@@@@@0000000@@@@@@@@.jpg")
    p_sped = str(ROOT / "datasets/sped/images/test/database/@0@0@@@@@0000000@@@@@@@@.jpg")
    n_amst = str(ROOT / "datasets/amstertime/images/test/queries/@0@0@@@@@0000@@@@@@@@.jpg")

    # =========================================================================
    # TOP CARD: STAGE I · DUAL-TEACHER TRAINING
    # =========================================================================
    box(ax, 0.22, 4.88, 11.56, 4.44, fc="#F5F6F8", r=0.18, zorder=1)
    box(ax, 0.42, 4.98, 2.55, 0.38, fc="#9E1B32", r=0.12, zorder=5)
    ax.text(1.70, 5.17, "Stage I · Training", fontsize=11.0, fontweight="bold", color="white", va="center", ha="center", zorder=6)

    photo(ax, q_sped, 0.48, 7.62, 1.38, 1.15, "#65A30D", overlay_ellipses=False)
    photo(ax, q_sped, 0.48, 5.95, 1.38, 1.15, "#2563EB", overlay_ellipses=True)

    # Frozen Teachers Row
    box(ax, 2.22, 7.45, 4.55, 1.66, fc="#E2EFF9", r=0.16, zorder=2)
    snowflake(ax, 2.48, 8.88, s=0.13)
    ax.text(4.60, 8.88, "Dual-Foundation Teachers (Frozen)", fontsize=10.2, fontweight="bold", ha="center", va="center", color="#0F172A")

    box(ax, 2.38, 7.60, 2.05, 1.00, fc="#C2E1F6", r=0.13, zorder=3)
    ax.text(3.40, 8.10, "MegaLoc\n(2D, 8448-D)", fontsize=9.0, fontweight="bold", ha="center", va="center", color="#0F172A")

    box(ax, 4.56, 7.60, 2.05, 1.00, fc="#C2E1F6", r=0.13, zorder=3)
    ax.text(5.26, 8.10, "Depth-Any.\nV2 (3D)", fontsize=8.8, fontweight="bold", ha="center", va="center", color="#0F172A")
    draw_mini_depth_map(ax, 5.98, 7.74, 0.52, 0.70, zorder=4)

    garrow(ax, 1.90, 8.18, 2.22, 8.18)

    # Teacher Aggregator Trapezoid
    trap(ax, 7.12, 7.45, 1.05, 1.66, fc="#CFE7F7", shrink=0.18, zorder=3)
    snowflake(ax, 7.36, 7.75, s=0.11)
    ax.text(7.68, 8.32, "Teacher\nTargets", fontsize=9.0, fontweight="bold", rotation=90, ha="center", va="center", color="#0F172A")
    garrow(ax, 6.77, 8.28, 7.12, 8.28)

    # Student Row
    box(ax, 2.22, 5.55, 4.55, 1.66, fc="#FAE3E1", r=0.16, zorder=2)
    flame(ax, 2.48, 5.75, s=0.13)
    ax.text(4.62, 5.75, "G³-VPR v3 (ViT-S + MulConvAdapter)", fontsize=9.5, fontweight="bold", ha="center", va="center", color="#0F172A")

    for i, (lbl, col) in enumerate([("Blk 1–6", "#C2E1F6"), ("Blk 7–12", "#F1C2BE"), ("Gated\nDRM", "#F1C2BE"), ("3D\nHead", "#C4E9DE")]):
        bx = 2.36 + i * 1.08
        box(ax, bx, 6.02, 0.98, 1.02, fc=col, r=0.11, zorder=3)
        ax.text(bx + 0.49, 6.53, lbl, fontsize=8.4, fontweight="bold", ha="center", va="center", color="#0F172A")

    garrow(ax, 1.90, 6.52, 2.22, 6.52)

    # Vertical Student Aggregator Stack with crisp polygon block arrows
    trap(ax, 7.04, 5.55, 0.80, 1.66, fc="#C6E9DF", shrink=0.14, zorder=3)
    flame(ax, 7.22, 5.78, s=0.09)
    ax.text(7.48, 6.42, "3D-DGSA", fontsize=8.6, fontweight="bold", rotation=90, ha="center", va="center", color="#064E3B")
    garrow(ax, 6.77, 6.40, 7.04, 6.40)

    box(ax, 8.10, 5.55, 0.68, 1.66, fc="#FDE6BE", r=0.12, zorder=3)
    flame(ax, 8.27, 5.78, s=0.09)
    ax.text(8.46, 6.42, "Nyström\nPMCI", fontsize=8.2, fontweight="bold", rotation=90, ha="center", va="center", color="#78350F")
    garrow(ax, 7.84, 6.40, 8.10, 6.40)

    trap(ax, 9.04, 5.55, 0.80, 1.66, fc="#E4DAFA", shrink=0.14, zorder=3)
    for si, (sh, sc) in enumerate([(0.34, "#7C3AED"), (0.62, "#8B5CF6"), (0.94, "#A78BFA")]):
        ax.add_patch(Rectangle((9.64 + si * 0.055, 6.38 - sh / 2), 0.04, sh, fc=sc, alpha=0.85, ec="none", zorder=4))
    flame(ax, 9.20, 5.78, s=0.09)
    ax.text(9.41, 6.42, "Matryoshka", fontsize=8.1, fontweight="bold", rotation=90, ha="center", va="center", color="#4C1D95")
    garrow(ax, 8.78, 6.40, 9.04, 6.40)

    # Right Feature Space Circles
    ax.add_patch(Circle((10.92, 8.32), 0.70, fc="#F0FDF4", ec="#65A30D", lw=1.6, ls="--", zorder=3))
    for dx, dy in [(-0.26, 0.20), (0.06, 0.34), (0.30, 0.12), (-0.10, -0.20), (0.22, -0.26)]:
        ax.add_patch(Circle((10.92 + dx, 8.32 + dy), 0.095, fc="#84CC16", ec="none", zorder=4))
    garrow(ax, 8.17, 8.28, 10.16, 8.28)

    ax.add_patch(Circle((10.92, 6.26), 0.70, fc="#EFF6FF", ec="#2563EB", lw=1.6, ls="--", zorder=3))
    t_pts = [(-0.25, 0.20), (0.08, 0.30), (0.28, 0.10), (-0.08, -0.18), (0.20, -0.26)]
    s_pts = [(-0.42, -0.05), (-0.14, 0.14), (0.40, -0.16), (-0.30, -0.36), (0.04, -0.45)]
    for (tx, ty), (sx, sy) in zip(t_pts, s_pts):
        ax.add_patch(Circle((10.92 + tx, 6.26 + ty), 0.075, fc="none", ec="#84CC16", lw=1.0, ls="--", zorder=4))
        ax.add_patch(Circle((10.92 + sx, 6.26 + sy), 0.09, fc="#3B82F6", ec="none", zorder=5))
        tarrow(ax, 10.92 + sx, 6.26 + sy, 10.92 + tx, 6.26 + ty, lw=0.85, ls="--")
    garrow(ax, 9.84, 6.40, 10.18, 6.40)

    ax.text(10.92, 7.29, r"$\mathcal{L}_{\mathrm{2D}}+\mathcal{L}_{\mathrm{3D}}+\mathcal{L}_{\mathrm{CSD}}$",
            fontsize=9.2, fontweight="bold", ha="center", va="center", color="#0F172A")
    tarrow(ax, 10.92, 7.45, 10.92, 7.58, lw=1.1)
    tarrow(ax, 10.92, 7.12, 10.92, 7.00, lw=1.1)
    ax.text(10.74, 5.15, "Matryoshka + MS Loss", fontsize=9.6, fontweight="bold", ha="center", va="center", color="#0F172A")

    # =========================================================================
    # BOTTOM CARD: STAGE II · BS=1 INFERENCE & G-OTR
    # =========================================================================
    box(ax, 0.22, 0.22, 11.56, 4.46, fc="#F5F6F8", r=0.18, zorder=1)
    box(ax, 8.78, 4.18, 2.78, 0.38, fc="#1E3A8A", r=0.12, zorder=5)
    ax.text(10.17, 4.37, "Stage II · BS=1 Inference", fontsize=10.8, fontweight="bold", color="white", va="center", ha="center", zorder=6)

    photo(ax, q_sped, 0.48, 3.35, 1.38, 1.08, "#65A30D", overlay_ellipses=True)
    photo(ax, p_sped, 0.48, 1.98, 1.38, 1.08, "#2563EB", overlay_ellipses=False)
    photo(ax, n_amst, 0.48, 0.62, 1.38, 1.08, "#EA580C", overlay_ellipses=False)

    # Top row of Bottom Card: Compact BS=1 Frozen Pipeline
    box(ax, 2.22, 2.75, 4.55, 1.66, fc="#E2EFF9", r=0.16, zorder=2)
    snowflake(ax, 2.48, 4.18, s=0.13)
    ax.text(4.60, 4.18, "G³-VPR v3 (36.1M, BS=1 Exact)", fontsize=9.6, fontweight="bold", ha="center", va="center", color="#0F172A")

    for i, (lbl, col) in enumerate([("ViT-S\n+ DRM", "#C2E1F6"), ("3D-DGSA\n16 Elip", "#C4E9DE"), ("Nyström\nPMCI", "#FDE6BE"), ("Matryo.\n128..4K", "#E4DAFA")]):
        bx = 2.36 + i * 1.08
        box(ax, bx, 2.92, 0.98, 1.02, fc=col, r=0.11, zorder=3)
        ax.text(bx + 0.49, 3.43, lbl, fontsize=8.2, fontweight="bold", ha="center", va="center", color="#0F172A")
    garrow(ax, 1.90, 3.85, 2.22, 3.85)

    # Compact Cascade Trapezoid
    trap(ax, 7.10, 2.75, 1.25, 1.66, fc="#E4DAFA", shrink=0.14, zorder=3)
    ax.text(7.72, 3.58, "AM-VPR\nCascade\n128→512\n→4096+3D", fontsize=7.9, fontweight="bold", ha="center", va="center", color="#4C1D95")
    garrow(ax, 6.77, 3.58, 7.10, 3.58)

    # Bottom-Left Lavender Inset — 100% ORTHOGONAL ZERO-CROSSING ROUTING!
    box(ax, 2.22, 0.42, 6.13, 1.96, fc="#ECE7F7", ec="#C4B5FD", lw=1.0, ls="--", r=0.14, zorder=2)

    # Left White Card (directly under 3D-DGSA): G-OTR + Mini Sinkhorn Heatmap
    box(ax, 2.40, 0.88, 2.82, 1.32, fc="#FFFFFF", ec="#A7F3D0", lw=0.9, r=0.11, zorder=3)
    ax.text(3.81, 2.02, "G-OTR (0 Extra FLOPs)", fontsize=8.8, fontweight="bold", ha="center", va="center", color="#065F46")
    draw_mini_sinkhorn_matrix(ax, 2.54, 1.02, 0.72, 0.76, zorder=4)
    ax.text(4.24, 1.49, "5-Iter Sinkhorn $T^*$\n+ 3D GW (<0.5 ms)", fontsize=7.7, ha="center", va="center", color="#111827")
    ax.text(4.24, 1.08, r"$W_2^2(\Sigma_i,\Sigma_j)$ No SVD", fontsize=7.4, ha="center", va="center", color="#0F766E")

    # Right White Card (directly under Nyström-PMCI): Nyström-PMCI + U/V Barcode Strips
    box(ax, 5.40, 0.88, 2.78, 1.32, fc="#FFFFFF", ec="#FDE68A", lw=0.9, r=0.11, zorder=3)
    ax.text(6.79, 2.02, "Nyström-PMCI", fontsize=8.8, fontweight="bold", ha="center", va="center", color="#92400E")
    draw_prototype_barcodes(ax, 5.54, 1.04, 0.82, 0.72, zorder=4)
    ax.text(7.24, 1.50, r"$x - \alpha W_{\mathrm{nuis}}(AU)$" + "\n" + r"$+ \beta W_{\mathrm{inv}}(AV)$",
            fontsize=7.6, ha="center", va="center", color="#111827")
    ax.text(7.24, 1.05, "128 Landmarks · SwAV", fontsize=7.2, ha="center", va="center", color="#92400E")

    ax.text(5.15, 0.56, "Zero-Cost 3D Optimal Transport & Batch-Invariant Whitening", fontsize=8.5, fontweight="bold", ha="center", va="center", color="#0F172A")

    # 100% Clean Orthogonal Zero-Crossing Connectors!
    # 1) Straight vertical dashed teal arrow: 3D-DGSA -> G-OTR
    tarrow(ax, 3.92, 2.92, 3.92, 2.22, color="#0D9488", lw=1.35, ls="--")
    # 2) Clean 90-degree elbow dashed amber arrow: Nyström-PMCI -> Nyström-PMCI card
    ax.plot([5.02, 5.02, 6.79], [2.92, 2.58, 2.58], color="#D97706", lw=1.3, ls="--", zorder=4)
    tarrow(ax, 6.79, 2.58, 6.79, 2.22, color="#D97706", lw=1.3, ls="--")
    # 3) Clean 90-degree solid teal return path: G-OTR -> right margin -> straight up into AM-VPR Cascade (ZERO crossing!)
    ax.plot([3.81, 3.81, 8.50, 8.50, 7.72], [0.88, 0.76, 0.76, 2.56, 2.56], color="#0D9488", lw=1.35, ls="-", zorder=4)
    tarrow(ax, 7.72, 2.56, 7.72, 2.86, color="#0D9488", lw=1.35, ls="-")

    # Right Side of Bottom Card: Anchor / Neg / Pos Circles + Conformal Mini-Curve!
    ax.add_patch(Circle((10.18, 3.38), 0.54, fc="#F0FDF4", ec="#65A30D", lw=1.5, ls="--", zorder=3))
    ax.add_patch(Circle((9.42, 2.12), 0.54, fc="#FFF7ED", ec="#EA580C", lw=1.5, ls="--", zorder=3))
    ax.add_patch(Circle((10.94, 2.12), 0.54, fc="#EFF6FF", ec="#2563EB", lw=1.5, ls="--", zorder=3))
    for dx, dy in [(-0.18, 0.15), (0.14, 0.20), (-0.12, -0.16), (0.18, -0.10)]:
        ax.add_patch(Circle((10.18 + dx, 3.38 + dy), 0.075, fc="#84CC16", zorder=4))
        ax.add_patch(Circle((9.42 + dx, 2.12 + dy), 0.075, fc="#EA580C", zorder=4))
        ax.add_patch(Circle((10.94 + dx, 2.12 + dy), 0.075, fc="#3B82F6", zorder=4))

    ax.text(11.12, 3.46, "Query", fontsize=9.6, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(9.42, 1.42, "Negative", fontsize=9.2, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(10.94, 1.42, "Positive", fontsize=9.2, fontweight="bold", ha="center", va="center", color="#0F172A")
    tarrow(ax, 9.70, 3.16, 9.38, 2.68, color="#111827", lw=1.2, rad=0.25)
    tarrow(ax, 10.66, 3.16, 10.98, 2.68, color="#111827", lw=1.2, ls="--", rad=-0.25)
    garrow(ax, 8.35, 3.45, 9.56, 3.45)

    # Mini Conformal Curve with shaded risk band
    ax.plot([9.15, 11.35], [0.58, 0.58], color="#0F172A", lw=1.1)
    ax.plot([9.15, 9.15], [0.58, 1.15], color="#0F172A", lw=1.1)
    xs = np.linspace(9.20, 11.28, 35)
    ys = 0.62 + 0.48 * (1.0 - np.exp(-1.8 * (xs - 9.20)))
    ax.fill_between(xs, np.maximum(0.58, ys - 0.06), ys + 0.04, color="#7C3AED", alpha=0.18, zorder=3)
    ax.plot(xs, ys, color="#6D28D9", lw=1.6, zorder=4)
    ax.text(10.25, 0.40, "Conformal Cascade (Risk ≤ α)", fontsize=8.8, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(8.98, 0.86, "R@1", fontsize=8.0, rotation=90, ha="center", va="center", color="#0F172A")

    plt.tight_layout(pad=0.12)
    png_p = OUT_DIR / "fig2_g3vpr_compact.png"
    pdf_p = OUT_DIR / "fig2_g3vpr_compact.pdf"
    svg_p = OUT_DIR / "fig2_g3vpr_compact.svg"
    fig.savefig(png_p, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_p, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_p, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if ARTIFACT_DIR.is_dir():
        shutil.copy2(png_p, ARTIFACT_DIR / "fig2_g3vpr_compact.png")
    print(f"[OK] Saved Crisp-Arrow Compact Figure 2 -> {png_p}, {pdf_p}, {svg_p}")


if __name__ == "__main__":
    main()
