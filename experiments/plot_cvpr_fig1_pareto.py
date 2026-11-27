"""
CVPR Figure 1: Single-Query VPR Pareto Efficiency Bubble Plot
Matches Stanford/CVPR Bubble Pareto aesthetic:
- Left crimson vertical accent bar (#9E1B32), bold crimson main title, bold black subtitle
- Clean white plot area with subtle light-gray gridlines, crisp inward ticks on all 4 spines
- Bubble area strictly proportional to Total Stored Parameters (M), including hidden PCA weights
- White center dot for BS=1 Invariant models vs. crisp dark-ringed 'x' marker for Batch-Fragile models
- Bottom-right gray bubble-size legend (20M, 36M, 82M, 107M, 230M) + invariance marker legend
- Reads real verified metrics directly from results/*.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


ROOT = Path(r"d:\Lab\D2VPR")
RESULTS_DIR = ROOT / "results"
ARTIFACT_COPY = Path(
    r"C:\Users\ADMIN\.gemini\antigravity\brain\59d5f9bb-49a4-4a21-9f40-e5671d200b1e\fig1_cvpr_pareto_efficiency.png"
)

# Color palette inspired by the Stanford/CVPR reference image
CRIMSON = "#9E1B32"
CRIMSON_LIGHT = "#C23B50"
AMBER = "#E8A21A"      # Batch-fragile D2-VPR family
TAN = "#C4A681"        # Batch-fragile CricaVPR
SAGE = "#97C57A"       # EigenPlaces
SKY = "#8FD0DC"        # SALAD
TEAL = "#139464"       # MegaLoc (Teacher)
GRAY_LEGEND = "#E6E6E6"


def load_verified_metrics() -> dict:
    with open(RESULTS_DIR / "exp2_hardware_profile.json", "r", encoding="utf-8") as f:
        hw = json.load(f)
    with open(RESULTS_DIR / "g3vpr_v2_sota_pitts.json", "r", encoding="utf-8") as f:
        pitts = json.load(f)
    with open(RESULTS_DIR / "exp1_diagnostic.json", "r", encoding="utf-8") as f:
        diag = json.load(f)

    hw_by_name = {row["model"]: row for row in hw["hardware_efficiency_rows"]}
    return {
        "hw": hw_by_name,
        "pitts_m5": pitts["results"]["pitts30k-val"],
        "diag": diag["cross_domain_forgetting_matrix"],
    }


def param_to_area(params_m: float) -> float:
    """Linearly map Total Stored Params (M) to matplotlib scatter area (pt^2)."""
    return params_m * 16.5


def build_figure() -> None:
    data = load_verified_metrics()
    hw = data["hw"]
    pitts_m5 = data["pitts_m5"]
    diag = data["diag"]

    # Verified G3-VPR (Ours, M5_full) Matryoshka frontier from g3vpr_v2_sota_pitts.json
    g3_params = hw["G3-VPR (Ours, 512-D Balanced)"]["effective_total_params_M"]  # 36.07M
    g3_dims = ["128", "256", "512", "4096"]
    g3_points = [
        {
            "label": f"G³-VPR ({d}-D)",
            "dim": d,
            "ram_gb": pitts_m5[d]["1M_map_RAM_GB"],
            "r1": pitts_m5[d]["R@1"],
            "params_m": g3_params,
            "invariant": True,
        }
        for d in g3_dims
    ]

    # Verified workspace baselines + clearly labeled literature SOTA references
    baselines = [
        {
            "name": "D²-VPR (Raw, BS=1)\n92.3% · 40.1 GB",
            "ram_gb": hw["D2VPR (Encoder, Raw)"]["faiss_1M_RAM_GB"],            # 40.05 GB
            "r1": diag["distillation.pth"]["pitts30k_val_sub_bs1"]["R@1"],       # 92.33%
            "params_m": hw["D2VPR (Encoder, Raw)"]["effective_total_params_M"],  # 38.24M
            "invariant": False,
            "color": AMBER,
            "label_offset": (20, -2),
            "ha": "left",
            "va": "center",
        },
        {
            "name": "D²-VPR + PCA4096*\n(BS>1, +44M PCA)",
            "ram_gb": hw["D2VPR (Encoder + PCA4096)"]["faiss_1M_RAM_GB"],            # 15.26 GB
            "r1": diag["distillation.pth"]["pitts30k_val_sub_multi_bs"]["R@1"],       # 94.33%
            "params_m": hw["D2VPR (Encoder + PCA4096)"]["effective_total_params_M"],  # 82.28M
            "invariant": False,
            "color": AMBER,
            "label_offset": (-24, 16),
            "ha": "right",
            "va": "bottom",
        },
        {
            "name": "CricaVPR†\n(10752-D, BS>1)",
            "ram_gb": hw["CricaVPR (Teacher)"]["faiss_1M_RAM_GB"],            # 40.05 GB
            "r1": 94.90,                                                       # Published CVPR'24 Pitts30k-test
            "params_m": hw["CricaVPR (Teacher)"]["effective_total_params_M"],  # 106.76M
            "invariant": False,
            "color": TAN,
            "label_offset": (26, 6),
            "ha": "left",
            "va": "center",
        },
        {
            "name": "EigenPlaces†\n(2048-D)",
            "ram_gb": round(2048 * 4 * 1e6 / (1024**3), 3),  # 7.629 GB
            "r1": 92.50,                                     # Published ICCV'23 Pitts30k-test
            "params_m": 25.6,                                # ResNet-50 + FC
            "invariant": True,
            "color": SAGE,
            "label_offset": (0, -20),
            "ha": "center",
            "va": "top",
        },
        {
            "name": "DINOv2-SALAD†\n(8448-D)",
            "ram_gb": round(8448 * 4 * 1e6 / (1024**3), 3),  # 31.471 GB
            "r1": 92.40,                                     # Published CVPR'24 Pitts30k-test
            "params_m": 88.0,                                # DINOv2-B + SALAD head
            "invariant": True,
            "color": SKY,
            "label_offset": (0, -27),
            "ha": "center",
            "va": "top",
        },
        {
            "name": "MegaLoc†\n(8448-D, Teacher)",
            "ram_gb": round(8448 * 4 * 1e6 / (1024**3), 3),  # 31.471 GB
            "r1": 94.10,                                     # Published CVPRW'25 Pitts30k-test
            "params_m": 88.0,                                # DINOv2-B + SALAD head
            "invariant": True,
            "color": TEAL,
            "label_offset": (-8, 27),
            "ha": "right",
            "va": "bottom",
        },
    ]

    # Configure typography
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.edgecolor": "#888888",
            "axes.linewidth": 0.9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )

    fig = plt.figure(figsize=(11.2, 6.0), dpi=300, facecolor="#FFFFFF")

    # Main plot axes leaving room for Stanford/CVPR header on top and tiny footnote below
    ax = fig.add_axes([0.10, 0.145, 0.86, 0.655], facecolor="#FFFFFF")

    # Left crimson accent bar + Title + Subtitle (exact Stanford/CVPR style)
    bar = mpatches.FancyBboxPatch(
        (0.022, 0.885),
        0.0085,
        0.075,
        boxstyle="square,pad=0",
        transform=fig.transFigure,
        facecolor=CRIMSON,
        edgecolor="none",
        clip_on=False,
    )
    fig.patches.append(bar)

    fig.text(
        0.044,
        0.922,
        "Efficient Single-Query Visual Place Recognition",
        fontsize=23.5,
        fontweight="bold",
        color=CRIMSON,
        va="center",
        ha="left",
    )
    fig.text(
        0.044,
        0.858,
        "G³-VPR achieves SOTA-class R@1 at 21× lower 1M-map memory with exact BS=1 invariance",
        fontsize=13.0,
        fontweight="bold",
        color="#111111",
        va="center",
        ha="left",
    )

    # Configure log2 X-axis and clean Y-axis
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.30, 118.0)
    ax.set_ylim(86.0, 96.8)

    x_ticks = [0.5, 1, 2, 4, 8, 16, 32, 64]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks], fontsize=10.5)
    ax.xaxis.set_minor_locator(plt.NullLocator())

    y_ticks = [87, 89, 91, 93, 95]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{t}" for t in y_ticks], fontsize=10.5)

    # Subtle light-gray gridlines & inward crisp ticks on all 4 sides
    ax.grid(True, which="major", color="#EBEBEB", linewidth=0.7, zorder=0)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=4.5,
        width=0.85,
        color="#888888",
        top=True,
        right=True,
        pad=6,
    )
    for spine in ax.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(0.9)

    ax.set_xlabel(r"1M-Map Memory (GB, FP32, $\log_2$ scale)", fontsize=11.5, color="#222222", labelpad=8)
    ax.set_ylabel("Pitts30k R@1 (%)", fontsize=11.5, color="#222222", labelpad=8)

    # Subtle shaded "Pareto Sweet-Spot" highlight band on the left (<= 2 GB)
    ax.axvspan(0.30, 2.35, color="#FDF4F5", alpha=0.55, zorder=0)
    ax.text(
        0.34,
        96.35,
        "Edge-Ready Regime (≤ 2 GB / 1M images)",
        fontsize=8.8,
        fontweight="bold",
        color=CRIMSON,
        alpha=0.85,
        va="top",
        ha="left",
        zorder=2,
    )

    # Plot baseline bubbles first (lower zorder)
    for b in baselines:
        ax.scatter(
            b["ram_gb"],
            b["r1"],
            s=param_to_area(b["params_m"]),
            color=b["color"],
            alpha=0.90,
            edgecolors="#FFFFFF",
            linewidths=1.6,
            zorder=3,
        )
        # Inner center marker: white dot for BS=1 Invariant, crisp white-ringed dark 'x' for Batch-Fragile
        if b["invariant"]:
            ax.scatter(
                b["ram_gb"],
                b["r1"],
                s=22,
                color="#FFFFFF",
                edgecolors="none",
                marker="o",
                zorder=4,
            )
        else:
            ax.scatter(
                b["ram_gb"],
                b["r1"],
                s=46,
                color="#FFFFFF",
                edgecolors="#222222",
                linewidths=1.0,
                marker="D",
                zorder=4,
            )

        ax.annotate(
            b["name"],
            xy=(b["ram_gb"], b["r1"]),
            xytext=b["label_offset"],
            textcoords="offset points",
            fontsize=9.3,
            color="#222222",
            ha=b["ha"],
            va=b["va"],
            linespacing=1.15,
            zorder=6,
        )

    # Plot G3-VPR Matryoshka dashed crimson Pareto frontier
    g3_x = [p["ram_gb"] for p in g3_points]
    g3_y = [p["r1"] for p in g3_points]
    ax.plot(
        g3_x,
        g3_y,
        color=CRIMSON,
        linestyle="--",
        linewidth=2.0,
        alpha=0.92,
        zorder=4,
    )

    # Plot G3-VPR bubbles on top
    g3_label_cfg = {
        "128": ("G³-VPR (128-D)\n89.3% · 0.48 GB", (16, -24), "left", "top"),
        "256": ("G³-VPR (256-D)\n91.4% · 0.95 GB", (-16, 18), "right", "bottom"),
        "512": ("G³-VPR (512-D, Sweet Spot)\n92.8% · 1.91 GB (21× less RAM)", (0, 22), "center", "bottom"),
        "4096": ("G³-VPR (4096-D)\n92.6% · 15.3 GB", (0, -24), "center", "top"),
    }

    for p in g3_points:
        is_sweet = p["dim"] == "512"
        ax.scatter(
            p["ram_gb"],
            p["r1"],
            s=param_to_area(p["params_m"]),
            color=CRIMSON if is_sweet else CRIMSON_LIGHT,
            alpha=0.96,
            edgecolors="#FFFFFF",
            linewidths=1.8 if is_sweet else 1.5,
            zorder=5,
        )
        # White center dot (BS=1 Invariant)
        ax.scatter(
            p["ram_gb"],
            p["r1"],
            s=24,
            color="#FFFFFF",
            edgecolors="none",
            marker="o",
            zorder=6,
        )
        txt, offset, ha, va = g3_label_cfg[p["dim"]]
        ax.annotate(
            txt,
            xy=(p["ram_gb"], p["r1"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.3 if not is_sweet else 9.6,
            fontweight="bold",
            color=CRIMSON,
            ha=ha,
            va=va,
            linespacing=1.15,
            zorder=7,
        )

    # Bottom-right Stanford-style gray bubble-size legend (20M, 36M, 82M, 107M, 230M)
    # Placed in clean empty bottom-right space (y ~ 87.6, x from 4.0 to 56.0)
    legend_params = [20, 36, 82, 107, 230]
    legend_xs = [4.0, 7.2, 13.0, 24.0, 48.0]
    legend_y = 87.65
    label_y = 89.25

    ax.text(
        1.85,
        88.15,
        "Total Stored\nParams (M)",
        fontsize=9.2,
        fontweight="bold",
        color="#333333",
        ha="center",
        va="center",
        zorder=5,
    )

    for px, pm in zip(legend_xs, legend_params):
        ax.scatter(
            px,
            legend_y,
            s=param_to_area(pm),
            color=GRAY_LEGEND,
            edgecolors="#CCCCCC",
            linewidths=0.7,
            zorder=2,
        )
        ax.text(
            px,
            label_y,
            f"{pm}M",
            fontsize=9.3,
            color="#333333",
            ha="center",
            va="bottom",
            zorder=5,
        )

    # Marker legend inside two of the gray bubbles (exact Stanford reference pattern!)
    # 36M bubble -> BS=1 Invariant (white circle with subtle dark ring for visibility on gray)
    ax.scatter(
        legend_xs[1],
        legend_y,
        s=26,
        color="#FFFFFF",
        edgecolors="#444444",
        linewidths=0.9,
        marker="o",
        zorder=4,
    )
    ax.text(
        legend_xs[1],
        86.52,
        "BS=1 Invariant",
        fontsize=8.9,
        color="#222222",
        ha="center",
        va="center",
        zorder=5,
    )

    # 82M bubble -> Batch-Fragile (diamond)
    ax.scatter(
        legend_xs[2],
        legend_y,
        s=42,
        color="#FFFFFF",
        edgecolors="#222222",
        linewidths=1.0,
        marker="D",
        zorder=4,
    )
    ax.text(
        legend_xs[2],
        86.52,
        "Batch-Fragile",
        fontsize=8.9,
        color="#222222",
        ha="center",
        va="center",
        zorder=5,
    )

    # Concise factual footnote at bottom (fits comfortably within figure margins)
    fig.text(
        0.10,
        0.025,
        "Bubble area = Total Stored Params (incl. hidden PCA). G³-VPR (36.1M, 0M PCA) & D²-VPR measured on Pitts30k-val; "
        "*D²-VPR+PCA4096: 82.3M stored params & multi-BS R@1; †published Pitts30k-test R@1.",
        fontsize=7.6,
        color="#555555",
        ha="left",
        va="center",
    )

    # Save PNG (300 DPI), PDF, SVG
    png_path = RESULTS_DIR / "fig1_cvpr_pareto_efficiency.png"
    pdf_path = RESULTS_DIR / "fig1_cvpr_pareto_efficiency.pdf"
    svg_path = RESULTS_DIR / "fig1_cvpr_pareto_efficiency.svg"

    fig.savefig(png_path, dpi=300, facecolor="#FFFFFF")
    fig.savefig(pdf_path, facecolor="#FFFFFF")
    fig.savefig(svg_path, facecolor="#FFFFFF")
    plt.close(fig)

    ARTIFACT_COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_path, ARTIFACT_COPY)
    print(f"Saved: {png_path}, {pdf_path}, {svg_path}")
    print(f"Copied PNG to: {ARTIFACT_COPY}")


if __name__ == "__main__":
    build_figure()
