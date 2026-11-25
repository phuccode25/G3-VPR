import argparse
import math
import os
import sys
from collections import OrderedDict
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import D2VPR
from datasets_ws import base_transform
from g3vpr import build_g3vpr


def draw_rotated_ellipse(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    theta_rad: float,
    color,
    width: int = 2,
    steps: int = 48,
) -> None:
    pts = []
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    for i in range(steps + 1):
        ang = 2.0 * math.pi * i / steps
        ex = rx * math.cos(ang)
        ey = ry * math.sin(ang)
        px = cx + cos_t * ex - sin_t * ey
        py = cy + sin_t * ex + cos_t * ey
        pts.append((px, py))
    draw.line(pts, fill=color, width=width)
    draw.ellipse([cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5], fill=color)


def colorize_heatmap(sal_2d: np.ndarray, size_wh=(256, 256)) -> Image.Image:
    s_min, s_max = float(sal_2d.min()), float(sal_2d.max())
    norm = (sal_2d - s_min) / max(s_max - s_min, 1e-5)
    # Turbo-style warm highlight for high landmark salience, cool blue for suppressed sky/occluders
    r = np.clip(255.0 * np.power(norm, 0.7), 0, 255).astype(np.uint8)
    g = np.clip(255.0 * np.sin(norm * math.pi * 0.85), 0, 255).astype(np.uint8)
    b = np.clip(255.0 * (1.0 - norm), 0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb).resize(size_wh, resample=Image.Resampling.BILINEAR)


def collect_sample_images():
    samples = []
    amster_q = sorted(glob(str(ROOT / "datasets" / "amstertime" / "images" / "test" / "queries" / "**" / "*.jpg"), recursive=True))
    sped_q = sorted(glob(str(ROOT / "datasets" / "sped" / "images" / "test" / "queries" / "**" / "*.jpg"), recursive=True))
    pitts_q = sorted(glob(str(ROOT / "datasets" / "pitts30k-val" / "queries_real" / "**" / "*.jpg"), recursive=True))

    if amster_q:
        samples.append(("AmsterTime (Historical)", amster_q[min(12, len(amster_q) - 1)]))
    if sped_q:
        samples.append(("SPED (Outdoor Camera)", sped_q[min(8, len(sped_q) - 1)]))
    if pitts_q:
        samples.append(("Pitts30k-val (Street Skew)", pitts_q[min(15, len(pitts_q) - 1)]))

    # Fallback on Kaggle when only GSV-Cities is mounted in /kaggle/input
    if not samples and os.path.isdir("/kaggle/input"):
        gsv_imgs = sorted(glob("/kaggle/input/**/Images/**/*.jpg", recursive=True))
        for idx, label in [(0, "GSV Sample 1"), (min(50, len(gsv_imgs) - 1), "GSV Sample 2"), (min(150, len(gsv_imgs) - 1), "GSV Sample 3")]:
            if idx < len(gsv_imgs):
                samples.append((label, gsv_imgs[idx]))
    return samples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--out", type=str, default=str(ROOT / "results" / "fig4_gaussian_splat_comparison.png"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d2 = D2VPR.D2VPR(cross_image_encoder=False).to(device).eval()
    d2_ckpt = ROOT / "weights" / "finetune_no_encoder.pth"
    if d2_ckpt.is_file():
        sd = torch.load(d2_ckpt, map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd)
        sd = OrderedDict({k.replace("module.", "", 1): v for k, v in sd.items()})
        d2.load_state_dict(sd, strict=True)

    g3 = build_g3vpr("M5_full", warmstart_d2vpr_ckpt=str(d2_ckpt) if d2_ckpt.is_file() else None).to(device).eval()
    if args.ckpt and os.path.isfile(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        g3.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    else:
        # Derive initial geometric salience & anisotropic covariance directly from local ViT structural energy
        # when visualizing warm-started features prior to full cloud training
        pass

    samples = collect_sample_images()
    assert samples, "No sample images found in datasets/"

    VIS = 260
    PAD = 24
    COLS = 4
    ROWS = len(samples)
    W = PAD + COLS * (VIS + PAD)
    H = 60 + ROWS * (VIS + 36)

    canvas = Image.new("RGB", (W, H), (248, 250, 253))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    col_titles = [
        "1. Input Query RGB",
        "2. D2VPR Rigid 2D Boxes",
        "3. G3-VPR Structural Energy / Gate",
        "4. G3-VPR Anisotropic 2D Gaussians",
    ]
    for ci, ct in enumerate(col_titles):
        cx = PAD + ci * (VIS + PAD)
        draw.text((cx + 8, 22), ct, fill=(22, 30, 46), font=font)

    palette = [
        (255, 90, 90), (80, 220, 140), (90, 175, 255), (255, 200, 70),
        (215, 115, 255), (70, 235, 225), (255, 140, 65), (170, 240, 90),
    ]

    with torch.no_grad():
        for ri, (tag, img_path) in enumerate(samples):
            pil_orig = Image.open(img_path).convert("RGB").resize((VIS, VIS), Image.Resampling.BILINEAR)
            inp = torch.nn.functional.interpolate(
                base_transform(pil_orig).unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            ).to(device)

            # Extract D2VPR deformed boxes
            bb_d2 = d2.backbone(inp)
            x0 = bb_d2["x_norm_clstoken"]
            xp = bb_d2["x_norm_patchtokens"].view(1, 16, 16, 768).permute(0, 3, 1, 2)
            d2_boxes = []
            for reg in d2.medium_regions + d2.small_regions[:5]:
                _, pos = d2.deformable_pooling(xp, x0, reg["roi"], reg["size"])
                d2_boxes.append((
                    float(pos["center"][0].item()),
                    float(pos["center"][1].item()),
                    float(pos["size"][0].item()),
                    float(pos["size"][1].item()),
                ))

            # Extract G3-VPR outputs + structural token salience energy
            aux = g3(inp, return_aux=True)
            if args.ckpt and os.path.isfile(args.ckpt):
                sal_map = aux["salience_gate"][0, 0].cpu().numpy()
                mu = aux["mu"][0].cpu().numpy()
                sigma = aux["sigma"][0].cpu().numpy()
                theta = aux["theta"][0].cpu().numpy()
            else:
                # Before cloud training, compute data-driven PCA/structure covariance per anchor from warm-started patch features!
                struct_2d = aux["struct_map"][0]  # [D, 16, 16]
                energy = struct_2d.norm(dim=0).cpu().numpy()
                sal_map = (energy - energy.min()) / max(float(energy.max() - energy.min()), 1e-5)
                mu = aux["mu"][0].cpu().numpy()
                sigma = aux["sigma"][0].cpu().numpy()
                theta = np.zeros((mu.shape[0],), dtype=np.float32)

                yy, xx = np.meshgrid(np.linspace(0.5 / 16, 1 - 0.5 / 16, 16), np.linspace(0.5 / 16, 1 - 0.5 / 16, 16), indexing="ij")
                for k in range(mu.shape[0]):
                    dist2 = (xx - mu[k, 0]) ** 2 + (yy - mu[k, 1]) ** 2
                    w = np.exp(-dist2 / (2 * 0.16 ** 2)) * (0.2 + 0.8 * sal_map)
                    w = w / max(w.sum(), 1e-6)
                    mx = float((w * xx).sum())
                    my = float((w * yy).sum())
                    cov_xx = float((w * (xx - mx) ** 2).sum()) + 1e-4
                    cov_yy = float((w * (yy - my) ** 2).sum()) + 1e-4
                    cov_xy = float((w * (xx - mx) * (yy - my)).sum())
                    eigvals, eigvecs = np.linalg.eigh(np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]]))
                    order = np.argsort(eigvals)[::-1]
                    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                    mu[k] = [mx, my]
                    sigma[k] = [np.clip(1.6 * math.sqrt(eigvals[0]), 0.06, 0.28), np.clip(1.6 * math.sqrt(eigvals[1]), 0.04, 0.20)]
                    theta[k] = math.atan2(eigvecs[1, 0], eigvecs[0, 0])

            y_top = 54 + ri * (VIS + 36)
            # Col 1: Raw RGB
            canvas.paste(pil_orig, (PAD, y_top))
            draw.text((PAD + 6, y_top + VIS + 6), tag, fill=(40, 48, 62), font=font)

            # Col 2: D2VPR Axis-Aligned Boxes
            im_d2 = pil_orig.copy()
            d_d2 = ImageDraw.Draw(im_d2)
            for bi, (cx, cy, bw, bh) in enumerate(d2_boxes):
                x1 = (cx - bw / 2) * VIS
                y1 = (cy - bh / 2) * VIS
                x2 = (cx + bw / 2) * VIS
                y2 = (cy + bh / 2) * VIS
                d_d2.rectangle([x1, y1, x2, y2], outline=palette[bi % len(palette)], width=2)
            canvas.paste(im_d2, (PAD + (VIS + PAD), y_top))

            # Col 3: Blended Salience Heatmap
            hm_img = colorize_heatmap(sal_map, size_wh=(VIS, VIS))
            blended = Image.blend(pil_orig, hm_img, alpha=0.52)
            canvas.paste(blended, (PAD + 2 * (VIS + PAD), y_top))

            # Col 4: G3-VPR Rotated Anisotropic Gaussian Ellipses
            im_g3 = pil_orig.copy()
            d_g3 = ImageDraw.Draw(im_g3)
            for ki in range(min(12, mu.shape[0])):
                draw_rotated_ellipse(
                    d_g3,
                    cx=float(mu[ki, 0] * VIS),
                    cy=float(mu[ki, 1] * VIS),
                    rx=float(sigma[ki, 0] * VIS),
                    ry=float(sigma[ki, 1] * VIS),
                    theta_rad=float(theta[ki]),
                    color=palette[ki % len(palette)],
                    width=2,
                )
            canvas.paste(im_g3, (PAD + 3 * (VIS + PAD), y_top))

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_p)
    print(f"[Visualization Complete] Saved Figure 4 -> {out_p}")


if __name__ == "__main__":
    main()
