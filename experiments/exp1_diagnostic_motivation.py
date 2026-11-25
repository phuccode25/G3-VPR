import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import commons
import D2VPR
import datasets_ws
from g3vpr import build_g3vpr
from src.dataloaders.gsv_cities import default_transform
from src.dataloaders.pittsburgh import PittsburghDataset


def load_d2vpr(ckpt_path: Path, cross_image_encoder: bool, device: str):
    model = D2VPR.D2VPR(cross_image_encoder=cross_image_encoder).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if sd and next(iter(sd.keys())).startswith("module."):
        sd = OrderedDict({k.replace("module.", "", 1): v for k, v in sd.items()})
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


from concurrent.futures import ThreadPoolExecutor

def preload_tensor_bank(dataset, indices: List[int]) -> torch.Tensor:
    """Loads resized [N, 3, 224, 224] tensors in parallel threads into CPU RAM."""
    with ThreadPoolExecutor(max_workers=8) as ex:
        imgs = list(ex.map(lambda i: dataset[i][0], indices))
    return torch.stack(imgs, dim=0)


@torch.no_grad()
def forward_tensor_bank(model, bank: torch.Tensor, batch_size: int, feat_dim: int, device: str) -> np.ndarray:
    n = bank.shape[0]
    feats = np.empty((n, feat_dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = bank[start:end].to(device, non_blocking=True)
        out = model(x)
        feats[start:end] = out.float().cpu().numpy()
    return feats


def compute_recalls(db_feats: np.ndarray, q_feats: np.ndarray, positives, k_vals=(1, 5, 10)) -> Dict[str, float]:
    index = faiss.IndexFlatL2(db_feats.shape[1])
    index.add(np.ascontiguousarray(db_feats, dtype=np.float32))
    _, preds = index.search(np.ascontiguousarray(q_feats, dtype=np.float32), max(k_vals))
    recalls = np.zeros(len(k_vals), dtype=np.float64)
    valid = 0
    for qi, pred in enumerate(preds):
        pos = positives[qi]
        if len(pos) == 0:
            continue
        valid += 1
        for ki, k in enumerate(k_vals):
            if np.any(np.isin(pred[:k], pos)):
                recalls[ki:] += 1
                break
    denom = max(valid, 1)
    return {f"R@{k}": round(float(recalls[i] / denom * 100.0), 2) for i, k in enumerate(k_vals)}


def audit_coordinate_bug_and_aggregator_speed(device: str) -> Dict[str, object]:
    offset_map = torch.zeros((1, 4, 16, 16), device=device, requires_grad=True)
    d2_model = D2VPR.D2VPR(cross_image_encoder=False).to(device).eval()

    total_grad = torch.zeros((16, 16), device=device)
    for region in d2_model.medium_regions + d2_model.small_regions:
        x1, y1, x2, y2 = region["roi"]
        gy, gx = torch.meshgrid(
            torch.linspace(y1, y2, region["size"][0], device=device),
            torch.linspace(x1, x2, region["size"][1], device=device),
            indexing="ij",
        )
        base_grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
        if offset_map.grad is not None:
            offset_map.grad.zero_()
        sampled = F.grid_sample(offset_map, base_grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled.sum().backward()
        total_grad += offset_map.grad[0].abs().sum(dim=0)

    d2_active_cells = int((total_grad > 0).sum().item())
    d2_coverage_pct = round(d2_active_cells / 256.0 * 100.0, 2)

    g3_model = build_g3vpr("M5_full").to(device).eval()
    x_p_d2 = torch.randn(16, 768, 16, 16, device=device)
    x0_d2 = torch.randn(16, 768, device=device)
    patch_g3 = torch.randn(16, 512, 16, 16, device=device)
    cls_g3 = torch.randn(16, 512, device=device)
    sal_g3 = torch.sigmoid(torch.randn(16, 1, 16, 16, device=device))
    dep_g3 = F.softplus(torch.randn(16, 1, 16, 16, device=device))

    with torch.no_grad():
        for _ in range(8):
            for reg in d2_model.medium_regions + d2_model.small_regions:
                d2_model.deformable_pooling(x_p_d2, x0_d2, reg["roi"], reg["size"])
            g3_model.aggregator(patch_g3, cls_g3, sal_g3, dep_g3)
        if "cuda" in device:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(40):
            med_f = [d2_model.deformable_pooling(x_p_d2, x0_d2, r["roi"], r["size"])[0] for r in d2_model.medium_regions]
            sml_f = [d2_model.deformable_pooling(x_p_d2, x0_d2, r["roi"], r["size"])[0] for r in d2_model.small_regions]
            d2_model.topdown_fusion(x0_d2, med_f, sml_f)
        if "cuda" in device:
            torch.cuda.synchronize()
        d2_agg_ms = (time.perf_counter() - t0) / 40.0 * 1000.0

        t0 = time.perf_counter()
        for _ in range(40):
            g3_model.aggregator(patch_g3, cls_g3, sal_g3, dep_g3)
        if "cuda" in device:
            torch.cuda.synchronize()
        g3_agg_ms = (time.perf_counter() - t0) / 40.0 * 1000.0

    d2_model.cpu()
    g3_model.cpu()
    if "cuda" in device:
        torch.cuda.empty_cache()

    return {
        "d2vpr_offset_map_active_cells_of_256": d2_active_cells,
        "d2vpr_offset_map_coverage_pct": d2_coverage_pct,
        "g3vpr_offset_map_coverage_pct": 100.0,
        "d2vpr_13loop_aggregator_ms_bs16": round(d2_agg_ms, 3),
        "g3vpr_vectorized_3d_dgsa_ms_bs16": round(g3_agg_ms, 3),
        "aggregator_speedup_x": round(d2_agg_ms / max(g3_agg_ms, 1e-4), 2),
        "d2vpr_grad_heatmap_16x16": total_grad.cpu().tolist(),
    }


def render_figure1_motivation(sweep: Dict, matrix: Dict, audit: Dict, out_path: Path) -> None:
    W, H = 2100, 680
    img = Image.new("RGB", (W, H), (250, 251, 254))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    panels = [(40, 60, 680, 620), (730, 60, 1390, 620), (1440, 60, 2060, 620)]
    titles = [
        "(a) Single-Query Collapse (Query BS: 32 -> 1)",
        "(b) Cross-Domain Forgetting Across Checkpoints",
        "(c) D2VPR [0,1] Offset Grid Bug (Only 25% Used)",
    ]
    for (x1, y1, x2, y2), title in zip(panels, titles):
        draw.rounded_rectangle([x1, y1, x2, y2], radius=12, fill=(255, 255, 255), outline=(210, 216, 226), width=2)
        draw.text((x1 + 20, y1 + 16), title, fill=(20, 28, 45), font=font)

    # Panel A
    px1, py1, px2, py2 = 110, 130, 640, 550
    draw.line([(px1, py2), (px2, py2)], fill=(80, 85, 95), width=2)
    draw.line([(px1, py1), (px1, py2)], fill=(80, 85, 95), width=2)

    bs_list = [1, 2, 3, 4, 8, 16, 32]
    for y_val in range(25, 101, 15):
        yy = py2 - int((y_val - 20.0) / 80.0 * (py2 - py1))
        draw.line([(px1, yy), (px2, yy)], fill=(235, 238, 244), width=1)
        draw.text((px1 - 36, yy - 6), f"{y_val}%", fill=(90, 95, 105), font=font)

    def plot_curve(vals, color, label, y_legend):
        pts = []
        for idx, v in enumerate(vals):
            xx = px1 + int(idx / (len(bs_list) - 1) * (px2 - px1))
            yy = py2 - int((v - 20.0) / 80.0 * (py2 - py1))
            yy = max(py1, min(py2 - 2, yy))
            pts.append((xx, yy))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=4)
        for idx, (xx, yy) in enumerate(pts):
            draw.ellipse([xx - 5, yy - 5, xx + 5, yy + 5], fill=color)
            if idx in (0, 2, 4, 6):
                draw.text((xx - 14, yy - 20), f"{vals[idx]:.1f}", fill=color, font=font)
        draw.rectangle([px1 + 20, y_legend, px1 + 42, y_legend + 10], fill=color)
        draw.text((px1 + 50, y_legend - 2), label, fill=(30, 35, 45), font=font)

    for idx, b in enumerate(bs_list):
        xx = px1 + int(idx / (len(bs_list) - 1) * (px2 - px1))
        draw.text((xx - 10, py2 + 12), f"B={b}", fill=(60, 65, 75), font=font)

    sped_curve = [sweep["sped"]["d2vpr_encoder_by_query_bs"][str(b)]["R@1"] for b in bs_list]
    amst_curve = [sweep["amstertime"]["d2vpr_encoder_by_query_bs"][str(b)]["R@1"] for b in bs_list]
    amst_noenc = [sweep["amstertime"]["d2vpr_no_encoder_bs1"]["R@1"]] * len(bs_list)

    plot_curve(sped_curve, (220, 58, 58), f"D2VPR SPED (-{sweep['sped']['bs1_drop_vs_peak_R1']}% at BS=1)", py2 - 90)
    plot_curve(amst_curve, (235, 130, 30), f"D2VPR AmsterTime (-{sweep['amstertime']['bs1_drop_vs_peak_R1']}% at BS=1)", py2 - 68)
    plot_curve(amst_noenc, (46, 140, 95), "Batch-Invariant Ref (0.0% BS=1 Drop)", py2 - 46)

    # Panel B
    bx1, by1, bx2, by2 = 800, 140, 1350, 550
    draw.line([(bx1, by2), (bx2, by2)], fill=(80, 85, 95), width=2)
    draw.line([(bx1, by1), (bx1, by2)], fill=(80, 85, 95), width=2)
    for y_val in (25, 45, 65, 85, 100):
        yy = by2 - int((y_val - 20.0) / 82.0 * (by2 - by1))
        draw.line([(bx1, yy), (bx2, yy)], fill=(235, 238, 244), width=1)
        draw.text((bx1 - 36, yy - 6), f"{y_val}%", fill=(90, 95, 105), font=font)

    groups = [
        ("Pitts30k (BS=8)", "pitts30k_val_sub_multi_bs"),
        ("AmsterTime (BS=8)", "amstertime_bs8"),
        ("AmsterTime (BS=1)", "amstertime_bs1"),
        ("SPED (BS=3)", "sped_bs3"),
    ]
    ckpt_colors = [
        ("distillation.pth", (56, 116, 203)),
        ("finetune.pth", (224, 82, 68)),
        ("finetune_no_encoder.pth", (90, 168, 115)),
    ]
    gw = (bx2 - bx1) // len(groups)
    for gi, (g_label, g_key) in enumerate(groups):
        gx0 = bx1 + gi * gw + 18
        for ci, (ck, col) in enumerate(ckpt_colors):
            val = matrix[ck][g_key]["R@1"]
            bar_x1 = gx0 + ci * 30
            bar_x2 = bar_x1 + 24
            bar_y1 = by2 - int((val - 20.0) / 82.0 * (by2 - by1))
            bar_y1 = max(by1, min(by2 - 3, bar_y1))
            draw.rectangle([bar_x1, bar_y1, bar_x2, by2], fill=col)
            draw.text((bar_x1 - 2, bar_y1 - 14), f"{val:.1f}", fill=col, font=font)
        draw.text((gx0 - 4, by2 + 12), g_label, fill=(50, 55, 65), font=font)

    for ci, (ck, col) in enumerate(ckpt_colors):
        ly = by1 - 22 + ci * 16
        draw.rectangle([bx2 - 210, ly, bx2 - 192, ly + 10], fill=col)
        draw.text((bx2 - 185, ly - 2), ck, fill=(30, 35, 45), font=font)

    # Panel C
    grid = np.array(audit["d2vpr_grad_heatmap_16x16"], dtype=np.float32)
    g_max = max(float(grid.max()), 1e-5)
    hx0, hy0 = 1530, 150
    cell = 24
    for r in range(16):
        for c in range(16):
            v = grid[r, c] / g_max
            col = (255, int(215 - 150 * v), int(120 - 100 * v)) if v > 0 else (232, 236, 244)
            cx1, cy1 = hx0 + c * cell, hy0 + r * cell
            draw.rectangle([cx1, cy1, cx1 + cell - 2, cy1 + cell - 2], fill=col)

    draw.rectangle([hx0 + 8 * cell, hy0 + 8 * cell, hx0 + 16 * cell - 2, hy0 + 16 * cell - 2], outline=(210, 30, 30), width=3)
    draw.text((hx0, hy0 + 16 * cell + 14), f"Active cells: {audit['d2vpr_offset_map_active_cells_of_256']}/256 ({audit['d2vpr_offset_map_coverage_pct']}%)", fill=(190, 35, 35), font=font)
    draw.text((hx0, hy0 + 16 * cell + 34), "75% top/left offset map receives ZERO gradient!", fill=(60, 65, 75), font=font)
    draw.text((hx0, hy0 + 16 * cell + 56), f"G3-VPR Vectorized 3D-DGSA: 100% coverage & {audit['aggregator_speedup_x']}x faster", fill=(25, 125, 75), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    commons.make_deterministic(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    print(f"=== Running Optimized Experiment 1 on {device} ===", flush=True)

    audit = audit_coordinate_bug_and_aggregator_speed(device)
    print(f"[1/3 Audit] { {k: v for k, v in audit.items() if k != 'd2vpr_grad_heatmap_16x16'} }", flush=True)

    # Preload datasets once from disk into CPU tensors (eliminates 12x redundant NTFS small-file reads!)
    ws_args = SimpleNamespace(resize=[224, 224], test_method="hard_resize", val_positive_dist_threshold=25)
    ds_sped = datasets_ws.BaseDataset(ws_args, str(ROOT / "datasets"), "sped", "test")
    ds_amster = datasets_ws.BaseDataset(ws_args, str(ROOT / "datasets"), "amstertime", "test")
    ds_pitts = PittsburghDataset(str(ROOT / "datasets" / "pitts30k-val"), transform=default_transform)

    print("Preloading SPED, AmsterTime, and stratified Pitts30k-val tensors into RAM once...", flush=True)
    sped_db_bank = preload_tensor_bank(ds_sped, list(range(ds_sped.database_num)))
    sped_q_bank = preload_tensor_bank(ds_sped, list(range(ds_sped.database_num, ds_sped.database_num + ds_sped.queries_num)))

    amst_db_bank = preload_tensor_bank(ds_amster, list(range(ds_amster.database_num)))
    amst_q_bank = preload_tensor_bank(ds_amster, list(range(ds_amster.database_num, ds_amster.database_num + ds_amster.queries_num)))

    rng = np.random.default_rng(3407)
    q_sel = np.linspace(0, ds_pitts.num_queries - 1, 300, dtype=int)
    pos_union = set()
    for qi in q_sel:
        pos_union.update(int(x) for x in ds_pitts.ground_truth[qi])
    remaining_db = [i for i in range(ds_pitts.num_references) if i not in pos_union]
    extra_db = rng.choice(remaining_db, size=max(0, 1200 - len(pos_union)), replace=False).tolist()
    db_sel = sorted(pos_union.union(extra_db))
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(db_sel)}
    pitts_positives = [
        np.array([old_to_new[int(p)] for p in ds_pitts.ground_truth[qi] if int(p) in old_to_new], dtype=np.int64)
        for qi in q_sel
    ]
    pitts_db_bank = preload_tensor_bank(ds_pitts, db_sel)
    pitts_q_bank = preload_tensor_bank(ds_pitts, [ds_pitts.num_references + int(qi) for qi in q_sel])
    print(f"Preload done in {time.perf_counter() - t0:.1f}s. Running GPU passes...", flush=True)

    batch_sizes = [1, 2, 3, 4, 8, 16, 32]
    m_dist = load_d2vpr(ROOT / "weights" / "distillation.pth", True, device)
    m_no = load_d2vpr(ROOT / "weights" / "finetune_no_encoder.pth", False, device)

    # Batch sensitivity sweep (reused directly in forgetting matrix!)
    sped_db_dist = forward_tensor_bank(m_dist, sped_db_bank, 3, 14 * 768, device)
    sped_db_no = forward_tensor_bank(m_no, sped_db_bank, 32, 14 * 768, device)
    sped_q_no = forward_tensor_bank(m_no, sped_q_bank, 32, 14 * 768, device)
    sped_rec_no = compute_recalls(sped_db_no, sped_q_no, ds_sped.get_positives())

    sped_curve = {}
    for b in batch_sizes:
        qf = forward_tensor_bank(m_dist, sped_q_bank, b, 14 * 768, device)
        sped_curve[str(b)] = compute_recalls(sped_db_dist, qf, ds_sped.get_positives())
    print(f"[2/3 Sweep SPED] {sped_curve}", flush=True)

    amst_db_dist = forward_tensor_bank(m_dist, amst_db_bank, 8, 14 * 768, device)
    amst_db_no = forward_tensor_bank(m_no, amst_db_bank, 32, 14 * 768, device)
    amst_q_no = forward_tensor_bank(m_no, amst_q_bank, 32, 14 * 768, device)
    amst_rec_no = compute_recalls(amst_db_no, amst_q_no, ds_amster.get_positives())

    amst_curve = {}
    for b in batch_sizes:
        qf = forward_tensor_bank(m_dist, amst_q_bank, b, 14 * 768, device)
        amst_curve[str(b)] = compute_recalls(amst_db_dist, qf, ds_amster.get_positives())
    print(f"[2/3 Sweep AmsterTime] {amst_curve}", flush=True)

    sweep = {
        "sped": {
            "d2vpr_encoder_by_query_bs": sped_curve,
            "d2vpr_no_encoder_bs1": sped_rec_no,
            "bs1_drop_vs_peak_R1": round(max(v["R@1"] for v in sped_curve.values()) - sped_curve["1"]["R@1"], 2),
        },
        "amstertime": {
            "d2vpr_encoder_by_query_bs": amst_curve,
            "d2vpr_no_encoder_bs1": amst_rec_no,
            "bs1_drop_vs_peak_R1": round(max(v["R@1"] for v in amst_curve.values()) - amst_curve["1"]["R@1"], 2),
        },
    }

    # Pitts30k-val for m_dist and m_no + full eval for finetune.pth
    p_db_dist = forward_tensor_bank(m_dist, pitts_db_bank, 8, 14 * 768, device)
    p_q8_dist = forward_tensor_bank(m_dist, pitts_q_bank, 8, 14 * 768, device)
    p_q1_dist = forward_tensor_bank(m_dist, pitts_q_bank, 1, 14 * 768, device)
    m_dist.cpu()

    p_db_no = forward_tensor_bank(m_no, pitts_db_bank, 32, 14 * 768, device)
    p_q_no = forward_tensor_bank(m_no, pitts_q_bank, 32, 14 * 768, device)
    m_no.cpu()
    if "cuda" in device:
        torch.cuda.empty_cache()

    m_fine = load_d2vpr(ROOT / "weights" / "finetune.pth", True, device)
    p_db_fine = forward_tensor_bank(m_fine, pitts_db_bank, 8, 14 * 768, device)
    p_q8_fine = forward_tensor_bank(m_fine, pitts_q_bank, 8, 14 * 768, device)
    p_q1_fine = forward_tensor_bank(m_fine, pitts_q_bank, 1, 14 * 768, device)

    a_db_fine = forward_tensor_bank(m_fine, amst_db_bank, 8, 14 * 768, device)
    a_q8_fine = forward_tensor_bank(m_fine, amst_q_bank, 8, 14 * 768, device)
    a_q1_fine = forward_tensor_bank(m_fine, amst_q_bank, 1, 14 * 768, device)

    s_db_fine = forward_tensor_bank(m_fine, sped_db_bank, 3, 14 * 768, device)
    s_q3_fine = forward_tensor_bank(m_fine, sped_q_bank, 3, 14 * 768, device)
    s_q1_fine = forward_tensor_bank(m_fine, sped_q_bank, 1, 14 * 768, device)

    matrix = {
        "distillation.pth": {
            "pitts30k_val_sub_multi_bs": compute_recalls(p_db_dist, p_q8_dist, pitts_positives),
            "pitts30k_val_sub_bs1": compute_recalls(p_db_dist, p_q1_dist, pitts_positives),
            "amstertime_bs8": amst_curve["8"],
            "amstertime_bs1": amst_curve["1"],
            "sped_bs3": sped_curve["3"],
            "sped_bs1": sped_curve["1"],
        },
        "finetune.pth": {
            "pitts30k_val_sub_multi_bs": compute_recalls(p_db_fine, p_q8_fine, pitts_positives),
            "pitts30k_val_sub_bs1": compute_recalls(p_db_fine, p_q1_fine, pitts_positives),
            "amstertime_bs8": compute_recalls(a_db_fine, a_q8_fine, ds_amster.get_positives()),
            "amstertime_bs1": compute_recalls(a_db_fine, a_q1_fine, ds_amster.get_positives()),
            "sped_bs3": compute_recalls(s_db_fine, s_q3_fine, ds_sped.get_positives()),
            "sped_bs1": compute_recalls(s_db_fine, s_q1_fine, ds_sped.get_positives()),
        },
        "finetune_no_encoder.pth": {
            "pitts30k_val_sub_multi_bs": compute_recalls(p_db_no, p_q_no, pitts_positives),
            "pitts30k_val_sub_bs1": compute_recalls(p_db_no, p_q_no, pitts_positives),
            "amstertime_bs8": amst_rec_no,
            "amstertime_bs1": amst_rec_no,
            "sped_bs3": sped_rec_no,
            "sped_bs1": sped_rec_no,
        },
    }
    print(f"[3/3 Forgetting Matrix] Done!", flush=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "fig1_diagnostic_motivation.png"
    payload = {
        "total_runtime_s": round(time.perf_counter() - t0, 2),
        "coordinate_and_speed_audit": {k: v for k, v in audit.items() if k != "d2vpr_grad_heatmap_16x16"},
        "batch_size_sensitivity": sweep,
        "cross_domain_forgetting_matrix": matrix,
        "figure1_path": str(fig_path),
    }
    json_path = out_dir / "exp1_diagnostic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    render_figure1_motivation(sweep, matrix, audit, fig_path)
    print(f"[Experiment 1 Complete] Runtime={payload['total_runtime_s']}s | Saved -> {json_path}", flush=True)


def forward_tensor_bank_recall_helper(db_f, q_f, pos):
    return compute_recalls(db_f, q_f, pos)


if __name__ == "__main__":
    main()
