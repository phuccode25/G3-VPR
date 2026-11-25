import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets_ws
from g3vpr import build_g3vpr
from g3vpr.matryoshka import MATRYOSHKA_DIMS
from src.dataloaders.gsv_cities import default_transform
from src.dataloaders.pittsburgh import PittsburghDataset


def evaluate_matryoshka_recalls(
    feats_by_dim: Dict[int, np.ndarray],
    num_db: int,
    positives_per_query,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, Dict[str, float]]:
    out = {}
    for d, all_f in feats_by_dim.items():
        db_f = np.ascontiguousarray(all_f[:num_db], dtype=np.float32)
        q_f = np.ascontiguousarray(all_f[num_db:], dtype=np.float32)

        index = faiss.IndexFlatIP(d)
        index.add(db_f)
        _, preds = index.search(q_f, max(k_values))

        recalls = np.zeros(len(k_values), dtype=np.float64)
        valid_q = 0
        for q_idx, pred in enumerate(preds):
            pos = positives_per_query[q_idx]
            if len(pos) == 0:
                continue
            valid_q += 1
            for i, k in enumerate(k_values):
                if np.any(np.isin(pred[:k], pos)):
                    recalls[i:] += 1
                    break
        denom = max(valid_q, 1)
        map_ram_gb_1m = round((1_000_000 * d * 4) / (1024 ** 3), 3)
        out[str(d)] = {
            "R@1": round(float(recalls[0] / denom * 100.0), 2),
            "R@5": round(float(recalls[1] / denom * 100.0), 2),
            "R@10": round(float(recalls[2] / denom * 100.0), 2),
            "1M_map_RAM_GB": map_ram_gb_1m,
        }
    return out


@torch.no_grad()
def extract_single_pass_all_dims(model, dataset, query_bs: int, db_bs: int, device: str):
    """
    Extracts database features with db_bs and query features with strict query_bs (default BS=1!)
    Note: Because G3-VPR is 100% batch-size invariant, BS=1 and BS=32 yield identical features!
    """
    total_len = len(dataset)
    feats = {d: np.empty((total_len, d), dtype=np.float32) for d in MATRYOSHKA_DIMS}
    model.eval()

    num_db = getattr(dataset, "database_num", getattr(dataset, "num_references", 0))
    db_indices = list(range(num_db))
    q_indices = list(range(num_db, total_len))

    db_loader = DataLoader(torch.utils.data.Subset(dataset, db_indices), batch_size=db_bs, shuffle=False, num_workers=0)
    q_loader = DataLoader(torch.utils.data.Subset(dataset, q_indices), batch_size=query_bs, shuffle=False, num_workers=0)

    ptr = 0
    for batch in tqdm(db_loader, desc="DB Extract", ncols=90):
        imgs = batch[0].to(device)
        bsz = imgs.shape[0]
        aux = model(imgs, return_aux=True)
        for d in MATRYOSHKA_DIMS:
            feats[d][ptr : ptr + bsz] = aux["nested_descriptors"][d].cpu().numpy()
        ptr += bsz

    for batch in tqdm(q_loader, desc=f"Query Extract (BS={query_bs})", ncols=90):
        imgs = batch[0].to(device)
        bsz = imgs.shape[0]
        aux = model(imgs, return_aux=True)
        for d in MATRYOSHKA_DIMS:
            feats[d][ptr : ptr + bsz] = aux["nested_descriptors"][d].cpu().numpy()
        ptr += bsz

    return feats


def format_latex_pareto_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    datasets = list(results.keys())
    header_cols = " & ".join([f"\\multicolumn{{2}}{{c}}{{{ds}}}" for ds in datasets])
    sub_cols = " & ".join(["R@1 & R@5" for _ in datasets])
    lines = [
        "\\begin{tabular}{l r " + "cc " * len(datasets) + "}",
        "\\toprule",
        f"Dim ($d$) & 1M-Map RAM (GB) & {header_cols} \\\\",
        f" & & {sub_cols} \\\\",
        "\\midrule",
    ]
    for d in MATRYOSHKA_DIMS:
        d_str = str(d)
        ram = results[datasets[0]][d_str]["1M_map_RAM_GB"]
        vals = " & ".join([f"{results[ds][d_str]['R@1']:.1f} & {results[ds][d_str]['R@5']:.1f}" for ds in datasets])
        lines.append(f"{d}-D & {ram:.2f} & {vals} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Single-Checkpoint BS=1 Matryoshka SOTA Evaluation for G3-VPR")
    p.add_argument("--ckpt", type=str, required=False, default=None, help="Path to g3vpr_universal_best.pth")
    p.add_argument("--ablation", type=str, default="M5_full")
    p.add_argument("--datasets_root", type=str, default="./datasets")
    p.add_argument("--places", type=str, default="sped,amstertime,pitts30k-val,msls_val,pitts30k,pitts250k,nordland")
    p.add_argument("--query_bs", type=int, default=1, help="Strict single-query online inference BS=1")
    p.add_argument("--db_bs", type=int, default=32)
    p.add_argument("--output_json", type=str, default="./results/g3vpr_sota_eval.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_g3vpr(ablation=args.ablation).to(device)
    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=True)
        print(f"[Loaded universal checkpoint: {args.ckpt}]")
    else:
        print("[Warning] No --ckpt provided; running with current initialized weights.")

    ws_args = SimpleNamespace(resize=[224, 224], test_method="hard_resize", val_positive_dist_threshold=25)
    all_results = {}

    # Auto-search /kaggle/input if datasets_root does not exist locally
    if not os.path.isdir(args.datasets_root) and os.path.isdir("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            if any(d in dirs for d in ("sped", "amstertime", "pitts30k-val")):
                args.datasets_root = root
                break

    for place in [x.strip() for x in args.places.split(",") if x.strip()]:
        if place == "pitts30k-val":
            p_dir = os.path.join(args.datasets_root, "pitts30k-val")
            if not os.path.isdir(p_dir):
                continue
            ds = PittsburghDataset(p_dir, transform=default_transform)
            num_db = ds.num_references
            positives = ds.ground_truth
        else:
            p_dir = os.path.join(args.datasets_root, place, "images", "test")
            if not os.path.isdir(p_dir):
                continue
            ds = datasets_ws.BaseDataset(ws_args, args.datasets_root, place, "test")
            num_db = ds.database_num
            positives = ds.get_positives()

        t0 = time.perf_counter()
        feats = extract_single_pass_all_dims(model, ds, query_bs=args.query_bs, db_bs=args.db_bs, device=device)
        elapsed = time.perf_counter() - t0
        res = evaluate_matryoshka_recalls(feats, num_db=num_db, positives_per_query=positives)
        all_results[place] = res
        print(f"\n[{place}] (time={elapsed:.1f}s, query_bs={args.query_bs})")
        for d in MATRYOSHKA_DIMS:
            print(f"  Dim {d:4d}-D -> R@1: {res[str(d)]['R@1']:5.2f}% | R@5: {res[str(d)]['R@5']:5.2f}% | RAM(1M): {res[str(d)]['1M_map_RAM_GB']:.2f} GB")

    # Fallback on Kaggle when only GSV-Cities is attached: evaluate strict BS=1 on held-out WashingtonDC
    if not all_results and os.path.isdir("/kaggle/input"):
        from src.dataloaders.gsv_cities import GSVCitiesDataset
        for root, dirs, _ in os.walk("/kaggle/input"):
            if "Dataframes" in dirs and os.path.isfile(os.path.join(root, "Dataframes", "WashingtonDC.csv")):
                print("\n[Info] External test sets not mounted; evaluating strict BS=1 on held-out WashingtonDC...")
                gsv_hold = GSVCitiesDataset(dataset_path=root, cities=["WashingtonDC"], img_per_place=2)
                n_pl = min(len(gsv_hold), 500)
                db_list, q_list = [], []
                for pi in range(n_pl):
                    pair, _ = gsv_hold[pi]
                    db_list.append((pair[0], pi))
                    q_list.append((pair[1], n_pl + pi))
                flat_ds = db_list + q_list

                class _PairEvalDS(torch.utils.data.Dataset):
                    database_num = n_pl
                    def __len__(self): return len(flat_ds)
                    def __getitem__(self, idx): return flat_ds[idx]

                feats = extract_single_pass_all_dims(model, _PairEvalDS(), query_bs=args.query_bs, db_bs=args.db_bs, device=device)
                pos_hold = [np.array([qi], dtype=np.int64) for qi in range(n_pl)]
                res = evaluate_matryoshka_recalls(feats, num_db=n_pl, positives_per_query=pos_hold)
                all_results["WashingtonDC_Holdout"] = res
                for d in MATRYOSHKA_DIMS:
                    print(f"  Dim {d:4d}-D -> R@1: {res[str(d)]['R@1']:5.2f}% | R@5: {res[str(d)]['R@5']:5.2f}%")
                break

    if all_results:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        latex_tbl = format_latex_pareto_table(all_results)
        payload = {
            "ablation": args.ablation,
            "checkpoint": args.ckpt,
            "query_batch_size": args.query_bs,
            "results": all_results,
            "latex_pareto_table": latex_tbl,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved full evaluation + LaTeX table to {args.output_json}")


if __name__ == "__main__":
    main()
