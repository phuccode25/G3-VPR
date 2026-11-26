import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import torch

from eval_g3vpr_sota import evaluate_matryoshka_recalls, extract_single_pass_all_dims
import datasets_ws
from g3vpr import build_g3vpr
from src.dataloaders.gsv_cities import default_transform
from src.dataloaders.pittsburgh import PittsburghDataset


ABLATION_ORDER = ["M1", "M2", "M3", "M4", "M5_full"]


def find_ckpt_for_ablation(ckpt_root: Path, abl: str) -> Path | None:
    candidates = [
        ckpt_root / f"ablation_{abl}" / "g3vpr_best_fp16.pth",
        ckpt_root / f"ablation_{abl}" / "g3vpr_universal_best.pth",
        ckpt_root / f"g3vpr_{abl}" / "g3vpr_universal_best.pth",
        ckpt_root / f"g3vpr_{abl}" / "g3vpr_best_fp16.pth",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Table-4 Controlled Ablation Evaluator at Strict BS=1")
    p.add_argument("--ckpt_root", type=str, default="./checkpoints")
    p.add_argument("--datasets_root", type=str, default="./datasets")
    p.add_argument("--places", type=str, default="sped,amstertime,pitts30k-val")
    p.add_argument("--query_bs", type=int, default=1)
    p.add_argument("--db_bs", type=int, default=32)
    p.add_argument("--output_json", type=str, default="./results/table4_ablation_eval.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_root = Path(args.ckpt_root)
    ws_args = SimpleNamespace(resize=[224, 224], test_method="hard_resize", val_positive_dist_threshold=25)

    datasets_map = {}
    for place in [x.strip() for x in args.places.split(",") if x.strip()]:
        if place == "pitts30k-val":
            p_dir = os.path.join(args.datasets_root, "pitts30k-val")
            if os.path.isdir(p_dir):
                ds = PittsburghDataset(p_dir, transform=default_transform)
                datasets_map[place] = (ds, ds.num_references, ds.ground_truth)
        else:
            p_dir = os.path.join(args.datasets_root, place, "images", "test")
            if os.path.isdir(p_dir):
                ds = datasets_ws.BaseDataset(ws_args, args.datasets_root, place, "test")
                datasets_map[place] = (ds, ds.database_num, ds.get_positives())

    summary: Dict[str, object] = {}
    for abl in ABLATION_ORDER:
        ck_path = find_ckpt_for_ablation(ckpt_root, abl)
        if ck_path is None:
            print(f"[Skip] Checkpoint for {abl} not found under {ckpt_root}")
            continue
        model = build_g3vpr(ablation=abl).to(device)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        sd = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in ck.get("model_state_dict", ck).items()}
        model.load_state_dict(sd, strict=True)
        model.eval()
        print(f"\n=== Evaluating Ablation {abl} ({ck_path.name}) ===")

        abl_res = {}
        for place, (ds, num_db, positives) in datasets_map.items():
            feats = extract_single_pass_all_dims(model, ds, query_bs=args.query_bs, db_bs=args.db_bs, device=device)
            rec = evaluate_matryoshka_recalls(feats, num_db=num_db, positives_per_query=positives)
            abl_res[place] = rec
            print(
                f"  [{place}] 4096-D R@1: {rec['4096']['R@1']:.2f}% | "
                f"256-D R@1: {rec['256']['R@1']:.2f}% | 128-D R@1: {rec['128']['R@1']:.2f}%"
            )
        summary[abl] = {"checkpoint": str(ck_path), "results": abl_res}
        del model
        if "cuda" in device:
            torch.cuda.empty_cache()

    if summary:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved Table-4 Ablation results -> {out_p}")


if __name__ == "__main__":
    main()
