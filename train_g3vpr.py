import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import datasets_ws
from g3vpr import DualTeacherEngine, G3VPRConfig, UnifiedG3VPRLoss, build_g3vpr
from src.dataloaders.gsv_cities import GSVCitiesDataset, default_transform
from src.dataloaders.mapillary_sls import MapillarySLSDataset as MSLSDataset
from src.dataloaders.pittsburgh import PittsburghDataset


def seed_all(seed: int = 3407) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_faiss_recalls(
    db_desc: np.ndarray,
    q_desc: np.ndarray,
    positives_per_query,
    k_values: List[int] = [1, 5, 10],
) -> Dict[int, float]:
    dim = db_desc.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(db_desc, dtype=np.float32))
    _, preds = index.search(np.ascontiguousarray(q_desc, dtype=np.float32), max(k_values))

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
    return {k: float(recalls[i] / denom * 100.0) for i, k in enumerate(k_values)}


def auto_detect_gsv_path(explicit_path: Optional[str]) -> str:
    if explicit_path and os.path.isdir(explicit_path) and os.path.isdir(os.path.join(explicit_path, "Dataframes")):
        return explicit_path
    candidates = [
        "/kaggle/input/gsv-cities",
        "/kaggle/input/datasets/amaralibey/gsv-cities",
        "./data/train/gsv-cities",
        "./datasets/gsv-cities",
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isdir(os.path.join(c, "Dataframes")):
            return c
    if os.path.isdir("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            if "Dataframes" in dirs and "Images" in dirs:
                return root
    raise FileNotFoundError(
        "GSV-Cities dataset not found. On Kaggle, click '+ Add Input' -> 'amaralibey/gsv-cities'."
    )


def auto_find_dataset_dir(default_path: Optional[str], target_folder_name: str) -> Optional[str]:
    if default_path and os.path.isdir(default_path):
        return default_path
    if os.path.isdir("/kaggle/input"):
        for root, dirs, _ in os.walk("/kaggle/input"):
            if os.path.basename(root).lower() == target_folder_name.lower():
                return root
            if target_folder_name in dirs:
                return os.path.join(root, target_folder_name)
    return default_path


@torch.no_grad()
def eval_dataset_multi_dim(model, ds, num_db: int, positives, eval_bs: int, num_workers: int, device: str, dims=(256, 4096)):
    loader = DataLoader(ds, batch_size=eval_bs, shuffle=False, num_workers=num_workers)
    feats = {d: np.empty((len(ds), d), dtype=np.float32) for d in dims}
    ptr = 0
    model.eval()
    for batch in loader:
        imgs = batch[0].to(device, non_blocking=True)
        bsz = imgs.shape[0]
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=("cuda" in device)):
            out = model(imgs, return_aux=True)
        for d in dims:
            feats[d][ptr : ptr + bsz] = out["nested_descriptors"][d].float().cpu().numpy()
        ptr += bsz

    return {d: compute_faiss_recalls(feats[d][:num_db], feats[d][num_db:], positives) for d in dims}


@torch.no_grad()
def run_universal_validation(model, args: argparse.Namespace, epoch: int, device: str) -> Dict[str, float]:
    epoch_metrics: Dict[str, float] = {"epoch": epoch}
    r1_scores_4096: List[float] = []

    if args.val_pitts30k and os.path.isdir(args.val_pitts30k):
        ds_pitts = PittsburghDataset(args.val_pitts30k, transform=default_transform)
        m = eval_dataset_multi_dim(model, ds_pitts, ds_pitts.num_references, ds_pitts.ground_truth, args.eval_bs, args.num_workers, device)
        epoch_metrics["pitts30k_val_R1_4096"] = round(m[4096][1], 2)
        epoch_metrics["pitts30k_val_R1_256"] = round(m[256][1], 2)
        r1_scores_4096.append(m[4096][1])

    if args.val_msls and os.path.isdir(args.val_msls):
        ds_msls = MSLSDataset(args.val_msls, transform=default_transform)
        m = eval_dataset_multi_dim(model, ds_msls, ds_msls.num_references, ds_msls.ground_truth, args.eval_bs, args.num_workers, device)
        epoch_metrics["msls_val_R1_4096"] = round(m[4096][1], 2)
        epoch_metrics["msls_val_R1_256"] = round(m[256][1], 2)
        r1_scores_4096.append(m[4096][1])

    if args.eval_datasets_folder and os.path.isdir(args.eval_datasets_folder):
        ws_args = SimpleNamespace(resize=[224, 224], test_method="hard_resize", val_positive_dist_threshold=25)
        for ds_name in ("sped", "amstertime"):
            if os.path.isdir(os.path.join(args.eval_datasets_folder, ds_name, "images", "test")):
                ds_ws = datasets_ws.BaseDataset(ws_args, args.eval_datasets_folder, ds_name, "test")
                m = eval_dataset_multi_dim(model, ds_ws, ds_ws.database_num, ds_ws.get_positives(), args.eval_bs, args.num_workers, device)
                epoch_metrics[f"{ds_name}_R1_4096"] = round(m[4096][1], 2)
                epoch_metrics[f"{ds_name}_R1_256"] = round(m[256][1], 2)
                r1_scores_4096.append(m[4096][1])

    # Fallback held-out city validation (WashingtonDC) when only GSV-Cities is mounted on Kaggle
    if not r1_scores_4096 and getattr(args, "_gsv_dir", None):
        w_csv = os.path.join(args._gsv_dir, "Dataframes", "WashingtonDC.csv")
        if os.path.isfile(w_csv):
            ds_holdout = GSVCitiesDataset(dataset_path=args._gsv_dir, cities=["WashingtonDC"], img_per_place=2)
            n_pl = min(len(ds_holdout), 400)
            db_imgs, q_imgs = [], []
            for pi in range(n_pl):
                pair_imgs, _ = ds_holdout[pi]
                db_imgs.append(pair_imgs[0])
                q_imgs.append(pair_imgs[1])
            all_hold = torch.stack(db_imgs + q_imgs, dim=0)
            model.eval()
            feats = {256: np.empty((2 * n_pl, 256), np.float32), 4096: np.empty((2 * n_pl, 4096), np.float32)}
            for st in range(0, 2 * n_pl, args.eval_bs):
                en = min(st + args.eval_bs, 2 * n_pl)
                with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=("cuda" in device)):
                    out = model(all_hold[st:en].to(device), return_aux=True)
                feats[256][st:en] = out["nested_descriptors"][256].float().cpu().numpy()
                feats[4096][st:en] = out["nested_descriptors"][4096].float().cpu().numpy()
            pos_hold = [np.array([qi], dtype=np.int64) for qi in range(n_pl)]
            m256 = compute_faiss_recalls(feats[256][:n_pl], feats[256][n_pl:], pos_hold)
            m4096 = compute_faiss_recalls(feats[4096][:n_pl], feats[4096][n_pl:], pos_hold)
            epoch_metrics["holdout_WashingtonDC_R1_4096"] = round(m4096[1], 2)
            epoch_metrics["holdout_WashingtonDC_R1_256"] = round(m256[1], 2)
            r1_scores_4096.append(m4096[1])

    if r1_scores_4096:
        universal_hm = len(r1_scores_4096) / sum(1.0 / max(s, 1e-3) for s in r1_scores_4096)
    else:
        universal_hm = 0.0
    epoch_metrics["universal_harmonic_R1"] = round(universal_hm, 2)
    return epoch_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train G3-VPR (Pure PyTorch AMP — Zero Lightning Dependency)")
    p.add_argument("--ablation", type=str, default="M5_full", choices=["M1", "M2", "M3", "M4", "M5_full"])
    p.add_argument("--gsv_path", type=str, default=None)
    p.add_argument("--cities", type=str, default="all", help="'all' or comma-separated city names")
    p.add_argument("--val_pitts30k", type=str, default="./datasets/pitts30k-val")
    p.add_argument("--val_msls", type=str, default="./datasets/msls-val")
    p.add_argument("--eval_datasets_folder", type=str, default="./datasets")
    p.add_argument("--dinov2_path", type=str, default=None)
    p.add_argument("--warmstart_ckpt", type=str, default=None)
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument("--teacher_2d", type=str, default="megaloc", choices=["megaloc", "cricavpr", "none"])
    p.add_argument("--teacher_3d", type=str, default="depth_anything_v2", choices=["depth_anything_v2", "vggt", "none"])
    p.add_argument("--cricavpr_ckpt", type=str, default=None)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--eval_bs", type=int, default=32)
    p.add_argument("--img_per_place", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--unfreeze_last_n", type=int, default=4)
    p.add_argument("--lambda_rel_2d", type=float, default=1.5)
    p.add_argument("--lambda_3d", type=float, default=0.5)
    p.add_argument("--lambda_proto", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--max_places", type=int, default=0, help="Cap places per epoch (0 = all)")
    p.add_argument("--save_dir", type=str, default="./checkpoints/g3vpr_M5_full")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gsv_dir = auto_detect_gsv_path(args.gsv_path)
    args._gsv_dir = gsv_dir
    args.val_pitts30k = auto_find_dataset_dir(args.val_pitts30k, "pitts30k-val")
    args.val_msls = auto_find_dataset_dir(args.val_msls, "msls-val")
    sped_dir = auto_find_dataset_dir(os.path.join(args.eval_datasets_folder, "sped"), "sped")
    if sped_dir and os.path.isdir(sped_dir):
        args.eval_datasets_folder = str(Path(sped_dir).parent)

    if args.cities == "all":
        all_csvs = [Path(p).stem for p in Path(gsv_dir).glob("Dataframes/*.csv")]
        cities = [c for c in all_csvs if c != "WashingtonDC"] if len(all_csvs) > 1 else "all"
    else:
        cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    base_gsv_ds = GSVCitiesDataset(dataset_path=gsv_dir, cities=cities, img_per_place=args.img_per_place)
    if args.max_places and args.max_places < len(base_gsv_ds):
        sel_idx = np.linspace(0, len(base_gsv_ds) - 1, args.max_places, dtype=int).tolist()
        train_ds = Subset(base_gsv_ds, sel_idx)
    else:
        train_ds = base_gsv_ds

    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=("cuda" in device),
        drop_last=True,
    )

    cfg = G3VPRConfig.from_ablation(
        ablation=args.ablation,
        pretrained_dinov2_path=args.dinov2_path,
        warmstart_d2vpr_ckpt=args.warmstart_ckpt,
    )
    model = build_g3vpr(
        ablation=args.ablation,
        pretrained_dinov2_path=args.dinov2_path,
        warmstart_d2vpr_ckpt=args.warmstart_ckpt,
    ).to(device)
    model.freeze_early_vit_blocks(unfreeze_last_n=args.unfreeze_last_n)

    start_epoch = 0
    best_universal_score = -1.0
    history: List[Dict[str, object]] = []
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_universal_score = float(ckpt.get("best_universal_score", -1.0))
        print(f"[Resumed from {args.resume_ckpt} at epoch {start_epoch}]")

    use_matryoshka = cfg.ablation.upper() == "M5_FULL"
    use_distill = cfg.ablation.upper() in ("M4", "M5_FULL")
    criterion = UnifiedG3VPRLoss(
        use_matryoshka=use_matryoshka,
        lambda_rel_2d=args.lambda_rel_2d if use_distill else 0.0,
        lambda_3d=args.lambda_3d if use_distill else 0.0,
        lambda_proto=args.lambda_proto if cfg.use_pmci else 0.0,
    ).to(device)

    teacher_engine: Optional[DualTeacherEngine] = None
    if use_distill and (args.teacher_2d != "none" or args.teacher_3d != "none"):
        # Put frozen teachers on cuda:1 if T4 x2 is enabled on Kaggle, freeing 100% of cuda:0 VRAM for Student!
        teacher_dev = "cuda:1" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else device
        print(f"[Loading Frozen Teachers ({args.teacher_2d}, {args.teacher_3d}) on {teacher_dev}]")
        teacher_engine = DualTeacherEngine(
            teacher_2d=args.teacher_2d,
            teacher_3d=args.teacher_3d,
            cricavpr_ckpt=args.cricavpr_ckpt,
            device=teacher_dev,
            use_fp16=True,
        )

    bb_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.vit.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.vit.")]
    optimizer = torch.optim.AdamW(
        [{"params": bb_params, "lr": args.lr * 0.25}, {"params": head_params, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.02
    )
    scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in device))

    print(f"=== Training G3-VPR ({args.ablation}) | Places={len(train_ds)} | BS={args.bs}x{args.img_per_place} | Device={device} ===")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t_ep = time.perf_counter()
        running_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=105)
        for images, labels in pbar:
            images = images.flatten(0, 1).to(device, non_blocking=True)
            labels = labels.flatten().to(device, non_blocking=True)

            t_2d, t_3d = None, None
            if teacher_engine is not None:
                t_dev = teacher_engine.device_str
                t_out = teacher_engine.extract_targets(images.to(t_dev, non_blocking=True))
                if t_out["teacher_2d_desc"] is not None:
                    t_2d = t_out["teacher_2d_desc"].to(device, non_blocking=True)
                if t_out["teacher_depth"] is not None:
                    t_3d = t_out["teacher_depth"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=("cuda" in device)):
                out = model(images, return_aux=True)
                loss_dict = criterion(out, labels=labels, teacher_2d_desc=t_2d, teacher_depth=t_3d)

            scaler.scale(loss_dict["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            steps += 1
            running_loss += float(loss_dict["loss"].item())
            pbar.set_postfix(
                loss=f"{running_loss / steps:.3f}",
                ms=f"{loss_dict['loss_ms'].item():.2f}",
                r2d=f"{loss_dict['loss_rel_2d'].item():.2f}",
                g3d=f"{loss_dict['loss_3d'].item():.2f}",
            )
            if args.fast_dev_run and steps >= 2:
                break

        scheduler.step()
        if hasattr(base_gsv_ds, "_refresh_dataframes") and not args.fast_dev_run:
            base_gsv_ds._refresh_dataframes()

        if args.fast_dev_run:
            print("[Fast Dev Run Passed]")
            break

        val_metrics = run_universal_validation(model, args, epoch, device)
        val_metrics["train_loss"] = round(running_loss / max(steps, 1), 4)
        val_metrics["epoch_time_s"] = round(time.perf_counter() - t_ep, 1)
        history.append(val_metrics)

        hm = val_metrics["universal_harmonic_R1"]
        is_best = hm >= best_universal_score
        if is_best:
            best_universal_score = hm

        ckpt_payload = {
            "epoch": epoch,
            "ablation": args.ablation,
            "best_universal_score": best_universal_score,
            "model_state_dict": model.state_dict(),
            "metrics": val_metrics,
        }
        torch.save(ckpt_payload, save_dir / "g3vpr_last.pth")
        if is_best:
            torch.save(ckpt_payload, save_dir / "g3vpr_universal_best.pth")
        with open(save_dir / "training_metrics.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(f"[Epoch {epoch + 1} Summary] {val_metrics}" + ("  <-- NEW BEST UNIVERSAL!" if is_best else ""))


if __name__ == "__main__":
    main()
