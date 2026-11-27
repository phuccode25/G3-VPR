import argparse
import json
import math
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
from src.dataloaders.gsv_cities import GSVCitiesDataset, build_eval_transform, build_train_transform, default_transform
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


def auto_detect_warmstart(explicit_ckpt: Optional[str], disable_warmstart: bool = False) -> Optional[str]:
    if disable_warmstart:
        return None
    if explicit_ckpt and os.path.isfile(explicit_ckpt):
        return explicit_ckpt
    for cand in (
        "./weights_init/vits_adapter_warmstart_fp16.pt",
        "./weights/distillation.pth",
        "./weights/finetune_no_encoder.pth",
    ):
        if os.path.isfile(cand):
            return cand
    return None


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
    r1_scores: List[float] = []

    eval_tf = build_eval_transform(getattr(args, "img_size", 224))
    if args.val_pitts30k and os.path.isdir(args.val_pitts30k):
        ds_pitts = PittsburghDataset(args.val_pitts30k, transform=eval_tf)
        m = eval_dataset_multi_dim(model, ds_pitts, ds_pitts.num_references, ds_pitts.ground_truth, args.eval_bs, args.num_workers, device)
        epoch_metrics["pitts30k_val_R1_4096"] = round(m[4096][1], 2)
        epoch_metrics["pitts30k_val_R1_256"] = round(m[256][1], 2)
        r1_scores.extend([m[4096][1], m[256][1]])

    if args.val_msls and os.path.isdir(args.val_msls):
        ds_msls = MSLSDataset(args.val_msls, transform=eval_tf)
        m = eval_dataset_multi_dim(model, ds_msls, ds_msls.num_references, ds_msls.ground_truth, args.eval_bs, args.num_workers, device)
        epoch_metrics["msls_val_R1_4096"] = round(m[4096][1], 2)
        epoch_metrics["msls_val_R1_256"] = round(m[256][1], 2)
        r1_scores.extend([m[4096][1], m[256][1]])

    # NOTE (CVPR Strict Hygiene): SPED & AmsterTime TEST sets are NEVER used for checkpoint selection!
    # Held-out WashingtonDC validation fallback on Kaggle when Pitts30k-val / MSLS-val are not mounted
    if not r1_scores and getattr(args, "_gsv_dir", None):
        wash_csv = os.path.join(args._gsv_dir, "Dataframes", "WashingtonDC.csv")
        if os.path.isfile(wash_csv):
            gsv_hold = GSVCitiesDataset(dataset_path=args._gsv_dir, cities=["WashingtonDC"], img_per_place=2, transform=eval_tf)
            n_places = min(len(gsv_hold), 500)
            # Fix deterministic RNG seed for held-out validation so epoch-to-epoch comparison is exact!
            rng_state = random.getstate()
            np_state = np.random.get_state()
            random.seed(12345)
            np.random.seed(12345)
            db_imgs, q_imgs = [], []
            for pi in range(n_places):
                pair, _ = gsv_hold[pi]
                db_imgs.append(pair[0])
                q_imgs.append(pair[1])
            random.setstate(rng_state)
            np.random.set_state(np_state)

            all_t = torch.stack(db_imgs + q_imgs)
            feats_hold = {256: [], 4096: []}
            model.eval()
            for st in range(0, len(all_t), args.eval_bs):
                batch_x = all_t[st : st + args.eval_bs].to(device)
                with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=("cuda" in device)):
                    out = model(batch_x, return_aux=True)
                for d in (256, 4096):
                    feats_hold[d].append(out["nested_descriptors"][d].float().cpu().numpy())
            pos_hold = [np.array([qi], dtype=np.int64) for qi in range(n_places)]
            for d in (256, 4096):
                arr = np.concatenate(feats_hold[d], axis=0)
                rec = compute_faiss_recalls(arr[:n_places], arr[n_places:], pos_hold)
                epoch_metrics[f"holdout_WashingtonDC_R1_{d}"] = round(rec[1], 2)
            # For M5_full track both 4096-D & 256-D; for non-Matryoshka ablations track 4096-D primarily
            if args.ablation.upper() == "M5_FULL":
                r1_scores.extend([epoch_metrics["holdout_WashingtonDC_R1_4096"], epoch_metrics["holdout_WashingtonDC_R1_256"]])
            else:
                r1_scores.append(epoch_metrics["holdout_WashingtonDC_R1_4096"])

    if r1_scores:
        inv_sum = sum(1.0 / max(s, 1e-3) for s in r1_scores)
        epoch_metrics["universal_harmonic_R1"] = round(len(r1_scores) / inv_sum, 2)
    else:
        epoch_metrics["universal_harmonic_R1"] = 0.0

    return epoch_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified G3-VPR v3 (CVPR) Training & Ablation Runner")
    p.add_argument(
        "--ablation",
        type=str,
        default="M5_full",
        choices=["M1", "M2a", "M2b", "M2", "M3", "M4a", "M4b", "M4c", "M4", "M5_full"],
    )
    p.add_argument("--gsv_path", type=str, default=None, help="Path to GSV-Cities root")
    p.add_argument("--cities", type=str, default="all", help="Comma-separated cities or 'all'")
    p.add_argument("--img_size", type=int, default=224, help="Input resolution (224 or 322)")
    p.add_argument("--no_aug", action="store_true", help="Disable RandAugment+Grayscale training augmentation")
    p.add_argument("--max_places", type=int, default=0, help="Optional cap on places per epoch (resampled each epoch!)")
    p.add_argument("--img_per_place", type=int, default=4)
    p.add_argument("--bs", type=int, default=32, help="Places per batch (total images = bs * img_per_place)")
    p.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    p.add_argument("--eval_bs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--unfreeze_last_n", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--devices", type=int, default=1)

    p.add_argument("--dinov2_path", type=str, default=None)
    p.add_argument("--warmstart_ckpt", type=str, default=None)
    p.add_argument("--no_warmstart", action="store_true", help="Disable automatic VPR ViT-S warmstart")
    p.add_argument("--resume_ckpt", type=str, default=None)

    p.add_argument("--teacher_2d", type=str, default="megaloc", choices=["megaloc", "cricavpr", "none"])
    p.add_argument("--teacher_3d", type=str, default="depth_anything_v2", choices=["depth_anything_v2", "vggt", "none"])
    p.add_argument("--cricavpr_ckpt", type=str, default=None)
    p.add_argument("--lambda_rel_2d", type=float, default=2.0)
    p.add_argument("--lambda_3d", type=float, default=0.5)
    p.add_argument("--lambda_proto", type=float, default=0.05)
    p.add_argument("--lambda_csd", type=float, default=0.25)

    p.add_argument("--val_pitts30k", type=str, default="./datasets/pitts30k-val")
    p.add_argument("--val_msls", type=str, default="./datasets/msls-val")
    p.add_argument("--eval_datasets_folder", type=str, default="./datasets")
    p.add_argument("--save_dir", type=str, default="./checkpoints/g3vpr_M5_full")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--fast_dev_run", action="store_true")
    return p.parse_args()


def build_epoch_loader(base_gsv_ds: GSVCitiesDataset, args: argparse.Namespace, epoch: int, device: str) -> DataLoader:
    if args.max_places and args.max_places < len(base_gsv_ds):
        rng = np.random.default_rng(args.seed + epoch * 997)
        sel_idx = rng.choice(len(base_gsv_ds), size=args.max_places, replace=False).tolist()
        train_ds = Subset(base_gsv_ds, sel_idx)
    else:
        train_ds = base_gsv_ds

    return DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=("cuda" in device),
        drop_last=True,
    )


def save_fp16_compact_checkpoint(payload: Dict[str, object], out_path: Path) -> None:
    sd = payload["model_state_dict"]  # type: ignore
    fp16_sd = {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}
    compact = {
        "epoch": payload["epoch"],
        "ablation": payload["ablation"],
        "best_universal_score": payload["best_universal_score"],
        "metrics": payload["metrics"],
        "model_state_dict": fp16_sd,
    }
    torch.save(compact, out_path)


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

    warmstart_path = auto_detect_warmstart(args.warmstart_ckpt, disable_warmstart=args.no_warmstart)
    if warmstart_path:
        print(f"[Warm-Start Enabled] Loading ViT-S + MulConvAdapter backbone from: {warmstart_path}")

    all_csvs = sorted([Path(p).stem for p in Path(gsv_dir).glob("Dataframes/*.csv")])
    if args.cities == "all":
        cities = [c for c in all_csvs if c != "WashingtonDC"] if len(all_csvs) > 1 else "all"
    else:
        alias_map = {"Toronto": "TRT", "Oslo": "OSL", "Prague": "PRG", "Paris": "PRS"}
        req = [alias_map.get(c.strip(), c.strip()) for c in args.cities.split(",") if c.strip()]
        valid_set = set(all_csvs)
        cities = [c for c in req if c in valid_set]
        if not cities:
            cities = [c for c in all_csvs if c != "WashingtonDC"] if len(all_csvs) > 1 else "all"

    train_tf = build_eval_transform(args.img_size) if args.no_aug else build_train_transform(args.img_size)
    base_gsv_ds = GSVCitiesDataset(
        dataset_path=gsv_dir, cities=cities, img_per_place=args.img_per_place, transform=train_tf
    )

    cfg = G3VPRConfig.from_ablation(
        ablation=args.ablation,
        pretrained_dinov2_path=args.dinov2_path,
        warmstart_d2vpr_ckpt=warmstart_path,
    )
    model = build_g3vpr(
        ablation=args.ablation,
        pretrained_dinov2_path=args.dinov2_path,
        warmstart_d2vpr_ckpt=warmstart_path,
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

    abl_u = cfg.ablation.upper()
    use_matryoshka = abl_u == "M5_FULL"
    use_2d_kd = abl_u in ("M4", "M4A", "M4C", "M5_FULL") and args.teacher_2d != "none"
    use_3d_kd = abl_u in ("M4", "M4B", "M4C", "M5_FULL") and args.teacher_3d != "none"
    teacher_dim = 10752 if args.teacher_2d.lower() == "cricavpr" else 8448
    criterion = UnifiedG3VPRLoss(
        use_matryoshka=use_matryoshka,
        lambda_rel_2d=args.lambda_rel_2d if use_2d_kd else 0.0,
        lambda_3d=args.lambda_3d if use_3d_kd else 0.0,
        lambda_proto=args.lambda_proto if cfg.use_pmci else 0.0,
        lambda_csd=args.lambda_csd if use_matryoshka else 0.0,
        teacher_2d_dim=teacher_dim,
    ).to(device)

    teacher_engine: Optional[DualTeacherEngine] = None
    if use_2d_kd or use_3d_kd:
        teacher_dev = "cuda:1" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else device
        t2d_name = args.teacher_2d if use_2d_kd else "none"
        t3d_name = args.teacher_3d if use_3d_kd else "none"
        print(f"[Loading Frozen Teachers ({t2d_name}, {t3d_name}) on {teacher_dev}]")
        teacher_engine = DualTeacherEngine(
            teacher_2d=t2d_name,
            teacher_3d=t3d_name,
            cricavpr_ckpt=args.cricavpr_ckpt,
            device=teacher_dev,
            use_fp16=True,
        )

    # Differential LR: gentle LR on pretrained ViT blocks, full LR on adapters, G3 heads & TeacherBridge!
    vit_block_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("backbone.vit.") and "adapter" not in n
    ]
    adapter_and_head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and (not n.startswith("backbone.vit.") or "adapter" in n)
    ] + [p for p in criterion.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": vit_block_params, "lr": args.lr * 0.15},
            {"params": adapter_and_head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    eff_places = min(args.max_places, len(base_gsv_ds)) if args.max_places else len(base_gsv_ds)
    steps_per_epoch = max(eff_places // max(args.bs, 1), 1)
    opt_steps_per_epoch = max(math.ceil(steps_per_epoch / max(args.grad_accum, 1)), 1)
    total_opt_steps = max(opt_steps_per_epoch * max(args.epochs - start_epoch, 1), 1)
    warmup_steps = max(int(0.06 * total_opt_steps), 1)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(total_opt_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", init_scale=4096.0, enabled=("cuda" in device))

    print(
        f"=== Training G3-VPR v2 ({args.ablation}) | Places/ep={eff_places} (Total pool={len(base_gsv_ds)}) "
        f"| BS={args.bs}x{args.img_per_place} | Accum={args.grad_accum} | Device={device} ==="
    )

    for epoch in range(start_epoch, args.epochs):
        train_loader = build_epoch_loader(base_gsv_ds, args, epoch, device)
        model.train()
        criterion.train()
        t_ep = time.perf_counter()
        running_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=105)
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (images, labels) in enumerate(pbar, start=1):
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

            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=("cuda" in device)):
                out = model(images, return_aux=True)
                loss_dict = criterion(out, labels=labels, teacher_2d_desc=t_2d, teacher_depth=t_3d)
                scaled_loss = loss_dict["loss"] / max(args.grad_accum, 1)

            scaler.scale(scaled_loss).backward()

            if step_idx % max(args.grad_accum, 1) == 0 or step_idx == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(criterion.parameters()), max_norm=5.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

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
            save_fp16_compact_checkpoint(ckpt_payload, save_dir / "g3vpr_best_fp16.pth")
        with open(save_dir / "training_metrics.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(f"[Epoch {epoch + 1} Summary] {val_metrics}" + ("  <-- NEW BEST UNIVERSAL!" if is_best else ""))


if __name__ == "__main__":
    main()
