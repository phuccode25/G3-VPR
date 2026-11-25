import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(r"d:\Lab\D2VPR")
sys.path.insert(0, str(ROOT))

import commons
import D2VPR
import datasets_ws
import network
import test
from loss import distillation_loss, loss_function


def strip_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if keys and keys[0].startswith("module."):
        return OrderedDict({k.replace("module.", "", 1): v for k, v in state_dict.items()})
    return state_dict


def load_model(ckpt_path: Path, cross_image_encoder: bool, device: str):
    model = D2VPR.D2VPR(cross_image_encoder=cross_image_encoder).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    sd = strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, f"Mismatch loading {ckpt_path}: missing={missing}, unexpected={unexpected}"
    model.eval()
    return model


def analyze_parameters():
    m_enc = D2VPR.D2VPR(cross_image_encoder=True)
    m_no = D2VPR.D2VPR(cross_image_encoder=False)
    teacher = network.CricaVPRNet()

    def group_counts(model):
        groups = {
            "backbone_base": 0,
            "adapter": 0,
            "drm_lin": 0,
            "drm_up_norm": 0,
            "tdda_offset_generator": 0,
            "tdda_topdown_fusion": 0,
            "tdda_pos_encoder": 0,
            "tdda_aggregation": 0,
            "cross_image_encoder": 0,
            "other": 0,
        }
        for name, param in model.named_parameters():
            n = param.numel()
            if "adapter" in name:
                groups["adapter"] += n
            elif name.startswith("backbone.lin"):
                groups["drm_lin"] += n
            elif name.startswith("backbone.up_norm"):
                groups["drm_up_norm"] += n
            elif name.startswith("backbone."):
                groups["backbone_base"] += n
            elif name.startswith("offset_generator"):
                groups["tdda_offset_generator"] += n
            elif name.startswith("topdown_fusion"):
                groups["tdda_topdown_fusion"] += n
            elif name.startswith("pos_encoder"):
                groups["tdda_pos_encoder"] += n
            elif name.startswith("aggregation"):
                groups["tdda_aggregation"] += n
            elif name.startswith("encoder."):
                groups["cross_image_encoder"] += n
            else:
                groups["other"] += n
        groups["drm_total"] = groups["drm_lin"] + groups["drm_up_norm"]
        groups["tdda_total"] = (
            groups["tdda_offset_generator"]
            + groups["tdda_topdown_fusion"]
            + groups["tdda_pos_encoder"]
            + groups["tdda_aggregation"]
        )
        total = sum(p.numel() for p in model.parameters())
        return groups, total

    enc_groups, enc_total = group_counts(m_enc)
    no_groups, no_total = group_counts(m_no)
    teacher_total = sum(p.numel() for p in teacher.parameters())
    teacher_backbone = sum(p.numel() for p in teacher.backbone.parameters())

    reduction_enc_pct = (1.0 - enc_total / teacher_total) * 100.0
    reduction_no_pct = (1.0 - no_total / teacher_total) * 100.0

    return {
        "d2vpr_encoder_breakdown_M": {k: round(v / 1e6, 4) for k, v in enc_groups.items()},
        "d2vpr_encoder_total_M": round(enc_total / 1e6, 4),
        "d2vpr_no_encoder_total_M": round(no_total / 1e6, 4),
        "teacher_cricavpr_total_M": round(teacher_total / 1e6, 4),
        "teacher_cricavpr_backbone_with_adapter_M": round(teacher_backbone / 1e6, 4),
        "param_reduction_encoder_vs_cricavpr_pct": round(reduction_enc_pct, 2),
        "param_reduction_no_encoder_vs_cricavpr_pct": round(reduction_no_pct, 2),
    }


def make_eval_args(infer_batch_size: int, device: str = "cuda"):
    return SimpleNamespace(
        resize=[224, 224],
        test_method="hard_resize",
        val_positive_dist_threshold=25,
        efficient_ram_testing=False,
        infer_batch_size=infer_batch_size,
        device=device,
        features_dim=14 * 768,
        recall_values=[1, 5, 10],
    )


def run_retrieval_evaluations(device: str = "cuda"):
    datasets_root = str(ROOT / "datasets")
    configs = [
        {
            "name": "D2VPR_no_encoder_SPED_bs32",
            "ckpt": ROOT / "weights" / "finetune_no_encoder.pth",
            "cross_image_encoder": False,
            "dataset": "sped",
            "batch_size": 32,
            "paper_ref": {"table": "Table 8 (no PCA)", "R@1": 86.2, "R@5": 93.2},
        },
        {
            "name": "D2VPR_no_encoder_AmsterTime_bs32",
            "ckpt": ROOT / "weights" / "finetune_no_encoder.pth",
            "cross_image_encoder": False,
            "dataset": "amstertime",
            "batch_size": 32,
            "paper_ref": {"table": "Table 8 (no PCA)", "R@1": 49.1, "R@5": 70.3},
        },
        {
            "name": "D2VPR_encoder_SPED_bs3",
            "ckpt": ROOT / "weights" / "distillation.pth",
            "cross_image_encoder": True,
            "dataset": "sped",
            "batch_size": 3,
            "paper_ref": {"table": "Table 9 (no PCA)", "R@1": 91.3, "R@5": 96.2},
        },
        {
            "name": "D2VPR_encoder_SPED_bs1",
            "ckpt": ROOT / "weights" / "distillation.pth",
            "cross_image_encoder": True,
            "dataset": "sped",
            "batch_size": 1,
            "paper_ref": {"table": "Table 8 (no PCA)", "R@1": 83.2, "R@5": 91.4},
        },
        {
            "name": "D2VPR_encoder_AmsterTime_bs8",
            "ckpt": ROOT / "weights" / "distillation.pth",
            "cross_image_encoder": True,
            "dataset": "amstertime",
            "batch_size": 8,
            "paper_ref": {"table": "Table 9 (no PCA)", "R@1": 62.9, "R@5": 80.7},
        },
        {
            "name": "D2VPR_encoder_AmsterTime_bs1",
            "ckpt": ROOT / "weights" / "distillation.pth",
            "cross_image_encoder": True,
            "dataset": "amstertime",
            "batch_size": 1,
            "paper_ref": {"table": "Table 8 (no PCA)", "R@1": 48.7, "R@5": 69.6},
        },
        {
            "name": "D2VPR_encoder_finetune_SPED_bs1",
            "ckpt": ROOT / "weights" / "finetune.pth",
            "cross_image_encoder": True,
            "dataset": "sped",
            "batch_size": 1,
            "paper_ref": {"table": "Table 8 (no PCA)", "R@1": 83.2, "R@5": 91.4},
        },
        {
            "name": "D2VPR_encoder_finetune_SPED_bs3",
            "ckpt": ROOT / "weights" / "finetune.pth",
            "cross_image_encoder": True,
            "dataset": "sped",
            "batch_size": 3,
            "paper_ref": {"table": "Table 9 (distillation.pth ref)", "R@1": 91.3, "R@5": 96.2},
        },
        {
            "name": "D2VPR_encoder_finetune_AmsterTime_bs1",
            "ckpt": ROOT / "weights" / "finetune.pth",
            "cross_image_encoder": True,
            "dataset": "amstertime",
            "batch_size": 1,
            "paper_ref": {"table": "Table 8 (no PCA)", "R@1": 48.7, "R@5": 69.6},
        },
        {
            "name": "D2VPR_encoder_finetune_AmsterTime_bs8",
            "ckpt": ROOT / "weights" / "finetune.pth",
            "cross_image_encoder": True,
            "dataset": "amstertime",
            "batch_size": 8,
            "paper_ref": {"table": "Table 9 (distillation.pth ref)", "R@1": 62.9, "R@5": 80.7},
        },
    ]

    results = []
    cached_models = {}
    cached_datasets = {}

    for cfg in configs:
        commons.make_deterministic(0)
        args = make_eval_args(cfg["batch_size"], device=device)
        mkey = (str(cfg["ckpt"]), cfg["cross_image_encoder"])
        if mkey not in cached_models:
            cached_models[mkey] = load_model(cfg["ckpt"], cfg["cross_image_encoder"], device)
        model = cached_models[mkey]

        dkey = cfg["dataset"]
        if dkey not in cached_datasets:
            cached_datasets[dkey] = datasets_ws.BaseDataset(args, datasets_root, dkey, "test")
        ds = cached_datasets[dkey]

        t0 = time.perf_counter()
        recalls, recalls_str = test.test(args, ds, model, args.test_method, pca=None)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        entry = {
            "name": cfg["name"],
            "dataset": cfg["dataset"],
            "batch_size": cfg["batch_size"],
            "cross_image_encoder": cfg["cross_image_encoder"],
            "db_num": int(ds.database_num),
            "queries_num": int(ds.queries_num),
            "R@1": round(float(recalls[0]), 2),
            "R@5": round(float(recalls[1]), 2),
            "R@10": round(float(recalls[2]), 2),
            "recalls_str": recalls_str,
            "wall_time_s": round(elapsed, 3),
            "paper_ref": cfg["paper_ref"],
        }
        results.append(entry)
        print(f"[EVAL] {cfg['name']}: {recalls_str} (time={elapsed:.2f}s) | Paper: {cfg['paper_ref']}")

    return results


def benchmark_amstertime_latency_and_vram(device: str = "cuda"):
    commons.make_deterministic(0)
    args = make_eval_args(infer_batch_size=32, device=device)
    ds = datasets_ws.BaseDataset(args, str(ROOT / "datasets"), "amstertime", "test")
    model = load_model(ROOT / "weights" / "distillation.pth", True, device)

    # Warmup
    with torch.no_grad():
        dummy = torch.randn(4, 3, 224, 224, device=device)
        _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    all_features = np.empty((len(ds), args.features_dim), dtype="float32")
    db_subset = Subset(ds, list(range(ds.database_num)))
    q_subset = Subset(ds, list(range(ds.database_num, ds.database_num + ds.queries_num)))
    db_loader = DataLoader(db_subset, batch_size=32, num_workers=0, pin_memory=(device == "cuda"))
    q_loader = DataLoader(q_subset, batch_size=32, num_workers=0, pin_memory=(device == "cuda"))

    t_extract_start = time.perf_counter()
    with torch.no_grad():
        for inputs, indices in db_loader:
            feats = model(inputs.to(device)).cpu().numpy()
            all_features[indices.numpy(), :] = feats
        for inputs, indices in q_loader:
            feats = model(inputs.to(device)).cpu().numpy()
            all_features[indices.numpy(), :] = feats
        if device == "cuda":
            torch.cuda.synchronize()
    extract_time_s = time.perf_counter() - t_extract_start

    peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2) if device == "cuda" else 0.0

    db_feats = all_features[: ds.database_num]
    q_feats = all_features[ds.database_num :]

    t_faiss_start = time.perf_counter()
    index = faiss.IndexFlatL2(args.features_dim)
    index.add(db_feats)
    _, _ = index.search(q_feats, 10)
    faiss_time_s = time.perf_counter() - t_faiss_start

    return {
        "dataset": "amstertime",
        "total_images": len(ds),
        "batch_size": 32,
        "feature_dim": args.features_dim,
        "feature_extraction_time_s": round(extract_time_s, 3),
        "faiss_index_and_search_time_s": round(faiss_time_s, 4),
        "total_inference_time_s": round(extract_time_s + faiss_time_s, 3),
        "throughput_img_per_s": round(len(ds) / extract_time_s, 2),
        "peak_vram_allocated_MB": round(peak_alloc_mb, 2),
        "peak_vram_reserved_MB": round(peak_reserved_mb, 2),
    }


def training_smoke_test(device: str = "cuda"):
    commons.make_deterministic(0)
    results = {}

    # Stage 1: Knowledge Distillation (MSE)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model_s1 = D2VPR.D2VPR(cross_image_encoder=True).to(device).train()
    opt_s1 = torch.optim.Adam(model_s1.parameters(), lr=2.5e-5)
    images = torch.randn(4, 3, 224, 224, device=device)
    teacher_target = torch.nn.functional.normalize(torch.randn(4, 14 * 768, device=device), p=2, dim=-1)
    desc_s1 = model_s1(images)
    loss_s1 = distillation_loss(desc_s1, teacher_target)
    opt_s1.zero_grad()
    loss_s1.backward()
    grad_norm_s1 = sum(p.grad.detach().norm().item() for p in model_s1.parameters() if p.grad is not None)
    opt_s1.step()
    if device == "cuda":
        torch.cuda.synchronize()
    results["stage1_distillation_smoke"] = {
        "batch_size": 4,
        "output_shape": list(desc_s1.shape),
        "loss": round(float(loss_s1.item()), 6),
        "loss_finite": bool(torch.isfinite(loss_s1).item()),
        "total_grad_norm": round(float(grad_norm_s1), 4),
        "peak_vram_allocated_MB": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if device == "cuda" else 0.0,
    }
    del model_s1, opt_s1, desc_s1, loss_s1

    # Stage 2: Fine-tuning (Multi-Similarity Loss with frozen early blocks)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model_s2 = D2VPR.D2VPR(cross_image_encoder=False).to(device).train()
    trainable_params = 0
    for name, param in model_s2.named_parameters():
        param.requires_grad = any(
            k in name
            for k in ["blocks.8", "blocks.9", "blocks.10", "blocks.11", "lin", "up_norm", "aggregation", "encoder", "topdown", "offset"]
        )
        if param.requires_grad:
            trainable_params += param.numel()
    opt_s2 = torch.optim.AdamW([p for p in model_s2.parameters() if p.requires_grad], lr=2e-4)
    labels = torch.tensor([0, 0, 1, 1], device=device)
    desc_s2 = model_s2(images)
    loss_s2 = loss_function(desc_s2, labels)
    opt_s2.zero_grad()
    loss_s2.backward()
    grad_norm_s2 = sum(p.grad.detach().norm().item() for p in model_s2.parameters() if p.grad is not None)
    opt_s2.step()
    if device == "cuda":
        torch.cuda.synchronize()
    results["stage2_finetune_smoke"] = {
        "batch_size": 4,
        "trainable_params_M": round(trainable_params / 1e6, 4),
        "output_shape": list(desc_s2.shape),
        "loss": round(float(loss_s2.item()), 6),
        "loss_finite": bool(torch.isfinite(loss_s2).item()),
        "total_grad_norm": round(float(grad_norm_s2), 4),
        "peak_vram_allocated_MB": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if device == "cuda" else 0.0,
    }
    return results


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU is required!"
    device_name = torch.cuda.get_device_name(0)
    print(f"Running D2-VPR benchmark reproduction on: {device_name} (PyTorch {torch.__version__})")

    summary = {
        "environment": {
            "gpu": device_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "faiss": faiss.__version__,
        },
        "parameters": analyze_parameters(),
        "retrieval": run_retrieval_evaluations("cuda"),
        "latency_and_vram": benchmark_amstertime_latency_and_vram("cuda"),
        "training_smoke": training_smoke_test("cuda"),
    }

    out_path = ROOT / "reproduction_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full reproduction results to {out_path}")
    print(json.dumps(summary, indent=2))
