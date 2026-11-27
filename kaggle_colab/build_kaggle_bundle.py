import base64
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
include_dirs = ["g3vpr", "backbone", "src", "experiments"]
include_files = [
    "commons.py",
    "D2VPR.py",
    "datasets_ws.py",
    "loss.py",
    "network.py",
    "parser.py",
    "test.py",
    "util.py",
    "train_g3vpr.py",
    "eval_g3vpr_sota.py",
    "eval_ablation_suite.py",
    "eval_cvpr_v3_suite.py",
    "requirements.txt",
]


def main() -> None:
    buf = io.BytesIO()
    zip_path = ROOT / "kaggle_colab" / "g3vpr_code_only.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=buf, mode="w:gz") as tf, zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in include_files:
            fp = ROOT / f
            if fp.is_file():
                tf.add(fp, arcname=f)
                zf.write(fp, arcname=f)
        for d in include_dirs:
            dp = ROOT / d
            for p in sorted(dp.rglob("*")):
                if (
                    p.is_file()
                    and "__pycache__" not in str(p)
                    and ".ipynb_checkpoints" not in str(p)
                    and p.suffix in (".py", ".txt", "")
                ):
                    rel = p.relative_to(ROOT).as_posix()
                    tf.add(p, arcname=rel)
                    zf.write(p, arcname=rel)

    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    b64_chunks = [b64_str[i : i + 100] for i in range(0, len(b64_str), 100)]

    cell1_source = (
        [
            "# Cell 1: Tu dong giai nen toan bo code G3-VPR v3 (CVPR All-in-One) ngay trong Kaggle/Colab\n",
            "import base64, io, os, subprocess, sys, tarfile, torch\n",
            "\n",
            'WORK_ROOT = "/kaggle/working/D2VPR" if os.path.isdir("/kaggle/working") else os.path.abspath("./D2VPR_cloud")\n',
            "os.makedirs(WORK_ROOT, exist_ok=True)\n",
            "\n",
            "ARCHIVE_B64 = (\n",
        ]
        + [f'    "{ch}"\n' for ch in b64_chunks]
        + [
            ")\n",
            'with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARCHIVE_B64)), mode="r:gz") as tf:\n',
            "    tf.extractall(WORK_ROOT)\n",
            "os.chdir(WORK_ROOT)\n",
            'print("[OK] Da giai nen toan bo du an G3-VPR v3 (CVPR) tai:", os.getcwd())\n',
            'print("PyTorch:", torch.__version__, "| CUDA GPU:", torch.cuda.is_available(), "| So GPU:", torch.cuda.device_count())\n',
            "if not torch.cuda.is_available():\n",
            '    print("CANH BAO: Ban chua bat GPU! Vao Settings ben phai -> Accelerator -> chon GPU T4 x2 va bat Internet ON!")\n',
            "else:\n",
            "    for i in range(torch.cuda.device_count()):\n",
            '        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)")\n',
            "\n",
            'subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)\n',
            'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pytorch-metric-learning"], check=True)\n',
            'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"], check=True)\n',
            'print("[OK] Moi truong thu vien G3-VPR v3 da san sang 100%!")',
        ]
    )

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# G3-VPR v3 (CVPR A*): Train + Ablation + Chấm SOTA + G-OTR + Conformal Cascade 100% Trên Kaggle (`T4 x2`)\n",
                    "**Toàn bộ nâng cấp chuẩn CVPR trong bản `v3`:**\n",
                    "- **Vá lỗi v2:** Bật `RandAugment + RandomGrayscale (20%) + ColorJitter` (khắc phục hoàn toàn lỗi rớt `AmsterTime`) + **cấm tuyệt đối rò rỉ tập Test** lúc chọn checkpoint.\n",
                    "- **3 Đột phá hợp nhất:** `G-OTR` (xác minh hình học 3D Top-20 không tốn FLOPs backbone), `Nyström-PMCI` (trừ nhiễu miền + SwAV chống chết nguyên mẫu), và `AM-VPR Conformal Cascade` (`CSD` + thoát sớm bảo chứng rủi ro).\n",
                    "- **Chấm điểm trực tiếp trên Kaggle:** Tự động chấm `G3-VPR v3` (`128..4096-D + G-OTR + Cascade`), so sánh **SOTA (`MegaLoc`, `DINOv2-SALAD`, `EigenPlaces`) ở `BS=1`**, chấm bảng **Ablation đồng ngân sách**, đo **FAISS 1M**, và xuất sẵn báo cáo **`CVPR_V3_SUMMARY_REPORT.md`** trong file ZIP Output!\n",
                    "\n",
                    "**Cách chạy trên Kaggle:**\n",
                    "1. **Settings** bên phải: chọn **Accelerator → `GPU T4 x2`** và bật **Internet → `ON`**.\n",
                    "2. **Input** bên phải: bấm **`+ Add Input`** → gắn **`amaralibey/gsv-cities`** (nếu bạn có gắn thêm `pitts30k` / `sped` / `amstertime` / `msls` thì Cell 6 sẽ tự nhận và chấm luôn!).\n",
                    "3. Bấm **`Save Version` → `Save & Run All (Commit)`** rồi tải file `g3vpr_paper_artifacts_v3.zip` về đối chiếu!",
                ],
            },
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell1_source},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 2: Tai trong so khoi dong am ViT-S+Adapter, DINOv2-S & Tu dong tim GSV-Cities + VPR test sets\n",
                    "import os, urllib.request\n",
                    'if os.path.isdir("/kaggle/working/D2VPR"):\n',
                    '    os.chdir("/kaggle/working/D2VPR")\n',
                    "\n",
                    'os.makedirs("./weights_init", exist_ok=True)\n',
                    'warm_ckpt = "./weights_init/vits_adapter_warmstart_fp16.pt"\n',
                    "if not os.path.isfile(warm_ckpt):\n",
                    "    for u in [\n",
                    '        "https://raw.githubusercontent.com/phuccode25/G3-VPR/main/weights_init/vits_adapter_warmstart_fp16.pt",\n',
                    '        "https://github.com/phuccode25/G3-VPR/raw/main/weights_init/vits_adapter_warmstart_fp16.pt",\n',
                    "    ]:\n",
                    "        try:\n",
                    '            print("Dang tai trong so khoi dong am ViT-S+Adapter tu:", u)\n',
                    "            urllib.request.urlretrieve(u, warm_ckpt)\n",
                    "            if os.path.isfile(warm_ckpt) and os.path.getsize(warm_ckpt) > 10_000_000:\n",
                    "                break\n",
                    "        except Exception as e:\n",
                    '            print("Thu URL tiep theo do:", e)\n',
                    'print("[OK] Warm-start ViT-S:", os.path.isfile(warm_ckpt))\n',
                    "\n",
                    'dinov2_ckpt = "./dinov2_vits14_pretrain.pth"\n',
                    "if not os.path.isfile(dinov2_ckpt):\n",
                    '    print("Dang tai trong so DINOv2 ViT-S/14...")\n',
                    "    urllib.request.urlretrieve(\n",
                    '        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",\n',
                    "        dinov2_ckpt\n",
                    "    )\n",
                    'print("[OK] DINOv2-S:", os.path.isfile(dinov2_ckpt))\n',
                    "\n",
                    "GSV_PATH = None\n",
                    'if os.path.isdir("/kaggle/input"):\n',
                    '    for r, dirs, _ in os.walk("/kaggle/input"):\n',
                    '        if "Dataframes" in dirs and "Images" in dirs:\n',
                    "            GSV_PATH = r\n",
                    "            break\n",
                    "if GSV_PATH is None:\n",
                    "    import kagglehub\n",
                    '    GSV_PATH = kagglehub.dataset_download("amaralibey/gsv-cities")\n',
                    'print("[OK] Da tim thay GSV-Cities tai:", GSV_PATH)',
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 3: Kiem tra kien truc G3-VPR v3 (Nystrom-PMCI + G-OTR), tinh bat bien BS=1 & toc do GPU T4\n",
                    "import os, subprocess, sys\n",
                    'if os.path.isdir("/kaggle/working/D2VPR"):\n',
                    '    os.chdir("/kaggle/working/D2VPR")\n',
                    'subprocess.run([sys.executable, "-u", "experiments/exp2_architecture_verify_and_profile.py"], check=True)',
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 4: Huan luyen Mo hinh Chinh G3-VPR v3 (M5_full) voi Data Aug + Nystrom-PMCI + CSD + Dual Teachers\n",
                    "import os, subprocess, sys\n",
                    'if os.path.isdir("/kaggle/working/D2VPR"):\n',
                    '    os.chdir("/kaggle/working/D2VPR")\n',
                    "\n",
                    'SAVE_DIR = "./checkpoints/g3vpr_M5_full"\n',
                    "cmd = [\n",
                    '    sys.executable, "-u", "train_g3vpr.py",\n',
                    '    "--ablation", "M5_full",\n',
                    '    "--gsv_path", GSV_PATH,\n',
                    '    "--dinov2_path", "./dinov2_vits14_pretrain.pth",\n',
                    '    "--teacher_2d", "megaloc",\n',
                    '    "--teacher_3d", "depth_anything_v2",\n',
                    '    "--bs", "32",\n',
                    '    "--grad_accum", "2",\n',
                    '    "--epochs", "8",\n',
                    '    "--max_places", "22000",\n',
                    '    "--lr", "3e-4",\n',
                    '    "--unfreeze_last_n", "6",\n',
                    '    "--save_dir", SAVE_DIR,\n',
                    "]\n",
                    'if os.path.isfile("./weights_init/vits_adapter_warmstart_fp16.pt"):\n',
                    '    cmd.extend(["--warmstart_ckpt", "./weights_init/vits_adapter_warmstart_fp16.pt"])\n',
                    'if os.path.isfile(f"{SAVE_DIR}/g3vpr_last.pth"):\n',
                    '    cmd.extend(["--resume_ckpt", f"{SAVE_DIR}/g3vpr_last.pth"])\n',
                    "\n",
                    'print("Running G3-VPR v3 M5_full:", " ".join(cmd))\n',
                    "subprocess.run(cmd, stdin=subprocess.DEVNULL, check=True)\n",
                    'if os.path.isfile(f"{SAVE_DIR}/g3vpr_last.pth"):\n',
                    '    os.remove(f"{SAVE_DIR}/g3vpr_last.pth")',
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 5: Bang Ablation Dong Ngan Sach & Tach Sach Bien (CVPR Table 4: 4 ep x 6000 places cho moi dong)\n",
                    "import os, subprocess, sys\n",
                    'if os.path.isdir("/kaggle/working/D2VPR"):\n',
                    '    os.chdir("/kaggle/working/D2VPR")\n',
                    "\n",
                    "ABLATION_SPECS = [\n",
                    '    ("M1", "M1", "none", "none", False),\n',
                    '    ("M2a_GatedDRM", "M2a", "none", "none", False),\n',
                    '    ("M2b_3DDGSA", "M2b", "none", "none", False),\n',
                    '    ("M3_NystromPMCI", "M3", "none", "none", False),\n',
                    '    ("M4a_2DKD_only", "M4a", "megaloc", "none", False),\n',
                    '    ("M4b_3DKD_only", "M4b", "none", "depth_anything_v2", False),\n',
                    '    ("M4c_DualKD", "M4c", "megaloc", "depth_anything_v2", False),\n',
                    '    ("M5_EqBudget", "M5_full", "megaloc", "depth_anything_v2", False),\n',
                    '    ("M5_NoWarmStart", "M5_full", "megaloc", "depth_anything_v2", True),\n',
                    "]\n",
                    "for tag, abl_cfg, t2d, t3d, no_ws in ABLATION_SPECS:\n",
                    '    out_d = f"./checkpoints/ablation_{tag}"\n',
                    "    abl_cmd = [\n",
                    '        sys.executable, "-u", "train_g3vpr.py",\n',
                    '        "--ablation", abl_cfg,\n',
                    '        "--gsv_path", GSV_PATH,\n',
                    '        "--cities", "all",\n',
                    '        "--dinov2_path", "./dinov2_vits14_pretrain.pth",\n',
                    '        "--teacher_2d", t2d,\n',
                    '        "--teacher_3d", t3d,\n',
                    '        "--bs", "32",\n',
                    '        "--grad_accum", "2",\n',
                    '        "--epochs", "4",\n',
                    '        "--max_places", "6000",\n',
                    '        "--lr", "3e-4",\n',
                    '        "--unfreeze_last_n", "6",\n',
                    '        "--save_dir", out_d,\n',
                    "    ]\n",
                    "    if no_ws:\n",
                    '        abl_cmd.append("--no_warmstart")\n',
                    '    elif os.path.isfile("./weights_init/vits_adapter_warmstart_fp16.pt"):\n',
                    '        abl_cmd.extend(["--warmstart_ckpt", "./weights_init/vits_adapter_warmstart_fp16.pt"])\n',
                    '    print(f"\\n=== Running Equal-Budget Ablation {tag} ({abl_cfg}, 2D={t2d}, 3D={t3d}, no_ws={no_ws}) ===")\n',
                    "    subprocess.run(abl_cmd, stdin=subprocess.DEVNULL, check=True)\n",
                    '    for rm_fn in ("g3vpr_last.pth", "g3vpr_universal_best.pth"):\n',
                    '        p_rm = os.path.join(out_d, rm_fn)\n',
                    "        if os.path.isfile(p_rm):\n",
                    "            os.remove(p_rm)",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 6: CHAM DIEM TRUC TIEP TREN KAGGLE (G3-VPR v3 + G-OTR + Conformal Cascade + SOTA Hub + Ablations + FAISS 1M)\n",
                    "import os, shutil, subprocess, sys\n",
                    'if os.path.isdir("/kaggle/working/D2VPR"):\n',
                    '    os.chdir("/kaggle/working/D2VPR")\n',
                    "\n",
                    'm5_ckpt = "./checkpoints/g3vpr_M5_full/g3vpr_universal_best.pth"\n',
                    "if os.path.isfile(m5_ckpt):\n",
                    "    subprocess.run([\n",
                    '        sys.executable, "-u", "eval_cvpr_v3_suite.py",\n',
                    '        "--ckpt", m5_ckpt,\n',
                    '        "--checkpoints_dir", "./checkpoints",\n',
                    '        "--query_bs", "1",\n',
                    '        "--run_sota_hub",\n',
                    '        "--run_faiss_1m",\n',
                    '        "--output_dir", "./results",\n',
                    "    ], check=True)\n",
                    "    subprocess.run([\n",
                    '        sys.executable, "-u", "experiments/visualize_gaussians.py",\n',
                    '        "--ckpt", m5_ckpt,\n',
                    "    ], check=True)\n",
                    "\n",
                    '    report_md = "./results/CVPR_V3_SUMMARY_REPORT.md"\n',
                    "    if os.path.isfile(report_md):\n",
                    '        print("\\n" + "=" * 80)\n',
                    '        print(open(report_md, encoding="utf-8").read())\n',
                    '        print("=" * 80)\n',
                    "\n",
                    'zip_out = "/kaggle/working/g3vpr_paper_artifacts_v3" if os.path.isdir("/kaggle/working") else "./g3vpr_paper_artifacts_v3"\n',
                    'shutil.rmtree("./export_bundle", ignore_errors=True)\n',
                    'os.makedirs("./export_bundle/results", exist_ok=True)\n',
                    'os.makedirs("./export_bundle/checkpoints/g3vpr_M5_full", exist_ok=True)\n',
                    'if os.path.isdir("./results"):\n',
                    '    shutil.copytree("./results", "./export_bundle/results", dirs_exist_ok=True)\n',
                    "\n",
                    'for main_fn in ("g3vpr_universal_best.pth", "g3vpr_best_fp16.pth", "training_metrics.json"):\n',
                    '    src_p = f"./checkpoints/g3vpr_M5_full/{main_fn}"\n',
                    "    if os.path.isfile(src_p):\n",
                    '        shutil.copy2(src_p, f"./export_bundle/checkpoints/g3vpr_M5_full/{main_fn}")\n',
                    "\n",
                    'for entry in sorted(os.listdir("./checkpoints")):\n',
                    '    if entry.startswith("ablation_"):\n',
                    '        abl_src = os.path.join("./checkpoints", entry)\n',
                    '        abl_dst = os.path.join("./export_bundle/checkpoints", entry)\n',
                    "        os.makedirs(abl_dst, exist_ok=True)\n",
                    '        for fn in ("g3vpr_best_fp16.pth", "training_metrics.json"):\n',
                    "            fp = os.path.join(abl_src, fn)\n",
                    "            if os.path.isfile(fp):\n",
                    "                shutil.copy2(fp, os.path.join(abl_dst, fn))\n",
                    "\n",
                    'archive_file = shutil.make_archive(zip_out, "zip", "./export_bundle")\n',
                    'print("[HOAN TAT G3-VPR V3] Tai file ket qua + bao cao doi chieu tai Output:", archive_file)',
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path = ROOT / "kaggle_colab" / "G3VPR_Kaggle_Colab_Runner.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Notebook size: {nb_path.stat().st_size / 1024:.1f} KB | Code zip size: {zip_path.stat().st_size / 1024:.1f} KB")

    with tempfile.TemporaryDirectory() as tmpd:
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64_str)), mode="r:gz") as tf:
            tf.extractall(tmpd)
        r1 = subprocess.run([sys.executable, "train_g3vpr.py", "--help"], cwd=tmpd, capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "eval_cvpr_v3_suite.py", "--help"], cwd=tmpd, capture_output=True, text=True)
        assert r1.returncode == 0, r1.stderr
        assert r2.returncode == 0, r2.stderr

        # Full isolated E2E v3 training + validation + G-OTR + Conformal Cascade + FP16 checkpoint verification!
        import pandas as pd
        from PIL import Image

        mock_gsv = Path(tmpd) / "mock_gsv"
        (mock_gsv / "Dataframes").mkdir(parents=True)
        for city in ("London", "WashingtonDC"):
            (mock_gsv / "Images" / city).mkdir(parents=True)
            rows = []
            for pid in range(1, 25):
                for v in range(4):
                    pano = f"pano{pid}_{v}"
                    yr, mo, nd, lat, lon = 2022, 5, 90 * v, "51.5", "-0.1"
                    pl_str = str(pid).zfill(7)
                    fn = f"{city}_{pl_str}_{str(yr).zfill(4)}_{str(mo).zfill(2)}_{str(nd).zfill(3)}_{lat}_{lon}_{pano}.jpg"
                    Image.new("RGB", (64, 64), ((pid * 11) % 255, (v * 50) % 255, 120)).save(mock_gsv / "Images" / city / fn)
                    rows.append({"place_id": pid, "city_id": city, "panoid": pano, "year": yr, "month": mo, "northdeg": nd, "lat": lat, "lon": lon})
            pd.DataFrame(rows).to_csv(mock_gsv / "Dataframes" / f"{city}.csv", index=False)

        r3 = subprocess.run(
            [
                sys.executable,
                "train_g3vpr.py",
                "--ablation", "M5_full",
                "--gsv_path", str(mock_gsv),
                "--no_warmstart",
                "--teacher_2d", "none",
                "--teacher_3d", "none",
                "--bs", "4",
                "--grad_accum", "2",
                "--epochs", "1",
                "--num_workers", "0",
                "--save_dir", str(Path(tmpd) / "ckpts"),
            ],
            cwd=tmpd,
            capture_output=True,
            text=True,
        )
        assert r3.returncode == 0, f"E2E v3 train failed:\nSTDOUT:\n{r3.stdout}\nSTDERR:\n{r3.stderr}"
        assert (Path(tmpd) / "ckpts" / "g3vpr_universal_best.pth").is_file(), "Checkpoint was not created!"
        assert (Path(tmpd) / "ckpts" / "g3vpr_best_fp16.pth").is_file(), "FP16 Checkpoint was not created!"

        r4 = subprocess.run(
            [
                sys.executable,
                "eval_cvpr_v3_suite.py",
                "--ckpt", str(Path(tmpd) / "ckpts" / "g3vpr_universal_best.pth"),
                "--datasets_root", str(mock_gsv),
                "--query_bs", "1",
                "--output_dir", str(Path(tmpd) / "res_v3"),
            ],
            cwd=tmpd,
            capture_output=True,
            text=True,
        )
        assert r4.returncode == 0, f"E2E v3 eval failed:\nSTDOUT:\n{r4.stdout}\nSTDERR:\n{r4.stderr}"
        assert (Path(tmpd) / "res_v3" / "cvpr_v3_full_results.json").is_file(), "v3 JSON missing!"
        assert (Path(tmpd) / "res_v3" / "CVPR_V3_SUMMARY_REPORT.md").is_file(), "v3 MD report missing!"
        print("Isolated fresh-directory E2E v3 Training + G-OTR + Conformal Cascade + Report Verification PASSED 100%!")


if __name__ == "__main__":
    main()
