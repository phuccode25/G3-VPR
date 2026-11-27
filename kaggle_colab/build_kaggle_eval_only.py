"""
Builds `kaggle_colab/G3VPR_v3_Kaggle_Eval_Only.ipynb` — a 100% standalone
Kaggle Evaluation-Only Notebook that:
  1. Unpacks all G3-VPR v3 code & installs faiss-cpu + pytorch-metric-learning.
  2. Auto-detects the user's v3 checkpoints from /kaggle/input (or previous Kaggle Output).
  3. Auto-detects mounted VPR datasets AND auto-downloads lightweight cross-domain
     benchmarks (SPED, AmsterTime, etc.) via gmberton/VPR-datasets-downloader if missing.
  4. Evaluates G3-VPR v3 (128..4096-D + G-OTR + Conformal Cascade) + all 9 Ablations
     + SOTA baselines (MegaLoc, DINOv2-SALAD, EigenPlaces) at strict BS=1 + FAISS 1M.
  5. Exports `CVPR_V3_SUMMARY_REPORT.md` & `g3vpr_v3_crossdomain_eval.zip`.
"""

import base64
import io
import json
import tarfile
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
    "eval_cvpr_v3_suite.py",
    "eval_g3vpr_sota.py",
    "requirements.txt",
]


def main() -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for f in include_files:
            fp = ROOT / f
            if fp.is_file():
                tf.add(fp, arcname=f)
        for d in include_dirs:
            dp = ROOT / d
            for p in sorted(dp.rglob("*")):
                if (
                    p.is_file()
                    and "__pycache__" not in str(p)
                    and ".ipynb_checkpoints" not in str(p)
                    and p.suffix in (".py", ".txt", "")
                ):
                    tf.add(p, arcname=p.relative_to(ROOT).as_posix())

    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    b64_chunks = [b64_str[i : i + 100] for i in range(0, len(b64_str), 100)]

    cell1_source = (
        [
            "# Cell 1: Giai nen code G3-VPR v3 & cai thu vien cham diem (khong can train lai!)\n",
            "import base64, io, os, subprocess, sys, tarfile, torch\n",
            "\n",
            'WORK_ROOT = "/kaggle/working/D2VPR_eval" if os.path.isdir("/kaggle/working") else os.path.abspath("./D2VPR_eval")\n',
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
            'print("[OK] Da giai nen code G3-VPR v3 tai:", os.getcwd())\n',
            'print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (hay bat GPU T4!)")\n',
            'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu", "gdown"], check=True)\n',
            'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pytorch-metric-learning"], check=True)\n',
            'print("[OK] San sang cham diem!")',
        ]
    )

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# G3-VPR v3 — Notebook Chấm Điểm Cross-Domain & Đối Chiếu SOTA Trực Tiếp Trên Kaggle\n",
                    "**Không cần train lại!** Notebook này tự động:\n",
                    "1. Tìm toàn bộ **10 checkpoint `G3-VPR v3`** bạn đã train (từ Output của notebook train trước hoặc Dataset bạn gắn vào Input).\n",
                    "2. Tự động tải về các bộ test khác miền chuẩn CVPR (**`SPED`**, **`AmsterTime`**, và tự nhận thêm **`Pitts30k` / `MSLS-val` / `Tokyo24/7` / `Nordland`** nếu bạn gắn vào Input).\n",
                    "3. Chấm ở chế độ **`BS=1` tuyệt đối** cho: **`G3-VPR v3` (`128..4096-D + G-OTR + AM-VPR Conformal Cascade`)**, **9 dòng Ablation đồng ngân sách**, và **3 đối thủ SOTA (`MegaLoc`, `DINOv2-SALAD`, `EigenPlaces`)**.\n",
                    "4. Xuất sẵn bảng báo cáo **`CVPR_V3_SUMMARY_REPORT.md`** + nén vào **`g3vpr_v3_crossdomain_eval.zip`** để tải về đối chiếu ngay!\n",
                    "\n",
                    "**Cách chạy nhanh nhất trên Kaggle:**\n",
                    "1. **Settings** bên phải: chọn **Accelerator → `GPU T4`** (hoặc `T4 x2`), bật **Internet → `ON`**.\n",
                    "2. **Input** bên phải: bấm **`+ Add Input` → tab `Your Work` → chọn Output của Notebook train `G3-VPR v3` vừa chạy xong** (hoặc upload `g3vpr_paper_artifacts_v3.zip` làm Dataset).\n",
                    "3. Bấm **`Save Version` → `Save & Run All`** (~10–20 phút là xong toàn bộ)!",
                ],
            },
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell1_source},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 2: Tu dong tim toan bo 10 checkpoint G3-VPR v3 trong /kaggle/input (hoac giai nen tu file zip neu co)\n",
                    "import os, zipfile\n",
                    'if os.path.isdir("/kaggle/working/D2VPR_eval"):\n',
                    '    os.chdir("/kaggle/working/D2VPR_eval")\n',
                    "\n",
                    "CKPT_ROOT = None\n",
                    "M5_CKPT = None\n",
                    "\n",
                    "# 1) Quet truc tiep /kaggle/input tim g3vpr_universal_best.pth / g3vpr_best_fp16.pth\n",
                    'if os.path.isdir("/kaggle/input"):\n',
                    '    for r, dirs, files in os.walk("/kaggle/input"):\n',
                    '        if "g3vpr_M5_full" in dirs:\n',
                    '            cand = os.path.join(r, "g3vpr_M5_full", "g3vpr_universal_best.pth")\n',
                    '            cand16 = os.path.join(r, "g3vpr_M5_full", "g3vpr_best_fp16.pth")\n',
                    "            if os.path.isfile(cand):\n",
                    "                CKPT_ROOT, M5_CKPT = r, cand\n",
                    "                break\n",
                    "            elif os.path.isfile(cand16):\n",
                    "                CKPT_ROOT, M5_CKPT = r, cand16\n",
                    "                break\n",
                    "        for fn in files:\n",
                    '            if fn.endswith(".zip") and "g3vpr_paper_artifacts_v3" in fn:\n',
                    '                print("Tim thay file zip artifacts, dang giai nen:", os.path.join(r, fn))\n',
                    '                with zipfile.ZipFile(os.path.join(r, fn), "r") as zf:\n',
                    '                    zf.extractall("./v3_extracted")\n',
                    '                CKPT_ROOT = "./v3_extracted/checkpoints"\n',
                    '                M5_CKPT = os.path.join(CKPT_ROOT, "g3vpr_M5_full", "g3vpr_universal_best.pth")\n',
                    "                break\n",
                    "        if M5_CKPT:\n",
                    "            break\n",
                    "\n",
                    "assert M5_CKPT and os.path.isfile(M5_CKPT), (\n",
                    '    "Khong tim thay checkpoint G3-VPR v3 trong /kaggle/input! "\n',
                    '    "Hay bam + Add Input -> Your Work -> gan Output cua notebook G3-VPR v3 (hoac dataset chua g3vpr_paper_artifacts_v3.zip)."\n',
                    ")\n",
                    'print("[OK] Main checkpoint M5_full:", M5_CKPT)\n',
                    'print("[OK] Thu muc chua cac ablation:", CKPT_ROOT)\n',
                    'print("Cac thu muc checkpoint tim thay:", sorted(os.listdir(CKPT_ROOT)))',
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 3: Tu dong tai cac bo test khac mien chuan CVPR (SPED, AmsterTime, SVOX, ...) neu chua gan trong /kaggle/input\n",
                    "import glob, os, subprocess, sys\n",
                    'if os.path.isdir("/kaggle/working/D2VPR_eval"):\n',
                    '    os.chdir("/kaggle/working/D2VPR_eval")\n',
                    "\n",
                    'DATASETS_DIR = os.path.abspath("./datasets")\n',
                    "os.makedirs(DATASETS_DIR, exist_ok=True)\n",
                    "\n",
                    "# Clone cong cu tai chuan VPR-datasets-downloader & cai du dependencies (py3_wget, utm, ...)\n",
                    'dl_repo = "./VPR-datasets-downloader"\n',
                    "if not os.path.isdir(dl_repo):\n",
                    '    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/gmberton/VPR-datasets-downloader.git", dl_repo], check=True)\n',
                    'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py3_wget", "utm", "pillow", "tqdm"], check=True)\n',
                    'avail_scripts = {os.path.basename(p): p for p in glob.glob(f"{dl_repo}/download_*.py")}\n',
                    'print("Cac script tai co san:", sorted(avail_scripts.keys()))\n',
                    "\n",
                    "# Tai cac bo benchmark chuan CVPR (SPED, AmsterTime, SVOX, Tokyo24/7)\n",
                    'for ds_name in ["sped", "amstertime", "svox", "tokyo247"]:\n',
                    '    target_dir = os.path.join(DATASETS_DIR, ds_name, "images", "test")\n',
                    "    if os.path.isdir(target_dir):\n",
                    '        print(f"[Da co san] {ds_name}")\n',
                    "        continue\n",
                    '    sc_path = avail_scripts.get(f"download_{ds_name}.py")\n',
                    "    if not sc_path:\n",
                    '        print(f"[Bo qua] Khong co script cho {ds_name}")\n',
                    "        continue\n",
                    '    print(f"\\n=== Dang tai bo benchmark chuan: {ds_name} ===")\n',
                    '    ret = subprocess.run([sys.executable, sc_path, "--datasets_folder", DATASETS_DIR], check=False)\n',
                    "    if ret.returncode != 0:\n",
                    '        print(f"[Canh bao] Khong tai tu dong duoc {ds_name}, tiep tuc bo khac.")\n',
                    "\n",
                    'print("\\n[OK] Danh sach thu muc trong ./datasets:", sorted(os.listdir(DATASETS_DIR)))',
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 4: Chay Cham Diem Toan Dien G3-VPR v3 + G-OTR + Conformal Cascade + 9 Ablations + SOTA Hub (BS=1) + FAISS 1M\n",
                    "import os, subprocess, sys\n",
                    'if os.path.isdir("/kaggle/working/D2VPR_eval"):\n',
                    '    os.chdir("/kaggle/working/D2VPR_eval")\n',
                    "\n",
                    "eval_cmd = [\n",
                    '    sys.executable, "-u", "eval_cvpr_v3_suite.py",\n',
                    '    "--ckpt", M5_CKPT,\n',
                    '    "--checkpoints_dir", CKPT_ROOT,\n',
                    '    "--datasets_root", "./datasets",\n',
                    '    "--query_bs", "1",\n',
                    '    "--db_bs", "64",\n',
                    '    "--run_sota_hub",\n',
                    '    "--run_faiss_1m",\n',
                    '    "--output_dir", "./results_crossdomain",\n',
                    "]\n",
                    'print("Running:", " ".join(eval_cmd))\n',
                    "subprocess.run(eval_cmd, check=True)",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 5: In toan bo bao cao doi chieu & Dong goi ZIP ket qua de tai ve may!\n",
                    "import os, shutil\n",
                    'if os.path.isdir("/kaggle/working/D2VPR_eval"):\n',
                    '    os.chdir("/kaggle/working/D2VPR_eval")\n',
                    "\n",
                    'report_md = "./results_crossdomain/CVPR_V3_SUMMARY_REPORT.md"\n',
                    "if os.path.isfile(report_md):\n",
                    '    print("=" * 85)\n',
                    '    print(open(report_md, encoding="utf-8").read())\n',
                    '    print("=" * 85)\n',
                    "\n",
                    'zip_out = "/kaggle/working/g3vpr_v3_crossdomain_eval" if os.path.isdir("/kaggle/working") else "./g3vpr_v3_crossdomain_eval"\n',
                    'archive = shutil.make_archive(zip_out, "zip", "./results_crossdomain")\n',
                    'print("\\n[HOAN TAT] Tai file ket qua tai muc Output ben phai:", archive)',
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

    nb_path = ROOT / "kaggle_colab" / "G3VPR_v3_Kaggle_Eval_Only.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"[OK] Built Kaggle Eval-Only Notebook -> {nb_path} ({nb_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
