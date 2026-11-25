import base64
import io
import json
import shutil
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
            "# Cell 1: Tu dong giai nen toan bo code G3-VPR ngay trong Kaggle/Colab (KHONG CAN GITHUB!)\n",
            "import base64, io, os, tarfile, torch\n",
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
            'print("[OK] Da giai nen toan bo du an tai:", os.getcwd())\n',
            'print("[OK] Cac file chinh:", sorted([f for f in os.listdir(".") if f.endswith(".py")]))\n',
            'print("PyTorch:", torch.__version__, "| CUDA GPU:", torch.cuda.is_available(), "| So GPU:", torch.cuda.device_count())\n',
            "if not torch.cuda.is_available():\n",
            '    print("CANH BAO: Ban chua bat GPU! Vao Settings ben phai -> Accelerator -> chon GPU T4 x2 va bat Internet ON!")\n',
            "else:\n",
            "    for i in range(torch.cuda.device_count()):\n",
            '        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)")\n',
            "\n",
            "!pip install -q pytorch-metric-learning faiss-cpu transformers timm einops",
        ]
    )

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# G3-VPR: Notebook Độc Lập 100% Cho Kaggle & Google Colab (Không Cần GitHub!)\n",
                    "**Cách chạy trên Kaggle chỉ với 3 thao tác:**\n",
                    "1. Ở cột **Settings** bên phải: chọn **Accelerator -> `GPU T4 x2`** (hoặc `P100`) và bật **Internet -> `ON`**.\n",
                    "2. Ở mục **Input** bên phải: bấm **`+ Add Input`** -> gắn **`amaralibey/gsv-cities`**.\n",
                    "3. Bấm **`Run All`** trên thanh công cụ! *(Toàn bộ file `.py` của dự án đã được nhúng sẵn tự giải nén ở Cell 1 — không cần đưa lên GitHub hay tải thêm code).* ",
                ],
            },
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell1_source},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 2: Tai trong so nen DINOv2 ViT-S/14 & Tu dong tim duong dan GSV-Cities\n",
                    "import os, urllib.request\n",
                    'if os.path.isdir("/kaggle/working/D2VPR"):\n',
                    '    os.chdir("/kaggle/working/D2VPR")\n',
                    "\n",
                    'dinov2_ckpt = "./dinov2_vits14_pretrain.pth"\n',
                    "if not os.path.isfile(dinov2_ckpt):\n",
                    '    print("Dang tai trong so DINOv2 ViT-S/14...")\n',
                    "    urllib.request.urlretrieve(\n",
                    '        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",\n',
                    "        dinov2_ckpt\n",
                    "    )\n",
                    'print("[OK] DINOv2-S san sang:", os.path.isfile(dinov2_ckpt))\n',
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
                    "# Cell 3: Kiem tra nhanh kien truc G3-VPR, tinh bat bien BS=1 & toc do GPU\n",
                    'if os.path.isdir("/kaggle/working/D2VPR"): os.chdir("/kaggle/working/D2VPR")\n',
                    "!python -u experiments/exp2_architecture_verify_and_profile.py",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 4: Huan luyen Mo hinh Chinh G3-VPR (M5_full) voi 2 Teachers (MegaLoc 2D + Depth-Anything-V2 3D)\n",
                    "# Mac dinh max_places=20000 (~80,000 anh/epoch) de chay gon trong 1 phien Kaggle T4.\n",
                    "# Neu muon train full 100% 62,000 dia diem, doi --max_places 0.\n",
                    "import os\n",
                    'if os.path.isdir("/kaggle/working/D2VPR"): os.chdir("/kaggle/working/D2VPR")\n',
                    'SAVE_DIR = "./checkpoints/g3vpr_M5_full"\n',
                    'RESUME_ARG = f"--resume_ckpt {SAVE_DIR}/g3vpr_last.pth" if os.path.isfile(f"{SAVE_DIR}/g3vpr_last.pth") else ""\n',
                    "\n",
                    "cmd = (\n",
                    '    f"python -u train_g3vpr.py "\n',
                    '    f"--ablation M5_full "\n',
                    '    f"--gsv_path \\"{GSV_PATH}\\" "\n',
                    '    f"--dinov2_path ./dinov2_vits14_pretrain.pth "\n',
                    '    f"{RESUME_ARG} "\n',
                    '    f"--teacher_2d megaloc "\n',
                    '    f"--teacher_3d depth_anything_v2 "\n',
                    '    f"--bs 64 --epochs 10 --max_places 20000 --lr 2e-4 --devices 1 "\n',
                    '    f"--save_dir \\"{SAVE_DIR}\\""\n',
                    ")\n",
                    'print("Running:", cmd)\n',
                    "!yes | {cmd}",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 5: Chay Bang Ablation Study (Table 4: M1 -> M2 -> M3 -> M4 -> M5_full)\n",
                    'if os.path.isdir("/kaggle/working/D2VPR"): os.chdir("/kaggle/working/D2VPR")\n',
                    'ABLATION_CITIES = "London,Boston,Melbourne,Chicago,Toronto"\n',
                    'for abl in ["M1", "M2", "M3", "M4", "M5_full"]:\n',
                    '    out_d = f"./checkpoints/ablation_{abl}"\n',
                    '    !yes | python -u train_g3vpr.py --ablation {abl} --gsv_path "{GSV_PATH}" --cities "{ABLATION_CITIES}" \\\n',
                    "        --dinov2_path ./dinov2_vits14_pretrain.pth --teacher_2d megaloc --teacher_3d depth_anything_v2 \\\n",
                    '        --bs 64 --epochs 5 --max_places 6000 --devices 1 --save_dir "{out_d}"',
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cell 6: Danh gia BS=1 Matryoshka (128D..4096D), ve Hinh 4 & Nen file ZIP ket qua tai ve may!\n",
                    "import os, shutil\n",
                    'if os.path.isdir("/kaggle/working/D2VPR"): os.chdir("/kaggle/working/D2VPR")\n',
                    '!python -u eval_g3vpr_sota.py --ckpt "./checkpoints/g3vpr_M5_full/g3vpr_universal_best.pth" --query_bs 1 --output_json ./results/g3vpr_sota_eval.json\n',
                    '!python -u experiments/visualize_gaussians.py --ckpt "./checkpoints/g3vpr_M5_full/g3vpr_universal_best.pth"\n',
                    "\n",
                    'zip_out = "/kaggle/working/g3vpr_paper_artifacts" if os.path.isdir("/kaggle/working") else "./g3vpr_paper_artifacts"\n',
                    'os.makedirs("./export_bundle/results", exist_ok=True)\n',
                    'os.makedirs("./export_bundle/checkpoints", exist_ok=True)\n',
                    'if os.path.isdir("./results"):\n',
                    '    shutil.copytree("./results", "./export_bundle/results", dirs_exist_ok=True)\n',
                    'best_p = "./checkpoints/g3vpr_M5_full/g3vpr_universal_best.pth"\n',
                    "if os.path.isfile(best_p):\n",
                    '    shutil.copy2(best_p, "./export_bundle/checkpoints/g3vpr_universal_best.pth")\n',
                    'for root, _, files in os.walk("./checkpoints"):\n',
                    "    for fn in files:\n",
                    '        if fn.endswith(".json") or fn.endswith(".csv"):\n',
                    '            rel_d = os.path.relpath(root, "./checkpoints")\n',
                    '            os.makedirs(f"./export_bundle/checkpoints/{rel_d}", exist_ok=True)\n',
                    '            shutil.copy2(os.path.join(root, fn), f"./export_bundle/checkpoints/{rel_d}/{fn}")\n',
                    'archive_file = shutil.make_archive(zip_out, "zip", "./export_bundle")\n',
                    'print("[HOAN TAT] Tai file ket qua tai muc Output ben phai:", archive_file)',
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
        r2 = subprocess.run([sys.executable, "eval_g3vpr_sota.py", "--help"], cwd=tmpd, capture_output=True, text=True)
        assert r1.returncode == 0, r1.stderr
        assert r2.returncode == 0, r2.stderr
        print("Isolated fresh-directory extraction & CLI verification PASSED 100%!")


if __name__ == "__main__":
    main()
