# Hướng dẫn Chạy Full Training & Ablation $\text{G}^3\text{-VPR}$ trên Kaggle / Google Colab

Vì GPU máy cá nhân (`RTX 3050 Laptop 4.29 GB VRAM`) phù hợp để chạy **Thí nghiệm Chẩn đoán (Experiment 1)**, **Đo đạc Phần cứng / Kiểm chứng Bất biến `BS=1` (Experiment 2)** và **Trực quan hóa Gaussian (Experiment 5)**, còn bước huấn luyện toàn tập **GSV-Cities (560,000 ảnh)** kèm **Dual-Foundation Teachers** (`MegaLoc` + `Depth-Anything-V2 / VGGT`) cần GPU 16GB–32GB VRAM, gói này đã đóng gói tự động hoàn toàn cho **Kaggle** và **Google Colab**.

---

## Cách 1: Chạy trên Kaggle (Nhanh nhất — Không tốn thời gian tải 560k ảnh!)

1. Mở [Kaggle Notebooks](https://www.kaggle.com/code) $\rightarrow$ **New Notebook** $\rightarrow$ **File > Import Notebook** và chọn file `kaggle_colab/G3VPR_Kaggle_Colab_Runner.ipynb`.
2. Ở thanh bên phải (**Settings**):
   - **Accelerator:** Chọn **`GPU T4 x2`** (32 GB tổng VRAM) hoặc `GPU P100` (16 GB VRAM).
   - **Internet:** Bật **`ON`** (để tải trọng số Teacher `MegaLoc` và `Depth-Anything-V2` tự động).
3. Bấm **+ Add Input** $\rightarrow$ tìm **`amaralibey/gsv-cities`** và bấm **Add**. (Tập dữ liệu huấn luyện 560,000 ảnh sẽ được gắn tức thì vào `/kaggle/input/gsv-cities` trong **0 giây**!).
4. Đẩy thư mục code dự án này lên GitHub cá nhân (private/public) hoặc nén thành `.zip` (`g3vpr/`, `backbone/`, `src/`, `datasets_ws.py`, `commons.py`, `util.py`, `test.py`, `train_g3vpr.py`, `eval_g3vpr_sota.py`, `experiments/`) rồi gắn vào Input Kaggle.
5. Bấm **Run All**:
   - **Cell 4** huấn luyện mô hình hợp nhất `M5_full` (tự động lưu `g3vpr_universal_best.pth` dựa trên trung bình điều hòa `Harmonic Mean R@1` chống quên liên miền và tự động resume nếu hết phiên 12 tiếng).
   - **Cell 5** chạy trọn bộ **Bảng Ablation 4 (`M1` $\rightarrow$ `M5_full`)**.
   - **Cell 6** xuất bảng điểm SOTA ở **`BS=1`** cho cả 6 mức chiều Matryoshka (`128, 256, 512, 1024, 2048, 4096`-D) ra định dạng **JSON + LaTeX** sẵn sàng dán thẳng vào bài báo!

---

## Cách 2: Chạy trên Google Colab (`L4 24GB` hoặc `A100 40GB` với Teacher `VGGT-1B`)

Nếu bạn có Colab Pro (`L4` / `A100`), bạn có thể đổi cờ `--teacher_3d vggt` ở **Cell 4** để kích hoạt trực tiếp mô hình nền tảng 3D **VGGT-1B (CVPR 2025)**:
```bash
pip install git+https://github.com/facebookresearch/vggt.git
python train_g3vpr.py \
    --ablation M5_full \
    --gsv_path "$GSV_PATH" \
    --dinov2_path ./dinov2_vits14_pretrain.pth \
    --warmstart_ckpt ./weights/finetune_no_encoder.pth \
    --teacher_2d megaloc \
    --teacher_3d vggt \
    --bs 64 --epochs 15 --lr 2e-4
```
