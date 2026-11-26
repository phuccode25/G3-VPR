# Hướng dẫn Chạy Full Training & Ablation $\text{G}^3\text{-VPR}$ v2 trên Kaggle / Google Colab

Bộ đóng gói **$\text{G}^3\text{-VPR}$ v2** đã nhúng sẵn 100% mã nguồn mới nhất vào ngay Cell 1 của Notebook [G3VPR_Kaggle_Colab_Runner.ipynb](file:///d:/Lab/D2VPR/kaggle_colab/G3VPR_Kaggle_Colab_Runner.ipynb) và tự động tải trọng số khởi động ấm `ViT-S + MulConvAdapter` (`46.65 MB`) từ GitHub ở Cell 2.

---

## Cách Chạy Trên Kaggle (`GPU T4 x2` — Khuyên Dùng Nhất)

1. Mở [Kaggle Notebooks](https://www.kaggle.com/code) $\rightarrow$ **New Notebook** $\rightarrow$ **File > Import Notebook** và tải file `kaggle_colab/G3VPR_Kaggle_Colab_Runner.ipynb` từ máy bạn lên (hoặc nhập từ nhánh `main` của `phuccode25/G3-VPR`).
2. Ở cột **Settings** bên phải:
   - **Accelerator:** Chọn **`GPU T4 x2`** (32 GB tổng VRAM: `cuda:0` cho Student, `cuda:1` cho 2 Teachers).
   - **Internet:** Bật **`ON`**.
3. Ở mục **Input** bên phải: bấm **`+ Add Input`** $\rightarrow$ tìm **`amaralibey/gsv-cities`** và bấm **Add**.
4. Bấm nút **`Save Version`** (góc trên bên phải) $\rightarrow$ chọn **`Save & Run All (Commit)`** $\rightarrow$ bấm **`Save`**:
   - **Cell 4:** Huấn luyện mô hình chính `M5_full` (lấy mẫu ngẫu nhiên phân tầng 22.000 địa điểm mỗi epoch qua 8 epoch, bao phủ toàn bộ 62.000 địa điểm `GSV-Cities`).
   - **Cell 5:** Chạy tự động trọn bộ **Bảng Ablation 4 (`M1`, `M2`, `M3`, `M4`)** bằng `subprocess.run` thuần Python.
   - **Cell 6:** Đánh giá `BS=1` Matryoshka (`128D..4096D`), vẽ Hình 4 và xuất gói nén **`g3vpr_paper_artifacts_v2.zip`** (chứa trọng số `M5_full` cùng trọn bộ trọng số nén `fp16` của `M1..M4` để chấm trên `SPED`, `AmsterTime`, `Pitts30k-val` ở máy cá nhân).
