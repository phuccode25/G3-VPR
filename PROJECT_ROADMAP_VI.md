# Sổ Tay Ngữ Cảnh & Kế Hoạch Tổng Thể Dự Án G³-VPR (Lưu Trữ Chống Quên Ngữ Cảnh)

> **Cập nhật lần cuối:** 26/11/2026  
> **Thư mục làm việc:** `d:\Lab\D2VPR`  
> **GitHub Repository:** https://github.com/phuccode25/G3-VPR  
> **Mục tiêu công bố:** Hội nghị hạng A\* (CVPR / ICCV / NeurIPS 2027)  
> **Tên bài báo:** *G³-VPR: Geometry-Grounded Gaussian Aggregation and Matryoshka Relational Distillation for Universal Single-Query Visual Place Recognition*

---

## 1. Câu Chuyện Nghiên Cứu (Motivation — Vì Sao Bài Báo Này Thuyết Phục Giám Khảo A\*?)

Bài toán **Visual Place Recognition (VPR)**: Cho robot/xe tự hành chụp **1 bức ảnh đơn lẻ (`BS=1`)** và tìm đúng vị trí trong cơ sở dữ liệu hàng triệu ảnh trải dài nhiều thời điểm (ngày/đêm, hiện đại/lịch sử 100 năm trước, nắng/mưa).

Chúng ta phát hiện và chứng minh bằng thực nghiệm **4 điểm yếu chí mạng** của mô hình SOTA hiện tại là **D²-VPR (AAAI 2026 — [D2VPR.py](file:///d:/Lab/D2VPR/D2VPR.py))**, đồng thời giải quyết triệt để từng điểm bằng **G³-VPR**:

| STT | Điểm yếu của bài cũ `D²-VPR` (Đã đo thật trên máy) | Giải pháp của `G³-VPR` (Code mới trong `g3vpr/`) |
| :-: | :--- | :--- |
| **1** | **Sụp đổ khi chạy thực tế 1 ảnh lẻ (`BS=1`):** Dùng `cross_image_encoder` (`11.03M` tham số) trộn thông tin chéo giữa các ảnh trong cùng lô (`BS=8` hoặc `BS=3`). Khi chạy lẻ `BS=1`, điểm R@1 trên `AmsterTime` **rơi từ `63.04%` xuống `29.16%` (`-33.88%`)**, trên `SPED` **rơi từ `91.43%` xuống `68.70%` (`-22.73%`)**. | **`PMCI` ([prototype_memory.py](file:///d:/Lab/D2VPR/g3vpr/prototype_memory.py)):** Dùng bộ nhớ **128 nguyên mẫu cảnh quan cố định** học sẵn trong mạng. Chạy `BS=1` hay `BS=32` đều cho kết quả **giống hệt nhau 100% (`0.00%` độ lệch)**. |
| **2** | **Học phố mới quên phố cũ (Phải chia 3 file trọng số):** Giai đoạn 2 (`finetune.pth`) bỏ mô hình Thầy làm tụt điểm ảnh lịch sử `AmsterTime` từ `63.04%` xuống `56.21%`, buộc tác giả phải tung ra 3 file `.pth` rời rạc cho từng tập test. | **Chưng cất Kép 2D + 3D ([losses.py](file:///d:/Lab/D2VPR/g3vpr/losses.py), [teachers.py](file:///d:/Lab/D2VPR/g3vpr/teachers.py)):** Học đồng thời từ **Thầy 2D `MegaLoc`** (ngữ nghĩa đa thời đại) và **Thầy 3D `Depth-Anything-V2`** (chiều sâu hình học) để tạo ra **1 file trọng số vạn năng duy nhất**. |
| **3** | **Ô chữ nhật 2D cứng nhắc & Lỗi chết 68.36% bản đồ:** Truyền nhầm tọa độ `[0, 1]` vào `F.grid_sample` (vốn cần `[-1, 1]`) làm **175/256 ô (`68.36%`) bị chết đạo hàm**, chạy vòng lặp dư 13 lần, và ô chữ nhật đứng thẳng không xoay theo vách tường nghiêng 3D được. | **`3D-DGSA` ([gaussian_aggregator.py](file:///d:/Lab/D2VPR/g3vpr/gaussian_aggregator.py)):** Dùng **16 đám mây Elip Gaussian 2D tự co giãn và xoay góc nghiêng** bám theo mặt tiền tòa nhà 3D, lọc bỏ bầu trời/xe cộ nhờ cổng `salience_gate`, phủ **100% ảnh** và **nhanh gấp `10.24 lần`** (`3.06 ms` vs `31.30 ms`). |
| **4** | **Bản đồ ngốn 40 GB RAM & Ma trận ẩn PCA 44M tham số:** Xuất vector `10.752` chiều (`40.05 GB` RAM / 1 triệu ảnh), hoặc nếu nén PCA xuống `4096` chiều phải lưu ma trận nặng `44.04M` tham số (tổng `82.28M`). | **Đầu ra Búp bê Nga `Matryoshka` ([matryoshka.py](file:///d:/Lab/D2VPR/g3vpr/matryoshka.py)):** Sắp xếp thông tin lồng nhau `{128, 256, 512, 1024, 2048, 4096}` chiều. Cắt thẳng `vector[:128]` không cần PCA, giảm RAM bản đồ **83.4 lần** (còn `0.48 GB`), tổng mạng chỉ **`30.97M` tham số**. |

---

## 2. Bản Đồ Toàn Bộ File Mã Nguồn & Kết Quả Hiện Có Trong `d:\Lab\D2VPR`

### A. Thư viện kiến trúc cốt lõi `g3vpr/`
* [g3vpr/backbones.py](file:///d:/Lab/D2VPR/g3vpr/backbones.py): `GeoSemanticGatedBackbone` — ViT-S/14 + `MulConvAdapter` hỗ trợ kích thước ảnh động `(H_p, W_p)`, tách tầng nông `[2, 4]` (cấu trúc 3D) và tầng sâu `[8, 12]` (ngữ nghĩa), dự đoán bản đồ độ sâu giả `pseudo_depth` và cổng lọc kiến trúc tĩnh `salience_gate`.
* [g3vpr/gaussian_aggregator.py](file:///d:/Lab/D2VPR/g3vpr/gaussian_aggregator.py): `Vectorized3DGaussianAggregator` (`3D-DGSA`) — 16 vùng lấy mẫu hình Elip Gaussian tự xoay nghiêng và co giãn trong 1 phép tính vector hóa duy nhất.
* [g3vpr/prototype_memory.py](file:///d:/Lab/D2VPR/g3vpr/prototype_memory.py): `PrototypeMemoryCrossInteraction` (`PMCI`) — Tự chú ý nội ảnh + chú ý chéo vào 128 nguyên mẫu trực giao (`batch_first=True`, bất biến tuyệt đối với `BS=1`).
* [g3vpr/matryoshka.py](file:///d:/Lab/D2VPR/g3vpr/matryoshka.py): `MatryoshkaProjectionHead` — Đầu chiếu phân tầng lồng nhau `128D .. 4096D` không cần PCA.
* [g3vpr/model.py](file:///d:/Lab/D2VPR/g3vpr/model.py): Lớp tổng hợp `G3VPR` và cấu hình `G3VPRConfig` hỗ trợ tự động 5 biến thể Ablation (`M1`, `M2`, `M3`, `M4`, `M5_full`).
* [g3vpr/losses.py](file:///d:/Lab/D2VPR/g3vpr/losses.py) & [g3vpr/teachers.py](file:///d:/Lab/D2VPR/g3vpr/teachers.py): Hàm mục tiêu hợp nhất `UnifiedG3VPRLoss` và bộ tải 2 Thầy đóng băng `DualTeacherEngine` (`MegaLoc` + `Depth-Anything-V2` / `VGGT`).

### B. Kịch bản huấn luyện, đánh giá & đóng gói Kaggle
* [train_g3vpr.py](file:///d:/Lab/D2VPR/train_g3vpr.py): Kịch bản huấn luyện chuẩn PyTorch AMP (`fp16`), tự động đặt 2 Thầy sang `cuda:1` trên Kaggle `T4 x2`.
* [eval_g3vpr_sota.py](file:///d:/Lab/D2VPR/eval_g3vpr_sota.py): Chấm điểm chuẩn `query_bs=1` đồng thời cho cả 6 mức chiều Matryoshka và xuất bảng LaTeX.
* [kaggle_colab/build_kaggle_bundle.py](file:///d:/Lab/D2VPR/kaggle_colab/build_kaggle_bundle.py) & [kaggle_colab/G3VPR_Kaggle_Colab_Runner.ipynb](file:///d:/Lab/D2VPR/kaggle_colab/G3VPR_Kaggle_Colab_Runner.ipynb): Bộ đóng gói toàn bộ mã nguồn vào Cell 1 của Notebook để chạy độc lập 100% trên Kaggle.

### C. Số liệu thực nghiệm & Hình vẽ đã hoàn thành
1. [results/exp1_diagnostic.json](file:///d:/Lab/D2VPR/results/exp1_diagnostic.json) & [results/fig1_diagnostic_motivation.png](file:///d:/Lab/D2VPR/results/fig1_diagnostic_motivation.png): Chứng minh lỗi sụp đổ `BS=1`, quên liên miền và chết 68.36% lưới tọa độ của `D²-VPR`.
2. [results/exp2_hardware_profile.json](file:///d:/Lab/D2VPR/results/exp2_hardware_profile.json) & [results/table_hardware_efficiency.md](file:///d:/Lab/D2VPR/results/table_hardware_efficiency.md): Chứng minh tốc độ trên Kaggle T4 đạt **`25.85 ms`** độ trễ `BS=1` (nhanh hơn 32.5%), thông lượng **`180.3 ảnh/giây`** (+36.1%), và **`30.97M` tham số** (-62.4% so với `D²-VPR + PCA`).
3. [checkpoints/g3vpr_M5_full/g3vpr_universal_best.pth](file:///d:/Lab/D2VPR/checkpoints/g3vpr_M5_full/g3vpr_universal_best.pth) & [results/g3vpr_sota_eval.json](file:///d:/Lab/D2VPR/results/g3vpr_sota_eval.json): Trọng số và bảng điểm từ lượt chạy **Kaggle Run v1**:
   * Đạt **`86.75%` R@1 (`4096-D`) / `83.50%` R@1 (`256-D`)** trên tập kiểm thử `WashingtonDC`.
   * Độ lệch giữa `BS=1, 2, 3, 4, 8, 16, 32` trên `SPED` bằng **`0.00%` tuyệt đối** (`55.02%` ở mọi `BS`).
   * Trên `Pitts30k-val` (`17.616` ảnh, `BS=1`): đạt **`70.89%` R@1 / `90.13%` R@5** ở `4096-D`, giữ **`61.59%` R@1 / `84.67%` R@5** ngay ở `128-D` (`0.48 GB` RAM).
   * Trên miền lạ khắc nghiệt: `SPED` đạt `55.02%` R@1, `AmsterTime` đạt `13.08%` R@1 (cần nâng cấp bản `v2` bên dưới để vượt `D²-VPR`).
4. [results/fig3_training_pareto_stability.png](file:///d:/Lab/D2VPR/results/fig3_training_pareto_stability.png) & [results/fig4_gaussian_splat_comparison.png](file:///d:/Lab/D2VPR/results/fig4_gaussian_splat_comparison.png): Biểu đồ đường cong Pareto + ổn định `BS=1` (Hình 3) và trực quan hóa Elip Gaussian xoay nghiêng (Hình 4).

---

## 3. Trạng Thái Giai Đoạn 2: Bản Nâng Cấp `G³-VPR v2` (ĐÃ CODE XONG & KIỂM CHỨNG TRÊN MÁY — SẴN SÀNG CHẠY KAGGLE V2!)

### Những gì đã nâng cấp và chứng minh ngay tại Bước 0 (Chưa cần train 1 bước nào!) của `G³-VPR v2`:
1. **Sửa dứt điểm 4 nút thắt của `v1`:**
   * **Bảo toàn không gian đẳng cự (Isometric Semi-Orthogonal Gated-DRM & Multi-Scale Matryoshka v2):** Tổng mô hình nặng **`36.01M` tham số** (vẫn nhỏ hơn `D²-VPR` `38.24M` và **nhẹ hơn `56.2%`** so với `D²-VPR + PCA4096` `82.28M`). Sửa `Signed GeM` thêm `eps = 1e-5` chống `NaN` đạo hàm `gem_p` và cổng `pose_scale = 0.05`.
   * **Chưng cất trực tiếp không gian `8.448` chiều `MegaLoc` (`MatryoshkaTeacherBridge`) + `NM-GRD` (`float32`):** Loại bỏ hoàn toàn lỗi nhân `0.07² = 0.0049` và hiện tượng tràn số `float16`.
   * **Trích xuất trọng số khởi động ấm thân ViT-S (`weights_init/vits_adapter_warmstart_fp16.pt` — `46.65 MB`):** Nhẹ dưới ngưỡng 100 MB của GitHub, tự động tải ở Cell 2 trên Kaggle.
   * **Cell 5 Ablation (`M1 → M4`) thuần Python `subprocess.run` + kịch bản chấm [eval_ablation_suite.py](file:///d:/Lab/D2VPR/eval_ablation_suite.py):** Xuất kèm đầy đủ file `g3vpr_best_fp16.pth` của cả 5 mô hình `M1..M5_full` trong gói `g3vpr_paper_artifacts_v2.zip`.
2. **Kết quả kiểm chứng thực tế tại Bước 0 (`Step-0`, 0 epoch huấn luyện) của `G³-VPR v2` so với `v1` (sau 8 epoch):**
   * **`SPED` (`BS=1`):** Ngay tại Bước 0 đã đạt **`76.94%` R@1 / `86.99%` R@5** ở `4096-D` và **`72.82%` R@1 / `84.68%` R@5** ngay ở `256-D` (vượt xa `55.02%` của `v1` tới **`+21.92%`** ngay trước khi bắt đầu train!).
   * **`AmsterTime` (`BS=1`):** Ngay tại Bước 0 đã đạt **`41.35%` R@1 / `62.71%` R@5** ở `4096-D` (vượt xa `13.08%` của `v1` tới **`+28.27%`**, và cao hơn luôn `29.16%` của `D²-VPR Encoder` ở `BS=1` ngay trước khi train!).

### Bước tiếp theo cho bạn:
1. Chạy [kaggle_colab/G3VPR_Kaggle_Colab_Runner.ipynb](file:///d:/Lab/D2VPR/kaggle_colab/G3VPR_Kaggle_Colab_Runner.ipynb) bản **v2** trên Kaggle (`GPU T4 x2`, bật `Internet ON`, gắn `amaralibey/gsv-cities`) -> bấm **`Save Version -> Save & Run All (Commit)`**.
2. Khi chạy xong, tải `g3vpr_paper_artifacts_v2.zip` ở mục **Output** về giải nén vào `d:\Lab\D2VPR`.
3. Chạy `eval_g3vpr_sota.py` + `eval_ablation_suite.py` tại `BS=1` để lấy trọn bộ bảng SOTA & Ablation cuối cùng rồi viết bản thảo LaTeX chuẩn hội nghị A\*.
