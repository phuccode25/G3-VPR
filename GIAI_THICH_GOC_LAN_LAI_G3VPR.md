# Giáo Trình "Hiểu Gốc Lẫn Lãi" Dự Án G³-VPR (Chống "Vibe Research" 100%)

> Tài liệu này được biên soạn để bạn nắm vững **từ nền tảng gốc của bài toán VPR, từng dòng code lỗi của bài báo cũ `D²-VPR` (AAAI 2026), luồng biến đổi Tensor chi tiết từng trạm của `G³-VPR`, toán học đằng sau 4 hàm Loss, bài học kỹ thuật từ `v1` lên `v2`, cho đến bộ câu hỏi phản biện gắt nhất của Giáo sư / Reviewer A\***.

---

## PHẦN 1: HIỂU PHẦN "GỐC" — Bài Toán VPR & Nền Tảng `D²-VPR` Hoạt Động Thế Nào?

### 1.1. Bản chất toán học của Visual Place Recognition (VPR)
VPR **không phải** bài toán phân loại (Classification) vì thế giới có vô hạn địa điểm mới mà lúc train mô hình chưa từng thấy. Thay vào đó, VPR là bài toán **Học Không Gian Đo Lường (Metric Learning / Image Retrieval)**:
* Mục tiêu: Xây dựng một hàm ánh xạ $f_\theta: \mathbb{R}^{3 \times H \times W} \to \mathbb{S}^{d-1}$ biến mỗi bức ảnh $I$ thành một **vector đơn vị** $\mathbf{z} \in \mathbb{R}^d$ (có độ dài L2 $\|\mathbf{z}\|_2 = 1$).
* **Giai đoạn Offline (Lập bản đồ):** Chạy toàn bộ ảnh bản đồ thành phố $\{I_1^{\text{db}}, \dots, I_N^{\text{db}}\}$ qua $f_\theta$ để lưu vào chỉ mục tìm kiếm **FAISS** (`faiss.IndexFlatIP(d)`).
* **Giai đoạn Online (Robot chạy thật ngoài đường):** Robot chụp **1 tấm ảnh truy vấn duy nhất (`Batch Size = 1`)** $I^q \to \mathbf{z}^q$, sau đó tính tích vô hướng (cũng chính là tương đồng Cosine vì $\|\mathbf{z}\|_2 = 1$):
  $$\operatorname{sim}(\mathbf{z}^q, \mathbf{z}_i^{\text{db}}) = \langle \mathbf{z}^q, \mathbf{z}_i^{\text{db}} \rangle = \sum_{c=1}^{d} z_c^q \cdot z_{i,c}^{\text{db}}$$
* Nếu địa điểm lấy ra ở Top-$K$ cách vị trí GPS thật $\le 25\text{m}$, ta tính là **đúng (`Recall@K`)**.

### 1.2. Bài báo gốc `D²-VPR` (AAAI 2026) đã làm gì?
Trước `D²-VPR`, mô hình `CricaVPR` (CVPR 2024) dùng mạng lớn **DINOv2 ViT-B/14** (`106.76M` tham số) nên quá nặng. `D²-VPR` muốn nén sang mạng nhỏ **DINOv2 ViT-S/14** (`384` chiều mỗi patch) qua 3 ý chính trong [D2VPR.py](file:///d:/Lab/D2VPR/D2VPR.py):
1. **Backbone + Adapter ([vision_transformer_small.py](file:///d:/Lab/D2VPR/backbone/vision_transformer_small.py)):** Cài các nhánh tích chập nhỏ `MulConvAdapter` (`1×1`, `3×3`, `5×5`) song song với tầng MLP của 12 khối ViT-S, rồi nối đặc trưng tầng thứ 2 và tầng thứ 12 (`384 + 384 = 768` chiều) chiếu qua 1 lớp tuyến tính tĩnh `self.lin = nn.Linear(1536, 768)` (gọi là `DRM`).
2. **Gom đặc trưng biến dạng `TDDA` ([D2VPR.py](file:///d:/Lab/D2VPR/D2VPR.py#L174-L283)):** Chia ảnh thành `1` vùng toàn cục + `4` ô vừa (`2×2`) + `9` ô nhỏ (`3×3`) = **14 vùng**. Dùng mạng `offset_generator` dịch chuyển các ô chữ nhật đó rồi lấy trung bình lũy thừa `GeM`.
3. **Trộn chéo ảnh `cross_image_encoder` ([D2VPR.py](file:///d:/Lab/D2VPR/D2VPR.py#L165-L171) & [D2VPR.py](file:///d:/Lab/D2VPR/D2VPR.py#L349-L357)):** Đưa tensor `[B, 14, 768]` vào `nn.TransformerEncoder(..., batch_first=False)` rồi trải phẳng thành vector `14 × 768 = 10.752` chiều!

---

## PHẦN 2: BẮT BỆNH TẬN GỐC — 4 Lỗi Toán Học & Code Của `D²-VPR` Mà Ta Đã Chứng Minh

Nếu ai hỏi bạn *"Tại sao phải sửa `D²-VPR`?"*, hãy chỉ thẳng vào đúng dòng code của họ:

### Lỗi 1: Gian lận lô ảnh (`batch_first=False`) làm sập mô hình khi chạy 1 ảnh (`BS=1`)
* **Dòng code bằng chứng:** [D2VPR.py](file:///d:/Lab/D2VPR/D2VPR.py#L166-L171) khởi tạo `TransformerEncoderLayer(..., batch_first=False)` và gọi `self.encoder(all_features)` trên tensor `[B, 14, 768]` tại [D2VPR.py](file:///d:/Lab/D2VPR/D2VPR.py#L351).
* **Bản chất toán học:** Trong PyTorch, khi `batch_first=False`, chiều đầu tiên `B` (số ảnh trong lô!) bị coi là **chiều dài câu (Sequence Length)**! Tức là bức ảnh số 1 làm phép Self-Attention nhìn sang bức ảnh số 2, số 3... số 8 trong cùng lô! Tệ hơn nữa, trong [eval_encoder.py](file:///d:/Lab/D2VPR/eval_encoder.py#L46-L70), tác giả gán cứng bằng tay `infer_batch_size = 3` cho `SPED`, `8` cho `AmsterTime`, `128` cho `MSLS`.
* **Hậu quả thực tế:** Khi robot ngoài đời gửi **1 bức ảnh (`B = 1`)**, chiều dài chuỗi bằng `1`, ma trận Attention suy biến thành ma trận đơn vị `[1.0]`, làm vector bị lệch hẳn khỏi phân phối của bản đồ DB (vốn được tạo ở `B = 8`), kéo điểm `AmsterTime` **rơi từ `63.04%` xuống `29.16%` (`-33.88%`)**!

### Lỗi 2: Lỗi truyền tọa độ `[0, 1]` thay vì `[-1, 1]` vào `F.grid_sample` làm chết 68.36% bản đồ
* **Dòng code bằng chứng:** Xem ngay lời chú thích sai lầm của tác giả gốc tại [D2VPR.py](file:///d:/Lab/D2VPR/D2VPR.py#L225-L234):
  > `# Here, we do not normalize the base_grid, because the offset generator can learn this coordinate transformation.`
* **Vì sao câu chú thích đó SAI TOÁN HỌC 100%?**
  * Hàm `F.grid_sample(offset_map, base_grid)` của PyTorch quy ước: tọa độ `(-1, -1)` là góc trên-trái, `(0, 0)` là **chính giữa bức ảnh**, và `(+1, +1)` là góc dưới-phải!
  * Khi tác giả truyền `base_grid` nằm trong đoạn `[0, 1]`, hàm `grid_sample` **chỉ đọc địa chỉ từ tâm ảnh `(0, 0)` đến góc dưới-phải `(+1, +1)`**!
  * `offset_generator` chỉ là tích chập `1×1` sinh ra *giá trị bên trong ô nhớ*, nó **không thể thay đổi địa chỉ đọc `base_grid`**! Do đó toàn bộ 3 góc phần tư còn lại (`x < 0` hoặc `y < 0`, ứng với **175 / 256 ô = `68.36%` diện tích ảnh**) không bao giờ được đọc tới và có **đạo hàm bằng `0` tuyệt đối mãi mãi**!
  * Chưa kể dòng `offset_map = self.offset_generator(fused_feat)` ([D2VPR.py](file:///d:/Lab/D2VPR/D2VPR.py#L223)) nằm bên trong vòng lặp `for` chạy lặp lại y hệt nhau **13 lần** cho cùng 1 bức ảnh!

### Lỗi 3: Huấn luyện 2 giai đoạn rời rạc gây "Quên thảm khốc" (Catastrophic Forgetting)
* **Dòng code bằng chứng:** Giai đoạn 1 ([distillation.py](file:///d:/Lab/D2VPR/distillation.py#L121-L123)) chỉ ép MSE theo Thầy `CricaVPR`. Sang Giai đoạn 2 ([finetune.py](file:///d:/Lab/D2VPR/finetune.py)), tác giả **tắt hoàn toàn mô hình Thầy**, chỉ chạy `MultiSimilarityLoss` trên ảnh đường phố hiện đại `GSV-Cities`.
* **Hậu quả:** Mạng quá khớp (overfit) vào màu sắc ảnh Google Street View hiện đại, làm điểm trên ảnh lịch sử đen trắng `AmsterTime` tụt từ `63.04%` xuống `56.21%`, buộc tác giả phải chia làm 3 file trọng số rời rạc (`distillation.pth`, `finetune.pth`, `finetune_no_encoder.pth`).

### Lỗi 4: Bùng nổ chiều `10.752` & Ma trận ẩn PCA `44.04M` tham số
* Ghép cứng `14 × 768 = 10.752` số thực (`float32` tốn `43 KB`/ảnh → **1 triệu ảnh ngốn `40.05 GB` RAM**). Muốn nén xuống `4096` chiều ở [eval_encoder.py](file:///d:/Lab/D2VPR/eval_encoder.py#L71-L77), họ dùng PCA ngoại tuyến cần lưu ma trận chiếu kích thước `10.752 × 4.096 = 44.040.192` tham số (nặng hơn cả mạng chính `38.24M`!).

---

## PHẦN 3: HIỂU PHẦN "LÃI" — Đường Đi Của 1 Bức Ảnh Qua 5 Trạm Của `G³-VPR v2`

Giả sử đưa 1 bức ảnh `x` kích thước `[B, 3, 224, 224]` vào [g3vpr/model.py](file:///d:/Lab/D2VPR/g3vpr/model.py#L169-L194). Hãy theo dõi chính xác từng Tensor biến đổi ra sao:

### Trạm 1: Thân mạng Tách Cổng Hình học – Ngữ nghĩa (`GeoSemanticGatedBackbone` — [backbones.py](file:///d:/Lab/D2VPR/g3vpr/backbones.py))
1. Chia ảnh `224×224` thành lưới `16×16 = 256` mảnh (patch `14×14`), cộng thêm 1 token toàn cục `CLS` → Tensor `[B, 257, 384]`. (Hỗ trợ cả ảnh chữ nhật bất kỳ `(H_p, W_p)` nhờ hàm vá động `_patch_adapter_dynamic_hw` tại [backbones.py](file:///d:/Lab/D2VPR/g3vpr/backbones.py#L10-L35)).
2. Đi qua 12 khối Transformer có gắn `MulConvAdapter`. Thay vì chỉ lấy tầng 2 và 12 trộn thô như bài cũ, ta tách bạch 2 luồng vật lý:
   * **Luồng Cấu trúc Hình học nông (Tầng `2` và `4`):** Nối lại thành `f_struct` `[B, 257, 768]` — giữ nguyên cạnh sắc nét, phối cảnh 3D và ranh giới vật thể.
   * **Luồng Ngữ nghĩa Bất biến sâu (Tầng `8` và `12`):** Nối lại thành `f_sem` `[B, 257, 768]` — bất biến với ngày/đêm, màu sắc/đen trắng.
3. **Đầu dự đoán Hình học 3D siêu nhẹ (`DepthwiseSeparableHead` — `0.18M` tham số):** Từ `f_struct` suy ra:
   * **Bản đồ Độ sâu giả bất biến tỷ lệ** $\hat{\mathbf{D}} \in \mathbb{R}_+^{B \times 1 \times 16 \times 16}$ (`pseudo_depth`).
   * **Cổng Độ nổi bật Kiến trúc tĩnh** $\mathbf{S}_{\text{geo}} \in (0, 1)^{B \times 1 \times 16 \times 16}$ (`salience_gate`): chỗ nào là bầu trời, mặt đường trơn, xe cộ động thì $\mathbf{S}_{\text{geo}} \to 0$; chỗ nào là mặt tiền tòa nhà cố định thì $\mathbf{S}_{\text{geo}} \to 1$.
4. **Chiếu đẳng cự trực giao (`nn.init.orthogonal_`):** Chiếu `f_sem` và `f_struct` về `512` chiều rồi trộn có kiểm soát:
   $$\mathbf{F}_{\text{fused}} = \operatorname{LayerNorm}\big(\mathbf{W}_{\text{sem}} \hat{\mathbf{f}}_{\text{sem}} + \tanh(\alpha) \cdot \mathbf{S}_{\text{geo}} \odot (\mathbf{W}_{\text{struct}} \hat{\mathbf{f}}_{\text{struct}})\big)$$
   *Vì sao phải khởi tạo trực giao `orthogonal_`?* Theo đại số tuyến tính, ma trận trực giao bảo toàn tích vô hướng $\langle \mathbf{Q}\mathbf{u}, \mathbf{Q}\mathbf{v} \rangle = \langle \mathbf{u}, \mathbf{v} \rangle$, giúp đặc trưng không bị xáo trộn nhiễu ngẫu nhiên ở Epoch 0!

### Trạm 2: Bộ Gom Đám Mây Elip Gaussian Định Hướng 3D (`3D-DGSA` — [gaussian_aggregator.py](file:///d:/Lab/D2VPR/g3vpr/gaussian_aggregator.py))
Thay vì chạy vòng lặp 13 lần với ô chữ nhật đứng thẳng bị lỗi tọa độ `[0, 1]`, ta làm trong **1 phép tính Tensor duy nhất**:
1. Ghép `[patch_map (512), salience_gate (1), depth_norm (1)]` qua tích chập chiều sâu `spatial_ctx` + `param_field` **đúng 1 lần duy nhất** sinh bản đồ biến dạng `5` kênh `[B, 5, 16, 16]`.
2. Đặt $K = 16$ tâm neo đa tỷ lệ (`4` góc phần tư `2×2` + `9` ô chi tiết `3×3` + `3` dải chân trời mặt tiền đường phố) và đổi sang **hệ tọa độ chuẩn xác `[-1, 1]`**: `sample_grid = anchor_centers * 2.0 - 1.0` ([gaussian_aggregator.py](file:///d:/Lab/D2VPR/g3vpr/gaussian_aggregator.py#L131)) → phủ **100% diện tích ảnh**, nhanh gấp **10.24 lần**!
3. Giải mã thông số cho mỗi đám mây Elip Gaussian thứ $k \in \{1 \dots 16\}$: tâm $\boldsymbol{\mu}_k = (\mu_{kx}, \mu_{ky})$, bán trục co giãn $(s_{kx}, s_{ky})$, và **góc xoay nghiêng** $\theta_k \in [-\frac{\pi}{3}, +\frac{\pi}{3}]$.
4. Xoay tọa độ từng điểm ảnh $\mathbf{p} = (x, y)$ vào hệ trục riêng của Elip thứ $k$ và tính trọng số Gaussian kết hợp cổng lọc 3D $\mathbf{S}_{\text{geo}}(\mathbf{p})$:
   $$\begin{bmatrix} \Delta x' \\ \Delta y' \end{bmatrix} = \begin{bmatrix} \cos\theta_k & \sin\theta_k \\ -\sin\theta_k & \cos\theta_k \end{bmatrix} (\mathbf{p} - \boldsymbol{\mu}_k), \quad w_k(\mathbf{p}) \propto \exp\!\left(-\frac{1}{2}\left[\left(\frac{\Delta x'}{s_{kx}}\right)^2 + \left(\frac{\Delta y'}{s_{ky}}\right)^2\right]\right) \cdot \big(0.15 + 0.85\,\mathbf{S}_{\text{geo}}(\mathbf{p})\big)$$
5. **Gom đặc trưng bằng GeM Đối Xứng Có Dấu (`Symmetric Signed GeM` — nâng cấp quan trọng ở `v2`):**
   * Đặc trưng sau `LayerNorm` có cả phần dương $\mathbf{x}_+ = \operatorname{ReLU}(\mathbf{x})$ và phần âm $\mathbf{x}_- = \operatorname{ReLU}(-\mathbf{x})$. Nếu cắt cụt giá trị âm như GeM cổ điển sẽ sinh độ lệch dương giả tạo (DC bias). Ta tính:
     $$\mathbf{g}_k = \left(\sum_{\mathbf{p}} \tilde{w}_k(\mathbf{p})\big(\mathbf{x}_+(\mathbf{p}) + \varepsilon\big)^p\right)^{1/p} - \left(\sum_{\mathbf{p}} \tilde{w}_k(\mathbf{p})\big(\mathbf{x}_-(\mathbf{p}) + \varepsilon\big)^p\right)^{1/p}$$
   * Cộng thêm mã hóa hình học 8 chiều `[cx, cy, sx, sy, cos(2θ), sin(2θ), mean_depth, mean_sal]` qua cổng nhỏ `tanh(pose_scale)` → Thu được **17 token vùng chuẩn hóa đơn vị** `[B, 17, 512]` (`1` toàn cục + `16` Elip vùng).

### Trạm 3: Bộ Nhớ Nguyên Mẫu Tương Tác Chéo (`PMCI` — [prototype_memory.py](file:///d:/Lab/D2VPR/g3vpr/prototype_memory.py))
Làm sao để có "trí tuệ toàn cảnh" như `cross_image_encoder` của `D²-VPR` mà **không được phép nhìn trộm ảnh khác trong cùng lô**?
* Ta khởi tạo một **Ngân hàng Bộ nhớ $M = 128$ Nguyên mẫu Địa-Ngữ cảnh trực giao** $\mathbf{P} \in \mathbb{R}^{128 \times 512}$ là tham số học cố định của mạng.
* Với **từng bức ảnh độc lập** (`batch_first=True`!), 17 token của bức ảnh đó thực hiện:
  1. **Self-Attention nội ảnh:** 17 vùng trong *cùng 1 bức ảnh* trao đổi thông tin với nhau.
  2. **Cross-Attention với 128 Nguyên mẫu $\mathbf{P}$:** Đối chiếu vùng ảnh đang nhìn với 128 "khái niệm kiến trúc chuẩn" trong bộ nhớ để tự cân bằng sáng/tối, cổ điển/hiện đại.
* Vì $\mathbf{P}$ cố định sau khi train, việc bạn truyền `BS = 1` hay `BS = 32` cho ra kết quả **trùng khớp tuyệt đối đến chữ số thập phân thứ 7 (`sai số = 1.3e-7`)**!

### Trạm 4: Đầu Chiếu Phổ Phân Cấp Trực Giao `Matryoshka` ([matryoshka.py](file:///d:/Lab/D2VPR/g3vpr/matryoshka.py))
Từ 17 token không gian `[B, 17, 512]`, mỗi token được chiếu qua **ma trận trực giao riêng của đúng vùng đó** (`[17, 192, 512]` → `3.264` chiều phân biệt không gian, không dùng chung một phễu hẹp!) rồi lắp ghép theo **thứ tự ưu tiên từ tổng quát đến chi tiết** thành vector `4096` chiều:
* `0 .. 128` (`128-D`): Tóm tắt toàn cảnh `CLS` + trung bình vùng (`0.48 GB` RAM / 1M ảnh).
* `128 .. 256` (`256-D`): Bổ sung bố cục `4` góc phần tư `2×2` (`0.95 GB` RAM).
* `256 .. 512` (`512-D`): Bổ sung `3` dải mặt tiền & đường chân trời (`1.91 GB` RAM).
* `512 .. 1024` (`1024-D`): Bổ sung `9` vùng chi tiết `3×3` (`3.81 GB` RAM).
* `1024 .. 4096` (`2048-D` & `4096-D`): Phần dư phổ trực giao tần số cao trên toàn bộ 17 vùng (`15.26 GB` RAM, **0 tham số PCA!**).

Khi suy luận thực tế, muốn dùng chiều $d$ nào chỉ cần lấy `F.normalize(full_raw[:, :d], p=2, dim=-1)`!

---

## PHẦN 4: TOÁN HỌC CỦA 4 HÀM LOSS HỢP NHẤT ([losses.py](file:///d:/Lab/D2VPR/g3vpr/losses.py))

Trong lúc huấn luyện (và **chỉ lúc huấn luyện**, lúc chạy thật không tốn 1 byte nào của Thầy!), ta tối ưu hàm mục tiêu tổng hợp:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MRL-MS}} + \lambda_{2\text{D}}\mathcal{L}_{2\text{D}} + \lambda_{3\text{D}}\mathcal{L}_{3\text{D}} + \lambda_{\text{proto}}\mathcal{L}_{\text{proto}}$$

1. **$\mathcal{L}_{\text{MRL-MS}}$ (Matryoshka Multi-Similarity Loss):** Áp dụng hàm mất mát đa tương đồng chuẩn VPR ($\alpha=2, \beta=50, \text{base}=0.5$) đồng thời lên cả **6 lát cắt** $d \in \{128, 256, 512, 1024, 2048, 4096\}$, ép mọi "con búp bê Nga" từ nhỏ nhất đến lớn nhất đều tự đứng vững độc lập!
2. **$\mathcal{L}_{2\text{D}}$ (Chưng cất Nền tảng 2D Lai từ Thầy `MegaLoc` `8448-D`):** Gồm 2 vế bổ trợ nhau:
   * **Cầu nối Không gian Con Trực tiếp (`MatryoshkaTeacherBridge`):** Gắn ma trận chiếu phụ chỉ dùng lúc train đưa `z[:256], z[:512], z[:1024], z[:4096]` lên khớp trực tiếp góc Cosine $1 - \cos(\hat{\mathbf{t}}_d, \mathbf{t}_{2\text{D}})$ với vector `8.448` chiều của `MegaLoc` (CVPR 2025). Mỗi ảnh nhận ngay `8.448` tín hiệu hướng dẫn trực tiếp!
   * **Chưng cất Quan hệ Trắc địa Che Dương Tính (`NM-GRD`):** Khớp ma trận Gram Cosine $4\|\mathbf{S}_S - \mathbf{S}_T\|_F^2$ và phân phối KL $T \cdot D_{\text{KL}}(p_T \| p_S)$ **chỉ tính trên các cặp ảnh khác địa điểm** (`labels[i] != labels[j]`, $T = 0.15$). Nhờ che các cặp trùng địa điểm đi, Softmax không bị dồn 99.9% vào 3 ảnh dương tính dễ mà buộc mạng trò phải học *mối liên hệ kiến trúc tinh tế giữa các con phố khác nhau* — chìa khóa chống quên miền lịch sử `AmsterTime`!
3. **$\mathcal{L}_{3\text{D}}$ (Chưng cất Hình học 3D bất biến tỷ lệ từ `Depth-Anything-V2`):** Dùng hàm **Eigen Scale-Invariant Log-Depth Loss**:
   $$\mathcal{L}_{\text{SI}} = \frac{1}{N}\sum_i \delta_i^2 - \frac{1}{2N^2}\left(\sum_i \delta_i\right)^2, \quad \delta_i = \log \hat{D}_i - \log D_i^*$$
   kèm giám sát biên độ sâu cho cổng $\mathbf{S}_{\text{geo}}$ để tự động dìm bầu trời (`depth ~ 0`) xuống thấp và nâng vùng rìa kiến trúc lên cao.
4. **$\mathcal{L}_{\text{proto}}$ (Đa dạng hóa Nguyên mẫu):** Phạt $\|\mathbf{P}\mathbf{P}^\top - \mathbf{I}_{128}\|_F^2$ để 128 nguyên mẫu trong `PMCI` vuông góc nhau, không bị suy biến trùng lặp.

---

## PHẦN 5: BÀI HỌC KỸ THUẬT ĐẮT GIÁ TỪ `v1` LÊN `v2` (Chứng Minh Bạn Thực Sự Hiểu Sâu!)

Nếu hội đồng hỏi *"Quá trình nghiên cứu và gỡ lỗi thực tế của bạn diễn ra thế nào?"*, đây chính là bằng chứng vàng chứng minh bạn làm nghiên cứu thật 100% chứ không phải "chép bài":

1. **Phát hiện nút thắt ở `v1`:** Ở lượt chạy `v1`, mô hình đã đạt `0.00%` độ lệch `BS=1` và `86.75%` trên `WashingtonDC`, nhưng sang `SPED` chỉ đạt `55.02%` (trong khi đo trực tiếp thân mạng ViT thô bên trong `v1` đã đạt `61.78%` — tức phần đầu gom làm tụt `-6.76%`!).
2. **Tìm ra 4 nguyên nhân toán học chính xác:**
   * **Sai lầm `T²` khi `T < 1`:** Ở `v1`, nhân KL với `0.07² = 0.0049` làm triệt tiêu 99.5% đạo hàm của Thầy `MegaLoc` (quy tắc nhân `T²` của Hinton chỉ đúng khi `T > 1` trên logit chưa chuẩn hóa; với tương đồng Cosine chia `T = 0.15`, muốn đạo hàm $\frac{\partial}{\partial S_S} = O(1)$ phải nhân với `T` chứ không phải `T²`!).
   * **Độ lệch dương (DC bias) và đạo hàm `0^p` của GeM:** Cắt `clamp(min=1e-6)` mất nửa âm của `LayerNorm`; còn `ReLU(x)^p` tại `x = 0` sinh đạo hàm theo số mũ $\frac{\partial (0^p)}{\partial p} = 0^p \ln(0) = \text{NaN}$! Thêm `(ReLU(x) + 10^{-5})^p` ở `Signed GeM` khử sạch cả DC bias lẫn lỗi `NaN`.
   * **Tỷ lệ Tín hiệu trên Nhiễu (SNR) của mã hóa vị trí:** Tọa độ `pose_encoder` chưa có cổng biên độ sẽ đè bẹp vector hình ảnh đơn vị; thêm cổng `tanh(pose_scale)` khởi tạo `0.05` + ma trận chiếu vùng trực giao riêng biệt giúp ngay tại **Bước 0 chưa train**, điểm `SPED` đã vọt từ `55.02%` lên **`76.94%`** và `AmsterTime` vọt từ `13.08%` lên **`41.35%`**!
   * **Tràn số `float16` dưới `GradScaler(65536)`:** Khi tổng Loss $\approx 10\text{–}17$, nhân `65536` vượt ngưỡng tối đa `65504` của `float16`. Hạ `init_scale = 4096` và ép tính chuẩn hóa L2 / Loss ở `float32` giúp huấn luyện hỗn hợp `fp16` mượt tuyệt đối.

---

## PHẦN 6: BỘ CÂU HỎI VẤN ĐÁP "GIÁO SƯ / REVIEWER HỎI XOÁY — BẠN ĐÁP GỌN"

### Câu 1: *"Tại sao không dùng luôn `D²-VPR` bản `finetune_no_encoder.pth` (vốn không bị tụt ở `BS=1`) mà phải sinh ra `G³-VPR`?"*
> **Trả lời:** Bản `finetune_no_encoder.pth` tuy không phụ thuộc `BS`, nhưng mang **3 điểm yếu lớn**: (1) Bị quên miền nghiêm trọng do cắt bỏ Encoder và Teacher lúc fine-tune (`AmsterTime` rớt từ `63.04%` xuống chỉ còn `49.23%`, `SPED` rớt từ `91.43%` xuống `86.00%`); (2) Vẫn dính lỗi chết `68.36%` bản đồ lấy mẫu `[0, 1]` và ô chữ nhật 2D mù phối cảnh 3D; (3) Xuất vector khổng lồ `10.752` chiều ngốn `40.05 GB` RAM / 1 triệu ảnh. `G³-VPR` giữ được ngữ cảnh toàn cục nhờ bộ nhớ 128 nguyên mẫu `PMCI` mà vẫn bất biến `BS=1`, đồng thời nén bản đồ nhỏ đi tới `83.4` lần nhờ `Matryoshka`.

### Câu 2: *"Tại sao gọi là '3D-Guided Anisotropic 2D Gaussian' mà lúc chạy thực tế lại chỉ cần đầu vào là 1 ảnh RGB 2D (không cần cảm biến LiDAR hay camera chiều sâu)?"*
> **Trả lời:** Thông tin 3D được **chưng cất nội hóa (distilled & internalized)** từ mô hình nền tảng `Depth-Anything-V2` vào nhánh nhẹ `DepthwiseSeparableHead` (`0.18M` tham số) gắn ngay trên tầng cấu trúc nông `[2, 4]` của ViT trong quá trình train. Nhờ vậy lúc chạy thực tế trên robot, mô hình tự suy ra độ sâu tương đối $\hat{\mathbf{D}}$ và cổng kiến trúc tĩnh $\mathbf{S}_{\text{geo}}$ trực tiếp từ 1 ảnh RGB với chi phí tính toán gần như bằng 0 (`< 0.4 ms`), không cần gắn LiDAR hay chạy mạng Depth lớn!

### Câu 3: *"Khởi động ấm thân `ViT-S + MulConvAdapter` từ trọng số đã học trên `GSV-Cities` có bị rò rỉ dữ liệu kiểm thử (Data Leakage) sang `AmsterTime`, `SPED` hay `Pitts30k` không?"*
> **Trả lời:** **Hoàn toàn không có bất kỳ sự rò rỉ nào!** Trọng số khởi động ấm đó chỉ chứa phần thân `ViT-S + MulConvAdapter` được huấn luyện **100% trên tập `GSV-Cities`** (chưa từng nhìn thấy một bức ảnh nào của `AmsterTime`, `SPED` hay `Pitts30k`). Hơn nữa, trong Bảng Ablation (`M1 → M5_full`), cả 5 biến thể đều xuất phát từ **cùng một điểm khởi tạo và cùng tập dữ liệu huấn luyện**, đảm bảo so sánh đóng góp khoa học công bằng tuyệt đối.

### Câu 4: *"Đầu ra Matryoshka khác gì so với việc lấy vector 4096 chiều rồi cắt đại 128 số đầu tiên, hoặc dùng PCA?"*
> **Trả lời:** Ở một mạng thông thường, thông tin được rải đều ngẫu nhiên khắp 4096 chiều nên nếu cắt 128 số đầu tiên, độ chính xác sẽ sụp đổ hoàn toàn. Còn với PCA, bạn phải thu thập hàng chục nghìn ảnh để tính trị riêng ngoại tuyến và phải lưu thêm ma trận chiếu nặng `44.04 triệu` tham số (`10752 × 4096`). Trong `G³-VPR`, cả **kiến trúc phân tầng từ thô đến tinh** (`128D` toàn cảnh → `256D` góc phần tư → `512D` mặt tiền → `1024D..4096D` chi tiết) lẫn **hàm Loss đa lát cắt `L_MRL-MS` + `MatryoshkaTeacherBridge`** đều ép thông tin quan trọng nhất dồn về các chiều đầu tiên ngay trong lúc học end-to-end, nên cắt ở đâu cũng là một vector hoàn chỉnh chuẩn L2 mà tốn **0 tham số phụ**!

### Câu 5: *"Bộ nhớ 128 nguyên mẫu `PMCI` đóng vai trò gì về mặt trực giác?"*
> **Trả lời:** Bạn có thể hình dung 128 nguyên mẫu trực giao đó như **"128 điểm mốc từ vựng kiến trúc toàn cầu"** (mái ngói cổ châu Âu, tường kính cao ốc, vòm cầu, hàng cây ven phố...). Khi một bức ảnh đen trắng năm 1920 từ `AmsterTime` và một bức ảnh màu năm 2024 chụp cùng tòa nhà đi vào mạng, cả hai đều đối chiếu (Cross-Attention) lên cùng các nguyên mẫu kiến trúc đó, giúp kéo hai biểu diễn về gần nhau bất chấp khoảng cách 100 năm thời gian mà không cần gom chùm ảnh lúc test!
