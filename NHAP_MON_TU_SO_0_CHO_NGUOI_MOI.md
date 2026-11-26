# Sách Học Từ Con Số 0: 7 Viên Gạch Nền Tảng Để Hiểu Sâu Toàn Bộ G³-VPR

> **Dành cho người muốn hiểu bản chất từ gốc rễ (không dùng thuật ngữ đánh đố):**  
> Sở dĩ đọc tài liệu nghiên cứu thấy "ngợp" là vì một bài báo A\* ghép **7 viên gạch cơ bản** lại cùng một lúc. Nếu bạn tách rời từng viên gạch ra học theo đúng thứ tự từ **Bài 1 → Bài 7** dưới đây, mỗi bài chỉ mất **5 phút** với ví dụ cộng trừ nhân chia cấp 3, bạn sẽ tự khắc đọc hiểu 100% file [GIAI_THICH_GOC_LAN_LAI_G3VPR.md](file:///d:/Lab/D2VPR/GIAI_THICH_GOC_LAN_LAI_G3VPR.md) và toàn bộ code Python!

---

## BÀI 1: Vector Đặc Trưng, Chuẩn Hóa L2, Góc Cosine & Kho FAISS Là Gì?

### 1. Máy tính không hiểu "hình ảnh", nó chỉ hiểu "dãy số" (Vector)
Khi bạn nhìn bức ảnh một ngôi nhà thờ cổ, não bạn thấy *"cao, có mái nhọn, tường gạch đỏ"*. Máy tính sẽ chấm điểm từng đặc điểm đó thành một dãy số, ví dụ chỉ **3 con số**:
* `[Độ cao, Độ nhọn mái, Độ đỏ của gạch]`
* Ảnh Nhà thờ A → Vector $\mathbf{a} = [3, 4, 0]$
* Ảnh Nhà thờ A chụp lúc trời hơi tối (mọi số bị mờ đi một nửa) → Vector $\mathbf{b} = [1.5, 2.0, 0]$
* Ảnh Bãi biển C → Vector $\mathbf{c} = [0, 0, 5]$

### 2. Tại sao phải "Chuẩn hóa L2" (`F.normalize`)?
Nhìn $\mathbf{a} = [3, 4, 0]$ và $\mathbf{b} = [1.5, 2.0, 0]$: rõ ràng là **cùng một nhà thờ**, chỉ khác độ sáng! Nếu tính khoảng cách thẳng thông thường thì chúng lệch nhau khá xa.
* **Độ dài L2** của vector $\mathbf{a} = [3, 4, 0]$ (định lý Pytago cấp 2) là: $\sqrt{3^2 + 4^2 + 0^2} = \sqrt{25} = 5$.
* **Chuẩn hóa L2** nghĩa là lấy từng số chia cho độ dài của nó để ép mũi tên dài đúng bằng **`1.0`**:
  * $\hat{\mathbf{a}} = [3/5, 4/5, 0] = [0.6, 0.8, 0]$
  * Độ dài của $\mathbf{b} = [1.5, 2.0, 0]$ là $\sqrt{1.5^2 + 2^2} = 2.5$ → Chia cho $2.5$ ta được $\hat{\mathbf{b}} = [0.6, 0.8, 0]$!
  * **Kỳ diệu chưa:** Sau khi chuẩn hóa L2, $\hat{\mathbf{a}}$ và $\hat{\mathbf{b}}$ trở nên **giống hệt nhau tuyệt đối**, triệt tiêu ảnh hưởng sáng/tối!

### 3. "Tương đồng Cosine" (`Cosine Similarity`) và `FAISS` là gì?
Khi các mũi tên đều dài bằng `1.0`, để biết 2 bức ảnh giống nhau cỡ nào, ta chỉ cần **nhân từng cặp số tương ứng rồi cộng lại** (gọi là tích vô hướng $\cos\theta$):
* Giữa $\hat{\mathbf{a}}$ và $\hat{\mathbf{b}}$ (cùng nhà thờ): $0.6 \times 0.6 + 0.8 \times 0.8 + 0 \times 0 = \mathbf{1.0}$ *(giống 100%, góc $0^\circ$)*.
* Giữa $\hat{\mathbf{a}}$ và Bãi biển $\hat{\mathbf{c}} = [0, 0, 1]$: $0.6 \times 0 + 0.8 \times 0 + 0 \times 1 = \mathbf{0.0}$ *(khác hoàn toàn, vuông góc $90^\circ$)*.
* **FAISS (`faiss.IndexFlatIP`):** Chỉ đơn giản là một thư viện siêu nhanh bằng C++ của Meta giúp thực hiện phép nhân cộng đó giữa **1 ảnh truy vấn** với **1.000.000 ảnh bản đồ** trong vài phần nghìn giây!
* *Trong dự án của ta:* Thay vì 3 con số, mỗi ảnh có `4.096` con số (hoặc cắt gọn còn `128` con số).

---

## BÀI 2: Mạng ViT (Vision Transformer), Patch `14×14`, Token `CLS` & `Adapter` Là Gì?

### 1. Cắt ảnh thành ô cờ (`Patch 14×14`)
Mạng Transformer vốn sinh ra để dịch câu văn (mỗi câu gồm nhiều **từ / token**). Để đưa bức ảnh vuông `224×224` điểm ảnh vào Transformer:
* Ta lấy kéo cắt bức ảnh thành các ô vuông nhỏ kích thước `14×14` điểm ảnh.
* Chiều ngang có $224 / 14 = 16$ ô; chiều dọc có $224 / 14 = 16$ ô → Tổng cộng có **$16 \times 16 = 256$ mảnh ghép (`patch tokens`)**. Mỗi mảnh ghép được biểu diễn bằng một dãy `384` số.

### 2. Token `CLS` ("Lớp trưởng tóm tắt") là gì?
Ngoài `256` mảnh ghép hình ảnh, người ta nhét thêm **1 nhân vật đặc biệt đứng ở vị trí số 0** gọi là **`CLS` token** (tổng cộng thành `1 + 256 = 257` token).
* `256` mảnh kia lo giữ chi tiết tại đúng ô đất của nó (góc trái có cái cây, góc phải có cửa sổ).
* Còn anh "lớp trưởng" `CLS` đi hỏi thăm khắp 256 mảnh qua 12 tầng để đúc kết thành **1 bản tóm tắt toàn cảnh chung của cả bức ảnh**.

### 3. Tầng nông (`[2, 4]`) vs Tầng sâu (`[8, 12]`) và `Adapter` là gì?
* Bức ảnh đi lần lượt qua **12 tầng (blocks)** từ tầng 1 đến tầng 12 của bộ não `DINOv2` (do Meta huấn luyện sẵn trên 142 triệu ảnh):
  * **Các tầng đầu (`2, 4` — Tầng nông):** Mắt mới nhìn vào ảnh nên thấy rất rõ **đường nét hình học, góc cạnh, độ sâu xa gần**, nhưng chưa hiểu "đây là tiệm bánh hay ngân hàng".
  * **Các tầng cuối (`8, 12` — Tầng sâu):** Đã suy luận sâu nên hiểu **ý nghĩa khái niệm** (dù ảnh màu năm 2024 hay ảnh đen trắng năm 1920 nó vẫn nhận ra "đây là dãy nhà cổ bên bờ kênh"), nhưng vị trí mép tường lại hơi nhòe đi.
  * *Đóng góp 1 của `G³-VPR`:* Lấy tầng `[2, 4]` để đo độ sâu 3D & lọc rác bầu trời, rồi ghép với tầng `[8, 12]` để giữ ngữ nghĩa vượt thời gian!
* **`MulConvAdapter` là gì?** Nếu sửa trực tiếp toàn bộ 22 triệu tham số của `DINOv2` từ đầu thì rất dễ làm "hỏng" kiến thức quý giá mà Meta đã dạy nó. Vì vậy ta gắn thêm các mạch nhánh nhỏ gọi là **Adapter** (chỉ nặng ~2.3 triệu tham số) nằm cạnh mỗi tầng để "bẻ lái nhẹ" kỹ năng nhìn chung sang chuyên ngành nhận diện đường phố!

---

## BÀI 3: `Attention` (Sự Chú Ý) & Tại Sao `batch_first=False` Của Bài Cũ Lại Gây Họa?

### 1. Hiểu `Attention` (`Query - Key - Value`) bằng ví dụ tìm sách
Mỗi mảnh ảnh đóng 3 vai:
* **`Query` (Câu hỏi):** *"Tôi là cánh cửa gỗ, quanh đây có mái vòm nào đi cặp với tôi không?"*
* **`Key` (Nhãn gáy sách):** Mỗi mảnh khác giơ biển lên: *"Tôi là bầu trời"*, *"Tôi là mái vòm đá"*.
* **`Value` (Nội dung):** Khi `Query` khớp với `Key` ("cửa gỗ" tìm thấy "mái vòm đá"), chúng hòa trộn `Value` lại để khẳng định: *"À, cửa gỗ nằm dưới mái vòm đá = Mặt tiền nhà thờ!"*.

### 2. Hai kiểu Attention trong `G³-VPR` (`PMCI`):
* **`Self-Attention` (Tự chú ý nội ảnh):** Các vùng *bên trong cùng 1 bức ảnh* nói chuyện với nhau.
* **`Cross-Attention` (Chú ý chéo với Bộ nhớ 128 Nguyên mẫu):** Bức ảnh đặt câu hỏi (`Query`), còn bộ sách tham khảo (`Key, Value`) là **128 khuôn mẫu kiến trúc cố định** nằm sẵn trong mạng.

### 3. Tại sao bài cũ `D²-VPR` bị sập ở `BS=1`?
* Trong lập trình GPU, để chạy cho nhanh, người ta thường xếp một chồng gồm `B` bức ảnh độc lập đi vào cùng lúc (gọi là **Batch Size = `B`**, ví dụ `B = 8` bức ảnh khác nhau).
* Đáng lẽ bức ảnh nào chỉ được tính toán bên trong bức ảnh đó! Nhưng ở `D2VPR.py`, tác giả bật nhầm cờ `batch_first=False`, khiến **bức ảnh số 1 quay sang làm phép `Self-Attention` với bức ảnh số 2, 3, ..., 8 đang đi cùng chuyến xe**!
* Giống hệt học sinh quen **quay sang bàn bên cạnh thảo luận bài (`B=8`)**. Đến khi đi thi thật ngoài đời, robot đi một mình chỉ có **1 bức ảnh (`B=1`)**, quay sang bên cạnh không có ai cả → điểm rơi thẳng từ `63%` xuống `29%`!
* Còn bộ nhớ `PMCI` của ta giống như học sinh **học thuộc sẵn 128 công thức chuẩn trong đầu** → ngồi thi 1 mình (`B=1`) hay ngồi phòng đông (`B=32`) đều làm bài giống hệt nhau 100%!

---

## BÀI 4: `grid_sample`, `GeM Pooling` & Vòng Elip Gaussian 2D Là Gì?

### 1. Hệ tọa độ `[-1, 1]` của `F.grid_sample`
Hãy nhớ lại trục tọa độ Oxy hồi cấp 2: gốc `(0, 0)` nằm **chính giữa tờ giấy**:
* Góc trên-trái là `(-1, -1)`; Góc trên-phải là `(+1, -1)`
* Góc dưới-trái là `(-1, +1)`; Góc dưới-phải là `(+1, +1)`

Bài cũ `D²-VPR` truyền tọa độ nằm từ `0` đến `1` vào hàm đọc ảnh `grid_sample` → nên máy tính chỉ đọc từ tâm `(0, 0)` đến góc dưới-phải `(+1, +1)`, bỏ hoang **3 góc còn lại (68.36% bức ảnh)**! Code `G³-VPR` đổi chuẩn xác bằng công thức `coord * 2 - 1` nên nhìn trọn 100% bức ảnh.

### 2. `GeM Pooling` (Trung bình lũy thừa tổng quát) dễ hiểu trong 20 giây
Giả sử trong 1 vùng ảnh có 4 ô nhỏ với độ nổi bật của biển hiệu là: `[1, 1, 1, 9]` (3 ô nền tường nhạt `1`, 1 ô trúng chữ biển hiệu rất nét `9`):
* Nếu lấy **Trung bình cộng (`p = 1`):** $(1 + 1 + 1 + 9) / 4 = \mathbf{3.0}$ → Chữ biển hiệu `9` bị 3 ô tường nhạt kéo tụt xuống lờ mờ!
* Nếu lấy **Lũy thừa bậc `p = 3` (`GeM`):** $\sqrt[3]{(1^3 + 1^3 + 1^3 + 9^3) / 4} = \sqrt[3]{732 / 4} = \sqrt[3]{183} \approx \mathbf{5.68}$!
* Khi `p` càng lớn, `GeM` càng **tự động làm nổi bật chi tiết đắt giá nhất (`9`)** mà vẫn mượt mà, có đạo hàm để mạng tự học được cả con số `p` tốt nhất!

### 3. Elip Gaussian 2D so với Ô chữ nhật
* Ô chữ nhật đứng thẳng giống như cái khuôn bánh vuông cứng: cắt ngang qua tường xiên là dính luôn bầu trời.
* **Elip Gaussian 2D** được định nghĩa bởi đúng 5 con số cực kỳ trực quan: tâm `(cx, cy)`, độ rộng-dài `(sx, sy)` và **góc nghiêng `θ`**. Nó tỏa trọng số êm dần từ tâm ra viền theo hình chuông úp và xoay nghiêng đúng theo chiều vách tường 3D!

---

## BÀI 5: Học Đo Lường (`Multi-Similarity Loss`) & Học Chưng Cất 2 Thầy (`Distillation`)

Làm sao máy tính "học" được? Ta cho nó một **Hàm Phạt (`Loss`)** — làm sai thì số phạt cao, làm đúng thì số phạt giảm về `0`:

1. **Phạt Đa Tương Đồng (`Multi-Similarity Loss`):**
   * Mỗi lượt học, ta đưa vào 32 địa điểm, mỗi địa điểm có 4 bức ảnh chụp ở các góc/năm khác nhau (`32 × 4 = 128` ảnh).
   * Hàm phạt ra lệnh: *"4 bức ảnh cùng 1 ngã tư phải kéo góc Cosine sát lên `1.0`; còn bức ảnh ở ngã tư khác (nhất là ngã tư nhìn hao hao giống làm dễ nhầm!) phải đẩy góc Cosine xuống thấp dưới `0.5`!"*.
2. **Học Chưng Cất Từ 2 Thầy Giáo Khổng Lồ (`Dual-Teacher Distillation`):**
   * Mô hình của ta nhỏ gọn (`36M` tham số) để chạy nhanh trên robot. Nhưng lúc đang ngồi "học ở trường" (trên Kaggle), ta mời **2 Giáo sư khổng lồ** (nặng hàng trăm triệu tham số) ngồi cạnh chấm bài:
     * **Thầy 1 (`MegaLoc` — Chuyên gia 2D):** Đã đi khắp thế giới, nhìn ảnh đen trắng năm 1920 vẫn biết nó giống tòa nhà nào. Ta lập một "cầu nối nháp" (`MatryoshkaTeacherBridge`) ép vector của trò hướng theo đúng la bàn `8.448` chiều của Thầy, đồng thời học bảng so sánh chéo giữa các con phố (`KL Divergence` — đo độ lệch giữa 2 bảng tỷ lệ phần trăm nhận định của Thầy và Trò).
     * **Thầy 2 (`Depth-Anything-V2` — Chuyên gia 3D):** Nhìn bức ảnh bất kỳ là biết ngay chỗ nào gần, chỗ nào xa tít tận chân trời. Trò nhìn theo Thầy 2 để học cách tự đoán độ sâu 3D và tự tắt vùng bầu trời.
   * **Khi tốt nghiệp đem đi chạy thật (`eval_g3vpr_sota.py`):** Cả 2 ông Thầy và cầu nối nháp đều cất ở nhà! Mô hình trò đi thi một mình, nhẹ tênh mà mang trọn nội công của 2 Thầy!

---

## BÀI 6: `PCA` Ngoại Tuyến vs. Búp Bê Nga `Matryoshka`

* **Cách cũ (`PCA`):** Giống như bạn nhét bừa bộn đồ đạc vào 1 chiếc vali khổng lồ (`10.752` ngăn). Lúc ra sân bay bị quá cân, bạn phải vác thêm một **cỗ máy ép thủy lực nặng 44 triệu linh kiện** (`ma trận PCA 10752 × 4096`) để ép nó nhỏ lại.
* **Cách của ta (`Matryoshka` — Búp bê Nga lồng nhau):** Ngay từ lúc xếp đồ (lúc train), ta quy định nghiêm ngặt:
  * **Ngăn đầu tiên (`0 → 128`):** Bắt buộc đựng hộ chiếu, ví tiền, chìa khóa (những thứ sống còn nhất để nhận diện toàn cảnh!).
  * **Ngăn thứ hai (`128 → 256`):** Đựng bố cục 4 góc phố.
  * **Ngăn thứ ba (`256 → 512`):** Đựng mặt tiền và đường chân trời.
  * **Các ngăn cuối (`512 → 4096`):** Đựng các chi tiết hoa văn nhỏ li ti.
  * Và ta chấm điểm phạt `Multi-Similarity Loss` trên **từng ngăn một**! Nhờ vậy lúc lên máy bay (chạy trên drone ít RAM), bạn chỉ cần **kéo khóa lấy đúng 128 ngăn đầu tiên (`z[:128]`)** — mất 0 giây tính toán, tốn 0 linh kiện phụ, nhẹ hơn **83.4 lần RAM**!

---

## BÀI 7: Đạo Hàm (`Gradient`), Lỗi Nổ Số `NaN` & Chế Độ `float16 AMP`

* **Đạo hàm (`Gradient`) là gì?** Hãy tưởng tượng bạn bị bịt mắt đứng trên sườn núi (độ cao là `Loss`) và muốn đi xuống thung lũng (`Loss = 0`). Bàn chân bạn dò xem *"bước sang trái hay sang phải thì dốc đi xuống?"* — độ dốc dưới chân bạn chính là **Đạo hàm (`grad`)**!
  * Ở `v1`, ta lỡ nhân nhầm hệ số `0.0049` vào độ dốc của Thầy `MegaLoc`, làm mặt đất phẳng lì như mặt bàn → mô hình đứng im không bước về phía Thầy được!
* **`float32` vs `float16 (AMP)` và `NaN`:**
  * Bình thường máy tính lưu mỗi con số bằng `32 bit` (`float32`), đếm được tới hàng tỷ tỷ nhưng tốn VRAM và chậm.
  * Để chạy nhanh gấp đôi trên GPU T4, ta bật chế độ **`float16` (`16 bit`)** — nhưng tờ giấy nháp `16 bit` chỉ ghi được con số tối đa là **`65.504`**! Nếu phép nhân nháp vượt quá `65.504`, máy tính ghi chữ `inf` (vô cực), rồi lấy `vô cực - vô cực` là nổ tung thành **`NaN` (Not a Number — hỏng toàn bộ mô hình)**!
  * Ở `v2`, ta sửa bằng cách: cho phần nhân ma trận nặng chạy ở `16 bit` cho nhanh, nhưng riêng bước chia độ dài L2, lũy thừa `GeM` (cộng thêm `0.00001` để không bao giờ tính `log(0)`!) và tính điểm phạt `Loss` thì chuyển sang giấy nháp `32 bit` rộng rãi → Vừa nhanh như tên bắn, vừa an toàn tuyệt đối 100%!

---

## Bạn Đã Sẵn Sàng Chưa? Cách Ghép 7 Viên Gạch Lại Chỉ Trong 1 Sơ Đồ Duy Nhất:

`Ảnh RGB (224×224)`  
$\downarrow$ **[Bài 2]** Cắt `256` mảnh `14×14` qua `ViT-S + Adapter`, tách tầng nông `[2,4]` (3D) & tầng sâu `[8,12]` (Ngữ nghĩa)  
$\downarrow$ **[Bài 4]** Dùng `16` vòng **Elip Gaussian xoay góc $\theta$** + **`GeM`** hút đặc trưng ở vùng kiến trúc tĩnh (tọa độ chuẩn `[-1,1]`)  
$\downarrow$ **[Bài 3]** Đối chiếu `Cross-Attention` với **128 Nguyên mẫu `PMCI`** (không nhìn trộm lô ảnh → chuẩn tuyệt đối ở `BS=1`!)  
$\downarrow$ **[Bài 6 + Bài 1]** Xếp vào **Búp bê Nga `Matryoshka` `4096` chiều** chuẩn hóa **L2**, học từ **2 Thầy (`MegaLoc` + `Depth-Anything-V2`)** **[Bài 5]** với chế độ **`fp16/fp32` an toàn** **[Bài 7]**!
