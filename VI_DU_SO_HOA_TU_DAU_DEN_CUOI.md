# Ví Dụ Số Hóa & Hình Vẽ Minh Họa Thuật Toán G³-VPR Từ Đầu Đến Cuối

> Tài liệu này thu nhỏ toàn bộ thuật toán **G³-VPR** về một bức ảnh bàn cờ **`2 × 2 = 4` ô** với các con số cụ thể bấm máy tính kiểm tra lại được ngay, đi trọn vẹn qua **6 Trạm** từ lúc ảnh đi vào đến lúc `FAISS` trả kết quả và `Loss` cập nhật trọng số.

---

## ĐẦU VÀO: Bức Ảnh Đường Phố Thu Nhỏ `2 × 2` (4 Ô Ảnh)

Giả sử robot chụp 1 tấm ảnh đường phố. Thay vì chia `16 × 16 = 256` ô như code thật, ta chia thành **`2 × 2 = 4` ô** để tính tay từng con số:

```text
y=0.0 +----------------------------+----------------------------+
      | Ô 1: Tâm (x=0.25, y=0.25)  | Ô 2: Tâm (x=0.75, y=0.25)  |
      |   🏛️ MÁI NHÀ CỔ (Tĩnh)     |   ☁️ BẦU TRỜI MÂY (Rác!)   |
y=0.5 +----------------------------+----------------------------+
      | Ô 3: Tâm (x=0.25, y=0.75)  | Ô 4: Tâm (x=0.75, y=0.75)  |
      |   🚚 XE TẢI ĐỖ (Vật cản!)  |   🧱 TƯỜNG GẠCH CỔ (Tĩnh)  |
y=1.0 +----------------------------+----------------------------+
     x=0.0                        x=0.5                        x=1.0
```

* **Tình huống thực tế oái oăm:** Dãy nhà cổ nằm **chéo góc $45^\circ$** từ `Ô 1` (trên-trái) xuống `Ô 4` (dưới-phải). Trong khi đó `Ô 2` là bầu trời trống rỗng và `Ô 3` bị một chiếc xe tải đậu chắn ngang!

---

## TRẠM 1: Thân Mạng `ViT` & Tách Cổng 3D (`GeoSemanticGatedBackbone` — `backbones.py`)

Giả sử mỗi ô chỉ biểu diễn bằng **2 con số đặc trưng**: `[Độ cổ kính, Độ gạch/khối cứng]` *(thực tế là `512` số)*.

1. **Tầng sâu `[8, 12]` rút ra Đặc trưng Ngữ nghĩa $\mathbf{F}_{\text{sem}}$:**
   * `Ô 1 (Mái nhà cổ)`: `[1.0, 0.4]`
   * `Ô 2 (Bầu trời)`: `[0.0, 0.0]`
   * `Ô 3 (Xe tải)`: `[0.2, 0.9]` *(Nguy hiểm: thùng xe vuông vức tạo số `0.9` gây nhiễu rất mạnh!)*
   * `Ô 4 (Tường gạch)`: `[0.8, 0.8]`
   * `CLS (Tóm tắt thô)`: `[0.5, 0.5]`

2. **Tầng nông `[2, 4]` đi qua đầu dự đoán 3D (`geo_head`) sinh ra Cổng Kiến trúc Tĩnh $\mathbf{S}_{\text{geo}} \in (0, 1)$:**
   * Nhìn độ sâu 3D, mạng thấy `Ô 2` xa vô tận (bầu trời) và `Ô 3` lồi sát mặt camera (xe tải chắn đường), nên bật công tắc:
   * `Ô 1 (Mái nhà)`: **$s_1 = 0.90$** *(Mở cổng!)*
   * `Ô 2 (Bầu trời)`: **$s_2 = 0.10$** *(Khóa cổng!)*
   * `Ô 3 (Xe tải)`: **$s_3 = 0.10$** *(Khóa cổng!)*
   * `Ô 4 (Tường gạch)`: **$s_4 = 0.90$** *(Mở cổng!)*

3. **Trộn đặc trưng $\mathbf{F}_{\text{fused}} = \mathbf{F}_{\text{sem}} + 0.25 \cdot s_i \cdot \mathbf{F}_{\text{struct}}$ và Chuẩn hóa độ dài Pytago (L2):**
   *(Với $\mathbf{F}_{\text{struct}}$ của 4 ô lần lượt là `[0.4, 0.0]`, `[0.0, 0.0]`, `[0.8, 0.8]`, `[0.4, 0.4]`)*:
   * `Ô 1`: $[1.0, 0.4] + 0.25 \times 0.90 \times [0.4, 0.0] = [1.09, 0.40] \xrightarrow{\text{chia } \sqrt{1.09^2 + 0.40^2} = 1.161} \hat{\mathbf{p}}_1 = \mathbf{[0.94,\; 0.34]}$
   * `Ô 2`: $\hat{\mathbf{p}}_2 = \mathbf{[0.00,\; 0.00]}$
   * `Ô 3`: $[0.2, 0.9] + 0.25 \times \mathbf{0.10} \times [0.8, 0.8] = [0.22, 0.92] \xrightarrow{\text{chia } \sqrt{0.22^2 + 0.92^2} = 0.946} \hat{\mathbf{p}}_3 = \mathbf{[0.23,\; 0.97]}$
   * `Ô 4`: $[0.8, 0.8] + 0.25 \times 0.90 \times [0.4, 0.4] = [0.89, 0.89] \xrightarrow{\text{chia } \sqrt{0.89^2 + 0.89^2} = 1.259} \hat{\mathbf{p}}_4 = \mathbf{[0.71,\; 0.71]}$

---

## TRẠM 2: Đám Mây Elip Gaussian Xoay Nghiêng (`3D-DGSA` — `gaussian_aggregator.py`)

Ta xét **1 đám mây Elip** đặt ở tâm ảnh $\boldsymbol{\mu} = (0.5, 0.5)$ để thấy sự lợi hại so với ô vuông của bài cũ:

1. **Đổi tọa độ tâm sang chuẩn `[-1, 1]` của PyTorch:**
   $$\text{sample\_grid} = 2 \times (0.5, 0.5) - 1 = (0.0,\; 0.0)$$
   *(Trúng phóc chính giữa bức ảnh! Bài cũ `D²-VPR` không nhân 2 trừ 1 nên truyền `(0.5, 0.5)` bị lệch tuốt xuống góc dưới-phải!).*

2. **Elip tự co giãn và xoay nghiêng $\theta = +45^\circ$ dọc theo dãy nhà cổ (`Ô 1` $\searrow$ `Ô 4`):**
   ```text
      +----------------------------+----------------------------+
      |   🏛️ Ô 1 (49% trọng số!)   |   ☁️ Ô 2 (chỉ 1%!)         |
      |        \  \  \             |                            |
      |         \  ( Elip xoay 45° ôm trọn dãy nhà cổ )         |
      |                   \  \  \  |                            |
      |   🚚 Ô 3 (chỉ 1%!)         |   🧱 Ô 4 (49% trọng số!)   |
      +----------------------------+----------------------------+
   ```
   * Bán trục vươn dài theo dãy nhà: $s_x = 0.50$; bán trục bóp hẹp ngang để né xe tải & bầu trời: $s_y = 0.15$.
   * Góc xoay $\theta = 45^\circ \implies \cos 45^\circ = 0.707,\; \sin 45^\circ = 0.707$.

3. **Bấm máy tính tính trọng số cho `Ô 4 (Tường gạch)` vs `Ô 3 (Xe tải)`:**
   * **Tại `Ô 4 (Tường gạch)`** có tọa độ $(0.75, 0.75) \implies$ độ lệch so với tâm $(0.5, 0.5)$ là $\Delta x = +0.25,\; \Delta y = +0.25$:
     * Xoay trục $45^\circ$: $\Delta x' = 0.707(0.25) + 0.707(0.25) = \mathbf{0.354}$, $\quad \Delta y' = -0.707(0.25) + 0.707(0.25) = \mathbf{0}$.
     * Độ xa Elip: $(0.354 \div 0.50)^2 + (0 \div 0.15)^2 = \mathbf{0.50}$.
     * Độ phủ hình chuông Gaussian: $G_4 = e^{-0.5 \times 0.50} = e^{-0.25} = \mathbf{0.78}$.
     * Nhân với Cổng tĩnh $(0.15 + 0.85 \times 0.90 = 0.915)$: $c_4 = 0.78 \times 0.915 = \mathbf{0.713}$ *(Tương tự `Ô 1` cũng được $c_1 = \mathbf{0.713}$)*.
   * **Tại `Ô 3 (Xe tải)`** có tọa độ $(0.25, 0.75) \implies \Delta x = -0.25,\; \Delta y = +0.25$:
     * Xoay trục $45^\circ$: $\Delta x' = \mathbf{0}$, $\quad \Delta y' = \mathbf{0.354}$ *(rơi đúng vào chiều hẹp $s_y = 0.15$ của Elip!)*.
     * Độ xa Elip: $(0 \div 0.50)^2 + (0.354 \div 0.15)^2 = \mathbf{5.56}$ *(Xa gấp 11 lần!)*.
     * Độ phủ hình chuông Gaussian: $G_3 = e^{-0.5 \times 5.56} = e^{-2.78} = \mathbf{0.062}$.
     * Nhân với Cổng tĩnh xe tải $(0.15 + 0.85 \times 0.10 = 0.235)$: $c_3 = 0.062 \times 0.235 = \mathbf{0.015}$ *(Tương tự `Ô 2` bầu trời cũng chỉ $c_2 = \mathbf{0.015}$)*.

4. **Chia tỷ lệ phần trăm (`norm_weights`):** Tổng $C = 0.713 + 0.015 + 0.015 + 0.713 = 1.456$:
   * **`Ô 1 (Mái nhà cổ)` = `49%`** | **`Ô 4 (Tường gạch)` = `49%`** | **`Ô 2 (Bầu trời)` = `1%`** | **`Ô 3 (Xe tải)` = `1%`**.
   * 👉 **Kết quả gom `GeM` ($p = 3$) rồi chuẩn hóa L2:**
     * Số thứ nhất: $\sqrt[3]{0.49 \times 0.94^3 + 0 + 0.01 \times 0.23^3 + 0.49 \times 0.71^3} = \sqrt[3]{0.582} = 0.835$
     * Số thứ hai: $\sqrt[3]{0.49 \times 0.34^3 + 0 + 0.01 \times 0.97^3 + 0.49 \times 0.71^3} = \sqrt[3]{0.203} = 0.588$
     * Chia Pytago $\sqrt{0.835^2 + 0.588^2} = 1.021 \implies$ **Token Vùng Elip $\mathbf{t}_1 = [0.82,\; 0.58]$** (cùng Token Toàn cảnh $\mathbf{t}_0 = [0.71,\; 0.71]$).
     *(Chiếc xe tải to tướng bị triệt tiêu mất 99% ảnh hưởng!)*

---

## TRẠM 3: Đối Chiếu Bộ Nhớ Nguyên Mẫu `PMCI` (`prototype_memory.py`)

Trong não robot cất sẵn **2 Nguyên mẫu vuông góc cố định** (thực tế là `128`):
* $\mathbf{m}_1 = [1.0, 0.0]$ *(Mẫu Nhà cổ)* và $\mathbf{m}_2 = [0.0, 1.0]$ *(Mẫu Khối đá)*.

Token $\mathbf{t}_1 = [0.82, 0.58]$ tự so khớp một mình với 2 nguyên mẫu ($\sim 56\%$ giống $\mathbf{m}_1$, $44\%$ giống $\mathbf{m}_2$), rút ra vector hiệu chỉnh $[0.56, 0.44]$, cộng nhẹ qua cổng `0.05` rồi chia Pytago:
$$\mathbf{t}_1' = \operatorname{L2Norm}\big([0.82, 0.58] + 0.05 \times [0.56, 0.44]\big) = \mathbf{[0.82,\; 0.57]}$$
* **Vì sao không bao giờ sập ở `BS=1`?** Vì phép tính trên không hề ngó sang bức ảnh nào khác đi chung lô!

---

## TRẠM 4: Đóng Gói Búp Bê Nga `Matryoshka` (`matryoshka.py`)

Từ Toàn cảnh $\mathbf{t}_0' = [0.71, 0.71]$ và Vùng Elip $\mathbf{t}_1' = [0.82, 0.57]$, ta xếp thành dãy **4 con số có thứ tự ưu tiên** (thực tế là `4096` số):
$$\mathbf{z}_{\text{raw}} = \big[\underbrace{0.77,\; 0.64}_{\text{Ngăn 1: Toàn cảnh}},\; \underbrace{0.82,\; 0.57}_{\text{Ngăn 2: Chi tiết vùng}}\big]$$

* **Chế độ Drone siêu nhẹ (cắt lấy 2 số đầu `z[:2]` rồi chia Pytago):** $\mathbf{z}^{(2\text{D})} = \mathbf{[0.77,\; 0.64]}$.
* **Chế độ Server đầy đủ (lấy đủ 4 số `z[:4]` rồi chia Pytago $\sqrt{2.0} = 1.414$):** $\mathbf{z}^{(4\text{D})} = \mathbf{[0.54,\; 0.45,\; 0.58,\; 0.40]}$.

---

## TRẠM 5: Tra Cứu Kho Bản Đồ `FAISS` (Ra Đáp Án Cuối Cùng!)

Trong ổ cứng Drone (`2-D`) lưu sẵn 2 địa điểm chụp từ 5 năm trước (lúc chưa có chiếc xe tải):
* **Địa điểm A (Đúng dãy nhà cổ đó):** $\mathbf{z}_A^{\text{db}} = [0.78,\; 0.63]$
* **Địa điểm B (Tòa cao ốc kính ở phố khác):** $\mathbf{z}_B^{\text{db}} = [0.10,\; 0.99]$

Lấy $\mathbf{z}^{(2\text{D})} = [0.77,\; 0.64]$ nhân cộng (Cosine) với từng địa điểm:
* **Điểm Địa điểm A:** $(0.77 \times 0.78) + (0.64 \times 0.63) = 0.6006 + 0.4032 = \mathbf{1.00}$ *(Khớp tuyệt đối!)*
* **Điểm Địa điểm B:** $(0.77 \times 0.10) + (0.64 \times 0.99) = 0.0770 + 0.6336 = \mathbf{0.71}$ *(Thấp hơn hẳn!)*

$\implies$ **Kết luận:** `1.00 > 0.71`, Robot báo chính xác vị trí **Địa điểm A**!

---

## TRẠM 6: Khi Đang Học Trên Kaggle Thì 4 Hàm Loss Chấm Phạt Thế Nào?

Giả sử lúc **mới bắt đầu học**, mạng chưa khôn nên đoán sai cổng bầu trời $s_2 = 0.80$ (lẽ ra phải `0.10`) và làm vector giống nhầm Địa điểm B tới `0.85` (trong khi Thầy `MegaLoc` bảo chỉ giống `0.20` thôi):
1. **`L_MRL-MS`:** Phạt vì điểm địa điểm sai `0.85` vượt quá ngưỡng `0.50` $\implies \mathcal{L}_{\text{MS}} = \frac{1}{2}\ln(1 + e^{2(0.85 - 0.50)}) = \mathbf{0.55}$.
2. **`L_2D` (Thầy `MegaLoc`):** Phạt độ lệch nhận định với Thầy: $4 \times (0.85 - 0.20)^2 = \mathbf{1.69}$ + ép hướng trực tiếp theo vector `8.448` chiều của Thầy.
3. **`L_3D` (Thầy `Depth-Anything`):** Phạt vì mở nhầm cổng bầu trời: $|0.80 - 0.10| = \mathbf{0.70}$.
4. **`L_proto`:** Phạt nếu 2 nguyên mẫu $\mathbf{m}_1, \mathbf{m}_2$ không vuông góc: $(\mathbf{m}_1 \cdot \mathbf{m}_2)^2 = \mathbf{0.00}$.

Tổng phạt $\mathcal{L}_{\text{total}} \implies$ Tính đạo hàm chỉnh lại trọng số mạng để bước sau tắt bầu trời, né xe tải và chấm đúng `1.00` như Trạm 1 → Trạm 5!
