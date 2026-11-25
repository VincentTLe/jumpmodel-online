# Giải thích Đơn giản Nhất & Toàn bộ Chứng minh trên 1 Trang Giấy

Không cần 23 file Lean hay hàng trăm trang ký hiệu. Toàn bộ nghiên cứu này thực chất chỉ xoay quanh **một hình ảnh vật lý duy nhất** và **3 chứng minh đại số sơ cấp (mỗi chứng minh đúng 5 dòng)** mà bất kỳ sinh viên năm 1–2 nào cũng tự kiểm tra bằng giấy bút được trong 10 phút.

---

## Phần 1: Nghiên cứu của chúng ta đang làm gì? (Giải thích trong 2 phút)

Hãy tưởng tượng bạn lắp cảm biến để tự động bật/tắt điều hòa phòng học, phát cảnh báo ô nhiễm bụi mịn cho thành phố, hoặc bật đèn cảnh báo rủi ro biến động thị trường:
- Hệ thống có 2 trạng thái: **`0` (Bình thường)** và **`1` (Cảnh báo / Kích hoạt)**.
- Nếu cứ thấy 1 phút cảm biến vọt lên là bật ngay, 1 phút sau dịu xuống lại tắt ngay ($\lambda = 0$), hệ thống sẽ **nhấp nháy liên tục** (hỏng máy nén điều hòa, làm người nhận phớt lờ cảnh báo).
- Để chống nhấp nháy, **Statistical Jump Model (SJM)** đặt một mức phạt $\lambda > 0$ cho mỗi lần đổi trạng thái.

### Hình ảnh cốt lõi: "Cây kim trong hộp $[-\lambda, +\lambda]$"

Mỗi khi có quan sát mới ở bước $t$, ta tính được mức bằng chứng $g_t = \ell_t(0) - \ell_t(1)$:
- $g_t > 0$: dữ liệu hôm nay nghiêng về trạng thái `1`.
- $g_t < 0$: dữ liệu hôm nay nghiêng về trạng thái `0`.

**Định lý 1** bên dưới chứng minh rằng toàn bộ thuật toán Quy hoạch động của SJM thực chất chỉ làm đúng một việc như một **cây kim trượt giữa hai bức tường chặn $-\lambda$ và $+\lambda$**:
- Bước mới: đẩy cây kim đi một đoạn $g_t$. Nếu chạm tường phải $+\lambda$ thì dừng ở $+\lambda$; nếu chạm tường trái $-\lambda$ thì dừng ở $-\lambda$.
- Nhìn vạch $0$ ở chính giữa hộp để phát tín hiệu: **kim $> 0$ thì báo `1`, kim $\le 0$ thì báo `0`**.

Từ đúng hình ảnh cái hộp $[-\lambda, +\lambda]$ này, **2 vấn đề lớn chưa được giải quyết trong thư viện Python `jumpmodels` hiện nay** hiện ra ngay lập tức:

1. **Vấn đề 1 — Chạy thực tế nên giữ kim liên tục (`Carried`) hay trượt cửa sổ $W$ bước (`Rolling-W`)?**
   - Trong code `predict_online` của thư viện `jumpmodels`, người ta giữ cây kim chạy tiếp từ ngày đầu tiên mãi mãi (**`Carried`**). Nhưng trong bài báo gốc (Shu et al., 2024), người ta lại cắt cửa sổ $W$ quan sát gần nhất, **nhấc cây kim đặt lại về vạch $0$ ở đầu cửa sổ** rồi tua tới cuối (**`Rolling-W`**).
   - **Định lý 2** chỉ ra: nếu cửa sổ $W$ đủ dài để trong đó cây kim kịp chạm tường $-\lambda$ hoặc $+\lambda$ một lần, cả hai cách **chập làm một 100%** (không cần tốn công tính lại cửa sổ!). Ngược lại, nếu $W$ ngắn mà đặt tường $\lambda$ rộng, việc mỗi bước lại nhấc kim về vạch $0$ làm cây kim lắc qua lắc lại quanh vạch $0$, khiến **tăng $\lambda$ lại làm cảnh báo nhấp nháy gấp đôi!**
2. **Vấn đề 2 — Chọn độ rộng bức tường $\lambda$ bằng bao nhiêu mà không cần đoán mò?**
   - Trong thư viện `jumpmodels`, tác giả gán mò `jump_penalty = 50.0` hoặc khuyên dùng lợi nhuận quá khứ để dò (rất dễ overfit).
   - Nhìn vào cái hộp: khi hệ thống đang yên ổn ở `0`, cây kim nằm sát tường trái $-\lambda$. Khi sự cố thật ập đến (mỗi bước đẩy kim sang phải một lực chuẩn $\Delta$), cây kim cần đi hết quãng đường $\lambda$ để vượt qua vạch $0$.
   - **Định lý 3** tính ra chính xác số bước trễ là $\lfloor \lambda / \Delta \rfloor + 1$. Vậy người vận hành chỉ cần nói **"Tôi chấp nhận trễ tối đa $B$ bước (ví dụ 5 phút hoặc 4 giờ)"**, công cụ của ta tìm ra ngay trần $\lambda < B\Delta$ để cắt giảm 66%–75% nhấp nháy giả mà không bao giờ cần nhãn thật!

---

## Phần 2: Ba Định lý và Chứng minh Đầy đủ cho Con người Đọc

Xét hai tâm cụm cố định $\mu_0, \mu_1 \in \mathbb{R}^p$ và mức phạt nhảy $\lambda > 0$. Với mỗi quan sát $x_t \in \mathbb{R}^p$, mất mát tại trạng thái $k \in \{0, 1\}$ là $\ell_t(k) = \frac{1}{2}\|x_t - \mu_k\|_2^2$, và chênh lệch mất mát là $g_t = \ell_t(0) - \ell_t(1)$. Tổng chi phí của một dãy trạng thái $s_{1:t} \in \{0,1\}^t$ là:
$$J_\lambda(s_{1:t}) = \sum_{i=1}^{t} \ell_i(s_i) + \lambda \sum_{i=2}^{t} \mathbf{1}\{s_i \ne s_{i-1}\}.$$
Gọi $V_t(k) = \min_{s_{1:t}:\, s_t = k} J_\lambda(s_{1:t})$ là chi phí nhỏ nhất kết thúc ở trạng thái $k$. Quy ước phát nhãn online tại thời điểm $t$ là $\hat{s}_t = 1$ nếu $V_t(1) < V_t(0)$, và $\hat{s}_t = 0$ nếu $V_t(0) \le V_t(1)$ (hòa chọn $0$, khớp chính xác hàm `argmin` của NumPy và `jumpmodels`).

---

### Định lý 1 (Đưa Quy hoạch động về Điểm tích lũy bị chặn 1 chiều)
Đặt hàm chặn $\operatorname{clip}_\lambda(z) = \max(-\lambda, \min(\lambda, z))$ và định nghĩa dãy điểm 1 chiều:
$$y_0 = 0, \qquad y_t = \operatorname{clip}_\lambda(y_{t-1} + g_t) \quad (t \ge 1).$$
Khi đó với mọi $t \ge 1$, hiệu chi phí tối ưu $D_t = V_t(0) - V_t(1)$ thỏa mãn $y_t = \operatorname{clip}_\lambda(D_t)$, và nhãn phát ra là $\hat{s}_t = \mathbf{1}\{y_t > 0\}$.

**Chứng minh (5 dòng):**
1. Tại $t = 1$, không có chi phí chuyển trạng thái nên $V_1(k) = \ell_1(k) \implies D_1 = g_1$, do đó $y_1 = \operatorname{clip}_\lambda(g_1) = \operatorname{clip}_\lambda(D_1)$.
2. Với $t \ge 2$, công thức truy hồi quy hoạch động cho $V_t(0) = \ell_t(0) + \min\{V_{t-1}(0),\, V_{t-1}(1) + \lambda\}$ và $V_t(1) = \ell_t(1) + \min\{V_{t-1}(1),\, V_{t-1}(0) + \lambda\}$.
3. Trừ hai vế và rút $V_{t-1}(1)$ ra khỏi cả hai phép $\min$, ta được:
   $$D_t = g_t + \min(D_{t-1}, \lambda) - \min(0, D_{t-1} + \lambda).$$
4. Xét trực tiếp 3 miền giá trị của $D_{t-1}$ ($< -\lambda$, thuộc $[-\lambda, \lambda]$, và $> \lambda$), biểu thức $\min(D_{t-1}, \lambda) - \min(0, D_{t-1} + \lambda)$ luôn bằng đúng $\operatorname{clip}_\lambda(D_{t-1})$.
5. Theo quy nạp, $\operatorname{clip}_\lambda(D_{t-1}) = y_{t-1}$, suy ra $D_t = g_t + y_{t-1}$ và $\operatorname{clip}_\lambda(D_t) = \operatorname{clip}_\lambda(y_{t-1} + g_t) = y_t$. Vì $\lambda > 0$, phép chặn ở $\pm\lambda$ giữ nguyên dấu chặt nên $V_t(1) < V_t(0) \iff D_t > 0 \iff y_t > 0$. $\blacksquare$

---

### Định lý 2 (Hội tụ bộ nhớ giữa `Carried` và `Rolling-W`, và Phản ví dụ nhảy giả)
Tại thời điểm $t \ge W$, so sánh điểm tích lũy từ đầu $y_t^{\text{car}}$ với điểm cửa sổ trượt $y_t^{(W)}$ (khởi tạo lại bằng $0$ trước bước $t - W + 1$ rồi chạy cùng công thức trên $g_{t-W+1}, \dots, g_t$):

1. **(Tính co & Chập quỹ đạo — Coalescence):** Gọi $u_j, v_j$ lần lượt là quỹ đạo của `Carried` và `Rolling-W` tại các bước $j \in \{t-W, \dots, t\}$ (với $u_{t-W} = y_{t-W}^{\text{car}} \in [-\lambda, \lambda]$ và $v_{t-W} = 0$). Khoảng cách $|u_j - v_j|$ không bao giờ tăng và luôn $\le \lambda$. Hơn nữa, nếu $u_{t-W} \ge 0$ và tồn tại một bước $j$ trong cửa sổ mà $v_{j-1} + g_j \ge \lambda$ hoặc $u_{j-1} + g_j \le -\lambda$ (tương tự cho $u_{t-W} \le 0$), thì $u_j = v_j$ và suy ra **$y_t^{\text{car}} = y_t^{(W)}$ tuyệt đối** (hai cơ chế phát cùng một nhãn).
2. **(Phản ví dụ cửa sổ trượt ngắn):** Tồn tại chuỗi tuần hoàn độ dài 8 với $g_t \in \{-1, +1\}$ và cửa sổ $W = 7$ sao cho khi tăng phạt từ $\lambda = 1.5$ lên $\lambda = 2.5$, số lần nhảy bên trong mỗi cửa sổ giảm về $0$, nhưng tỷ lệ đổi quyết định của `Rolling-7` **tăng gấp đôi từ $2/8$ (25%) lên $4/8$ (50%)**, trong khi `Carried` chỉ đổi $2/8$ (25%).

**Chứng minh:**
- *Ý 1 (4 dòng):* Với mọi hằng số $g$, hàm $f(x) = \operatorname{clip}_\lambda(x + g)$ là hàm tăng và co Lipschitz bậc 1: $|f(a) - f(b)| \le |a - b|$. Tại đầu cửa sổ $j_0 = t - W$, vì $u_{j_0} \in [-\lambda, \lambda]$ và $v_{j_0} = 0$ nên $|u_{j_0} - v_{j_0}| \le \lambda$. Giả sử $u_{j_0} \ge v_{j_0} = 0$ (trường hợp $u_{j_0} \le 0$ đối xứng hoàn toàn); do tính tăng nên $u_{j-1} \ge v_{j-1}$ với mọi $j > j_0$. Ngay khi $v_{j-1} + g_j \ge \lambda$ thì $u_{j-1} + g_j \ge v_{j-1} + g_j \ge \lambda \implies u_j = v_j = +\lambda$; còn nếu $u_{j-1} + g_j \le -\lambda$ thì $v_{j-1} + g_j \le u_{j-1} + g_j \le -\lambda \implies u_j = v_j = -\lambda$. Khi đã bằng nhau tại bước $j$, mọi bước sau tới $t$ đều nhận chung đầu vào nên $y_t^{\text{car}} = y_t^{(W)}$.
- *Ý 2 (Kiểm tra tay trực tiếp):* Lặp lại chu kỳ 8 bước `[-1, -1, -1, +1, -1, +1, +1, +1]` với $W = 7$:
  - Ở $\lambda = 2.5$: vì tổng đại số trên 7 bước bất kỳ có độ lớn tối đa $\le 3$ nhưng mọi đường đổi trạng thái bên trong cửa sổ tốn ít nhất $\lambda = 2.5$ phí phạt nên nghiệm tối ưu trên từng cửa sổ đều phẳng (toàn `0` hoặc toàn `1`), tức nhãn cuối chính là dấu của tổng 7 phần tử. Tính tổng 7 phần tử trượt qua 8 pha của chu kỳ cho lần lượt `[-1, +1, +1, +1, -1, +1, -1, -1]` $\implies$ dãy nhãn `0, 1, 1, 1, 0, 1, 0, 0`, có **4 lần đổi nhãn** trên mỗi chu kỳ 8 bước (50%).
  - Ở $\lambda = 1.5$: tính trực tiếp $y_7$ theo Định lý 1 qua 8 pha cho dãy nhãn `0, 0, 1, 1, 1, 0, 0, 0`, chỉ có **2 lần đổi nhãn** (25%). Với `Carried` ở $\lambda = 2.5$, điểm tích lũy giữ nguyên qua biên chu kỳ chỉ đổi dấu đúng 1 lần lên và 1 lần xuống (**2 lần đổi / 8 bước**). $\blacksquare$

---

### Định lý 3 (Độ trễ Phản ứng Bậc thang & Ngưỡng Mức phạt theo Ngân sách Độ trễ)
Gọi $\Delta = \frac{1}{2}\|\mu_1 - \mu_0\|_2^2 > 0$ là khoảng cách mất mát một bước giữa hai tâm cụm.
1. Giả sử bộ giải mã `Carried` đang ổn định ở trạng thái `0` ($y_{t_0} = -\lambda$) và từ bước $t_0 + 1$ trở đi nhận chuỗi tín hiệu sạch ở tâm trạng thái `1` ($x_t = \mu_1$, tức $g_t = +\Delta$). Số bước cần thiết để bộ giải mã chuyển sang phát nhãn `1` là:
   $$d_{0 \to 1}(\lambda) = \left\lfloor \frac{\lambda}{\Delta} \right\rfloor + 1.$$
2. Ngược lại, nếu đang ổn định ở trạng thái `1` ($y_{t_0} = +\lambda$) và nhận $x_t = \mu_0$ ($g_t = -\Delta$), số bước cần thiết để quay về phát nhãn `0` là:
   $$d_{1 \to 0}(\lambda) = \max\!\left(1,\, \left\lceil \frac{\lambda}{\Delta} \right\rceil\right).$$
3. Với mọi ngân sách độ trễ nguyên dương $B \ge 1$, điều kiện để độ trễ xấu nhất $\max\{d_{0 \to 1}(\lambda),\, d_{1 \to 0}(\lambda)\} \le B$ tương đương chính xác với:
   $$\lambda < B \cdot \Delta.$$

**Chứng minh (5 dòng):**
1. Xuất phát từ $y_{t_0} = -\lambda$, mỗi bước nhận $g_t = +\Delta > 0$ làm điểm tăng thêm $\Delta$ cho tới khi chạm trần $+\lambda$; cụ thể sau $k \ge 1$ bước, $y_{t_0+k} = \min(\lambda,\, -\lambda + k\Delta)$.
2. Vì $\lambda > 0$, điều kiện phát nhãn `1` ($y_{t_0+k} > 0$) xảy ra khi và chỉ khi $-\lambda + k\Delta > 0 \iff k > \lambda / \Delta$. Số nguyên dương nhỏ nhất lớn hơn thực sự $\lambda / \Delta$ chính là $\lfloor \lambda / \Delta \rfloor + 1$.
3. Đối xứng, xuất phát từ $y_{t_0} = +\lambda$ và nhận $g_t = -\Delta$, sau $k \ge 1$ bước điểm là $y_{t_0+k} = \max(-\lambda,\, \lambda - k\Delta)$. Bộ giải mã phát nhãn `0` ngay khi $y_{t_0+k} \le 0 \iff \lambda - k\Delta \le 0 \iff k \ge \lambda / \Delta$, có nghiệm nguyên $k \ge 1$ nhỏ nhất là $\max(1, \lceil \lambda / \Delta \rceil)$.
4. Với mọi số thực dương $r = \lambda / \Delta > 0$, ta luôn có $\lceil r \rceil \le \lfloor r \rfloor + 1$, nên độ trễ xấu nhất luôn là $d_{0 \to 1}(\lambda) = \lfloor \lambda / \Delta \rfloor + 1$.
5. Do đó $\max(d_{0 \to 1}, d_{1 \to 0}) \le B \iff \lfloor \lambda / \Delta \rfloor + 1 \le B \iff \lfloor \lambda / \Delta \rfloor \le B - 1 \iff \lambda / \Delta < B \iff \lambda < B\Delta$. $\blacksquare$
