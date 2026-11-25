# `jm-online` — Streaming Inference & Label-Free Penalty Calibration for Statistical Jump Models

Một thư viện Python nhỏ gọn (~450 dòng mã nguồn cốt lõi, **chỉ phụ thuộc `numpy`**) bổ sung trực tiếp hai mảnh ghép còn khuyết cho hệ sinh thái [`Yizhan-Oliver-Shu/jump-models`](https://github.com/Yizhan-Oliver-Shu/jump-models):

1. **Giải mã streaming từng bước thời gian thực (`step(x_t)`) & So sánh `Carried-Score` vs. `Rolling-W`:**
   - Trong `jumpmodels`, hàm `predict_online(X)` tích lũy chi phí quy hoạch động từ $t = 0$ đến vô cùng (`carried`), trong khi nhiều bài báo thực nghiệm lại trượt một cửa sổ cố định $W$ quan sát rồi lấy nhãn cuối (`rolling-W`).
   - `OnlineJumpDecoder` hỗ trợ cả hai chế độ ở dạng streaming từng điểm dữ liệu với bộ nhớ $O(K)$, tương thích 100% với cả `JumpModel` lẫn `SparseJumpModel` của Shu, và không bắt cài `matplotlib` hay `scikit-learn` (giải quyết trực tiếp Issue #5 của `jump-models` trên Python 3.13+).
2. **Chọn mức phạt `jump_penalty` ($\lambda$) theo Ngân sách Độ trễ — Không cần nhãn thật, không cần backtest lợi nhuận:**
   - Thay vì chọn mò `jump_penalty = 50.0` hoặc tối ưu hóa lợi nhuận lịch sử dễ bị overfit, `jm-online` chuẩn hóa $\lambda = \eta \cdot s_{\text{loss}}$ (theo trung vị chênh lệch mất mát trên tập huấn luyện) và tính **độ trễ phản ứng bậc thang xấu nhất** $d_{\max}(\lambda)$ (mở rộng trực tiếp định lý trong [`../core-theory-reference/Delay.lean`](../core-theory-reference/Delay.lean) sang không gian đặc trưng thực).
   - Người vận hành chỉ cần đặt ngân sách độ trễ vật lý chấp nhận được (ví dụ: $\le 5$ phút cho điều hòa tòa nhà, $\le 4$ giờ cho cảnh báo bụi mịn, $\le 3$ ngày cho cảnh báo rủi ro biến động), hàm `select_penalty_by_delay_budget()` sẽ tự động chọn $\lambda$ giảm tối đa dao động nhấp nháy giả.

---

## Cấu trúc thư mục (đọc hết toàn bộ code trong 15 phút)

```text
jumpmodel-online/
  pyproject.toml                 ← Cấu hình gói chuẩn (chỉ cần numpy>=1.24)
  src/jm_online/
    decoder.py                   ← Bộ giải mã streaming/rolling/offline & bộ fit JM thuần NumPy
    diagnostics.py               ← Độ trễ bậc thang, chọn λ theo độ trễ, so sánh Carried vs. Rolling-W
    plotting.py                  ← Vẽ đường cong Pareto Độ trễ – Tần suất nhảy (nạp matplotlib khi gọi)
  examples/
    run_real_benchmarks.py       ← Chạy thực nghiệm trên 3 bộ dữ liệu cảm biến & rủi ro công khai
  tests/
    test_jm_online.py            ← Kiểm thử khớp 100% với công thức của Shu & các định lý/phản ví dụ Lean
  data/                          ← Dữ liệu công khai (UCI Occupancy CC-BY-4.0, UCI Beijing PM2.5 CC-BY-4.0, FRED VIX)
  outputs/                       ← Bảng tổng hợp CSV/JSON và 6 biểu đồ PNG
```

---

## Cách dùng nhanh (Quickstart)

```python
from jm_online import (
    OnlineJumpDecoder,
    compare_carried_vs_rolling,
    evaluate_penalty_curve,
    fit_jump_model,
    select_penalty_by_delay_budget,
)

# 1. Fit tâm cụm trên tập huấn luyện (hoặc bọc trực tiếp model đã fit từ `jumpmodels` bằng `OnlineJumpDecoder.from_jumpmodel(jm)`)
fit = fit_jump_model(X_train, n_components=2, jump_penalty=2.0, random_state=42)

# 2. Chọn mức phạt lambda tự động theo ngân sách độ trễ <= 4 bước (không cần nhãn y!)
curve = evaluate_penalty_curve(X_train, fit.centers_)
chosen = select_penalty_by_delay_budget(curve, max_delay_steps=4)

# 3. Giải mã thời gian thực từng quan sát mới (O(K^2) thời gian, O(K) bộ nhớ)
decoder = OnlineJumpDecoder(fit.centers_, jump_penalty=chosen.jump_penalty)
for x_t in X_live_stream:
    state_t = decoder.step(x_t, mode="carried")

# 4. Kiểm tra xem cửa sổ trượt độ dài W có bị nhảy giả so với predict_online không
rows = compare_carried_vs_rolling(X_live_stream, fit.centers_, chosen.jump_penalty, windows=[7, 15, 63, 168])
```

---

## Kết quả thực nghiệm trên 3 bài toán giám sát thực tế (Out-of-Sample)

Chạy lại toàn bộ bằng:
```powershell
python -m pytest -v
python examples\run_real_benchmarks.py
```

### 1. Hiệu quả của bộ chọn $\lambda$ không nhãn (Label-Free Delay-Budget Calibration)

| Bài toán thực tế | Tập Train | Tập Test (chưa từng xem) | Ngân sách độ trễ đặt trước | $\eta$ được chọn ($\lambda$) | Số lần nhảy khi $\lambda=0$ | Số lần nhảy với $\lambda$ đã chọn | Giảm nhấp nháy giả | Ý nghĩa vật lý trên tập Test |
|---|---:|---:|---|---:|---:|---:|---:|---|
| **Cảm biến phòng / HVAC** *(UCI Occupancy, 1 phút/mẫu)* | 8,143 | 12,417 | $\le 5$ phút | `4.0` (`20.38`) | 54 *(6.26/ngày)* | **14** *(1.62/ngày)* | **−74.1%** | Độ chính xác không giám sát so với camera giữ kín đạt **83.92%**, độ trễ thực tế trung vị khi có người bước vào phòng là **1.0 phút**. |
| **Cảnh báo ô nhiễm bụi mịn** *(UCI Beijing PM2.5, 1 giờ/mẫu)* | 26,275 | 17,520 | $\le 4$ giờ | `4.0` (`11.14`) | 831 *(34.65/tháng)* | **281** *(11.72/tháng)* | **−66.2%** | Tách rõ chế độ không khí sạch (trung vị **28 $\mu\text{g/m}^3$**) và đợt mù khói độc hại (trung vị **112 $\mu\text{g/m}^3$**). |
| **Giám sát rủi ro biến động** *(FRED CBOE VIX, 1 ngày/mẫu)* | 4,265 *(1990–2006)* | 5,038 *(2007–2026)* | $\le 3$ phiên | `3.0` (`4.65`) | 259 *(12.96/năm)* | **65** *(3.25/năm)* | **−74.9%** | Tách chế độ thị trường êm (trung vị VIX **13.5**) và chế độ căng thẳng (trung vị VIX **19.1**) qua 20 năm ngoài mẫu. |

### 2. Xác nhận thực nghiệm hiện tượng Toán học: `Carried-Score` (`predict_online`) vs. `Rolling-W`

Đúng như dự đoán từ lý thuyết điểm tích lũy bị chặn (`Model.lean` & phản ví dụ cửa sổ trượt):
- **Khi cửa sổ $W$ ngắn ($W = 7$):** Do mất trí nhớ mỗi khi khởi tạo lại đầu cửa sổ, `Rolling-7` tự sinh thêm **+14 cú nhảy thừa** trên cảm biến tòa nhà (`24` vs `10`), **+190 cú nhảy cảnh báo thừa** trên cảm biến bụi mịn Bắc Kinh (`471` vs `281`), và **+46 cú nhảy thừa** trên VIX (`111` vs `65`).
- **Khi cửa sổ $W$ vượt ngưỡng bão hòa (*coalescence*):** Đến $W = 127$ phút (Occupancy), $W = 168$ giờ (PM2.5) và $W = 63$ ngày (VIX), xác suất bất đồng giữa `Rolling-W` và `Carried-Score` giảm về **đúng 0.00%** — chứng minh rằng bộ giải mã streaming `Carried-Score` với bộ nhớ $O(1)$ cho kết quả giống hệt cửa sổ trượt dài mà không tốn chi phí tính lại lịch sử và không bao giờ dính bẫy nhảy giả của cửa sổ ngắn.

---

## Hướng phát triển & Công bố tiếp theo

1. **Đóng góp mã nguồn mở / Bài báo phần mềm (JOSS):** Đưa thư mục `jumpmodel-online` lên một repo GitHub công khai độc lập (có GitHub Actions chạy `pytest`, ví dụ notebook tương tác với `jumpmodels`), hoặc mở Issue/PR trao đổi trực tiếp với tác giả `Yizhan-Oliver-Shu/jump-models`. Sau khi duy trì mở đủ chuẩn lịch sử phát triển, có thể nộp lên *Journal of Open Source Software (JOSS)*.
2. **Bài báo ngắn Toán ứng dụng & Thống kê (SIURO / arXiv):** Kết hợp 3 định lý cốt lõi trong [`../core-theory-reference/`](../core-theory-reference/) với công thức độ trễ bậc thang tổng quát và thực nghiệm cảm biến ở trên thành một bài báo 8 trang cực gọn — không phụ thuộc vào dữ liệu tài chính trả phí và hoàn toàn tái lập 100%.
