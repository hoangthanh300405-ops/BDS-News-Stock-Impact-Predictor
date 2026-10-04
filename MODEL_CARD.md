# Thẻ mô hình — FINAL "Đọc tin, có nên làm theo?"

| | |
|---|---|
| **Nhiệm vụ** | Với một tin tức về một cổ phiếu BĐS (mã, phiên), đánh giá: nếu nhà đầu tư nhỏ lẻ làm theo tin (tin tốt → mua, tin xấu → bán nếu đang giữ) ở giá đóng cửa phiên đọc tin và giữ 5 phiên, sau phí 0,4%, có lãi không |
| **Người dùng dự kiến** | Sinh viên, giảng viên, nhà đầu tư cá nhân muốn hiểu tác động của tin tức. **Không phải công cụ khuyến nghị đầu tư** |
| **Mô hình** | Trung bình phân vị của 3 mô hình: Logistic Regression (C = 0,1), Random Forest (500 cây, ≥ 50 mẫu/lá), LightGBM hồi quy Huber trên mức lãi. Trọng số thời gian 0,5^(tuổi/250 phiên) |
| **Đầu vào** (57 đặc trưng) | Nội dung tin (nhãn LLM qwen2.5:7b: hướng, mức độ, loại sự kiện); loại bài (bộ lọc NLP); độ mới và "uy tín" của tin (chỉ dùng kết quả đã biết); giá đã chạy theo hướng tin; bối cảnh thị trường và nhiệt độ tin; thanh khoản và rủi ro; hồ sơ doanh nghiệp |
| **Đầu ra** | Điểm 0–100 và 3 chế độ khuyến nghị: tiêu chuẩn (top 10%), thận trọng (cả 3 mô hình cùng top 20%), tự tin hai phía (top/bottom 15%, phần giữa không khuyến nghị) |
| **Dữ liệu** | 47.272 bài báo CafeF, Vietstock, Kenh14 (01/2024 – 09/2026); giá đóng cửa và khối lượng 83 mã; 11.380 lần (mã, phiên) có tin với hướng ≠ 0 |
| **Đánh giá** | Walk-forward 6 quý (test 2025Q2 → 2026Q3, 6.686 tin), purge + embargo; mô hình và chế độ chốt trước khi chạy; ngưỡng từ val; bootstrap theo khối 5 phiên |

## Hiệu năng trên tập test

| Chế độ | Tỷ lệ tin có khuyến nghị | Tỷ lệ đúng | Mốc |
|---|---|---|---|
| Tiêu chuẩn | 13% | 60,0% | luôn làm theo 45,6% |
| Thận trọng | 11% | 62,3% | luôn làm theo 45,6% |
| Tự tin hai phía 10% / 15% | 24% / 30% | 65,1% / 64,0% | không bao giờ làm theo 54,4% |
| Toàn bộ tin (AUC) | 100% | AUC 0,599 | 0,5 |

Mô phỏng danh mục (100 triệu, 10 vị thế, chế độ thận trọng): +59,4%. Mua theo mọi tin tốt: −23,6%; giữ rổ BĐS: −21,4%.

## Giới hạn và rủi ro

- **Một giai đoạn test** (6 quý, thị trường BĐS chủ yếu đi xuống). Hiệu năng dao động mạnh theo quý: 2025Q3 thất bại (AUC 0,48, precision 40%).
- **Chỉ đúng khi mô hình tự tin.** Trên toàn bộ tin, accuracy khoảng 57%, so với mốc 54,4%.
- **Giả định giao dịch:** khớp ở giá đóng cửa, chưa tính trượt giá (kịch bản phí 0,8% vẫn lãi), không bán khống.
- **Nhãn LLM không có nhãn người:**
  - Kiểm chứng gián tiếp cho thấy nhãn lệch dương (51–59% bài PR hoặc nhận định bị chấm tích cực) và qwen bỏ sót phần lớn tin xấu.
  - Mô hình dùng nhãn như một đặc trưng và được chấm bằng lợi suất thật.
- **Không dùng cho:** quyết định đầu tư thật, cổ phiếu ngoài 83 mã BĐS, dữ liệu sau 09/2026 mà không huấn luyện lại.

## Mô hình phụ: bối cảnh ngành

Dashboard có thêm ô "Bối cảnh ngành" (`src/sector/`). Ô này dùng Logistic Regression trên tin toàn ngành và xu hướng rổ BĐS để đánh giá 20 phiên tới (AUC test 0,62, khoảng tin cậy 95% là 0,40–0,80). Ô này **chỉ để tham khảo**: không tham gia vào khuyến nghị của mô hình FINAL và không được đánh giá như một sản phẩm.

## Tái lập

`py -3.12 -m pip install -r requirements.txt` rồi `py -3.12 -m src.run_all` (khoảng 20 phút nếu đã có vector ngữ nghĩa, khoảng 35 phút nếu chưa). 15 unit test kiểm tra việc không rò rỉ tương lai: `py -3.12 -m pytest -q`.
