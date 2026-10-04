# Hướng dẫn sử dụng — "Đọc tin, có nên mua?"

## 1. Xem kết quả (không cần cài gì)

Mở `outputs/dashboard/doc_tin_co_nen_mua.html` bằng trình duyệt (Chrome, Edge, Firefox). Dashboard gồm 6 phần:

1. **Bốn ô tóm tắt:** tỷ lệ lệnh có lãi và lợi nhuận cả kỳ của ba nhà đầu tư:
   - A: mua theo mọi tin tốt;
   - B: không đọc báo, giữ rổ BĐS;
   - C: chỉ làm theo khi mô hình khuyên.
2. **Giá chạy trước khi báo đưa tin:** đường lợi suất bất thường quanh ngày có tin. Nhà đầu tư chỉ giao dịch được từ đường nét đứt trở đi.
3. **100 triệu đồng sau 6 quý:** giá trị danh mục của A, B và hai chế độ của C.
4. **Tra từng mã:** chọn mã để xem giá và các tin.
   - Chấm đặc: mô hình khuyên mua.
   - Chấm rỗng: mô hình khuyên đứng ngoài.
   - Tam giác: tin xấu.
   - Bấm vào chấm hoặc dòng trong bảng để xem chi tiết: giá đã chạy bao nhiêu, điểm mô hình, lãi hay lỗ thật sau 5 phiên.
5. **Bối cảnh ngành (tham khảo):** tình hình chung của ngành ở các phiên cuối.
   - Số liệu mô tả: số tin ngành, sắc thái tin so với 20 phiên qua, tỷ lệ tin xấu, tin rủi ro pháp lý, rổ BĐS 20 phiên qua, tỷ lệ mã tăng.
   - Nhận định của mô hình phụ cho 20 phiên tới: "Thuận lợi / Trung tính / Kém thuận lợi". Độ tin cậy thấp (AUC 0,62, khoảng tin cậy chứa 0,5).
   - Chỉ để hiểu bối cảnh. **Không dùng để bỏ qua khuyến nghị từng mã**: kiểm tra cho thấy lọc theo ngành không làm tăng tỷ lệ đúng.
6. **Khuyến nghị cho tin mới nhất:** các tin trong những phiên cuối của dữ liệu. Chọn chế độ ở ô "Chế độ".

## 2. Ba chế độ khuyến nghị

| Chế độ | Khi nào mô hình lên tiếng | Nên dùng khi |
|---|---|---|
| **Tiêu chuẩn** | Tin nằm trong 10% điểm cao nhất → "MUA" (tin tốt) hoặc "BÁN nếu đang giữ" (tin xấu) | Muốn có khoảng 1 tín hiệu cho mỗi 8 tin |
| **Thận trọng** | Cả 3 mô hình thành phần cùng xếp tin vào 20% cao nhất | Muốn ít tín hiệu nhưng chắc hơn (62% đúng trên test) |
| **Tự tin hai phía** | 15% điểm cao nhất → làm theo; 15% thấp nhất → "ĐỪNG mua" / "ĐỪNG bán tháo"; phần giữa → "Không đủ tự tin" | Muốn biết cả tin nên theo và tin nên tránh (64–65% đúng) |

**Điểm (0–100)** là thứ hạng của tin so với các tin trong quý gần nhất, không phải xác suất lãi.

## 3. Chạy lại từ đầu với dữ liệu mới

1. Cài Python 3.12, rồi chạy `py -3.12 -m pip install -r requirements.txt`.
2. Đặt dữ liệu vào `data final/` theo đúng cấu trúc hiện có: `keyword/`, `law/`, `company/`, `Stock/`.
3. Chạy `py -3.12 -m src.run_all`. Lần đầu mất khoảng 35 phút vì phải mã hóa ngữ nghĩa bài báo. Nếu chỉ thêm giá, không thêm tin: `--skip-embed`.
4. Mở lại dashboard. Bảng khuyến nghị mới nhất nằm ở `outputs/reports/khuyen_nghi_moi_nhat.csv`.

## 4. Đọc kết quả cho đúng

- Mô hình chỉ đạt 60–65% **khi nó đủ tự tin**. Không nên áp dụng cho mọi tin.
- Kết quả được đo trên 6 quý 2025–2026, thị trường BĐS chủ yếu đi xuống. Chưa được kiểm chứng ở giai đoạn khác.
- **Đây là sản phẩm học thuật (môn Nhập môn Khoa học dữ liệu), không phải khuyến nghị đầu tư.**
