# Khung báo cáo — Tin tức bất động sản và nhà đầu tư nhỏ lẻ

Tên đề xuất: **"Đọc tin, có nên mua? Phân tích và dự báo tác động của tin tức lên cổ phiếu bất động sản Việt Nam"**

Các kết quả được sinh sau khi chạy các module tương ứng trong `src/`
và được lưu tại `outputs/reports/`.

Câu chuyện xuyên suốt, 4 bước:

1. Tin tức BĐS được thị trường hấp thụ rất nhanh.
2. Khi nhà đầu tư đọc được thì giá đã chạy.
3. Làm theo báo một cách máy móc thì thua cả tung đồng xu.
4. Nhưng lọc đúng tin (theo bối cảnh thị trường và mức giá đã chạy) thì có lợi thế đo được.

---

## Chương 1. Giới thiệu (2–3 trang)

- **Bối cảnh:**
  - Nhà đầu tư cá nhân chiếm phần lớn giao dịch trên HOSE/HNX.
  - Nguồn thông tin chính của họ là báo tài chính.
  - Cổ phiếu BĐS nhạy với tin pháp lý, tín dụng và dự án.
- **Câu hỏi nghiên cứu:**
  - **RQ-A (phân tích):** tin tức phản ánh vào giá khi nào, và loại tin nào còn tác động sau khi đọc?
  - **RQ-B (dự báo):** sau khi đọc một tin, nhà đầu tư có nên làm theo không, và mô hình có giúp chọn tin tốt hơn làm theo mọi tin không?
- **Đóng góp:**
  1. Bộ dữ liệu 47 nghìn bài có nhãn tác động theo mã.
  2. Quy trình NLP chưng cất nhãn LLM.
  3. Bằng chứng "báo trễ so với giá" trên thị trường Việt Nam.
  4. Mô hình lọc tin cho nhà đầu tư nhỏ lẻ, đánh giá ngoài mẫu và mô phỏng danh mục.

## Chương 2. Tổng quan tài liệu (3–4 trang)

- **Tin tức và giá:** Tetlock (2007, 2011), Chan (2003), Boudoukh et al. (2019), Ke–Kelly–Xiu (2019), Lopez-Lira & Tang (2023).
- **Sự chú ý và nhà đầu tư nhỏ lẻ:** Barber & Odean (2008), Da–Engelberg–Gao (2011).
- **Thị trường Việt Nam:** Vu et al. (2023).
- **Phương pháp event study:** Brown & Warner (1985), MacKinlay (1997), Kolari & Pynnönen (2010).
- **LLM gán nhãn và hiệu chỉnh:** Gilardi et al. (2023), Egami et al. (2023).

## Chương 3. Dữ liệu (4–5 trang)

- **Nguồn:**
  - Tin: CafeF 71%, Vietstock 24%, Kenh14 5%; 01/2024 – 09/2026.
  - Giá: 83 mã, giá đóng cửa và khối lượng.
  - Hồ sơ doanh nghiệp.
  - Dùng hình `p0_*.png` và bảng `p0_data_contract.md`.
- **Tiền xử lý:**
  - Khử trùng lặp theo URL chuẩn hóa.
  - Múi giờ Việt Nam.
  - **Chốt phiên 14:45:** lệnh ATC; kiểm chứng khớp bảng trust score 99,91%.
  - Cờ mã kém thanh khoản.
  - Benchmark đồng trọng số (lý do: VIC tăng 10 lần).
- **Nhãn LLM (qwen2.5:7b):**
  - Mô tả pipeline và prompt gán nhãn: `src/label/classify_all.py`.
  - Phân bố nhãn có xu hướng lệch dương; cần trình bày kết quả kiểm chứng nhãn và các hạn chế liên quan.
- **Hạn chế dữ liệu:**
  - Chỉ có giá đóng cửa, không có VN-Index.
  - Một LLM gán nhãn.

## Chương 4. Phương pháp (6–8 trang)

4.1. **Tầng NLP:**
- **Kiểm chứng nhãn nhiều chiều** (thay tập vàng, `src.nlp.label_validation`):
  - V1: qwen tự nhất quán giữa hai luồng gán nhãn.
  - V2: qwen so với GPT-4.1-mini (3.434 bài).
  - V3: trọng tài thứ ba gán mù 150 bài chọn phân tầng, ghi xác suất chọn π.
  - V4: soi bằng từ khóa tiêu đề.
  - V5–V6: phản ứng giá.
- Chưng cất nhãn LLM bằng tf-idf + Logistic Regression.
- Bộ lọc mức liên quan bằng giám sát yếu.

4.2. **Phân tích sự kiện:**
- AR = lợi suất − rổ BĐS − xu hướng riêng của mã.
- t-test theo phiên.
- Hiệu chỉnh kiểm định bội Benjamini–Hochberg.
- **Nêu rõ vì sao phải trừ xu hướng riêng:** kết quả thay đổi khi trừ (T29).

4.3. **Bài toán nhà đầu tư nhỏ lẻ:**
- Đơn vị quan sát, thời điểm giao dịch, hai target, xử lý trần/sàn, không bán khống.
- Quy trình được triển khai trong các module `src/retail/` và `src/features/`.

4.4. **Đặc trưng:** 7 nhóm đặc trưng, tuân thủ nguyên tắc chỉ sử dụng thông tin đã có đến thời điểm dự báo.

4.5. **Mô hình:**
- Vòng 1: LightGBM, Logistic, quy tắc kinh nghiệm.
- So sánh 10 thuật toán.
- Vòng 3: E3 = kết hợp Logistic, Random Forest và hồi quy xếp hạng; quy tắc đồng thuận.

4.6. **Đánh giá:**
- Walk-forward 6 quý, purge và embargo.
- Ngưỡng lấy từ val.
- **Cấu hình chính thức chốt trước.**
- Bootstrap theo khối; số quý thắng mốc.
- **Lý do không dùng accuracy làm chỉ số chính:** mốc "không bao giờ làm theo" đã đạt 54,4%.

4.7. **Mô phỏng danh mục:** 100 triệu, 10 vị thế, phí 0,4%.

## Chương 5. Kết quả phân tích (5–6 trang)

5.1. **Giá chạy trước tin:**
- Hình `r8_car_path.png` và `r9_lag.png`.
- 10 phiên trước tin +0,74%; 5 phiên sau khi giao dịch được +0,01%.

5.2. **Loại tin (bảng `retail_A2_by_type.csv`):**
- Tin thị trường tốt đảo chiều −0,8%/20 phiên (t = −3,0).
- Tin xấu về nợ/trái phiếu tiếp tục giảm.
- Sau hiệu chỉnh BH, không loại nào có ý nghĩa.

5.3. **Tin chính sách và sự chú ý:** khối lượng tăng quanh tin; tin xấu → khối lượng giảm sau đó.

5.4. **Kết quả rỗng có giá trị:**
- Tin không cải thiện dự báo chiều giá phiên sau.
- Mô hình chỉ dùng giá và khối lượng đạt 61–62% khi chọn lọc.

## Chương 6. Kết quả dự báo (6–8 trang)

6.1. **NLP:**
- Chưng cất nhãn: 75%.
- Kiểm chứng nhãn:
  - κ loại sự kiện 0,61 (hai LLM), 0,74 / 0,60 (so với trọng tài).
  - Ngược chiều hiếm (2,6–4,1%).
  - Lệch dương 51–59%; qwen bỏ sót 88% tin xấu.

6.2. **Nhà đầu tư nhỏ lẻ:**
- Bảng đầy đủ vòng 1 được sinh bởi `src.retail.full_table` và lưu trong `outputs/reports/`.
- So sánh 10 thuật toán để đánh giá mức độ ảnh hưởng của lựa chọn mô hình tới hiệu năng.

6.3. **Mô hình cuối cùng (`final_retail.csv`), kết quả chính của báo cáo:**
- AUC 0,599. Tiêu chuẩn: 60,0% đúng ở 13% số tin. Thận trọng: 62,3% ở 11%.
- **Tự tin hai phía: 65,1% ở 24% số tin** (target thắng rổ: 61,4% ở 18%, ≥ 55% ở cả 6/6 quý).
- Luôn ghi kèm độ phủ và mốc so sánh (45,6% / 54,4%).
- Các vòng 1–4 và so sánh 10 thuật toán đưa vào phụ lục B: cho thấy quá trình và mức trần AUC 0,55–0,60.

6.4. **Mô hình dựa vào gì (A3, `r10_importance_groups.png`):**
- Bối cảnh thị trường chiếm 50% tín hiệu.
- Giá đã chạy chiếm 43% với target thắng rổ.

6.5. **Mô phỏng danh mục và độ vững:**
- `r11_portfolio.png`, `retail_portfolio.csv`, `r12_robustness.png`.
- C hơn A và B ở 11/11 kịch bản.
- Khi chỉ giữ 1 phiên, C gần như hòa vốn.

## Chương 7. Thảo luận (2–3 trang)

- **Ý nghĩa cho nhà đầu tư nhỏ lẻ:**
  - Đừng đuổi theo tin bình luận thị trường.
  - Tin xấu đáng nghe hơn tin tốt.
  - Bối cảnh thị trường quan trọng hơn nội dung từng tin.
- **Vì sao tin tức không giúp dự báo chiều giá nhưng giúp chọn tin:** hai câu hỏi khác nhau.
- **Giới hạn:**
  - Một giai đoạn test, thị trường chủ yếu đi xuống.
  - Không có nhãn người; trọng tài là một LLM (mẫu 150 bài).
  - Giả định khớp giá đóng cửa, chưa tính trượt giá.
  - Không phải khuyến nghị đầu tư.
- **Hướng mở:**
  - Mở rộng giai đoạn.
  - PhoBERT.
  - Gán tay một mẫu nhỏ để hiệu chỉnh nhãn LLM (DSL/PPI).

## Chương 8. Kết luận (1 trang)

## Phụ lục

- A. Kiến trúc hệ thống và cấu trúc thư mục.
- B. Các vòng thử nghiệm không được chọn: vòng 2, hướng B/C và thử nghiệm v3 chiều giá.
- C. Dashboard "Đọc tin, có nên mua?" (ảnh chụp và mô tả).
- D. Hướng dẫn chạy lại (`README.md`).

---

### Checklist trước khi nộp

- [ ] Ghi rõ trong phần giới hạn: nhãn được kiểm chứng bằng LLM và phản ứng giá, không có nhãn người.
- [ ] Mọi con số trong báo cáo được đối chiếu với các output được sinh từ pipeline và các script tương ứng trong `src/`.
- [ ] Mỗi kết quả chọn lọc (top q%) đều ghi kèm **độ phủ** và **khoảng tin cậy**.
- [ ] Ghi rõ những cấu hình nào được chốt trước và những con số nào chỉ là tham khảo trên test.
