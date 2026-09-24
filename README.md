# BDS-News-Stock-Impact-Predictor
# BDS-News-Stock-Impact-Predictor

Dự đoán chiều biến động giá cổ phiếu bất động sản (BĐS) từ tin tức nhà đất.

## Mục tiêu

Dự đoán ở 2 cấp độ:

- **Mã cổ phiếu cụ thể** (Model A): tin này làm giá mã VHM, NVL, DXG... tăng hay giảm?
- **Toàn ngành BĐS** (Model B): tin này làm cả ngành tăng hay giảm?

Điểm khác biệt: mỗi tin được **gán trọng số theo độ tin cậy của nguồn** (tính bằng backtest, không đặt tay) và kết hợp **company profile** (phân khúc, độ nhạy với tin tức).

## Cách tiếp cận

Phương án cân bằng: data-driven trust ranking + company profile, tách riêng tin ảnh hưởng **specific** (1 công ty) và **ngành** (toàn thị trường).

## Pipeline

| Bước | Nội dung | Công cụ |
|------|----------|---------|
| 1 | Gán nhãn `pham_vi_anh_huong` (specific / nganh / khong_lien_quan), loại `khong_lien_quan` | `classify_news.py` (Gemini) |
| 2 | Cào company profile, tính độ nhạy tin tức | Script cào Vietstock/CafeF |
| 3 | Thu thập giá từng mã, tính benchmark ngành BĐS | Script lấy giá |
| 4 | Backtest nhãn impact vs biến động giá thực tế (+1 / +3 ngày) | Python |
| 5 | Merge feature table | pandas |
| 6 | Train Model A + Model B | scikit-learn |
| 7 | Đánh giá: LOO CV, accuracy/F1, feature importance | scikit-learn |

## Feature chính

- `article_labels`: nhãn tin từ Gemini (impact, phạm vi ảnh hưởng)
- `company_profile`: phân khúc, độ nhạy tin tức
- `source_trust_score`: `trust_specific` và `trust_nganh` (tính qua backtest)

## Mô hình

- Model A và Model B: Logistic Regression hoặc Decision Tree
- Đánh giá bằng Leave-One-Out Cross-Validation riêng cho từng model

## Đầu ra

- `company_profile.csv`
- `source_trust_score.csv`
- 2 model đã train
- Báo cáo tổng hợp

## Cấu trúc thư mục (gợi ý)

```
BDS-News-Stock-Impact-Predictor/
├── data/
│   ├── raw/              # tin tức, giá
│   ├── processed/        # article_labels, feature table
│   ├── company_profile.csv
│   └── source_trust_score.csv
├── src/
│   ├── classify_news.py
│   ├── scrape_profile.py
│   ├── fetch_prices.py
│   ├── backtest.py
│   ├── build_features.py
│   └── train.py
├── models/               # Model A, Model B
├── reports/
└── README.md
```

## Hạn chế cần lưu ý

- Dataset hiện chỉ **92 bài** (CafeF, VnExpress). Sau khi tách specific/ngành, mỗi nhóm còn nhỏ hơn, cần crawl thêm nhiều tuần.
- Trust score với cỡ mẫu nhỏ **chưa có ý nghĩa thống kê**, chỉ nên xem là tham khảo.
- Gemini free tier có rate limit, bước gán nhãn cần chạy theo batch.
- Company profile là dữ liệu bán tĩnh, cần quy trình cập nhật định kỳ.

## Việc cần chốt

- [ ] Benchmark ngành BĐS: dùng chỉ số có sẵn hay tự tính từ rổ mã theo dõi
- [ ] Khung backtest: +1 ngày hay +3 ngày (thử cả hai)

## Cài đặt & chạy

```bash
pip install pandas scikit-learn google-generativeai
export GEMINI_API_KEY="your_key"

python src/classify_news.py
python src/scrape_profile.py
python src/fetch_prices.py
python src/backtest.py
python src/build_features.py
python src/train.py
```
