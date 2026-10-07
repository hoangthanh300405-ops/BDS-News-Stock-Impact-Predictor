# BDS-News-Stock-Impact-Predictor

**Đọc tin, có nên mua?** — hệ thống phân tích tác động của tin tức bất động sản lên cổ phiếu BĐS Việt Nam từ góc nhìn nhà đầu tư nhỏ lẻ.

Project kết hợp dữ liệu tin tức, dữ liệu giá cổ phiếu, NLP và machine learning để nghiên cứu câu hỏi:

> Sau khi một nhà đầu tư đọc được một tin về cổ phiếu bất động sản, việc làm theo hướng của tin có còn mang lại lợi thế hay không?

Project được xây dựng cho mục đích **nghiên cứu và học thuật**, không phải công cụ khuyến nghị đầu tư thực tế.

---

## 1. Tổng quan

Pipeline gồm các bước chính:

```text
Dữ liệu giá + tin đã gán nhãn
            ↓
      Làm sạch dữ liệu
            ↓
  Ánh xạ tin → phiên giao dịch
            ↓
    NLP & đặc trưng văn bản
            ↓
      Gom bài thành story
            ↓
  Tạo sự kiện (mã, phiên)
            ↓
   Walk-forward prediction
            ↓
 Phân tích / mô phỏng danh mục
            ↓
          Dashboard
```

Dữ liệu tin được thu thập từ:

- CafeF
- Vietstock
- Kenh14

và chia thành ba nhóm:

- `keyword`
- `law`
- `company`

---

## 2. Cấu trúc repository

```text
BDS-News-Stock-Impact-Predictor/
│
├── data/
│   ├── crawl/          # dữ liệu tin thô
│   ├── classified/     # dữ liệu tin đã gán nhãn
│   ├── manual/         # dữ liệu kiểm chứng nhãn
│   └── stock/          # dữ liệu giá và trust score
│
├── docs/
│   ├── MODEL_CARD.md
│   └── KHUNG_BAO_CAO.md
│
├── src/
│   ├── analysis/
│   ├── crawl/
│   ├── data/
│   ├── eval/
│   ├── features/
│   ├── label/
│   ├── models/
│   ├── nlp/
│   ├── retail/
│   ├── sector/
│   ├── config.py
│   ├── package.py
│   ├── run_all.py
│   └── run_pipeline.py
│
├── tests/
│
├── .gitignore
├── requirements.txt
└── README.md
```

Hai thư mục sau **không được lưu sẵn trong repository**:

```text
data/clean/
outputs/
```

Chúng được tạo tự động khi chạy pipeline.

---

## 3. Các file chính

| File | Chức năng |
|---|---|
| `src/config.py` | Quản lý đường dẫn dữ liệu, timezone, cutoff phiên và các hằng số chung |
| `src/run_pipeline.py` | Chạy pipeline xử lý dữ liệu ban đầu |
| `src/run_all.py` | Chạy toàn bộ hệ thống theo đúng thứ tự |
| `src/data/sessions.py` | Ánh xạ thời gian đăng tin sang phiên giao dịch |
| `src/label/classify_all.py` | Gán nhãn dữ liệu tin tức |
| `src/nlp/label_validation.py` | Kiểm chứng nhãn giữa Qwen, GPT-4.1-mini và các nguồn tham chiếu |
| `src/nlp/embed.py` | Sinh embedding cho bài báo |
| `src/nlp/stories.py` | Gom các bài liên quan thành story |
| `src/retail/events.py` | Tạo tập sự kiện cho bài toán nhà đầu tư nhỏ lẻ |
| `src/retail/run_final.py` | Huấn luyện và đánh giá mô hình cuối |
| `src/retail/predict_latest.py` | Sinh dự đoán cho tin mới nhất |
| `src/retail/portfolio.py` | Mô phỏng danh mục |
| `src/retail/robustness.py` | Kiểm tra độ vững của mô hình/mô phỏng |
| `src/retail/dashboard.py` | Sinh dashboard HTML |
| `src/sector/run_sector.py` | Phân tích bối cảnh ngành |
| `src/package.py` | Đóng gói project thành file ZIP |
| `docs/MODEL_CARD.md` | Mô tả mô hình, dữ liệu, hiệu năng và giới hạn |
| `docs/KHUNG_BAO_CAO.md` | Khung nội dung cho báo cáo nghiên cứu |

## 4. Dữ liệu

### 4.1. Tin tức đã gán nhãn

Các file đầu vào chính nằm trong:

```text
data/classified/
```

gồm:

```text
cafef_company_classified.csv
cafef_keyword_classified.csv
cafef_law_classified.csv

kenh14_company_classified.csv
kenh14_keywords_classified.csv
kenh14_law_classified.csv

vietstock_company_classified.csv
vietstock_keyword_classified.csv
vietstock_law_classified.csv
```

Đây là 9 bộ dữ liệu tương ứng với:

```text
3 loại tin × 3 nguồn báo
```

---

### 4.2. Dữ liệu giá

Dữ liệu cổ phiếu nằm trong:

```text
data/stock/
```

với các file:

```text
Company_Profile_BDS_REAL_DATA__1_.xlsx
Bang_Trust_Score.xlsx
```

`Company_Profile_BDS_REAL_DATA__1_.xlsx` chứa dữ liệu giá, khối lượng giao dịch và hồ sơ doanh nghiệp.

`Bang_Trust_Score.xlsx` được sử dụng làm dữ liệu tham chiếu cho một số bước kiểm chứng dữ liệu.

---

### 4.3. Dữ liệu kiểm chứng nhãn

Các file trong:

```text
data/manual/
```

bao gồm:

```text
gpt41mini_labels.csv
ref_labels.csv
ref_sample.csv
ref_key.csv
team_sheet.xlsx
team_key.parquet
```

Trong đó `gpt41mini_labels.csv` là bộ nhãn độc lập được tạo bằng GPT-4.1-mini để phục vụ **cross-validation / agreement analysis** với bộ nhãn chính.

File này không phải output của pipeline.

---

## 5. Cài đặt

Project được phát triển với Python 3.12.

Cài dependencies:

```bash
py -3.12 -m pip install -r requirements.txt
```

Hoặc:

```bash
python -m pip install -r requirements.txt
```

Mọi lệnh bên dưới được chạy từ thư mục gốc của repository.

---

## 6. Chạy pipeline dữ liệu

Chạy:

```bash
py -3.12 -m src.run_pipeline
```

hoặc:

```bash
python -m src.run_pipeline
```

Pipeline thực hiện lần lượt:

```text
P4 — dữ liệu giá và lịch phiên
 ↓
P1/P2 — làm sạch và hợp nhất tin
 ↓
P3 — luồng pháp lý / chính sách
 ↓
P6 — tạo target
```

Các file trung gian được sinh tự động trong:

```text
data/clean/
```

Ví dụ:

```text
trading_calendar.parquet
stock_prices_clean.parquet
benchmark_nganh.parquet
company_profile.parquet

articles_clean.parquet
article_label.parquet
article_ticker_label.parquet
```

Các file trong `data/clean/` là generated data và không cần commit lên repository.

---

## 7. Ánh xạ tin sang phiên giao dịch

Project sử dụng timezone:

```text
Asia/Ho_Chi_Minh
```

Cutoff của phiên giao dịch:

```text
14:45
```

Quy tắc:

```text
Tin đăng trước 14:45
→ phiên giao dịch cùng ngày

Tin đăng từ 14:45 trở đi
→ phiên giao dịch tiếp theo
```

Tin đăng vào ngày không giao dịch cũng được chuyển sang phiên giao dịch kế tiếp.

Logic nằm tại:

```text
src/data/sessions.py
```

---

## 8. NLP và kiểm chứng nhãn

### Gán nhãn

Pipeline gán nhãn dữ liệu nằm tại:

```text
src/label/classify_all.py
```

Dữ liệu thô:

```text
data/crawl/
```

Dữ liệu sau gán nhãn:

```text
data/classified/
```

---

### Kiểm chứng nhãn

Module:

```text
src/nlp/label_validation.py
```

thực hiện nhiều hướng kiểm chứng, bao gồm:

- tính nhất quán giữa các luồng gán nhãn;
- so sánh Qwen với GPT-4.1-mini;
- so sánh với nhãn referee nếu có;
- kiểm tra bằng từ khóa rõ nghĩa;
- kiểm tra liên hệ giữa nhãn và phản ứng giá.

Chạy:

```bash
python -m src.nlp.label_validation
```

Bộ nhãn GPT-4.1-mini được lấy từ:

```text
data/manual/gpt41mini_labels.csv
```

---

## 9. Chạy toàn bộ hệ thống

Sau khi dữ liệu đầu vào đã được chuẩn bị:

```bash
py -3.12 -m src.run_all
```

Hoặc:

```bash
python -m src.run_all
```

Flow chính hiện tại gồm:

```text
Data pipeline
    ↓
NLP classifiers
    ↓
Text embedding
    ↓
Story clustering
    ↓
Label validation
    ↓
Retail events
    ↓
Final walk-forward model
    ↓
Latest prediction
    ↓
Portfolio simulation
    ↓
Robustness analysis
    ↓
Additional analytics
    ↓
Sector context
    ↓
Dashboard
```

---

### Dùng lại embedding đã có

Nếu embedding đã được tạo trước:

```bash
python -m src.run_all --skip-embed
```

---

### Chạy thêm các thí nghiệm nghiên cứu

```bash
python -m src.run_all --full
```

`--full` bổ sung các module nghiên cứu như:

- data contract / EDA;
- mô hình dự báo chiều giá;
- attention model;
- policy event study;
- các phiên bản retail model;
- so sánh nhiều thuật toán;
- permutation importance;
- learning curve.

---

## 10. Các module chính

### Dữ liệu

```text
src/data/
```

Xử lý:

- giá cổ phiếu;
- lịch giao dịch;
- tin tức;
- chính sách;
- target;
- ánh xạ tin sang phiên.

### Feature engineering

```text
src/features/
```

Bao gồm:

- đặc trưng giá;
- đặc trưng tin;
- attention features;
- calibration;
- các biến phục vụ bài toán continuation.

### NLP

```text
src/nlp/
```

Bao gồm:

- chưng cất nhãn;
- embedding;
- story clustering;
- kiểm chứng nhãn;
- đánh giá mẫu gán tay.

### Retail investor model

```text
src/retail/
```

Đây là phần chính của project, tập trung vào câu hỏi:

> Sau khi đọc một tin, nhà đầu tư có nên làm theo hướng của tin hay không?

Mô hình cuối:

```bash
python -m src.retail.run_final
```

### Sector model

```text
src/sector/
```

Mô hình này cung cấp thêm bối cảnh toàn ngành và được sử dụng như một phân tích bổ sung.

---

## 11. Output

Repository không lưu sẵn `outputs/`.

Sau khi chạy code, hệ thống tự tạo:

```text
outputs/
├── reports/
├── figures/
├── predictions/
└── dashboard/
```

Trong đó:

```text
outputs/reports/
```

chứa các bảng và báo cáo kết quả;

```text
outputs/figures/
```

chứa các biểu đồ;

```text
outputs/predictions/
```

chứa prediction từ các mô hình;

và:

```text
outputs/dashboard/doc_tin_co_nen_mua.html
```

là dashboard cuối cùng.

Các file trong `outputs/` là **generated artifacts**, vì vậy không cần upload thủ công lên repository.

---

## 12. Dashboard

Sau khi các bước cần thiết đã chạy:

```bash
python -m src.retail.dashboard
```

Dashboard được sinh tại:

```text
outputs/dashboard/doc_tin_co_nen_mua.html
```

Mở file bằng trình duyệt để xem kết quả.

---

## 13. Testing

Chạy unit test bằng:

```bash
py -3.12 -m pytest -q
```

hoặc:

```bash
python -m pytest -q
```

Các test kiểm tra những phần quan trọng của pipeline như:

- ánh xạ phiên;
- cutoff 14:45;
- timezone;
- feature construction;
- calibration;
- tránh sử dụng thông tin tương lai.

---

## 14. Tái lập project

Một workflow cơ bản trên máy mới:

```bash
py -3.12 -m pip install -r requirements.txt

py -3.12 -m src.run_pipeline

py -3.12 -m pytest -q

py -3.12 -m src.run_all
```

Sau khi hoàn tất, kết quả được sinh trong:

```text
outputs/
```

và dashboard nằm tại:

```text
outputs/dashboard/doc_tin_co_nen_mua.html
```

---

## 15. Documentation

Thông tin chi tiết về mô hình cuối:

```text
docs/MODEL_CARD.md
```

Khung báo cáo nghiên cứu:

```text
docs/KHUNG_BAO_CAO.md
```

---

## 16. Lưu ý

Project này được xây dựng phục vụ mục đích **nghiên cứu và học thuật**.

Các kết quả prediction, backtest và portfolio simulation:

- phụ thuộc vào dữ liệu và giai đoạn nghiên cứu;
- không đảm bảo hiệu quả trong tương lai;
- không phải khuyến nghị mua hoặc bán chứng khoán.

Khi đánh giá mô hình cần xem đồng thời hiệu năng ngoài mẫu, độ phủ của tín hiệu, robustness và các giả định của mô phỏng.