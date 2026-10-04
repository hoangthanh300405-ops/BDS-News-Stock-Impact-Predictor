# Đọc tin, có nên mua? — BDS-News-Stock-Impact-Predictor (SPINE)

Hệ thống phân tích và dự báo tác động của **tin tức bất động sản** lên **83 cổ phiếu BĐS Việt Nam** (HOSE/HNX),
từ góc nhìn **nhà đầu tư nhỏ lẻ đọc báo**: đọc một tin xong, có nên làm theo không?
Thiết kế: [`pipeline_thiet_ke_du_an/SPINE_Thiet_Ke_Chi_Tiet.md`](pipeline_thiet_ke_du_an/SPINE_Thiet_Ke_Chi_Tiet.md) (§0.4) ·
Kết quả: [`outputs/reports/KET_QUA_TONG_HOP.md`](outputs/reports/KET_QUA_TONG_HOP.md) (§0) ·
Khung báo cáo: [`pipeline_thiet_ke_du_an/KHUNG_BAO_CAO.md`](pipeline_thiet_ke_du_an/KHUNG_BAO_CAO.md)

## Sản phẩm

| Thành phần | Ở đâu |
|---|---|
| **Dashboard** "Đọc tin, có nên mua?": so sánh 3 nhà đầu tư, tra từng mã, bối cảnh ngành (tham khảo), khuyến nghị cho tin mới nhất | `outputs/dashboard/doc_tin_co_nen_mua.html` (mở bằng trình duyệt) |
| **Khuyến nghị cho tin mới nhất** (3 chế độ: tiêu chuẩn / thận trọng / tự tin hai phía) | `outputs/reports/khuyen_nghi_moi_nhat.csv` |
| **Mô hình cuối cùng** và kết quả walk-forward | `src/retail/run_final.py` → `outputs/reports/final_retail.csv` |
| Mô hình phụ **bối cảnh ngành** (chỉ tham khảo, không tác động khuyến nghị) | `src/sector/` → `outputs/reports/sector_tune_moi_nhat.csv` (KET_QUA §3h) |
| Báo cáo kết quả, biểu đồ | `outputs/reports/`, `outputs/figures/` |

**Kết quả chính** (tập test 2025Q2–2026Q3, 6.686 tin, chưa từng dùng để chọn gì):
- Khi mô hình đủ tự tin, **60–65% khuyến nghị đúng**: tiêu chuẩn 60,0%, thận trọng 62,3%, tự tin hai phía 65,1% ở 24% số tin. Làm theo mọi tin chỉ đúng 45,6%.
- Danh mục 100 triệu (chế độ thận trọng): **+59%**, trong khi mua theo mọi tin **−24%** và giữ rổ BĐS **−21%**.
- Đây là mô phỏng học thuật, **không phải khuyến nghị đầu tư**.

## Cài đặt

Python 3.12, chạy trên CPU (không cần GPU):

```bash
py -3.12 -m pip install -r requirements.txt
```

Dữ liệu gốc đặt trong `data final/` (chỉ đọc). Chạy toàn bộ sản phẩm bằng một lệnh:

```bash
py -3.12 -m src.run_all              # ~35 phút (lâu nhất: mã hóa ngữ nghĩa 47 nghìn bài báo, ~13 phút)
py -3.12 -m src.run_all --skip-embed # dùng lại vector ngữ nghĩa đã có (~20 phút)
py -3.12 -m src.run_all --full       # thêm mọi thí nghiệm nghiên cứu (~2 giờ)
py -3.12 -m pytest -q                # 15 test chống rò rỉ tương lai
```

## Chạy từng bước

Mọi lệnh chạy từ thư mục gốc của project.

| Bước | Lệnh | Thời gian | Đầu ra |
|---|---|---|---|
| Tầng 1: giá, tin, luồng pháp lý, target | `py -3.12 -m src.run_pipeline` | ~30 giây | `data/clean/*.parquet` |
| **MÔ HÌNH CUỐI CÙNG (nhà đầu tư nhỏ lẻ)** | `py -3.12 -m src.retail.run_final` | ~5 phút | `outputs/reports/final_retail.csv` |
| **Khuyến nghị cho tin mới nhất** | `py -3.12 -m src.retail.predict_latest` | ~1 phút | `outputs/reports/khuyen_nghi_moi_nhat.csv` |
| Mô hình phụ: bối cảnh ngành, bản đầu (tham khảo) | `py -3.12 -m src.sector.run_sector` | ~1 phút | `outputs/reports/sector_ket_qua.md` |
| Mô hình phụ: tinh chỉnh + bối cảnh ngành mới nhất | `py -3.12 -m src.sector.tune_sector` | ~8 phút | `outputs/reports/sector_tune_*.csv` |
| Thêm dữ liệu có tốt hơn không? (đường cong học của mô hình chính) | `py -3.12 -m src.retail.learning_curve` | ~8 phút | `outputs/reports/learning_curve.csv` |
| Kiểm thử | `py -3.12 -m pytest -q` | ~5 giây | 15 test |
| P0: data contract + biểu đồ EDA | `py -3.12 -m src.eval.p0_report` | ~20 giây | `outputs/reports/p0_data_contract.md`, `outputs/figures/p0_*.png` |
| Baseline B0–B2 | `py -3.12 -m src.models.run_baselines` | ~5 phút | `outputs/reports/baselines.csv` |
| RQ1: giá so với giá + tin | `py -3.12 -m src.models.run_rq1` | ~8 phút | `outputs/reports/rq1_price_vs_news.csv` |
| Chẩn đoán thời điểm tác động của tin | `py -3.12 -m src.models.diag_news_timing` | ~1 phút | in ra màn hình |
| Mức 2: Rank IC, big_move, LRI/PPI, Đ2 | `py -3.12 -m src.models.run_level2` | ~8 phút | `outputs/reports/level2.csv` |
| Phản ứng của giá với tin (event study doanh nghiệp) | `py -3.12 -m src.analysis.reaction` | ~1 phút | `outputs/reports/reaction.md`, `outputs/figures/r1–r2_*.png` |
| Output tăng/không tăng × đồng thuận/mâu thuẫn (walk-forward và 70/10/20) | `py -3.12 -m src.models.run_updown` | ~5 phút | `outputs/reports/updown_agreement.csv` |
| Cải thiện output (chọn theo validation, stacking, tín hiệu mạnh) | `py -3.12 -m src.models.run_updown_v2` | ~10 phút | `outputs/reports/updown_v2.csv` |
| **Hướng B: dự báo sự chú ý (thanh khoản, biến động)** | `py -3.12 -m src.models.run_attention` | ~8 phút | `outputs/reports/attention.csv` |
| Hướng B: khối lượng giao dịch quanh phiên có tin | `py -3.12 -m src.analysis.attention_event` | ~1 phút | `outputs/reports/attention_event.csv`, `outputs/figures/r5_*.png` |
| **Hướng C1: đi tiếp / đảo chiều sau biến động mạnh** | `py -3.12 -m src.models.run_c1` | ~3 phút | `outputs/reports/c1_continuation.csv`, `c1_descriptive.csv` |
| **Hướng C: mô hình văn bản (tin trước giờ mở cửa → phản ứng trong phiên)** | `py -3.12 -m src.models.run_text` (thêm `--fund` = chỉ tin cơ bản) | ~5 phút | `outputs/reports/text_model*.csv`, `text_top_terms*.md` |
| Hướng C2: loại tin & độ mới | `py -3.12 -m src.analysis.news_novelty_check` | ~2 phút | in ra màn hình |
| **Bảng output hợp nhất (mã × phiên, dự đoán cạnh thực tế)** | `py -3.12 -m src.analysis.output_table` | ~10 giây | `outputs/reports/bang_output.csv` |
| **Cấu hình cuối (kết quả chính cho báo cáo)** | `py -3.12 -m src.models.run_final` | ~3 phút | `outputs/reports/final_selective.csv` |
| Vòng cải tiến cuối (C1 lưới rộng, C2 bagging, C3 lọc đồng thuận B) | `py -3.12 -m src.models.run_final_v2` | ~15 phút | `outputs/reports/final_v2.csv` |
| Diễn giải: permutation importance | `py -3.12 -m src.analysis.importance` | ~3 phút | `outputs/reports/importance.csv`, `outputs/figures/r4_importance.png` |
| Thử nghiệm v3: target dấu lợi suất + đặc trưng mới (phụ lục) | `py -3.12 -m src.models.run_direction_v3` | ~10 phút | `outputs/reports/direction_v3.csv` |
| Dự đoán có chọn lọc (độ phủ ↔ độ chính xác) | `py -3.12 -m src.analysis.selective` | ~10 giây | `outputs/reports/selective.csv` |
| Chiều tăng/giảm & kết hợp A/B | `py -3.12 -m src.analysis.direction_agreement` | ~10 giây | `outputs/reports/direction_agreement.csv` |
| NLP: tạo tập vàng gán tay (`--xlsx-only` = chỉ ghi lại file hướng dẫn, giữ mẫu) | `py -3.12 -m src.nlp.gold_sample` | ~10 giây | `data/gold/tap_vang_gan_nhan.xlsx`, `gold_key.parquet` |
| NLP: chưng cất nhãn LLM + bộ lọc mức liên quan | `py -3.12 -m src.nlp.classifiers` | ~3 phút | `data/clean/article_nlp.parquet`, `outputs/reports/nlp_classifiers.csv` |
| **NLP: kiểm chứng nhãn nhiều chiều** (thay tập vàng; `--export` xuất mẫu mù cho trọng tài) | `py -3.12 -m src.nlp.label_validation` | ~1 phút | `outputs/reports/label_validation.md`, `outputs/figures/r13_label_car.png` |
| NLP: chấm tập vàng (chỉ khi có nhãn người) | `py -3.12 -m src.nlp.gold_eval` | ~5 giây | `outputs/reports/gold_eval.csv` |
| **Nhà đầu tư nhỏ lẻ: bảng sự kiện "đọc tin → làm theo?"** (cần chạy classifiers trước) | `py -3.12 -m src.retail.events` | ~20 giây | `data/clean/retail_events.parquet` |
| **Nhà đầu tư nhỏ lẻ: mô hình + so sánh A/B/C** | `py -3.12 -m src.retail.run_retail` | ~2 phút | `outputs/reports/retail_*.csv`, `outputs/figures/r6–r7_*.png` |
| Nhà đầu tư nhỏ lẻ: vòng cải tiến v2 (chọn theo val) | `py -3.12 -m src.retail.run_retail_v2` | ~10 phút | `outputs/reports/retail_v2_*.csv` |
| Nhà đầu tư nhỏ lẻ: so sánh 10 thuật toán (RF, Extra Trees, HGB, SVM, k-NN, NB, MLP, kết hợp…) | `py -3.12 -m src.retail.run_models_compare` | ~15 phút | `outputs/reports/retail_models_compare.csv` |
| Nhà đầu tư nhỏ lẻ: bảng kết quả đầy đủ (vòng 1) | `py -3.12 -m src.retail.full_table` | ~2 giây | `outputs/reports/BANG_KET_QUA_DAY_DU.csv/.md` |
| **Nhà đầu tư nhỏ lẻ: vòng 3 (E3 kết hợp + học xếp hạng + đồng thuận)** | `py -3.12 -m src.retail.run_retail_v3` | ~10 phút | `outputs/reports/retail_v3.csv`, `retail_v3_quarters.csv` |
| Phân tích A1–A2: báo trễ so với giá, loại tin còn dư địa | `py -3.12 -m src.analysis.retail_analytics` | ~30 giây | `outputs/reports/retail_A1_car.csv`, `retail_A2_by_type.csv`, `outputs/figures/r8–r9_*.png` |
| Phân tích A3: mô hình E3 dựa vào nhóm thông tin nào | `py -3.12 -m src.analysis.retail_importance` | ~5 phút | `outputs/reports/retail_A3_importance.csv`, `outputs/figures/r10_*.png` |
| **P4: mô phỏng danh mục 100 triệu** (cần chạy run_retail_v3 trước) | `py -3.12 -m src.retail.portfolio` | ~2 phút | `outputs/reports/retail_portfolio.csv`, `outputs/figures/r11_portfolio.png` |
| P5: kiểm tra độ vững của mô phỏng danh mục | `py -3.12 -m src.retail.robustness` | ~5 phút | `outputs/reports/retail_robustness.csv`, `outputs/figures/r12_robustness.png` |
| **Dashboard "Đọc tin, có nên mua?"** (sau run_final, predict_latest, retail_analytics, robustness, sector) | `py -3.12 -m src.retail.dashboard` | ~1 phút | `outputs/dashboard/doc_tin_co_nen_mua.html` |
| NLP: mã hóa ngữ nghĩa bài báo (multilingual-e5-small, CPU) | `py -3.12 -m src.nlp.embed` | ~15 phút | `data/clean/article_emb.npy` |
| NLP: gom tin thành câu chuyện (`--inspect` để xem các cặp quanh ngưỡng) | `py -3.12 -m src.nlp.stories` | ~1 phút | `data/clean/stories.parquet` |
| Phân tích A4–A9: câu chuyện, nguồn, thời điểm, thị trường, thanh khoản, khối lượng | `py -3.12 -m src.analysis.retail_analytics2` | ~1 phút | `outputs/reports/retail_A4_A9.csv`, `outputs/figures/r14_heterogeneity.png` |
| Nhà đầu tư nhỏ lẻ: vòng 4 (E4: 5 mô hình + ngữ nghĩa + câu chuyện + trọng số thời gian) | `py -3.12 -m src.retail.run_retail_v4` | ~25 phút | `outputs/reports/retail_v4.csv` |
| Event study chính sách | `py -3.12 -m src.analysis.policy_event_study` | ~30 giây | `outputs/reports/policy_event_study.md`, `outputs/figures/r3_*.png` |

**Tổng hợp kết quả:** [`outputs/reports/KET_QUA_TONG_HOP.md`](outputs/reports/KET_QUA_TONG_HOP.md)

## Cấu trúc

```
data final/            dữ liệu gốc — CHỈ ĐỌC, pipeline không ghi vào đây
data/clean/            Tầng 1 (parquet): articles_clean, article_label, article_ticker_label, legal_streams,
                       policy_events, trading_calendar, stock_prices_clean, benchmark_nganh, company_profile,
                       target_A, target_B
src/config.py          đường dẫn, hằng số, quy ước (chốt phiên 14:45, h ∈ {1, 5}, ngưỡng ±0,5σ…)
src/data/              P1 tin tức, P3 luồng pháp lý, P4 giá, P6 target, ánh xạ tin -> phiên
src/features/          đặc trưng giá (F5), tin tức (F1, F6, F7, F3), hiệu chỉnh nhãn Đ2
src/eval/              walk-forward 6 fold (purge + embargo), chỉ số, kiểm định DM (HLN), bootstrap khối, Rank IC
src/models/            engine (B0, B1, LogReg, LightGBM), các script thí nghiệm
src/nlp/               tập vàng gán tay, chưng cất nhãn LLM (tf-idf), bộ lọc mức liên quan
src/retail/            góc nhìn nhà đầu tư nhỏ lẻ: "đọc tin xong, có nên làm theo?"
data/gold/             tập vàng (file gán cho nhóm + đáp án LLM — KHÔNG giao đáp án cho người gán)
tests/                 unit test: chốt phiên, múi giờ, không rò rỉ tương lai (đặc trưng, target, fold, Đ2)
outputs/               bảng kết quả (reports/) và dự đoán từng dòng (predictions/)
crawl_data/            code crawler (*.py) + CSV keyword thô của CafeF/Kenh14 (chỉ để tham khảo cách thu thập)
dist/                  2 gói: *_code_v1.0.zip (chỉ code) và *_v1.0.zip (đầy đủ) — py -3.12 -m src.package
_luu_tru/              tài liệu thiết kế cũ (HiDRA-Net, kế hoạch gốc, backup SPINE) — không dùng trong pipeline
```

## Quy ước quan trọng

- **Không nhìn trộm tương lai.** Đặc trưng tại phiên t chỉ dùng tin đăng trước 14:45 phiên t và giá đến hết phiên t. Bảng hiệu chỉnh Đ2, chuẩn hóa và siêu tham số đều học lại trong từng fold, chỉ từ train. Các unit test trong `tests/` kiểm tra những điều này. Đừng xóa chúng.
- **Tập test (2025Q2 → 2026Q3) không dùng để chọn bất cứ thứ gì:** không chọn đặc trưng, không chọn horizon, không chọn ngưỡng.
- **So sánh mô hình** trên cùng mẫu, cùng fold. Chỉ số chính: balanced accuracy, MCC, log-loss, Rank IC. Không so macro-F1 với B0, vì B0 chỉ đoán một lớp nên macro-F1 của nó thấp một cách cơ học.
