"""Chạy TOÀN BỘ sản phẩm theo đúng thứ tự, từ dữ liệu gốc trong `data final/` đến dashboard.

    py -3.12 -m src.run_all              # sản phẩm chính (~40 phút, lâu nhất là bước mã hóa ngữ nghĩa bài báo)
    py -3.12 -m src.run_all --skip-embed # dùng lại vector ngữ nghĩa đã có (~20 phút)
    py -3.12 -m src.run_all --full       # thêm mọi thí nghiệm nghiên cứu (vòng 1–4, so sánh 10 thuật toán…; ~2 giờ)
Dừng ngay khi một bước lỗi và in tên bước đó.
"""
import subprocess
import sys
import time

CORE = [
    ("Tầng 1: giá, tin, luồng pháp lý, target", "src.run_pipeline"),
    ("NLP: chưng cất nhãn LLM + bộ lọc mức liên quan", "src.nlp.classifiers"),
    ("NLP: mã hóa ngữ nghĩa bài báo (e5)", "src.nlp.embed"),
    ("NLP: gom tin thành câu chuyện", "src.nlp.stories"),
    ("NLP: kiểm chứng nhãn LLM nhiều chiều", "src.nlp.label_validation"),
    ("Nhà đầu tư: bảng sự kiện", "src.retail.events"),
    ("Nhà đầu tư: MÔ HÌNH CUỐI CÙNG (walk-forward)", "src.retail.run_final"),
    ("Nhà đầu tư: khuyến nghị cho tin mới nhất", "src.retail.predict_latest"),
    ("Nhà đầu tư: mô phỏng danh mục", "src.retail.portfolio"),
    ("Nhà đầu tư: kiểm tra độ vững", "src.retail.robustness"),
    ("Phân tích A1–A2", "src.analysis.retail_analytics"),
    ("Phân tích A4–A9", "src.analysis.retail_analytics2"),
    ("Mô hình phụ (tham khảo): toàn ngành, bản đầu", "src.sector.run_sector"),
    ("Mô hình phụ (tham khảo): tinh chỉnh + bối cảnh ngành mới nhất", "src.sector.tune_sector"),
    ("Dashboard", "src.retail.dashboard"),
]
RESEARCH = [
    ("P0: data contract + EDA", "src.eval.p0_report"),
    ("Chiều giá: cấu hình cuối (§2f)", "src.models.run_final"),
    ("Sự chú ý (§3b)", "src.models.run_attention"),
    ("Event study chính sách (§4)", "src.analysis.policy_event_study"),
    ("Nhà đầu tư vòng 1 + bảng đầy đủ", "src.retail.run_retail"),
    ("Nhà đầu tư: bảng đầy đủ vòng 1", "src.retail.full_table"),
    ("Nhà đầu tư vòng 2", "src.retail.run_retail_v2"),
    ("Nhà đầu tư vòng 3", "src.retail.run_retail_v3"),
    ("Nhà đầu tư vòng 4", "src.retail.run_retail_v4"),
    ("So sánh 10 thuật toán", "src.retail.run_models_compare"),
    ("Phân tích A3 (dựa vào gì)", "src.analysis.retail_importance"),
    ("Đường cong học: thêm dữ liệu có tốt hơn không", "src.retail.learning_curve"),
]


def main():
    steps = [s for s in CORE if not ("--skip-embed" in sys.argv and s[1] == "src.nlp.embed")]
    if "--full" in sys.argv:
        steps = steps[:-1] + RESEARCH + steps[-1:]
    t_all = time.time()
    for i, (name, mod) in enumerate(steps, 1):
        t0 = time.time()
        print(f"[{i}/{len(steps)}] {name} ({mod})", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", mod], capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
            sys.exit(f"LỖI ở bước {i}: {name}")
        print(f"      xong sau {time.time() - t0:.0f} giây", flush=True)
    print(f"Hoàn tất {len(steps)} bước sau {(time.time() - t_all) / 60:.1f} phút. "
          f"Mở outputs/dashboard/doc_tin_co_nen_mua.html để xem kết quả.")


if __name__ == "__main__":
    main()
