"""P3 — Tách ba luồng tin pháp lý & dựng đợt sự kiện chính sách (§6 P3, §3.5, §17.2).

Bài có event_type ∈ {LEGAL, POLICY}:
  co_van_ban (regex tên văn bản trên title + description + su_kien_chinh — KHÔNG quét content)
     True  -> luồng 'chinh_sach' ; giai_doan suy từ title + su_kien_chinh
     False -> khớp chủ đề vụ án / cưỡng chế / tranh chấp / thủ tục sổ đỏ ? 'rui_ro_phap_ly' : 'khac'
Đầu ra: data/clean/legal_streams.parquet (1 dòng / bài LEGAL-POLICY), policy_events.parquet (đợt, k = 5)
"""
import re

import numpy as np
import pandas as pd

from src.config import CLEAN

RE_DOC = re.compile(r"Luật (Đất đai|Nhà ở|Kinh doanh bất động sản|Các tổ chức tín dụng|Thuế|Quy hoạch|Xây dựng|Đầu tư|"
                    r"Phát triển đô thị)|Nghị định (số )?\d+|Thông tư (số )?\d+|Nghị quyết (số )?\d+", re.IGNORECASE)
RE_RISK = re.compile(r"lừa đảo|chiếm đoạt|khởi tố|bắt tạm giam|truy tố|xét xử|tòa án|giả mạo|làm giả|phong tỏa|kê biên|"
                     r"cưỡng chế|thu hồi đất|thu hồi dự án|đình chỉ|xử phạt|vi phạm|tranh chấp|khiếu nại|kiện|"
                     r"cư dân phản đối|sổ đỏ|sổ hồng|giấy chứng nhận|sang tên|cấp sổ|thủ tục", re.IGNORECASE)
# thứ tự vòng đời: lấy giai đoạn muộn nhất khi khớp nhiều giai đoạn (§17.2)
STAGES = [("du_thao", r"dự thảo|đề xuất|lấy ý kiến|trình Quốc hội|trình Chính phủ"),
          ("thong_qua", r"thông qua|biểu quyết"),
          ("ban_hanh", r"ban hành|ký ban hành"),
          ("sua_doi", r"sửa đổi|bổ sung"),
          ("co_hieu_luc", r"có hiệu lực|hiệu lực từ|chính thức áp dụng|bắt đầu áp dụng"),
          ("huong_dan", r"hướng dẫn|quy định chi tiết")]
STAGE_GROUP = {"du_thao": "de_xuat", "thong_qua": "ban_hanh", "ban_hanh": "ban_hanh", "sua_doi": "ban_hanh",
               "co_hieu_luc": "hieu_luc", "huong_dan": "hieu_luc"}
EVENT_K = 5          # ngưỡng ngày sự kiện: k nhỏ nhất sao cho ngày sự kiện ≤ 30% số phiên (§6 P3)


def build(verbose=True):
    art = pd.read_parquet(CLEAN / "articles_clean.parquet")
    lab = pd.read_parquet(CLEAN / "article_label.parquet")
    score = lab.groupby("article_id")["impact_score"].mean()
    et = lab.groupby("article_id")["event_type"].agg(lambda s: s.mode().iat[0])
    a = art.set_index("article_id").join(et.rename("et"))
    a = a[a["et"].isin(["LEGAL", "POLICY"])].copy()
    head = a["title"].fillna("") + " . " + a["description"].fillna("") + " . " + a["su_kien_chinh"].fillna("")
    a["co_van_ban"] = head.str.contains(RE_DOC)
    a["risk_topic"] = head.str.contains(RE_RISK)
    a["stream"] = np.select([a["co_van_ban"], a["risk_topic"]], ["chinh_sach", "rui_ro_phap_ly"], "khac")
    st = a["title"].fillna("") + " " + a["su_kien_chinh"].fillna("")
    a["giai_doan"] = ""
    for name, pat in STAGES:                       # sau ghi đè trước -> giữ giai đoạn muộn nhất
        a.loc[st.str.contains(pat, case=False, regex=True), "giai_doan"] = name
    a["nhom_giai_doan"] = a["giai_doan"].map(STAGE_GROUP).fillna("")
    a["impact_score"] = score.reindex(a.index)
    out = a.reset_index()[["article_id", "session", "et", "stream", "co_van_ban", "risk_topic", "giai_doan",
                           "nhom_giai_doan", "impact_score"]]
    out.to_parquet(CLEAN / "legal_streams.parquet", index=False)

    # đợt sự kiện chính sách: phiên có ≥ k bài luồng chính sách; các phiên cách nhau ≤ 2 ngày lịch -> cùng đợt
    cal = pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"]
    cnt = out[out["stream"] == "chinh_sach"].groupby("session").size().reindex(cal, fill_value=0)
    days = cnt[cnt >= EVENT_K].index
    gap = pd.Series(days).diff().dt.days.fillna(99)
    event_id = (gap > 2).cumsum()
    ev = pd.DataFrame({"date": days, "event_id": event_id.values, "n_bai": cnt[days].values})
    ev.to_parquet(CLEAN / "policy_events.parquet", index=False)

    if verbose:
        print(f"[P3] {len(out)} bài LEGAL/POLICY | luồng: {out['stream'].value_counts(normalize=True).round(3).to_dict()}")
        pol = out[out["stream"] == "chinh_sach"]
        print(f"[P3] luồng chính sách xác định được giai đoạn: {(pol['giai_doan'] != '').mean():.1%} | "
              f"tỷ lệ nhãn âm: chính sách {(pol['impact_score'] < 0).mean():.1%}, "
              f"rủi ro pháp lý {(out.loc[out['stream'] == 'rui_ro_phap_ly', 'impact_score'] < 0).mean():.1%}")
        print(f"[P3] đợt chính sách (k = {EVENT_K}): {ev['event_id'].nunique()} đợt, {len(ev)} ngày sự kiện "
              f"({len(ev) / len(cal):.0%} số phiên)")
    return out, ev


if __name__ == "__main__":
    build()
