"""Trụ cột 1(b)(c) — CHƯNG CẤT nhãn LLM thành mô hình văn bản nhẹ + BỘ LỌC MỨC LIÊN QUAN.

(b) Chưng cất: tf-idf (tiêu đề + mô tả + 300 ký tự đầu nội dung) + Logistic Regression, học lại loại sự kiện (14 lớp)
    và dấu sắc thái (3 lớp) mà qwen2.5:7b đã gán. Chia THEO THỜI GIAN: train < 01/06/2025, test ≥ 01/06/2025;
    C chọn trên 3 tháng cuối của train. Bài thuộc TẬP VÀNG bị loại khỏi train (để chấm công bằng).
(c) Mức liên quan: LLM không gán nhãn này -> dùng GIÁM SÁT YẾU (weak supervision): luật từ khóa + siêu dữ liệu gán nhãn
    cho các bài "chắc chắn", huấn luyện mô hình văn bản trên các bài đó, rồi dự đoán cho mọi bài; đánh giá trên tập vàng.
Đầu ra: outputs/reports/nlp_classifiers.csv, data/clean/article_nlp.parquet (dự đoán cho mọi bài),
        outputs/reports/nlp_top_terms.md, cột distil_* trong data/manual/team_key.parquet.
Dùng:  python -m src.nlp.classifiers
"""
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import CLEAN, ROOT
from src.nlp.manual_sample import llm_article_labels

OUT, GOLD = ROOT / "outputs", ROOT / "data" / "manual"
SPLIT, VAL_START = "2025-06-01", "2025-03-01"
FIRM_EVENTS = {"EARNINGS", "CAPITAL_RAISE", "M_AND_A", "DEBT_BOND", "MANAGEMENT", "INSIDER_TRADING", "LEGAL"}
RE_MARKET = re.compile(r"vn-?index|khối ngoại|phiên giao dịch|thanh khoản (thị trường|toàn sàn)|sắc (xanh|đỏ)|"
                       r"nhóm cổ phiếu|chỉ số|điểm|tự doanh|bán ròng|mua ròng|chốt phiên|đóng cửa phiên", re.I)
RE_PROMO = re.compile(r"mở bán|ra mắt|chiết khấu|ưu đãi|tiện ích|căn hộ mẫu|booking|giỏ hàng|thanh toán chỉ|"
                      r"sở hữu ngay|vị trí vàng|đẳng cấp|thượng lưu|an cư", re.I)
RE_POLICY = re.compile(r"luật|nghị định|thông tư|nghị quyết|quy hoạch|bộ xây dựng|chính phủ|quốc hội|thủ tướng|"
                       r"lãi suất|tín dụng|ngân hàng nhà nước", re.I)


def corpus():
    art = pd.read_parquet(CLEAN / "articles_clean.parquet",
                          columns=["article_id", "session", "src", "title", "description", "content", "short_content"])
    d = art[~art["short_content"]].merge(llm_article_labels(), on="article_id")
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet", columns=["article_id", "n_tickers_in_article"])
    d = d.merge(atl.groupby("article_id")["n_tickers_in_article"].max().rename("n_tick"), on="article_id", how="left")
    d["n_tick"] = d["n_tick"].fillna(0)
    d["text"] = (d["title"].fillna("") + ". " + d["description"].fillna("") + ". " + d["content"].fillna("").str[:300]).str.lower()
    d["sent"] = np.sign(d["llm_score"].round(6)).astype(int)
    gold = set(pd.read_parquet(GOLD / "team_key.parquet")["article_id"]) if (GOLD / "team_key.parquet").exists() else set()
    d["is_gold"] = d["article_id"].isin(gold)
    return d.dropna(subset=["session"]).reset_index(drop=True)


def weak_relevance(d):
    """Nhãn yếu cho các bài 'chắc chắn'; phần còn lại để trống (NaN)."""
    head = d["title"].fillna("") + " " + d["description"].fillna("")
    m_market = head.str.contains(RE_MARKET) & (d["llm_event_type"].isin(["MARKET", "ANALYST"]) | (d["n_tick"] >= 3))
    m_firm = d["llm_event_type"].isin(FIRM_EVENTS) & d["n_tick"].between(1, 2) & ~head.str.contains(RE_MARKET)
    m_promo = (d["llm_event_type"] == "PROJECT") & head.str.contains(RE_PROMO) & (d["n_tick"] == 0)
    m_policy = d["llm_event_type"].isin(["POLICY", "INTEREST_RATE", "BANKING_CREDIT"]) & head.str.contains(RE_POLICY) & (d["n_tick"] == 0)
    lab = pd.Series(np.nan, index=d.index, dtype=object)
    for mask, name in ((m_policy, "CHINH_SACH_NGANH"), (m_promo, "QUANG_BA"), (m_firm, "SU_KIEN_DN"), (m_market, "BINH_LUAN_TT")):
        lab[mask & lab.isna()] = name
    return lab


def train_eval(d, y, name, rows, terms):
    tr_all = d[(d["session"] < SPLIT) & ~d["is_gold"] & d[y].notna()]
    te = d[(d["session"] >= SPLIT) & d[y].notna()]
    tr, va = tr_all[tr_all["session"] < VAL_START], tr_all[tr_all["session"] >= VAL_START]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6, max_features=100000, sublinear_tf=True).fit(tr["text"])
    best, bf = None, -1
    for C in (0.5, 2, 8):
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(vec.transform(tr["text"]), tr[y].astype(str))
        f = f1_score(va[y].astype(str), m.predict(vec.transform(va["text"])), average="macro")
        if f > bf:
            best, bf = C, f
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6, max_features=100000, sublinear_tf=True).fit(tr_all["text"])
    m = LogisticRegression(C=best, max_iter=3000, class_weight="balanced").fit(vec.transform(tr_all["text"]), tr_all[y].astype(str))
    t0 = time.time(); pred = m.predict(vec.transform(te["text"])); dt = time.time() - t0
    maj = tr_all[y].astype(str).value_counts().index[0]
    rows.append({"nhiệm vụ": name, "nhãn huấn luyện": "LLM" if y != "weak_rel" else "giám sát yếu", "n train": len(tr_all),
                 "n test": len(te), "accuracy": accuracy_score(te[y].astype(str), pred),
                 "đoán lớp phổ biến": accuracy_score(te[y].astype(str), [maj] * len(te)),
                 "macro-F1": f1_score(te[y].astype(str), pred, average="macro"), "C": best,
                 "tốc độ (bài/giây, CPU)": len(te) / max(dt, 1e-6)})
    rep = classification_report(te[y].astype(str), pred, output_dict=True, zero_division=0)
    per = {k: round(v["f1-score"], 3) for k, v in rep.items() if k not in ("accuracy", "macro avg", "weighted avg")}
    vocab = np.array(vec.get_feature_names_out())
    terms[name] = {c: list(vocab[np.argsort(m.coef_[i])[-10:][::-1]]) for i, c in enumerate(m.classes_)} if len(m.classes_) > 2 else {}
    return m, vec, per


def main():
    d = corpus()
    d["weak_rel"] = weak_relevance(d)
    print(f"Kho: {len(d):,} bài | bài có nhãn yếu mức liên quan: {d['weak_rel'].notna().sum():,} "
          f"{d['weak_rel'].value_counts().to_dict()}")
    rows, terms, per = [], {}, {}
    models = {}
    for y, name in (("llm_event_type", "Loại sự kiện (14 lớp)"), ("sent", "Dấu sắc thái (3 lớp)"),
                    ("weak_rel", "Mức liên quan (4 lớp)")):
        models[y] = train_eval(d, y, name, rows, terms)
        per[name] = models[y][2]
        print(f"  xong {name}")
    # dự đoán cho mọi bài (dùng cho trụ cột 2) + cho tập vàng
    out = d[["article_id", "session", "src"]].copy()
    for y, col in (("llm_event_type", "pred_event_type"), ("sent", "pred_sent"), ("weak_rel", "pred_relevance")):
        m, vec, _ = models[y]
        out[col] = m.predict(vec.transform(d["text"]))
        if y == "weak_rel":
            P = m.predict_proba(vec.transform(d["text"]))
            out["p_relevance_max"] = P.max(axis=1)
    out.to_parquet(CLEAN / "article_nlp.parquet", index=False)
    if (GOLD / "team_key.parquet").exists():
        key = pd.read_parquet(GOLD / "team_key.parquet").drop(columns=["distil_event_type", "distil_sent", "distil_relevance"], errors="ignore")
        p = out.set_index("article_id")
        key["distil_event_type"] = key["article_id"].map(p["pred_event_type"])
        key["distil_sent"] = key["article_id"].map(p["pred_sent"])
        key["distil_relevance"] = key["article_id"].map(p["pred_relevance"])
        key.to_parquet(GOLD / "team_key.parquet", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "reports" / "nlp_classifiers.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "reports" / "nlp_top_terms.md", "w", encoding="utf-8") as fh:
        fh.write("# Từ/cụm từ đặc trưng nhất của mỗi lớp (mô hình tf-idf)\n\n")
        for task, cl in terms.items():
            fh.write(f"## {task}\n\n")
            for c, words in cl.items():
                fh.write(f"- **{c}**: {', '.join(words)}\n")
            fh.write("\n")
        fh.write("## F1 theo lớp (tập test theo thời gian)\n\n")
        for task, p in per.items():
            fh.write(f"- **{task}**: " + ", ".join(f"{k} {v}" for k, v in p.items()) + "\n")
    print(f"\nDự đoán mức liên quan cho toàn kho: {out['pred_relevance'].value_counts(normalize=True).round(3).to_dict()}")
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))
    for task, p in per.items():
        print(f"F1 theo lớp — {task}: {p}")


if __name__ == "__main__":
    main()
