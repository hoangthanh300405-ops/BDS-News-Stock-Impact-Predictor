"""Kiểm tra nhanh: tác động của tin có phụ thuộc ĐỘ MỚI (Tetlock 2011) và LOẠI SỰ KIỆN (Boudoukh et al. 2019)?

Độ mới của bài n về mã i = 1 − cosine TF-IDF lớn nhất giữa (tiêu đề + mô tả) của bài n và mọi bài về mã i trong
5 phiên trước. Chỉ dùng tin TRƯỚC GIỜ MỞ CỬA (phép thử sạch: nhà đầu tư đọc tin trước khi phiên bắt đầu).
Đo: Rank IC giữa sắc thái nhãn LLM và AR phiên phản ứng, tách theo nhóm độ mới / loại tin.
Dùng:  python -m src.analysis.news_novelty_check
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import CLEAN

FUND = {"tai_chinh_dn", "phap_ly", "chinh_sach", "lai_suat_tin_dung"}      # sự kiện cơ bản
EVT = {"EARNINGS": "Kết quả KD", "CAPITAL_RAISE": "Phát hành/vốn", "M_AND_A": "M&A", "LEGAL": "Pháp lý",
       "DEBT_BOND": "Nợ/trái phiếu", "MANAGEMENT": "Nhân sự/QT", "INSIDER_TRADING": "GD nội bộ",
       "PROJECT": "Dự án", "MARKET": "Thị trường", "POLICY": "Chính sách"}


def main():
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet")
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session", "published_at", "title", "description"])
    cal = pd.read_parquet(CLEAN / "trading_calendar.parquet").set_index("date")["trading_index"]
    c = atl[~atl["price_report"]].merge(art, on="article_id").dropna(subset=["session", "impact_score"])
    loc = c["published_at"].dt.tz_localize(None)
    c["pre_open"] = (loc.dt.normalize() != c["session"]) | ((loc.dt.hour * 60 + loc.dt.minute) < 9 * 60)
    c["ti"] = c["session"].map(cal)
    # ---- độ mới: TF-IDF trên tiêu đề + mô tả (mỗi bài một vector)
    docs = art.set_index("article_id").loc[c["article_id"].unique()]
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True)
    X = normalize(tf.fit_transform((docs["title"].fillna("") + " " + docs["description"].fillna("")).str.lower()))
    row = pd.Series(range(len(docs)), index=docs.index)
    nov = {}
    for t, g in c.groupby("ticker"):
        g = g.drop_duplicates("article_id").sort_values("ti")
        idx, tis = row[g["article_id"]].to_numpy(), g["ti"].to_numpy()
        S = (X[idx] @ X[idx].T).toarray()
        for a in range(len(g)):
            prior = (tis < tis[a]) & (tis >= tis[a] - 5)                   # bài về cùng mã, 5 phiên trước
            nov[(g["article_id"].iat[a], t)] = 1 - (S[a, prior].max() if prior.any() else 0.0)
    c["novelty"] = [nov.get((a, t), np.nan) for a, t in zip(c["article_id"], c["ticker"])]
    # ---- AR phiên phản ứng
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet", columns=["date", "ticker", "log_return"])
    b = pd.read_parquet(CLEAN / "benchmark_nganh.parquet")
    rB = b[b["method"] == "ew_liquid"].set_index("date")["log_return"]
    ar = px.pivot_table(index="date", columns="ticker", values="log_return").sub(rB, axis=0).stack().rename("ar0").reset_index()
    p = c[c["pre_open"]].merge(ar, left_on=["session", "ticker"], right_on=["date", "ticker"])
    p["abs_ar0"] = p["ar0"].abs()

    def ic(d, y="ar0"):
        s = d.groupby("session").apply(lambda x: spearmanr(x["impact_score"], x[y])[0]
                                       if len(x) >= 5 and x["impact_score"].nunique() > 1 else np.nan, include_groups=False).dropna()
        return f"{s.mean():+.3f} (t={s.mean() / (s.std() / np.sqrt(len(s))):+.1f}, {len(d):,} bài)"

    print(f"Tin trước giờ mở cửa: {len(p):,} (bài × mã) | độ mới trung vị {p['novelty'].median():.2f}")
    p["nov_grp"] = pd.qcut(p["novelty"], 3, labels=["CŨ (lặp lại)", "trung bình", "MỚI"])
    print("\nTheo ĐỘ MỚI — IC(sắc thái, AR phiên phản ứng) | |AR| trung bình:")
    for g, d in p.groupby("nov_grp", observed=True):
        print(f"  {g:<14} {ic(d)} | |AR| {100 * d['abs_ar0'].mean():.2f}%")
    print("\nTheo LOẠI SỰ KIỆN — IC(sắc thái, AR phiên phản ứng):")
    for et, d in p.groupby("event_type"):
        if len(d) > 300 and et in EVT:
            print(f"  {EVT[et]:<14} {ic(d)}")
    print("\nTin cơ bản + MỚI:", ic(p[p["nhom_tin"].isin(FUND) & (p["nov_grp"] == "MỚI")]))
    print("Tin thị trường/dự án + CŨ:", ic(p[~p["nhom_tin"].isin(FUND) & (p["nov_grp"] == "CŨ (lặp lại)")]))


if __name__ == "__main__":
    main()
