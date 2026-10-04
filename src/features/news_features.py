"""Đặc trưng tin tức lõi — mức 1 (§7.2 F1, §7.7 F6, §7.8 F7 rút gọn).

Nửa đời cố định λ = 3 phiên (lưới λ và hiệu chỉnh Đ2 thuộc mức 2). Nhãn LLM THÔ.
Trọng số của bài đăng ở phiên t−τ khi tính cho phiên t:  w(τ) = 2^(−τ/λ),  0 ≤ τ ≤ 3λ.
Tin ở phiên t đều đăng trước 14:45 phiên t (P1), nên đặc trưng tại t không nhìn thấy tương lai.

  INT   = Σ w                       (cường độ: lượng tin có trọng số)
  TONE  = Σ w·s / max(1, Σ w)       (sắc thái: trung bình điểm có trọng số)
  NEG   = Σ w·[s<0] / max(1, Σ w)   (tỷ trọng tin xấu — tin âm có thể mang nhiều thông tin hơn)
"""
import numpy as np
import pandas as pd

from src.config import CLEAN

LAMBDA = 3
GROUPS = ["du_an_ha_tang", "chinh_sach", "thi_truong", "phap_ly", "tai_chinh_dn", "lai_suat_tin_dung"]
B_NEWS = ["TONE_all", "INTz_all", "TONE_phap_ly", "TONE_chinh_sach", "breadth_net"]
A_NEWS = ["TONE_i", "NEG_i", "INT_i", "mention_z", "news_today", "TONE_gap", "price_report_share",
          "n_tickers_mean", "TONE_all", "INTz_all"]
LEGAL = ["LRI_tone", "LRI_share", "PPI_share", "PPI_tone", "is_policy_day"]          # F3 (mức 2)


def _kernel(lam=LAMBDA):
    tau = np.arange(3 * lam + 1)
    return 2.0 ** (-tau / lam)


def _decay(daily: pd.DataFrame, n_sessions: int, lam=LAMBDA) -> pd.DataFrame:
    """daily: index = trading_index (int), cột = đại lượng cộng dồn theo phiên. Trả về tổng có trọng số nửa đời."""
    full = daily.reindex(range(n_sessions), fill_value=0.0).to_numpy(float)
    k = _kernel(lam)
    out = np.zeros_like(full)
    for tau, w in enumerate(k):                      # out[t] = Σ_τ w(τ)·x[t−τ]
        out[tau:] += w * full[: n_sessions - tau]
    return pd.DataFrame(out, index=range(n_sessions), columns=daily.columns)


def _calendar():
    cal = pd.read_parquet(CLEAN / "trading_calendar.parquet")
    return cal.set_index("date")["trading_index"], pd.DatetimeIndex(cal["date"])


def _industry_articles(ti, ind_scores=None):
    """1 dòng / bài cấp ngành (keyword + law; bài có cả hai -> trung bình điểm).
    ind_scores: tùy chọn — Series article_id -> điểm đã hiệu chỉnh (Đ2) thay cho nhãn thô."""
    lab = pd.read_parquet(CLEAN / "article_label.parquet")
    lab = lab[lab["typ"].isin(["keyword", "law"])]
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session"])
    a = lab.groupby("article_id").agg(s=("impact_score", "mean"), nhom=("nhom_tin", "first")).reset_index()
    if ind_scores is not None:
        a["s"] = a["article_id"].map(ind_scores)
    a = a.merge(art, on="article_id").dropna(subset=["session", "s"])
    a["ti"] = a["session"].map(ti)
    return a


def industry_news(ind_scores=None, tick_scores=None):
    ti, cal = _calendar()
    n = len(cal)
    a = _industry_articles(ti, ind_scores)
    a["neg"] = (a["s"] < 0).astype(float)
    daily = {"cnt_all": a.groupby("ti").size(), "sum_all": a.groupby("ti")["s"].sum(),
             "neg_all": a.groupby("ti")["neg"].sum()}
    for g in GROUPS:
        sub = a[a["nhom"] == g]
        daily[f"cnt_{g}"] = sub.groupby("ti").size()
        daily[f"sum_{g}"] = sub.groupby("ti")["s"].sum()
    daily = pd.DataFrame(daily).fillna(0)
    D = _decay(daily, n)
    f = pd.DataFrame(index=range(n))
    f["INT_all"] = D["cnt_all"]
    f["TONE_all"] = D["sum_all"] / D["cnt_all"].clip(lower=1)
    f["NEG_all"] = D["neg_all"] / D["cnt_all"].clip(lower=1)
    for g in GROUPS:
        f[f"TONE_{g}"] = D[f"sum_{g}"] / D[f"cnt_{g}"].clip(lower=1)
        f[f"INT_{g}"] = D[f"cnt_{g}"]
    raw = daily["cnt_all"].reindex(range(n), fill_value=0)
    f["INTz_all"] = (raw - raw.rolling(60, min_periods=20).mean().shift(1)) / raw.rolling(60, min_periods=20).std().shift(1)

    # F7 rút gọn — độ rộng tin doanh nghiệp: (số mã có tin tốt − số mã có tin xấu) / số mã có tin, theo nửa đời
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet", columns=["article_id", "ticker", "impact_score"])
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session"])
    c = atl.merge(art, on="article_id").dropna(subset=["session"])
    if tick_scores is not None:
        c["impact_score"] = pd.MultiIndex.from_frame(c[["article_id", "ticker"]]).map(tick_scores)
    c["ti"] = c["session"].map(ti)
    per = c.groupby(["ti", "ticker"])["impact_score"].mean().reset_index()
    br = pd.DataFrame({"pos": per.assign(v=per.impact_score > 0).groupby("ti")["v"].sum(),
                       "neg": per.assign(v=per.impact_score < 0).groupby("ti")["v"].sum(),
                       "n": per.groupby("ti").size()}).astype(float)
    B = _decay(br, n)
    f["breadth_net"] = (B["pos"] - B["neg"]) / B["n"].clip(lower=1)
    f = pd.concat([f, legal_features(ti, n, daily[["cnt_all"]], ind_scores)], axis=1)
    f["date"] = cal
    return f


def legal_features(ti, n, daily_total, ind_scores=None, lam_lri=LAMBDA, lam_ppi=10):
    """F3 (§7.4): LRI — rủi ro pháp lý (λ = 3), PPI — áp lực chính sách (λ = 10: kỳ vọng tác động dài hơn).
    Cả hai chia cho tổng lượng tin cùng cửa sổ (như chỉ số EPU, T19) để không nhảy theo số bài crawl được."""
    ls = pd.read_parquet(CLEAN / "legal_streams.parquet").dropna(subset=["session"])
    if ind_scores is not None:
        ls["impact_score"] = ls["article_id"].map(ind_scores).fillna(ls["impact_score"])
    ls["ti"] = ls["session"].map(ti)
    risk, pol = ls[ls["stream"] == "rui_ro_phap_ly"], ls[ls["stream"] == "chinh_sach"]
    lri = _decay(pd.DataFrame({"c": risk.groupby("ti").size(), "s": risk.groupby("ti")["impact_score"].sum()}).fillna(0),
                 n, lam_lri)
    ppi = _decay(pd.DataFrame({"c": pol.groupby("ti").size(), "s": pol.groupby("ti")["impact_score"].sum()}).fillna(0),
                 n, lam_ppi)
    ev = pd.read_parquet(CLEAN / "policy_events.parquet")
    tot_lri = _decay(daily_total, n, lam_lri)["cnt_all"]           # tổng lượng tin, cùng nửa đời với tử số
    tot_ppi = _decay(daily_total, n, lam_ppi)["cnt_all"]
    out = pd.DataFrame(index=range(n))
    out["LRI_tone"] = lri["s"] / lri["c"].clip(lower=1)
    out["LRI_share"] = lri["c"] / tot_lri.clip(lower=1)
    out["PPI_share"] = ppi["c"] / tot_ppi.clip(lower=1)
    out["PPI_tone"] = ppi["s"] / ppi["c"].clip(lower=1)
    out["is_policy_day"] = 0.0
    out.loc[ev["date"].map(ti).dropna().astype(int).values, "is_policy_day"] = 1.0
    return out


def ticker_news(ind_scores=None, tick_scores=None):
    ti, cal = _calendar()
    n = len(cal)
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet",
                          columns=["article_id", "ticker", "impact_score", "price_report", "n_tickers_in_article"])
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session"])
    c = atl.merge(art, on="article_id")
    if tick_scores is not None:
        c["impact_score"] = pd.MultiIndex.from_frame(c[["article_id", "ticker"]]).map(tick_scores)
    c = c.dropna(subset=["session", "impact_score"])
    c["ti"] = c["session"].map(ti)
    c["neg"] = (c["impact_score"] < 0).astype(float)
    c["pr"] = c["price_report"].astype(float)
    g = c.groupby(["ticker", "ti"]).agg(cnt=("impact_score", "size"), s=("impact_score", "sum"), neg=("neg", "sum"),
                                        pr=("pr", "sum"), ntk=("n_tickers_in_article", "sum"))
    out = []
    for t, d in g.groupby(level=0):
        d = d.droplevel(0)
        D = _decay(d, n)
        raw = d["cnt"].reindex(range(n), fill_value=0)
        f = pd.DataFrame({
            "ticker": t, "date": cal,
            "INT_i": D["cnt"].values,
            "TONE_i": (D["s"] / D["cnt"].clip(lower=1)).values,
            "NEG_i": (D["neg"] / D["cnt"].clip(lower=1)).values,
            "price_report_share": (D["pr"] / D["cnt"].clip(lower=1)).values,
            "n_tickers_mean": (D["ntk"] / D["cnt"].clip(lower=1)).values,
            "news_today": (raw > 0).astype(float).values,
            "mention_z": ((raw - raw.rolling(20, min_periods=10).mean().shift(1))
                          / raw.rolling(20, min_periods=10).std().shift(1).replace(0, np.nan)).fillna(0).values,
        })
        out.append(f)
    f = pd.concat(out, ignore_index=True)
    ind = industry_news(ind_scores, tick_scores)[["date", "TONE_all", "INTz_all"]]
    f = f.merge(ind, on="date", how="left")
    f["TONE_gap"] = f["TONE_i"] - f["TONE_all"]
    return f
