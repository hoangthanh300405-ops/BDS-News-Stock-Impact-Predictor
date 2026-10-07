"""Bài toán VĂN BẢN — từ nội dung tin đăng TRƯỚC GIỜ MỞ CỬA, dự báo phản ứng của CHÍNH PHIÊN ĐÓ.

Đơn vị   : (mã i, phiên t) có ít nhất một bài về mã i đăng trước giờ mở cửa phiên t (tối hôm trước sau 14:45, cuối
           tuần/ngày nghỉ, hoặc sáng trước 9:00); bỏ bài chỉ thuật lại biến động giá. Văn bản = nối tiêu đề + mô tả.
Target   : react_big — |AR_t| > 1σ (phản ứng giá mạnh);  vol_spike0 — giá trị GD phiên t vượt TB 20 phiên trước > 1σ;
           up0 — AR_t > 0 (chiều phản ứng)
Baseline : CHỈ thông tin đến hết phiên t−1 (lúc tin xuất hiện, phiên t chưa diễn ra) -> phép thử công bằng cho tin tức
So sánh  : M0 baseline | M1 + nhãn LLM | M2 + văn bản (tf-idf) | M3 + LLM + văn bản — cùng LogReg, cùng 6 fold;
           tf-idf và chuẩn hóa fit trên train của từng fold; C chọn trên validation.
Dùng:  python -m src.models.run_text
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import CLEAN, ROOT
from src.eval.metrics import dm_test
from src.eval.walk_forward import make_folds
from src.features.attention_features import _volume_panel
from src.features.price_features import features_A

OUT = ROOT / "outputs"
FUND = {"tai_chinh_dn", "phap_ly", "chinh_sach", "lai_suat_tin_dung"}
EVENTS = ["EARNINGS", "LEGAL", "CAPITAL_RAISE", "M_AND_A", "DEBT_BOND", "MARKET", "PROJECT", "POLICY", "MANAGEMENT"]
BASE = ["sigma_ar60", "abs_r_prev", "r_5d_prev", "lv_z20_prev", "spike_rate60", "illiquid", "log_mcap", "dow"]
LLM = ["n_bai", "tone", "abs_tone", "neg_share", "has_fund", "n_tickers_mean"] + [f"ev_{e}" for e in EVENTS]
TARGETS = {"react_big": "Phản ứng giá mạnh (|AR| > 1σ)", "vol_spike0": "Thanh khoản đột biến trong phiên",
           "up0": "Chiều phản ứng (AR > 0)"}


def build():
    cal = pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"]
    nxt = pd.Series(cal.values[1:], index=cal.values[:-1])               # phiên kế tiếp
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet")
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session", "published_at", "title", "description"])
    c = atl[~atl["price_report"]].merge(art, on="article_id").dropna(subset=["session", "impact_score"])
    loc = c["published_at"].dt.tz_localize(None)
    c = c[(loc.dt.normalize() != c["session"]) | ((loc.dt.hour * 60 + loc.dt.minute) < 540)].copy()   # trước giờ mở cửa
    c["text"] = (c["title"].fillna("") + ". " + c["description"].fillna("")).str.lower()
    for e in EVENTS:
        c[f"ev_{e}"] = (c["event_type"] == e).astype(float)
    c["fund"] = c["nhom_tin"].isin(FUND).astype(float)
    c["neg"] = (c["impact_score"] < 0).astype(float)
    g = c.groupby(["ticker", "session"])
    ev = g.agg(text=("text", lambda s: " ".join(dict.fromkeys(s))), n_bai=("article_id", "nunique"),
               tone=("impact_score", "mean"), neg_share=("neg", "mean"), has_fund=("fund", "max"),
               n_tickers_mean=("n_tickers_in_article", "mean"), **{f"ev_{e}": (f"ev_{e}", "max") for e in EVENTS})
    ev = ev.reset_index().rename(columns={"session": "date"})
    ev["abs_tone"] = ev["tone"].abs()
    # baseline: thông tin đến hết phiên t−1, gán sang phiên t
    fa = features_A()[["date", "ticker", "sigma_ar60", "r_1d", "r_5d", "illiquid", "log_mcap"]]
    vol, _ = _volume_panel()
    prev = fa.merge(vol[["date", "ticker", "lv_z20", "spike_rate60"]], on=["date", "ticker"], how="left")
    prev["date"] = prev["date"].map(nxt)
    prev = prev.dropna(subset=["date"]).rename(columns={"r_1d": "r_prev", "r_5d": "r_5d_prev", "lv_z20": "lv_z20_prev"})
    prev["abs_r_prev"] = prev["r_prev"].abs()
    ev = ev.merge(prev, on=["date", "ticker"], how="left")
    ev["dow"] = ev["date"].dt.dayofweek.astype(float)
    # target tại phiên t
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet", columns=["date", "ticker", "log_return"])
    b = pd.read_parquet(CLEAN / "benchmark_nganh.parquet")
    rB = b[b["method"] == "ew_liquid"].set_index("date")["log_return"]
    ar = px.pivot_table(index="date", columns="ticker", values="log_return").reindex(rB.index).sub(rB, axis=0)
    z = ar / ar.rolling(60, min_periods=40).std().shift(1)
    tz = pd.concat({"ar0": ar, "z0": z}, axis=1).stack(level=1, future_stack=True).reset_index().rename(columns={"level_1": "ticker"})
    ev = ev.merge(tz, on=["date", "ticker"], how="left").merge(vol[["date", "ticker", "lv_z20"]], on=["date", "ticker"], how="left")
    ev["react_big"] = (ev["z0"].abs() > 1).astype(float).where(ev["z0"].notna())
    ev["vol_spike0"] = (ev["lv_z20"] > 1).astype(float).where(ev["lv_z20"].notna())
    ev["up0"] = np.where(ev["ar0"] > 0, 1.0, np.where(ev["ar0"] < 0, 0.0, np.nan))
    return ev[ev["date"] >= "2024-04-01"].reset_index(drop=True)


def fit_predict(tr, va, te, target, use_llm, use_text):
    dense = BASE + (LLM if use_llm else [])
    sc = StandardScaler().fit(tr[dense])
    parts = lambda d, tf: [sp.csr_matrix(sc.transform(d[dense]))] + ([tf.transform(d["text"])] if tf else [])
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, max_features=40000, sublinear_tf=True).fit(tr["text"]) if use_text else None
    Xtr, Xva = sp.hstack(parts(tr, tf)).tocsr(), sp.hstack(parts(va, tf)).tocsr()
    best, bl = None, np.inf
    for C in (0.03, 0.1, 0.3, 1.0):
        m = LogisticRegression(C=C, max_iter=3000).fit(Xtr, tr[target])
        l = log_loss(va[target], m.predict_proba(Xva)[:, 1])
        if l < bl:
            best, bl = C, l
    fit = pd.concat([tr, va])
    sc = StandardScaler().fit(fit[dense])
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, max_features=40000, sublinear_tf=True).fit(fit["text"]) if use_text else None
    m = LogisticRegression(C=best, max_iter=3000).fit(sp.hstack(parts(fit, tf)).tocsr(), fit[target])
    return m.predict_proba(sp.hstack(parts(te, tf)).tocsr())[:, 1], m, tf, dense


def boot_auc_diff(t, a, b, n=1000, block=20, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.array(sorted(t["date"].unique())); groups = t.groupby("date").indices
    T = len(dates); out = []
    for _ in range(n):
        st = rng.integers(0, T - block + 1, size=int(np.ceil(T / block)))
        idx = np.concatenate([groups[x] for x in np.concatenate([dates[s:s + block] for s in st])[:T]])
        y = t["y"].to_numpy()[idx]
        if y.min() == y.max():
            continue
        out.append(roc_auc_score(y, t[a].to_numpy()[idx]) - roc_auc_score(y, t[b].to_numpy()[idx]))
    return np.percentile(out, [2.5, 97.5])


def top_terms(m, tf, dense, k=15):
    coef = m.coef_[0][len(dense):]
    vocab = np.array(tf.get_feature_names_out())
    o = np.argsort(coef)
    return list(vocab[o[-k:][::-1]]), list(vocab[o[:k]])


def main(fund_only=False):
    ev = build()
    tag = ""
    if fund_only:                              # giả thuyết C2 (Boudoukh et al. 2019): chỉ tin SỰ KIỆN CƠ BẢN
        ev = ev[ev["has_fund"] == 1].reset_index(drop=True)
        tag = "_fund"
    print(f"Sự kiện (mã, phiên có tin trước giờ mở cửa{' — CHỈ TIN CƠ BẢN' if fund_only else ''}): {len(ev):,} | tỷ lệ: " +
          ", ".join(f"{k} {ev[k].mean():.1%}" for k in TARGETS))
    specs = {"M0 chỉ giá (đến t−1)": (False, False), "M1 + nhãn LLM": (True, False),
             "M2 + văn bản": (False, True), "M3 + LLM + văn bản": (True, True)}
    rows, terms, preds = [], {}, []
    for target in TARGETS:
        d = ev.dropna(subset=BASE + [target]).copy()
        d[BASE + LLM] = d[BASE + LLM].fillna(0.0)
        t_all = []
        for f in make_folds(1):
            tr, va, te = d[d["date"].isin(f.train)], d[d["date"].isin(f.val)], d[d["date"].isin(f.test)]
            out = te[["date", "ticker", target]].rename(columns={target: "y"}).copy()
            for name, (ul, ut) in specs.items():
                out[name], m, tf, dense = fit_predict(tr, va, te, target, ul, ut)
                if f.k == 6 and ut and not ul:
                    terms[target] = top_terms(m, tf, dense)
            t_all.append(out)
        t = pd.concat(t_all, ignore_index=True)
        preds.append(t.assign(target=target))
        ll = lambda col: pd.Series(-(t["y"] * np.log(np.clip(t[col], 1e-6, 1)) + (1 - t["y"]) * np.log(np.clip(1 - t[col], 1e-6, 1))))
        ref = "M0 chỉ giá (đến t−1)"
        for name in specs:
            r = {"target": TARGETS[target], "mô hình": name, "n test": len(t), "tỷ lệ xảy ra": t["y"].mean(),
                 "AUC test": roc_auc_score(t["y"], t[name])}
            if name != ref:
                lo, hi = boot_auc_diff(t, name, ref)
                r["ΔAUC vs M0"], r["KTC95"] = r["AUC test"] - roc_auc_score(t["y"], t[ref]), f"[{lo:+.4f}, {hi:+.4f}]"
                r["DM vs M0"], r["p_DM"] = dm_test(ll(name).groupby(t["date"]).mean(), ll(ref).groupby(t["date"]).mean(), 1)
            rows.append(r)
        print(f"  xong {target}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "reports" / f"text_model{tag}.csv", index=False, encoding="utf-8-sig")
    pd.concat(preds, ignore_index=True).to_parquet(OUT / "predictions" / f"text_model{tag}.parquet", index=False)
    with open(OUT / "reports" / f"text_top_terms{tag}.md", "w", encoding="utf-8") as fh:
        for tg, (pos, neg) in terms.items():
            fh.write(f"## {TARGETS[tg]}\n\n**Tăng xác suất:** {', '.join(pos)}\n\n**Giảm xác suất:** {', '.join(neg)}\n\n")
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(res.to_string(index=False))
    for tg, (pos, neg) in terms.items():
        print(f"\n[{TARGETS[tg]}] TĂNG xác suất: {', '.join(pos[:12])}\n   GIẢM xác suất: {', '.join(neg[:12])}")


if __name__ == "__main__":
    import sys
    main(fund_only="--fund" in sys.argv)
