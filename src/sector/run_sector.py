"""MÔ HÌNH TOÀN NGÀNH: tin tức toàn ngành BĐS -> rổ BĐS 5 phiên tới có lãi sau phí không. Chốt trước khi chạy (30/09/2026).

Đơn vị: một PHIÊN t (không phải một mã). Câu hỏi của nhà đầu tư nhỏ lẻ: "đọc hết tin ngành hôm nay, có nên mua rổ BĐS
(hoặc quỹ/ETF BĐS) ở giá đóng cửa hôm nay, giữ 5 phiên không?"
Mục tiêu (Y)  : log(rổ[t+5]/rổ[t]) > log(1,004)  (rổ = EW liquid, cùng rổ dùng làm mốc ở mô hình chính; phí 0,4% như Y1).
Thông tin     : tin có phiên ≤ t (đăng trước 14:45 ngày t); giá, thanh khoản, độ rộng thị trường đến hết phiên t−1.
Đặc trưng     : NEWS (tin toàn ngành: số lượng, sắc thái theo nhóm tin, tỷ lệ tin xấu, chính sách, rủi ro pháp lý, tin về mã)
                + PRICE (động lượng, biến động, drawdown của rổ; độ rộng; thanh khoản).
Mô hình       : như mô hình chính — trung bình phân vị (so với val) của LR (C=0,1), RF (500 cây), LightGBM hồi quy Huber
                trên lợi suất 5 phiên; trọng số thời gian 0,5^(tuổi/250). Vì chỉ có vài trăm phiên nên cây nông hơn
                (RF min_samples_leaf = 20; LGBM num_leaves = 7, min_child_samples = 20).
Đánh giá      : walk-forward 6 quý (test 2025Q2 → 2026Q3), purge + embargo; ngưỡng lấy từ val.
                So sánh: mốc lớp đa số, quy tắc động lượng (r20 > 0), quy tắc sắc thái tin (tone5 > trung vị train),
                cùng mô hình chỉ dùng giá / chỉ dùng tin (đo giá trị của tin tức).
Khuyến nghị   : "MUA rổ" khi điểm ∈ top q của val, "ĐỨNG NGOÀI" khi ∈ bottom q, còn lại không khuyến nghị (q = 20%, 30%).
Đầu ra        : outputs/reports/sector_*.csv|md, outputs/predictions/sector.parquet, outputs/figures/r20_sector.png
Dùng          : python -m src.sector.run_sector
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CLEAN, ROOT
from src.eval.walk_forward import make_folds

H, COST, SEED, HALF_LIFE = 5, 0.004, 42, 250
BASKET = "ew_liquid"
REP, PRED, FIG = ROOT / "outputs" / "reports", ROOT / "outputs" / "predictions", ROOT / "outputs" / "figures"
COMPS = ["LR", "RF", "REG"]
TWO_SIDED = (0.20, 0.30)
GROUPS = {"tt": ["thi_truong"], "dn": ["tai_chinh_dn"], "da": ["du_an_ha_tang"], "cs": ["chinh_sach", "phap_ly"],
          "ls": ["lai_suat_tin_dung"]}


# ---------------------------------------------------------------- dữ liệu theo phiên
def build():
    cal = pd.DatetimeIndex(pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"])
    b = pd.read_parquet(CLEAN / "benchmark_nganh.parquet").pivot(index="date", columns="method", values="value").reindex(cal)
    L = np.log(b[BASKET])
    d = pd.DataFrame(index=cal)
    d.index.name = "date"
    # --- mục tiêu: mua đóng cửa t, bán đóng cửa t+5
    d["fwd"] = L.shift(-H) - L
    d["y"] = (d["fwd"] > np.log(1 + COST)).astype(float).where(d["fwd"].notna())
    # --- giá: chỉ đến hết phiên t−1
    Lp, r = L.shift(1), L.diff().shift(1)
    for w in (1, 5, 20, 60):
        d[f"p_r{w}"] = Lp - Lp.shift(w)
    d["p_vol20"] = r.rolling(20).std()
    d["p_ma20"] = Lp - Lp.rolling(20).mean()
    d["p_dd60"] = Lp - Lp.rolling(60).max()
    LB = np.log(b["bds_index"]).shift(1)
    d["p_big_small20"] = (LB - LB.shift(20)) - d["p_r20"]          # vốn hóa lớn so với rổ đều
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet")
    px = px[px["suspended"].eq(0)]
    c = px.pivot(index="date", columns="ticker", values="close").reindex(cal)
    v = px.pivot(index="date", columns="ticker", values="value_bn").reindex(cal)
    ret = np.log(c).diff()
    d["p_breadth1"] = (ret > 0).sum(1).div(ret.notna().sum(1)).shift(1)
    d["p_breadth5"] = d["p_breadth1"].rolling(5).mean()
    d["p_above_ma20"] = (c > c.rolling(20, min_periods=15).mean()).sum(1).div(c.notna().sum(1)).shift(1)
    tv = v.sum(1, min_count=1)
    d["p_turn5_60"] = np.log(tv.rolling(5).mean() / tv.rolling(60, min_periods=40).mean()).shift(1)
    # --- tin: phiên ≤ t
    lab = pd.read_parquet(CLEAN / "article_label.parquet")
    art = (lab.groupby("article_id").agg(score=("impact_score", "mean"), nhom=("nhom_tin", "first"))
           .join(pd.read_parquet(CLEAN / "article_nlp.parquet").set_index("article_id")[["session", "pred_relevance"]], how="inner"))
    art = art[art["session"].isin(cal)]
    g = art.groupby("session")

    def daily(mask=None):
        a = art if mask is None else art[mask]
        gg = a.groupby("session")["score"]
        return pd.DataFrame({"n": gg.size(), "s": gg.sum(), "neg": gg.apply(lambda x: (x < 0).sum()),
                             "pos": gg.apply(lambda x: (x > 0).sum())}).reindex(cal, fill_value=0)

    def roll(D, w):
        return D.rolling(w, min_periods=1).sum()

    D = daily()
    for w in (1, 5, 20):
        R = roll(D, w)
        d[f"n_cnt{w}"] = np.log1p(R["n"])
        d[f"n_tone{w}"] = R["s"] / R["n"].replace(0, np.nan)
        d[f"n_neg{w}"] = R["neg"] / R["n"].replace(0, np.nan)
        d[f"n_pos{w}"] = R["pos"] / R["n"].replace(0, np.nan)
    R60 = roll(D, 60)
    d["n_tone_shift"] = d["n_tone5"] - R60["s"] / R60["n"].replace(0, np.nan)
    d["n_abn_cnt"] = np.log((roll(D, 5)["n"] + 1) / (R60["n"] / 12 + 1))
    for k, grp in GROUPS.items():
        R = roll(daily(art["nhom"].isin(grp)), 5)
        d[f"n_tone5_{k}"] = (R["s"] / R["n"].replace(0, np.nan)).fillna(0)
        d[f"n_share5_{k}"] = R["n"] / roll(D, 5)["n"].replace(0, np.nan)
    R = roll(daily(art["pred_relevance"].ne("QUANG_BA")), 5)
    d["n_tone5_noPR"] = R["s"] / R["n"].replace(0, np.nan)
    tk = pd.read_parquet(CLEAN / "article_ticker_label.parquet")[["article_id", "impact_score", "price_report"]].merge(
        art[["session"]], left_on="article_id", right_index=True)
    T = tk.groupby("session").agg(n=("impact_score", "size"), s=("impact_score", "sum"),
                                  pr=("price_report", "sum")).reindex(cal, fill_value=0)
    T5 = T.rolling(5, min_periods=1).sum()
    d["n_firm_tone5"] = T5["s"] / T5["n"].replace(0, np.nan)
    d["n_firm_cnt5"] = np.log1p(T5["n"])
    d["n_price_report5"] = T5["pr"] / T5["n"].replace(0, np.nan)
    pol = pd.read_parquet(CLEAN / "policy_events.parquet").groupby("date")["n_bai"].sum().reindex(cal, fill_value=0)
    d["n_policy5"] = np.log1p(pol.rolling(5, min_periods=1).sum())
    lg = pd.read_parquet(CLEAN / "legal_streams.parquet")
    risk = lg[lg["stream"].eq("rui_ro_phap_ly")].groupby("session").size().reindex(cal, fill_value=0)
    d["n_legal_risk5"] = np.log1p(risk.rolling(5, min_periods=1).sum())
    d["n_last_session"] = g.size().reindex(cal).notna()          # chỉ để lọc phiên mới nhất có tin
    return d.reset_index()


PRICE = ["p_r1", "p_r5", "p_r20", "p_r60", "p_vol20", "p_ma20", "p_dd60", "p_big_small20", "p_breadth1", "p_breadth5",
         "p_above_ma20", "p_turn5_60"]
NEWS = (["n_cnt1", "n_cnt5", "n_cnt20", "n_tone1", "n_tone5", "n_tone20", "n_neg5", "n_neg20", "n_pos5", "n_tone_shift",
         "n_abn_cnt", "n_tone5_noPR", "n_firm_tone5", "n_firm_cnt5", "n_price_report5", "n_policy5", "n_legal_risk5"]
        + [f"n_{a}5_{k}" for k in GROUPS for a in ("tone", "share")])
FEATSETS = {"ALL": PRICE + NEWS, "PRICE": PRICE, "NEWS": NEWS}


# ---------------------------------------------------------------- mô hình
def weights(dates, end):
    age = (pd.to_datetime(end) - pd.to_datetime(dates)).dt.days.to_numpy() * 250 / 365
    return 0.5 ** (age / HALF_LIFE)


def fit_score(kind, tr, te, feats, w):
    med = tr[feats].median()
    Xtr, Xte = tr[feats].fillna(med).fillna(0), te[feats].fillna(med).fillna(0)
    if kind == "LR":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        return m.fit(Xtr, tr["y"].astype(int), logisticregression__sample_weight=w).predict_proba(Xte)[:, 1]
    if kind == "RF":
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=SEED)
        return m.fit(Xtr, tr["y"].astype(int), sample_weight=w).predict_proba(Xte)[:, 1]
    lo, hi = tr["fwd"].quantile([0.01, 0.99])
    m = LGBMRegressor(objective="huber", alpha=0.02, num_leaves=7, learning_rate=0.03, n_estimators=200,
                      min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                      verbose=-1)
    return m.fit(Xtr, tr["fwd"].clip(lo, hi), sample_weight=w).predict(Xte)


def pct_vs(ref, x):
    return np.searchsorted(np.sort(ref), x, side="right") / len(ref)


def run(d):
    lab = d[d["y"].notna()].reset_index(drop=True)
    new = d[d["y"].isna() & d["n_last_session"] & (d["date"] > lab["date"].max())].copy()
    out = []
    folds = make_folds(H + 1)
    for f in folds:
        tr, va = lab[lab.date.isin(f.train)], lab[lab.date.isin(f.val)].copy()
        fit, te = lab[lab.date.isin(f.fit)], lab[lab.date.isin(f.test)].copy()
        wt, wf = weights(tr.date, tr.date.max()), weights(fit.date, fit.date.max())
        for fs, feats in FEATSETS.items():
            for k in COMPS:
                va[f"{fs}|{k}"] = fit_score(k, tr, va, feats, wt)
                te[f"{fs}|{k}"] = fit_score(k, fit, te, feats, wf)
                if fs == "ALL" and f is folds[-1]:
                    new[f"ALL|{k}"] = fit_score(k, fit, new, feats, wf)
            for x in (va, te) + ((new,) if fs == "ALL" and f is folds[-1] else ()):
                x[f"s|{fs}"] = np.mean([pct_vs(va[f"{fs}|{k}"], x[f"{fs}|{k}"]) for k in COMPS], axis=0)
        sv = va["s|ALL"]
        for x in (te,) + ((new,) if f is folds[-1] else ()):
            for q in TWO_SIDED:
                t = int(q * 100)
                x[f"hi{t}"], x[f"lo{t}"] = x["s|ALL"] >= np.quantile(sv, 1 - q), x["s|ALL"] <= np.quantile(sv, q)
        te["tone_rule"] = te["n_tone5"] > tr["n_tone5"].median()
        te["mom_rule"] = te["p_r20"] > 0
        te["train_rate"] = tr["y"].mean()
        te["fold"], te["quy_test"] = f.k, f.test_q
        out.append(te)
        print(f"fold {f.k} {f.test_q}: train {len(tr)} | val {len(va)} | fit {len(fit)} | test {len(te)}", flush=True)
    return pd.concat(out, ignore_index=True), new


# ---------------------------------------------------------------- đánh giá
def block_boot_auc(res, col, n=2000, block=10):
    """CI 95% của AUC, lấy mẫu lại theo khối 10 phiên liên tiếp (mục tiêu 5 phiên chồng lấn)."""
    rng = np.random.default_rng(SEED)
    y, s, T = res["y"].to_numpy(), res[col].to_numpy(), len(res)
    vals = []
    for _ in range(n):
        st = rng.integers(0, T - block, size=T // block + 1)
        idx = (st[:, None] + np.arange(block)).ravel()[:T]
        if 0 < y[idx].mean() < 1:
            vals.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(vals, [2.5, 97.5])


def evaluate(res):
    y = res["y"].astype(int)
    maj = int(y.mean() >= 0.5)
    rows = [{"cách dự báo": "Mốc: luôn đoán lớp đa số của test (" + ("tăng" if maj else "không lãi") + ")",
             "độ phủ": 1.0, "accuracy": (y == maj).mean()},
            {"cách dự báo": "Mốc: đoán theo lớp đa số của train", "độ phủ": 1.0,
             "accuracy": (y == (res["train_rate"] >= 0.5).astype(int)).mean()}]
    for name, col in (("Quy tắc động lượng: rổ tăng 20 phiên qua → mua", "mom_rule"),
                      ("Quy tắc tin: sắc thái tin 5 phiên > trung vị train → mua", "tone_rule")):
        p = res[col].astype(int)
        rows.append({"cách dự báo": name, "độ phủ": 1.0, "accuracy": accuracy_score(y, p),
                     "balanced acc": balanced_accuracy_score(y, p), "macro-F1": f1_score(y, p, average="macro")})
    for fs, name in (("PRICE", "Mô hình chỉ dùng giá"), ("NEWS", "Mô hình chỉ dùng tin"), ("ALL", "★ MÔ HÌNH NGÀNH (tin + giá)")):
        s = res[f"s|{fs}"]
        p = (s > 0.5).astype(int)
        lo, hi = block_boot_auc(res, f"s|{fs}")
        aq = [roc_auc_score(g["y"], g[f"s|{fs}"]) for _, g in res.groupby("fold")]
        rows.append({"cách dự báo": name + " · mọi phiên", "độ phủ": 1.0, "AUC": roc_auc_score(y, s),
                     "CI95 AUC": f"[{lo:.3f}, {hi:.3f}]", "AUC quý thấp–cao": f"{min(aq):.3f}–{max(aq):.3f}",
                     "số quý AUC > 0,5": sum(a > 0.5 for a in aq), "accuracy": accuracy_score(y, p),
                     "balanced acc": balanced_accuracy_score(y, p), "macro-F1": f1_score(y, p, average="macro")})
    for k in COMPS:
        rows.append({"cách dự báo": f"  thành phần {k} (tin + giá)", "độ phủ": 1.0, "AUC": roc_auc_score(y, res[f"ALL|{k}"])})
    for q in TWO_SIDED:
        t = int(q * 100)
        hi, lo = res[f"hi{t}"].astype(bool), res[f"lo{t}"].astype(bool)
        cov = hi | lo
        pred = hi[cov].astype(int)
        per_q = [accuracy_score(g["y"][g[f"hi{t}"] | g[f"lo{t}"]].astype(int), g[f"hi{t}"][g[f"hi{t}"] | g[f"lo{t}"]].astype(int))
                 for _, g in res.groupby("fold") if (g[f"hi{t}"] | g[f"lo{t}"]).any()]
        rows.append({"cách dự báo": f"★ Tự tin hai phía {t}%: MUA / ĐỨNG NGOÀI / không khuyến nghị", "độ phủ": cov.mean(),
                     "accuracy": accuracy_score(y[cov], pred), "số phiên có khuyến nghị": int(cov.sum()),
                     "precision MUA": y[hi].mean(), "ĐỨNG NGOÀI đúng": 1 - y[lo].mean(),
                     "lợi suất 5 phiên TB khi MUA": res.loc[hi, "fwd"].mean(),
                     "lợi suất 5 phiên TB khi ĐỨNG NGOÀI": res.loc[lo, "fwd"].mean(),
                     "accuracy quý thấp/cao": f"{min(per_q):.3f}/{max(per_q):.3f}"})
    return pd.DataFrame(rows)


def backtest(res):
    """Kiểm tra 5 phiên một lần (không chồng lấn); trung bình 5 điểm xuất phát. Phí 0,4% mỗi lần vào rổ."""
    out = []
    for off in range(H):
        r = res.iloc[off::H]
        for name, hold in (("Giữ rổ BĐS suốt kỳ", pd.Series(True, index=r.index)),
                           ("Mô hình ngành: giữ rổ khi điểm > trung vị val", r["s|ALL"] > 0.5),
                           ("Mô hình ngành: chỉ giữ khi MUA (top 30%)", r["hi30"].astype(bool)),
                           ("Quy tắc động lượng", r["mom_rule"].astype(bool))):
            enter = hold & ~hold.shift(1, fill_value=False)          # phí chỉ khi vào rổ (đang giữ thì không mất phí)
            lr = r.loc[hold, "fwd"].sum() + np.log(1 - COST) * enter.sum()
            out.append({"chiến lược": name, "offset": off, "lợi suất": np.expm1(lr), "tỷ lệ thời gian cầm rổ": hold.mean()})
    return pd.DataFrame(out).groupby("chiến lược", sort=False)[["lợi suất", "tỷ lệ thời gian cầm rổ"]].mean().reset_index()


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q = res.groupby("quy_test").apply(lambda g: pd.Series({k: roc_auc_score(g["y"], g[f"s|{k}"]) for k in FEATSETS}))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(q))
    for i, (k, lab) in enumerate((("ALL", "Tin + giá"), ("PRICE", "Chỉ giá"), ("NEWS", "Chỉ tin"))):
        ax.bar(x + (i - 1) * 0.26, q[k], 0.24, label=lab)
    ax.axhline(0.5, color="grey", lw=1, ls="--")
    ax.set_xticks(x, q.index)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.2, 0.9)
    ax.set_title("Mô hình toàn ngành — AUC theo quý test (rổ BĐS 5 phiên có lãi sau phí)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "r20_sector.png", dpi=150)


def main():
    d = build()
    res, new = run(d)
    ev, bt = evaluate(res), backtest(res)
    qt = res.groupby("quy_test").apply(lambda g: pd.Series({
        "số phiên": len(g), "tỷ lệ phiên rổ có lãi": g["y"].mean(), "lợi suất rổ cả quý": np.expm1(g["fwd"].iloc[::H].sum()),
        "AUC tin+giá": roc_auc_score(g["y"], g["s|ALL"]), "AUC chỉ giá": roc_auc_score(g["y"], g["s|PRICE"]),
        "AUC chỉ tin": roc_auc_score(g["y"], g["s|NEWS"]),
        "accuracy hai phía 30%": accuracy_score(g["y"][g["hi30"] | g["lo30"]].astype(int), g["hi30"][g["hi30"] | g["lo30"]].astype(int))
        if (g["hi30"] | g["lo30"]).any() else np.nan})).reset_index()
    new["khuyến nghị"] = np.where(new["hi20"], "MUA rổ (tự tin cao)", np.where(new["hi30"], "MUA rổ",
                         np.where(new["lo20"], "ĐỨNG NGOÀI (tự tin cao)", np.where(new["lo30"], "ĐỨNG NGOÀI", "Không khuyến nghị"))))
    new["điểm 0–100"] = (100 * new["s|ALL"]).round(0)
    latest = new[["date", "điểm 0–100", "khuyến nghị", "n_cnt1", "n_tone1", "n_tone5", "p_r5", "p_r20"]]
    PRED.mkdir(parents=True, exist_ok=True)
    res.to_parquet(PRED / "sector.parquet", index=False)
    for name, t in (("sector_ket_qua", ev), ("sector_quy", qt), ("sector_backtest", bt), ("sector_moi_nhat", latest)):
        t.to_csv(REP / f"{name}.csv", index=False, encoding="utf-8-sig")
    (REP / "sector_ket_qua.md").write_text("\n\n".join(t.round(3).to_markdown(index=False) for t in (ev, qt, bt, latest)),
                                           encoding="utf-8")
    figure(res)
    with pd.option_context("display.width", 320, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        for t in (ev, qt, bt, latest):
            print(t.to_string(index=False), "\n")
    print("n test:", len(res), "| tỷ lệ gốc:", round(res["y"].mean(), 3), "| test:", res.date.min().date(), "→", res.date.max().date())


if __name__ == "__main__":
    main()
