"""TINH CHỈNH RIÊNG cho mô hình TOÀN NGÀNH (không theo cấu hình mô hình chính). Chốt lưới + quy tắc chọn trước khi chạy (01/10/2026).

Tính chất của bài toán ngành → hướng tinh chỉnh:
  • Ít mẫu (180–550 phiên train)          → mô hình đơn giản, điều chuẩn mạnh, bộ đặc trưng gọn.
  • Thị trường theo "chế độ" kéo dài      → trí nhớ ngắn: trọng số thời gian với chu kỳ bán rã 60/125/250 phiên hoặc không (0).
  • Xu hướng ngành đi theo tháng          → thử kỳ nắm giữ H = 5, 10, 20 phiên (cùng phí 0,4%).
Lưới (240 cấu hình) = 4 bộ đặc trưng × 5 mô hình × 4 trọng số thời gian × 3 kỳ nắm giữ
  đặc trưng : PRICE (12), NEWS (29), GON (14 biến chậm: động lượng, độ rộng, drawdown + sắc thái tin 5/20 phiên), ALL (41)
  mô hình   : LR C ∈ {0,01; 0,1; 1}, RF (300 cây, ≥ 30 mẫu/lá), LightGBM phân loại rất nông (4 lá, 150 vòng)
CHỌN CẤU HÌNH (walk-forward lồng nhau, KHÔNG dùng quý test):
  Với quý test k, chấm mọi cấu hình trên các "quý val trong" 2024Q4 … (quý ngay trước k); mỗi quý val trong
  được dự báo bởi mô hình train trên dữ liệu trước nó (purge H+1 + embargo 5). Tiêu chí: AUC trung bình các quý val trong.
  Mô hình chính thức (★) = trung bình phân vị của 5 cấu hình tốt nhất trong kỳ nắm giữ H tốt nhất (ổn định hơn 1 cấu hình).
  Đối chứng: cấu hình tốt nhất đơn lẻ; mô hình ngành bản đầu (run_sector); quy tắc động lượng; lớp đa số.
Lưu ý trung thực: bản đầu của mô hình ngành đã được chấm trên cùng các quý test. Lưới trên được đặt ra từ tính chất bài
toán, việc chọn chỉ dựa vào val trong; nhưng không còn tập giữ lại nào hoàn toàn "chưa nhìn".
Đầu ra: outputs/reports/sector_tune_*.csv, outputs/figures/r22_sector_tuned_100tr.png
Dùng  : python -m src.sector.tune_sector
"""
import itertools

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CLEAN, ROOT
from src.eval.walk_forward import EMBARGO, FIRST_SAMPLE, LAST_NEWS, TEST_QUARTERS
from src.sector.run_sector import BASKET, COST, NEWS, PRICE, SEED, build

REP, FIG = ROOT / "outputs" / "reports", ROOT / "outputs" / "figures"
VON = 100_000_000
GON = ["p_r20", "p_r60", "p_dd60", "p_vol20", "p_breadth5", "p_above_ma20", "p_turn5_60", "p_big_small20",
       "n_tone5", "n_tone20", "n_tone_shift", "n_neg20", "n_firm_tone5", "n_legal_risk5"]
FEATS = {"PRICE": PRICE, "NEWS": NEWS, "GON": GON, "ALL": PRICE + NEWS}
MODELS = ["LR0.01", "LR0.1", "LR1", "RF", "LGB"]
HLS = [60, 125, 250, 0]            # 0 = không dùng trọng số thời gian
HS = [5, 10, 20]
INNER_FIRST = pd.Period("2024Q4", "Q")
TOPK = 5


def fit_predict(model, tr, te, feats, y, hl):
    med = tr[feats].median()
    Xtr, Xte = tr[feats].fillna(med).fillna(0), te[feats].fillna(med).fillna(0)
    yt = tr[y].astype(int)
    if yt.nunique() < 2:
        return np.full(len(te), 0.5)
    w = np.ones(len(tr)) if not hl else 0.5 ** ((tr["pos"].max() - tr["pos"]).to_numpy() / hl)
    if model.startswith("LR"):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=float(model[2:]), max_iter=3000))
        return m.fit(Xtr, yt, logisticregression__sample_weight=w).predict_proba(Xte)[:, 1]
    if model == "RF":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=30, max_features="sqrt", n_jobs=-1, random_state=SEED)
        return m.fit(Xtr, yt, sample_weight=w).predict_proba(Xte)[:, 1]
    m = LGBMClassifier(num_leaves=4, n_estimators=150, learning_rate=0.03, min_child_samples=30, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    return m.fit(Xtr, yt, sample_weight=w).predict_proba(Xte)[:, 1]


def pct_vs(ref, x):
    return np.searchsorted(np.sort(ref), x, side="right") / len(ref)


def main():
    d = build().drop(columns=["fwd", "y"])
    cal = pd.DatetimeIndex(pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"])
    L = np.log(pd.read_parquet(CLEAN / "benchmark_nganh.parquet").query("method == @BASKET").set_index("date")["value"]
               .reindex(cal))
    for H in HS:
        d[f"fwd{H}"] = (L.shift(-H) - L).to_numpy()
        d[f"y{H}"] = (d[f"fwd{H}"] > np.log(1 + COST)).astype(float).where(d[f"fwd{H}"].notna())
    d["pos"] = np.arange(len(d))
    d["q"] = d["date"].dt.to_period("Q")
    d = d[d["date"] >= FIRST_SAMPLE].reset_index(drop=True)
    pos0 = {q: d.loc[d["q"] == q, "pos"].min() for q in d["q"].unique()}

    def train_before(q, H):
        return d[(d["pos"] + H + 1 + EMBARGO < pos0[q]) & d[f"y{H}"].notna()]

    tests = [pd.Period(t, "Q") for t in TEST_QUARTERS]
    inner_qs = [q for q in sorted(d["q"].unique()) if INNER_FIRST <= q < tests[-1]]
    configs = list(itertools.product(FEATS, MODELS, HLS, HS))
    # ---------- 1) chấm mọi cấu hình trên từng quý val trong (mỗi quý chỉ tính một lần, dùng lại cho mọi fold)
    inner = {}   # (cfg, q) -> (auc, scores)
    for i, cfg in enumerate(configs):
        fs, mdl, hl, H = cfg
        for q in inner_qs:
            tr, va = train_before(q, H), d[(d["q"] == q) & d[f"y{H}"].notna()]
            s = fit_predict(mdl, tr, va, FEATS[fs], f"y{H}", hl)
            auc = roc_auc_score(va[f"y{H}"], s) if va[f"y{H}"].nunique() == 2 else np.nan
            inner[(cfg, q)] = (auc, s)
        if i % 40 == 39:
            print(f"đã chấm {i + 1}/{len(configs)} cấu hình", flush=True)
    # ---------- 2) mỗi quý test: chọn bằng val trong, fit lại, dự báo test
    out, picks = [], []
    # phiên mới: sau phiên cuối có kết quả 5 phiên, đến hết lịch tin (cùng tập phiên với khuyến nghị tin mới nhất)
    new = d[(d["date"] > d.loc[d["y5"].notna(), "date"].max()) & (d["date"] <= LAST_NEWS + pd.Timedelta(days=1))].copy()
    for k, tq in enumerate(tests, 1):
        qs = [q for q in inner_qs if q < tq]
        score = pd.Series({cfg: np.nanmean([inner[(cfg, q)][0] for q in qs]) for cfg in configs})
        H = score.groupby(level=3).apply(lambda s: s.sort_values(ascending=False).head(TOPK).mean()).idxmax()
        top = score[[c for c in configs if c[3] == H]].sort_values(ascending=False).head(TOPK)
        best1 = score.idxmax()
        te = d[(d["q"] == tq) & (d["date"] <= LAST_NEWS)].copy()
        te = te[te[f"y{H}"].notna()] if tq != tests[-1] else te[te[f"y{H}"].notna()]
        ref_q = tq - 1
        pcts, pcts_new = [], []
        for cfg in top.index:
            fs, mdl, hl, _ = cfg
            fit = train_before(tq, H)
            s = fit_predict(mdl, fit, te, FEATS[fs], f"y{H}", hl)
            pcts.append(pct_vs(inner[(cfg, ref_q)][1], s))
            if k == len(tests):
                pcts_new.append(pct_vs(inner[(cfg, ref_q)][1], fit_predict(mdl, fit, new, FEATS[fs], f"y{H}", hl)))
        te["score"] = np.mean(pcts, axis=0)
        # ngưỡng hai phía từ phân phối điểm ensemble trên quý val ngay trước (không nhìn test)
        ref = np.mean([pct_vs(inner[(c, ref_q)][1], inner[(c, ref_q)][1]) for c in top.index], axis=0)
        te["hi"], te["lo"] = te["score"] >= np.quantile(ref, 0.70), te["score"] <= np.quantile(ref, 0.30)
        fs1, m1, hl1, H1 = best1
        tb = d[(d["q"] == tq) & d[f"y{H1}"].notna()]
        s1 = fit_predict(m1, train_before(tq, H1), tb, FEATS[fs1], f"y{H1}", hl1)
        te["H"], te["y"], te["fwd"] = H, te[f"y{H}"], te[f"fwd{H}"]
        te["fold"], te["quy_test"] = k, str(tq)
        out.append(te)
        if k == len(tests):
            new["score"] = np.mean(pcts_new, axis=0)
            new["hi"], new["lo"] = new["score"] >= np.quantile(ref, 0.70), new["score"] <= np.quantile(ref, 0.30)
            new["H"] = H
        picks.append({"quý test": str(tq), "số quý val trong": len(qs), "kỳ nắm giữ H chọn": H,
                      "5 cấu hình (đặc trưng · mô hình · bán rã)": "; ".join(f"{c[0]}·{c[1]}·{c[2]}" for c in top.index),
                      "AUC val trong TB (top 5)": top.mean(),
                      "AUC test ★ (top 5)": roc_auc_score(te["y"], te["score"]),
                      "cấu hình tốt nhất đơn lẻ": f"{fs1}·{m1}·{hl1}·H{H1}", "AUC val trong (đơn lẻ)": score.max(),
                      "AUC test (đơn lẻ)": roc_auc_score(tb[f"y{H1}"], s1),
                      "tỷ lệ phiên có lãi (test)": te["y"].mean()})
        print(f"{tq}: H={H} | val trong {top.mean():.3f} | test ★ {picks[-1]['AUC test ★ (top 5)']:.3f}", flush=True)
    res = pd.concat(out, ignore_index=True)
    pk = pd.DataFrame(picks)
    # ---------- 3) tổng hợp
    y = res["y"].astype(int)
    first = pd.read_parquet(ROOT / "outputs" / "predictions" / "sector.parquet")
    cov = res["hi"] | res["lo"]
    mom = (res["p_r20"] > 0).astype(int)
    summ = pd.DataFrame([
        {"cách dự báo": "Mốc: luôn đoán lớp đa số của test", "AUC": np.nan, "accuracy": max(y.mean(), 1 - y.mean()), "độ phủ": 1.0},
        {"cách dự báo": "Quy tắc động lượng (r20 > 0), cùng mục tiêu H đã chọn", "AUC": np.nan,
         "accuracy": accuracy_score(y, mom), "balanced acc": balanced_accuracy_score(y, mom), "độ phủ": 1.0},
        {"cách dự báo": "Mô hình ngành bản đầu (H = 5, cấu hình mô hình chính)", "AUC": roc_auc_score(first["y"], first["s|ALL"]),
         "accuracy": accuracy_score(first["y"], first["s|ALL"] > 0.5), "độ phủ": 1.0},
        {"cách dự báo": "★ Mô hình ngành tinh chỉnh (top 5 cấu hình) · mọi phiên", "AUC": roc_auc_score(y, res["score"]),
         "accuracy": accuracy_score(y, res["score"] > 0.5), "balanced acc": balanced_accuracy_score(y, res["score"] > 0.5),
         "độ phủ": 1.0, "số quý AUC > 0,5": int((pk["AUC test ★ (top 5)"] > 0.5).sum())},
        {"cách dự báo": "★ Tinh chỉnh · tự tin hai phía 30% (MUA / ĐỨNG NGOÀI)", "AUC": np.nan,
         "accuracy": accuracy_score(y[cov], res.loc[cov, "hi"].astype(int)), "độ phủ": cov.mean(),
         "precision MUA": y[res["hi"]].mean(), "ĐỨNG NGOÀI đúng": 1 - y[res["lo"]].mean()},
    ])
    # ---------- 4) demo 100 triệu: mỗi lần quyết định giữ rổ H phiên (H của quý đó) hoặc giữ tiền H phiên
    res = res.sort_values("date").reset_index(drop=True)

    def sim(rule, off):
        i, v, holding, path, n_in, wins, periods = off, 1.0, False, [], 0, 0, 0
        while i < len(res):
            r = res.iloc[i]
            h = bool(rule(r))
            if h:
                if not holding:
                    v *= 1 - COST; n_in += 1
                v *= np.exp(r["fwd"]); periods += 1; wins += r["fwd"] > np.log(1 + COST)
            holding = h
            path.append((r["date"], v))
            i += int(r["H"])
        return v, path, n_in, periods, wins

    rules = {"Giữ rổ BĐS suốt kỳ": lambda r: True,
             "★ Tinh chỉnh: giữ khi điểm > trung vị": lambda r: r["score"] > 0.5,
             "★ Tinh chỉnh: chỉ giữ khi MUA (top 30%)": lambda r: r["hi"],
             "Quy tắc động lượng (rổ tăng 20 phiên)": lambda r: r["p_r20"] > 0}
    demo, curves = [], {}
    for name, rule in rules.items():
        runs = [sim(rule, off) for off in range(5)]
        v0, path, n_in, periods, wins = runs[0]
        eq = pd.Series([VON] + [VON * p[1] for p in path])
        demo.append({"chiến lược": name, "vốn cuối (triệu)": VON * v0 / 1e6, "lợi suất": v0 - 1, "số lần vào rổ": n_in,
                     "kỳ cầm rổ": periods, "kỳ cầm rổ có lãi": wins / periods if periods else np.nan,
                     "sụt giảm lớn nhất": (eq / eq.cummax() - 1).min(),
                     "lợi suất 5 cách bắt đầu (thấp–cao)": f"{min(r[0] for r in runs) - 1:+.1%} → {max(r[0] for r in runs) - 1:+.1%}"})
        curves[name] = path
    demo = pd.DataFrame(demo)
    new["khuyến nghị"] = np.where(new["hi"], "MUA rổ", np.where(new["lo"], "ĐỨNG NGOÀI", "Không khuyến nghị"))
    new["điểm 0–100"] = (100 * new["score"]).round(0)
    latest = new[["date", "H", "điểm 0–100", "khuyến nghị"]].rename(columns={"H": "kỳ nắm giữ (phiên)"})
    for name, t in (("sector_tune_chon", pk), ("sector_tune_ket_qua", summ), ("sector_tune_demo_100tr", demo),
                    ("sector_tune_moi_nhat", latest)):
        t.to_csv(REP / f"{name}.csv", index=False, encoding="utf-8-sig")
    res.to_parquet(ROOT / "outputs" / "predictions" / "sector_tuned.parquet", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, path in curves.items():
        xs = [res["date"].iloc[0]] + [p[0] for p in path]
        ax.step(xs, [100] + [100 * p[1] for p in path], where="post", lw=2 if "★" in name else 1.3, label=name)
    ax.axhline(100, color="grey", lw=1, ls="--")
    ax.set_ylabel("Vốn (triệu đồng)")
    ax.set_title("Demo 100 triệu — mô hình ngành đã tinh chỉnh, 04/2025 → 09/2026 (phí 0,4%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "r22_sector_tuned_100tr.png", dpi=150)
    with pd.option_context("display.width", 320, "display.max_columns", 30, "display.max_colwidth", 90,
                           "display.float_format", "{:.3f}".format):
        for t in (pk, summ, demo, latest):
            print(t.to_string(index=False), "\n")


if __name__ == "__main__":
    main()
