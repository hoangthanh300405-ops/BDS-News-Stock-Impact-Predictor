"""Nhà đầu tư nhỏ lẻ — VÒNG 3 (vòng cải tiến cuối). Cấu hình chính thức CHỐT TRƯỚC khi chạy: E3 (kết hợp).

Ứng viên (mọi ngưỡng lấy từ val; không chọn gì theo test):
  ref-LGBM  LightGBM đặc trưng v1 (mô hình chính vòng 1)
  LR3       Logistic, đặc trưng v3 (= v2 + hồ sơ doanh nghiệp + nhiệt độ tin tức trong phiên)
  RF3       Random Forest, đặc trưng v3
  REG3      HỌC ĐỂ XẾP HẠNG: LightGBM hồi quy (Huber) dự đoán MỨC lãi (winsorize 1–99%), rồi xếp hạng
  E3 ★      Kết hợp: trung bình PHÂN VỊ (so với phân phối trên val) của LR3, RF3, REG3
  CONS      Đồng thuận: chỉ làm theo khi LR3, RF3 và REG3 cùng xếp tin vào top q
  CONS-AB   (target tuyệt đối) chỉ làm theo khi E3 của cả hai target (có lãi & thắng rổ) cùng xếp vào top q
Đầu ra: outputs/reports/retail_v3.csv (+ .md), retail_v3_quarters.csv
Dùng:  python -m src.retail.run_retail_v3
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.config import CLEAN, ROOT
from src.eval.walk_forward import make_folds
from src.retail.events import FEATURES, FEATURES_V3, H
from src.retail.run_retail import SEED, TARGETS, block_boot, cls_metrics, lgbm, logreg, prep

REP = ROOT / "outputs" / "reports"
QS = (0.1, 0.2, 0.3)
OFFICIAL = "E3"


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=SEED)


def reg():
    return LGBMRegressor(objective="huber", alpha=0.02, num_leaves=15, learning_rate=0.03, n_estimators=300,
                         min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
                         random_state=SEED, verbose=-1)


def fit_score(kind, tr, te, y, cont):
    feats = FEATURES if kind == "ref-LGBM" else FEATURES_V3
    med = tr[feats].replace([np.inf, -np.inf], np.nan).median()
    Xtr, Xte = prep(tr, feats, med), prep(te, feats, med)
    if kind == "REG3":
        lo, hi = tr[cont].quantile([0.01, 0.99])
        return reg().fit(Xtr, tr[cont].clip(lo, hi)).predict(Xte)
    make = {"ref-LGBM": lgbm, "LR3": logreg, "RF3": rf}[kind]
    return make().fit(Xtr, tr[y].astype(int)).predict_proba(Xte)[:, 1]


def pct_vs(ref, x):
    """Phân vị của x so với phân phối ref (val) — đưa điểm của các mô hình về cùng thang."""
    r = np.sort(ref)
    return np.searchsorted(r, x, side="right") / len(r)


def run(ev, y, cont):
    ev = ev[ev[y].notna()].copy()
    out = []
    base_models = ["ref-LGBM", "LR3", "RF3", "REG3"]
    for f in make_folds(H + 1):
        tr, va = ev[ev["date"].isin(f.train)], ev[ev["date"].isin(f.val)].copy()
        fit, te = ev[ev["date"].isin(f.fit)], ev[ev["date"].isin(f.test)].copy()
        for k in base_models:
            va[f"s|{k}"] = fit_score(k, tr, va, y, cont)
            te[f"s|{k}"] = fit_score(k, fit, te, y, cont)
        # E3: trung bình phân vị (so với val) của LR3, RF3, REG3
        for d in (va, te):
            d["s|E3"] = np.mean([pct_vs(va[f"s|{k}"], d[f"s|{k}"]) for k in ("LR3", "RF3", "REG3")], axis=0)
        for k in base_models + ["E3"]:
            for q in QS:
                thr = np.quantile(va[f"s|{k}"], 1 - q)
                te[f"top{int(q * 100)}|{k}"] = te[f"s|{k}"] >= thr
                va[f"top{int(q * 100)}|{k}"] = va[f"s|{k}"] >= thr
        for q in QS:
            t = int(q * 100)
            te[f"top{t}|CONS"] = te[[f"top{t}|{k}" for k in ("LR3", "RF3", "REG3")]].all(axis=1)
        te["fold"], te["quy_test"] = f.k, f.test_q
        out.append(te)
    return pd.concat(out, ignore_index=True)


def table(res, key, y, pnl, cands):
    rows = []
    A = res[y].mean()
    for k in cands:
        auc = roc_auc_score(res[y], res[f"s|{k}"]) if f"s|{k}" in res else np.nan
        for q in QS:
            dec = res[f"top{int(q * 100)}|{k}"].astype(bool)
            if dec.sum() < 30:
                continue
            ci = block_boot(res["date"], res[y].where(dec), res[y])
            wins = sum(res.loc[dec & (res["fold"] == j), y].mean() > res.loc[res["fold"] == j, y].mean()
                       for j in res["fold"].unique())
            rows.append({"target": key, "mô hình": ("★ " if k == OFFICIAL else "") + k, "quy tắc": f"top {q:.0%}",
                         "AUC": auc, "khuyên làm theo": dec.mean(), **cls_metrics(res[y], dec),
                         "lãi TB/lệnh": res.loc[dec, pnl].mean(), "precision − A": res.loc[dec, y].mean() - A,
                         "CI95 (precision − A)": f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", "số quý thắng A": int(wins)})
    return rows


def main():
    ev = pd.read_parquet(CLEAN / "retail_events.parquet")
    conts = {"tuyet_doi": f"pnl_{H}", "tuong_doi": "rel_pnl"}
    res_by, rows = {}, []
    for key, (y, pnl) in TARGETS.items():
        res_by[key] = run(ev, y, conts[key])
        print(f"xong {key}")
    # đồng thuận giữa hai target (cho target tuyệt đối): E3 có lãi & E3 thắng rổ cùng top q
    a, r = res_by["tuyet_doi"], res_by["tuong_doi"].set_index(["date", "ticker"])
    ix = pd.MultiIndex.from_frame(a[["date", "ticker"]])
    for q in QS:
        t = int(q * 100)
        a[f"top{t}|CONS-AB"] = a[f"top{t}|E3"] & r[f"top{t}|E3"].reindex(ix).fillna(False).to_numpy()
    pred_dir = ROOT / "outputs" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for key, r_ in res_by.items():                      # dự đoán test (dùng cho mô phỏng danh mục, dashboard)
        keep = ["date", "ticker", "quy_test", "fold", "d", "title", "event_type", "entry_shift", f"ret_{H}", f"pnl_{H}",
                "rel_pnl", "follow_ok", "follow_rel", "mkt_ret_H", "d_r5", "frac_comm", "frac_firm"]
        keep += [c for c in r_ if c.startswith(("s|", "top"))]
        r_[keep].to_parquet(pred_dir / f"retail_v3_{key}.parquet", index=False)
    for key, (y, pnl) in TARGETS.items():
        cands = ["ref-LGBM", "LR3", "RF3", "REG3", "E3", "CONS"] + (["CONS-AB"] if key == "tuyet_doi" else [])
        rows += table(res_by[key], key, y, pnl, cands)
    out = pd.DataFrame(rows)
    out.to_csv(REP / "retail_v3.csv", index=False, encoding="utf-8-sig")
    (REP / "retail_v3.md").write_text(out.round(3).to_markdown(index=False), encoding="utf-8")
    pq = []
    for key, (y, _) in TARGETS.items():
        g = res_by[key]
        for qt, x in g.groupby("quy_test"):
            dec = x[f"top30|{OFFICIAL}"].astype(bool)
            pq.append({"target": key, "quý": qt, "tỷ lệ gốc": x[y].mean(), "AUC E3": roc_auc_score(x[y], x[f"s|{OFFICIAL}"]),
                       "precision top 30%": x.loc[dec, y].mean(), **{k: v for k, v in cls_metrics(x[y], dec).items()
                                                                   if k in ("accuracy", "macro-F1")}})
    pd.DataFrame(pq).to_csv(REP / "retail_v3_quarters.csv", index=False, encoding="utf-8-sig")
    cols = ["target", "mô hình", "quy tắc", "AUC", "khuyên làm theo", "accuracy", "balanced acc", "precision (làm theo)",
            "recall (làm theo)", "F1 (làm theo)", "macro-F1", "lãi TB/lệnh", "CI95 (precision − A)", "số quý thắng A"]
    with pd.option_context("display.width", 300, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(out[cols].to_string(index=False)); print(); print(pd.DataFrame(pq).to_string(index=False))


if __name__ == "__main__":
    main()
