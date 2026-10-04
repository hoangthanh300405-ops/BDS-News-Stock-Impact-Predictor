"""MÔ HÌNH CUỐI CÙNG cho nhà đầu tư nhỏ lẻ (FINAL). Chốt trước khi chạy (30/09/2026).

Mô hình: E3 có trọng số thời gian = trung bình PHÂN VỊ (so với val) của LR, Random Forest, LightGBM hồi quy Huber
(mức lãi), đặc trưng v3, trọng số 0,5^(tuổi/250 phiên). Đây là E3 (vòng 3) + thay đổi duy nhất có ích ở vòng 4.
Ba chế độ khuyến nghị (ngưỡng đều lấy từ phân phối điểm trên VAL, không nhìn test):
  • Tiêu chuẩn  : khuyên LÀM THEO khi điểm ∈ top 10%.
  • Thận trọng  : khuyên LÀM THEO khi cả 3 mô hình cùng xếp tin vào top 20%.
  • Tự tin hai phía (q = 10/15/20%): top q → "làm theo", bottom q → "đừng làm theo", còn lại KHÔNG khuyến nghị.
    Chỉ số: accuracy trên các tin có khuyến nghị + độ phủ (dự đoán có chọn lọc, Geifman & El-Yaniv 2017).
Mốc: luôn làm theo (A); không bao giờ làm theo; E3 không trọng số (vòng 3).
Đầu ra: outputs/reports/final_retail.csv, final_retail_quarters.csv; outputs/predictions/final_{target}.parquet
Dùng:  python -m src.retail.run_final
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.config import CLEAN, ROOT
from src.eval.walk_forward import make_folds
from src.retail.events import FEATURES_V3, H
from src.retail.run_retail import TARGETS, block_boot, cls_metrics
from src.retail.run_retail_v4 import CONTS, fit_score, pct_vs, weights

REP, PRED = ROOT / "outputs" / "reports", ROOT / "outputs" / "predictions"
COMPS = ["LR", "RF", "REG"]
TWO_SIDED = (0.10, 0.15, 0.20)


def run(ev, y, cont):
    ev = ev[ev[y].notna()].reset_index(drop=True)
    out = []
    for f in make_folds(H + 1):
        tr, va = ev[ev["date"].isin(f.train)], ev[ev["date"].isin(f.val)].copy()
        fit, te = ev[ev["date"].isin(f.fit)], ev[ev["date"].isin(f.test)].copy()
        for pre, (wt, wf) in {"": (weights(tr["date"], tr["date"].max()), weights(fit["date"], fit["date"].max())),
                              "nw_": (np.ones(len(tr)), np.ones(len(fit)))}.items():
            for k in COMPS:
                va[f"s|{pre}{k}"] = fit_score(k, tr, va, y, cont, FEATURES_V3, wt)
                te[f"s|{pre}{k}"] = fit_score(k, fit, te, y, cont, FEATURES_V3, wf)
            for d in (va, te):
                d[f"s|{pre}E"] = np.mean([pct_vs(va[f"s|{pre}{k}"], d[f"s|{pre}{k}"]) for k in COMPS], axis=0)
        s, sv = te["s|E"], va["s|E"]
        te["std"] = s >= np.quantile(sv, 0.90)
        te["std_nw"] = te["s|nw_E"] >= np.quantile(va["s|nw_E"], 0.90)
        te["cons"] = np.all([te[f"s|{k}"] >= np.quantile(va[f"s|{k}"], 0.80) for k in COMPS], axis=0)
        for q in TWO_SIDED:
            t = int(q * 100)
            te[f"hi{t}"], te[f"lo{t}"] = s >= np.quantile(sv, 1 - q), s <= np.quantile(sv, q)
        # tên cột tương thích với portfolio / robustness / dashboard (điểm = mô hình FINAL)
        te["s|E3"], te["top10|E3"], te["top20|CONS"] = s, te["std"], te["cons"]
        te["top20|E3"] = s >= np.quantile(sv, 0.80)
        te["fold"], te["quy_test"] = f.k, f.test_q
        out.append(te)
    return pd.concat(out, ignore_index=True)


def summarize(res, key, y, pnl):
    rows, A = [], res[y].mean()
    yy = res[y].astype(int)
    base = {"target": key, "n test": len(res), "tỷ lệ gốc": A}
    rows.append({**base, "chế độ": "Mốc: luôn làm theo (A)", "độ phủ": 1.0, "accuracy trên tin có khuyến nghị": A,
                 "precision (làm theo)": A, "lãi TB khi làm theo": res[pnl].mean()})
    rows.append({**base, "chế độ": "Mốc: không bao giờ làm theo", "độ phủ": 1.0, "accuracy trên tin có khuyến nghị": 1 - A})
    for name, col, sc in (("E3 vòng 3 · top 10% (đối chứng)", "std_nw", "s|nw_E"), ("★ Tiêu chuẩn · top 10%", "std", "s|E"),
                          ("★ Thận trọng · 3 mô hình đồng thuận top 20%", "cons", "s|E")):
        dec = res[col].astype(bool)
        ci = block_boot(res["date"], res[y].where(dec), res[y])
        wins = sum(res.loc[dec & (res["fold"] == j), y].mean() > res.loc[res["fold"] == j, y].mean() for j in res["fold"].unique())
        rows.append({**base, "chế độ": name, "AUC (toàn bộ tin)": roc_auc_score(yy, res[sc]), "độ phủ": dec.mean(),
                     "accuracy trên tin có khuyến nghị": res.loc[dec, y].mean(), "precision (làm theo)": res.loc[dec, y].mean(),
                     "lãi TB khi làm theo": res.loc[dec, pnl].mean(), "CI95 (precision − A)": f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                     "số quý thắng A": int(wins), **{f"toàn bộ tin · {k}": v for k, v in cls_metrics(yy, dec).items()
                                                     if k in ("accuracy", "macro-F1")}})
    for q in TWO_SIDED:
        t = int(q * 100)
        hi, lo = res[f"hi{t}"].astype(bool), res[f"lo{t}"].astype(bool)
        cov = hi | lo
        pred = np.where(hi, 1, 0)[cov]
        acc = accuracy_score(yy[cov], pred)
        # KTC cho accuracy trên phần có khuyến nghị: bootstrap khối phiên, so với "đoán lớp đa số" trên cùng phần đó
        correct = pd.Series(np.where(cov, (np.where(hi, 1, 0) == yy).astype(float), np.nan))
        maj = pd.Series(np.where(cov, (yy == int(A >= 0.5)).astype(float), np.nan))
        ci = block_boot(res["date"], correct, maj)
        per_q = [accuracy_score(g[y].astype(int)[g[f"hi{t}"] | g[f"lo{t}"]],
                                np.where(g[f"hi{t}"], 1, 0)[(g[f"hi{t}"] | g[f"lo{t}"]).to_numpy()]) for _, g in res.groupby("fold")]
        rows.append({**base, "chế độ": f"★ Tự tin hai phía · top/bottom {t}%", "độ phủ": cov.mean(),
                     "accuracy trên tin có khuyến nghị": acc, "precision (làm theo)": yy[hi].mean(),
                     "NPV (đừng làm theo đúng)": 1 - yy[lo].mean(), "lãi TB khi làm theo": res.loc[hi, pnl].mean(),
                     "lãi TB nếu làm theo các tin bị khuyên tránh": res.loc[lo, pnl].mean(),
                     "CI95 (accuracy − đoán lớp đa số)": f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                     "accuracy quý thấp/cao": f"{min(per_q):.3f}/{max(per_q):.3f}",
                     "số quý ≥ 55%": int(sum(a >= 0.55 for a in per_q))})
    return rows


def main():
    ev = pd.read_parquet(CLEAN / "retail_events.parquet")
    PRED.mkdir(parents=True, exist_ok=True)
    rows, pq = [], []
    for key, (y, pnl) in TARGETS.items():
        res = run(ev, y, CONTS[key])
        rows += summarize(res, key, y, pnl)
        keep = ["date", "ticker", "quy_test", "fold", "d", "title", "event_type", "entry_shift", f"ret_{H}", f"pnl_{H}",
                "rel_pnl", "follow_ok", "follow_rel", "mkt_ret_H", "d_r5", "frac_comm", "frac_firm"]
        res[keep + [c for c in res if c.startswith("s|") or c in ("std", "std_nw", "cons", "top10|E3", "top20|E3", "top20|CONS") or c[:2] in ("hi", "lo")]].to_parquet(
            PRED / f"final_{key}.parquet", index=False)
        for qt, g in res.groupby("quy_test"):
            cov = g["hi15"] | g["lo15"]
            pq.append({"target": key, "quý": qt, "tỷ lệ gốc": g[y].mean(), "AUC": roc_auc_score(g[y], g["s|E"]),
                       "precision tiêu chuẩn": g.loc[g["std"], y].mean(), "precision thận trọng": g.loc[g["cons"], y].mean(),
                       "accuracy hai phía 15%": accuracy_score(g[y].astype(int)[cov], np.where(g["hi15"], 1, 0)[cov.to_numpy()])})
        print(f"xong {key}")
    out = pd.DataFrame(rows)
    out.to_csv(REP / "final_retail.csv", index=False, encoding="utf-8-sig")
    (REP / "final_retail.md").write_text(out.round(3).to_markdown(index=False), encoding="utf-8")
    pd.DataFrame(pq).to_csv(REP / "final_retail_quarters.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 300, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False)); print(pd.DataFrame(pq).to_string(index=False))


if __name__ == "__main__":
    main()
