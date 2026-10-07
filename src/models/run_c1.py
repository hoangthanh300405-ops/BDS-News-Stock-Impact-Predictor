"""C1 — dự báo ĐI TIẾP / ĐẢO CHIỀU sau biến động mạnh: mô hình "giá" so với "giá + TIN" (ngoài mẫu).

Cùng quy trình như mọi thí nghiệm: walk-forward 6 fold (purge + embargo), chọn siêu tham số trên validation,
tập test chỉ để báo cáo. Kiểm định: ΔAUC có KTC bootstrap khối theo phiên, DM (HLN) trên log-loss theo phiên.
Kèm "lợi suất theo chiều biến động" của nhóm mô hình dự báo đi tiếp vs đảo chiều (ý nghĩa kinh tế).
Dùng:  python -m src.models.run_c1
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import ROOT
from src.eval.metrics import BINARY, block_bootstrap_diff, dm_test, row_logloss
from src.features.c1_features import C1_NEWS, C1_PRICE, build_c1
from src.models.engine import LGBM, LogReg, run

OUT = ROOT / "outputs"


def descriptive(d):
    rows = []
    for h in (1, 5):
        for g, x in d.dropna(subset=[f"cont_{h}"]).groupby("nhom"):
            r = x[f"cont_ret_{h}"] * 100
            by_day = r.groupby(x["date"]).mean()
            rows.append({"h": h, "nhóm": g, "n": len(x), "P(đi tiếp)": x[f"cont_{h}"].mean(),
                         "lợi suất theo chiều biến động (%)": r.mean(),
                         "t (gom cụm theo phiên)": by_day.mean() / (by_day.std() / np.sqrt(len(by_day)))})
    return pd.DataFrame(rows)


def main():
    d = build_c1()
    print(f"Sự kiện biến động mạnh (|z| > 1,5): {len(d):,} | nhóm: {d['nhom'].value_counts().to_dict()}")
    desc = descriptive(d)
    rows, preds = [], []
    for h in (1, 5):
        tgt = f"cont_{h}"
        dd = d.dropna(subset=C1_PRICE + C1_NEWS + [tgt])
        for algo_name, algo in (("LogReg", LogReg), ("LightGBM", LGBM)):
            P = {"giá": run(dd, C1_PRICE, tgt, h, algo, "gia", classes=BINARY, return_val=True),
                 "giá + tin": run(dd, C1_PRICE + C1_NEWS, tgt, h, algo, "gia_tin", classes=BINARY, return_val=True)}
            T = {k: v[v["part"] == "test"].merge(dd[["date", "ticker", f"cont_ret_{h}", "nhom"]], on=["date", "ticker"])
                 for k, v in P.items()}
            ref = T["giá"]; ref_daily = row_logloss(ref).groupby(ref["date"]).mean()
            for k, t in T.items():
                v = P[k][P[k]["part"] == "val"]
                pred_cont = t["p_1"] >= t["p_1"].median()               # nửa được dự báo "đi tiếp" nhiều hơn
                r = {"h": h, "thuật toán": algo_name, "đặc trưng": k, "n test": len(t),
                     "AUC val": roc_auc_score(v["y"], v["p_1"]), "AUC test": roc_auc_score(t["y"], t["p_1"]),
                     "tỷ lệ đi tiếp": t["y"].mean(),
                     "lợi suất nhóm dự báo ĐI TIẾP (%)": 100 * t.loc[pred_cont, f"cont_ret_{h}"].mean(),
                     "lợi suất nhóm dự báo ĐẢO CHIỀU (%)": 100 * t.loc[~pred_cont, f"cont_ret_{h}"].mean()}
                if k != "giá":
                    daily = row_logloss(t).groupby(t["date"]).mean()
                    r["DM vs giá"], r["p_DM"] = dm_test(daily, ref_daily, h)
                    dlt, lo, hi = block_bootstrap_diff(t, ref, metric="auc", n_boot=1000)
                    r["ΔAUC vs giá"], r["KTC95"] = dlt, f"[{lo:+.4f}, {hi:+.4f}]"
                rows.append(r)
                preds.append(t.assign(h=h, algo=algo_name, spec=k))
            print(f"  xong h={h} {algo_name}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "reports" / "c1_continuation.csv", index=False, encoding="utf-8-sig")
    desc.to_csv(OUT / "reports" / "c1_descriptive.csv", index=False, encoding="utf-8-sig")
    pd.concat(preds, ignore_index=True).to_parquet(OUT / "predictions" / "c1.parquet", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print("\nMÔ TẢ (toàn mẫu):"); print(desc.to_string(index=False))
        print("\nNGOÀI MẪU:"); print(res.to_string(index=False))


if __name__ == "__main__":
    main()
