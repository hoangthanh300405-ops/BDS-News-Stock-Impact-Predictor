"""RQ1 — Tin tức có cải thiện dự đoán so với chỉ dùng giá không? (mức 1)   Dùng:  python -m src.models.run_rq1

So sánh trên CÙNG mẫu, CÙNG fold:
  B2  chỉ giá          (mốc)
  B3  chỉ tin
  A1  giá + tin        (SPINE lõi, nhãn thô, λ = 3)
Kiểm định: DM (HLN) trên log-loss theo phiên, bootstrap khối cho chênh lệch balanced accuracy — đều so với B2.
"""
import time

import numpy as np
import pandas as pd

from src.config import CLEAN, HORIZONS, ROOT
from src.eval.metrics import block_bootstrap_diff, dm_test, row_logloss, summarize
from src.features.news_features import A_NEWS, B_NEWS, industry_news, ticker_news
from src.features.price_features import A_PRICE, B_PRICE_CORE, features_A, features_B, sample_A
from src.models.engine import LGBM, LogReg, run

OUT = ROOT / "outputs"


def data_B():
    d = features_B().merge(industry_news(), on="date")
    return d.merge(pd.read_parquet(CLEAN / "target_B.parquet"), on="date")


def data_A():
    d = sample_A().merge(features_A(), on=["date", "ticker"]).merge(ticker_news().drop(columns=["TONE_all", "INTz_all"]),
                                                                    on=["date", "ticker"], how="left")
    d = d.merge(industry_news()[["date", "TONE_all", "INTz_all"]], on="date", how="left")
    return d.merge(pd.read_parquet(CLEAN / "target_A.parquet"), on=["date", "ticker"])


def compare(level, h, data, price, news, target, n_boot):
    data = data.dropna(subset=price + news + [target])
    specs = {"B2 giá": price, "B3 tin": news, "A1 giá + tin": price + news}
    rows, preds = [], {}
    for algo_name, algo in (("LogReg", LogReg), ("LightGBM", LGBM)):
        P = {k: run(data, cols, target, h, algo, f"{k} — {algo_name}") for k, cols in specs.items()}
        ref = P["B2 giá"]
        ref_daily = row_logloss(ref).groupby(ref["date"]).mean()
        for k, p in P.items():
            s = summarize(p)
            s.update(level=level, h=h, algo=algo_name, model=k,
                     ba_theo_fold=" ".join(f"{summarize(g)['bal_acc']:.2f}" for _, g in p.groupby("fold")))
            if k != "B2 giá":
                daily = row_logloss(p).groupby(p["date"]).mean()
                s["DM_vs_B2"], s["p_DM"] = dm_test(daily, ref_daily, h)
                d, lo, hi = block_bootstrap_diff(p, ref, metric="bal_acc", n_boot=n_boot)
                s["dBA_vs_B2"], s["dBA_CI95"] = d, f"[{lo:+.3f}, {hi:+.3f}]"
            rows.append(s)
            preds[s["model"] + "|" + algo_name] = p.assign(level=level, h=h, algo=algo_name, spec=k)
    return rows, preds


def main():
    dB, dA = data_B(), data_A()
    print(f"Model B: {len(dB)} phiên | Model A: {len(dA)} cặp")
    results, allp = [], []
    for h in HORIZONS:
        for level, data, price, news, target, nb in (("B ngành", dB, B_PRICE_CORE, B_NEWS, f"yB_{h}", 1000),
                                                     ("A mã", dA, A_PRICE, A_NEWS, f"y_{h}", 300)):
            t0 = time.time()
            rows, preds = compare(level, h, data, price, news, target, nb)
            results += rows; allp += list(preds.values())
            print(f"  xong {level} h={h} ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(results)[["level", "h", "algo", "model", "n", "bal_acc", "mcc", "macro_f1", "log_loss",
                                 "DM_vs_B2", "p_DM", "dBA_vs_B2", "dBA_CI95", "ba_theo_fold"]]
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "reports" / "rq1_price_vs_news.csv", index=False, encoding="utf-8-sig")
    pd.concat(allp, ignore_index=True).to_parquet(OUT / "predictions" / "rq1.parquet", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
