"""Bước 4 — Baseline B0–B2 cho Model B (ngành) và Model A (mã), h ∈ {1, 5}.   Dùng:  python -m src.models.run_baselines

B0  lớp đa số của train (sàn tuyệt đối)
B1  động lượng: z của lợi suất 5 phiên vừa qua (ngành: benchmark; mã: AR)
B2  chỉ đặc trưng giá (F5) — LogReg và LightGBM   → mốc để đo "tin tức có giúp không" (RQ1)
"""
import time

import numpy as np
import pandas as pd

from src.config import CLEAN, HORIZONS, ROOT
from src.eval.metrics import block_bootstrap_diff, dm_test, row_logloss, summarize
from src.features.price_features import A_PRICE, B_PRICE_CORE, features_A, features_B, sample_A
from src.models.engine import LGBM, LogReg, Majority, Momentum, run

OUT = ROOT / "outputs"


def data_B():
    f = features_B()
    t = pd.read_parquet(CLEAN / "target_B.parquet")
    d = f.merge(t, on="date")
    d["mom_z"] = d["rB_5d"] / (d["sigmaB_20"] * np.sqrt(5))
    return d


def data_A():
    s = sample_A()
    f = features_A()
    t = pd.read_parquet(CLEAN / "target_A.parquet")
    d = s.merge(f, on=["date", "ticker"]).merge(t, on=["date", "ticker"])
    d["mom_z"] = d["ar_5d"] / (d["sigma_ar60"] * np.sqrt(5))
    return d


def evaluate(level, h, data, price_cols, target, n_boot):
    specs = [("B0 lớp đa số", ["mom_z"], Majority), ("B1 động lượng", ["mom_z"], Momentum),
             ("B2 giá — LogReg", price_cols, LogReg), ("B2 giá — LightGBM", price_cols, LGBM)]
    # cùng một tập mẫu cho mọi baseline: bỏ dòng thiếu bất kỳ cột nào được dùng
    used = sorted(set(sum([c for _, c, _ in specs], [])))
    data = data.dropna(subset=used + [target])
    preds = {name: run(data, cols, target, h, cls, name) for name, cols, cls in specs}
    base = preds["B0 lớp đa số"]
    base_daily = row_logloss(base).groupby(base["date"]).mean()
    rows = []
    for name, p in preds.items():
        s = summarize(p)
        s.update(level=level, h=h, model=name)
        s["macro_f1_theo_fold"] = " ".join(f"{summarize(g)['macro_f1']:.2f}" for _, g in p.groupby("fold"))
        if name != "B0 lớp đa số":
            daily = row_logloss(p).groupby(p["date"]).mean()
            s["DM_vs_B0"], s["p_DM"] = dm_test(daily, base_daily, h)
            # balanced accuracy: đoán mù luôn = 1/3, nên chênh lệch so với B0 có nghĩa (macro-F1 thì không:
            # B0 chỉ đoán một lớp nên macro-F1 của nó thấp một cách cơ học)
            d, lo, hi = block_bootstrap_diff(p, base, metric="bal_acc", n_boot=n_boot)
            s["dBA_vs_B0"], s["dBA_CI95"] = d, f"[{lo:+.3f}, {hi:+.3f}]"
        rows.append(s)
    allp = pd.concat(preds.values(), ignore_index=True).assign(level=level, h=h)
    return rows, allp


def main():
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    results, allpreds = [], []
    dB, dA = data_B(), data_A()
    print(f"Model B: {len(dB)} phiên | Model A: {len(dA)} cặp (mã, phiên) trong mẫu có tin")
    for h in HORIZONS:
        for level, data, cols, target, nb in (("B ngành", dB, B_PRICE_CORE, f"yB_{h}", 1000),
                                              ("A mã", dA, A_PRICE, f"y_{h}", 300)):
            t0 = time.time()
            rows, p = evaluate(level, h, data, cols, target, nb)
            results += rows; allpreds.append(p)
            print(f"  xong {level} h={h} ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(results)
    cols = ["level", "h", "model", "n", "macro_f1", "bal_acc", "mcc", "recall_giam", "log_loss",
            "DM_vs_B0", "p_DM", "dBA_vs_B0", "dBA_CI95", "macro_f1_theo_fold"]
    res = res[cols]
    res.to_csv(OUT / "reports" / "baselines.csv", index=False, encoding="utf-8-sig")
    pd.concat(allpreds, ignore_index=True).to_parquet(OUT / "predictions" / "baselines.parquet", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
