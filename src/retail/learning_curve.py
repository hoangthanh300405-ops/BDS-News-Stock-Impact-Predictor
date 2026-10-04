"""Thêm dữ liệu có làm mô hình FINAL tốt hơn không? (đường cong học, slide "Would more data help?")

Giữ nguyên 6 quý test walk-forward; chỉ lấy ngẫu nhiên 25/50/75/100% số sự kiện của tập train và tập fit
(3 hạt giống cho mỗi mức < 100%), huấn luyện lại đúng cấu hình FINAL (LR + RF + LightGBM hồi quy, trọng số thời gian)
và đo AUC trên toàn bộ tập test cùng precision của quy tắc tiêu chuẩn (top 10%).
Đầu ra: outputs/reports/learning_curve.csv
Dùng:  python -m src.retail.learning_curve   (~8 phút)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import CLEAN, ROOT
from src.eval.walk_forward import make_folds
from src.retail.events import FEATURES_V3, H
from src.retail.run_retail import TARGETS
from src.retail.run_retail_v4 import CONTS, fit_score, pct_vs, weights

COMPS = ("LR", "RF", "REG")
FRACS = (0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2)


def run_once(d, y, cont, frac, seed):
    rng = np.random.default_rng(seed)
    out, sizes = [], []
    for f in make_folds(H + 1):
        tr, va = d[d.date.isin(f.train)], d[d.date.isin(f.val)].copy()
        fit, te = d[d.date.isin(f.fit)], d[d.date.isin(f.test)].copy()
        if frac < 1:
            tr = tr.sample(frac=frac, random_state=int(rng.integers(1e9)))
            fit = fit.sample(frac=frac, random_state=int(rng.integers(1e9)))
        wt, wf = weights(tr.date, tr.date.max()), weights(fit.date, fit.date.max())
        for k in COMPS:
            va[k] = fit_score(k, tr, va, y, cont, FEATURES_V3, wt)
            te[k] = fit_score(k, fit, te, y, cont, FEATURES_V3, wf)
        sv = np.mean([pct_vs(va[k], va[k]) for k in COMPS], axis=0)
        te["E"] = np.mean([pct_vs(va[k], te[k]) for k in COMPS], axis=0)
        te["std"] = te["E"] >= np.quantile(sv, 0.9)
        out.append(te); sizes.append(len(fit))
    r = pd.concat(out)
    return {"frac": frac, "seed": seed, "mean_fit_events": np.mean(sizes),
            "auc": roc_auc_score(r[y], r["E"]), "precision_standard": r.loc[r["std"], y].mean()}


def main():
    ev = pd.read_parquet(CLEAN / "retail_events.parquet")
    y, _ = TARGETS["tuyet_doi"]
    d = ev[ev[y].notna()].reset_index(drop=True)
    rows = []
    for frac in FRACS:
        for seed in (SEEDS if frac < 1 else (0,)):
            rows.append(run_once(d, y, CONTS["tuyet_doi"], frac, seed))
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "reports" / "learning_curve.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(res.groupby("frac")[["mean_fit_events", "auc", "precision_standard"]].agg(["mean", "min", "max"]))


if __name__ == "__main__":
    main()
