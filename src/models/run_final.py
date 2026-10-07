"""Cấu hình CUỐI (chốt trước khi xem test): mô hình 3 lớp theo thiết kế §8 (Model A, LightGBM, target y_h),
bộ đặc trưng S1 hoặc S2 chọn theo VALIDATION, output = chiều tăng/giảm có chọn lọc.

  Điểm        = p_up − p_down;  chiều = dấu(điểm);  đo trên chiều giá THỰC của cổ phiếu (bỏ phiên đứng giá)
  Chọn S1/S2  : AUC trên validation (gộp 6 fold) của điểm so với chiều giá thực
  Có chọn lọc : chỉ báo khi |điểm| ≥ phân vị (1 − độ phủ) của |điểm| trên TEST các fold TRƯỚC (chỉ quá khứ);
                fold 1 không có quá khứ -> báo cáo trên fold 2–6
Dùng:  python -m src.models.run_final
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import HORIZONS, ROOT
from src.models.engine import LGBM, run
from src.models.run_direction_v3 import SETS, boot, data

OUT = ROOT / "outputs"
COVER = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]


def main():
    from src.config import CLEAN
    d = data().merge(pd.read_parquet(CLEAN / "target_A.parquet", columns=["date", "ticker", "y_1", "y_5"]),
                     on=["date", "ticker"])
    rows, allp, chosen = [], [], {}
    for h in HORIZONS:
        res_h = {}
        for sname, cols in SETS.items():
            p = run(d, cols, f"y_{h}", h, LGBM, sname, return_val=True)
            p = p.merge(d[["date", "ticker", f"r_{h}"]], on=["date", "ticker"])
            p["score"] = p["p_up"] - p["p_down"]
            p = p[p[f"r_{h}"] != 0].dropna(subset=[f"r_{h}"])
            p["up"] = (p[f"r_{h}"] > 0).astype(int)
            v = p[p["part"] == "val"]
            res_h[sname] = (roc_auc_score(v["up"], v["score"]), p)
        best = max(res_h, key=lambda s: res_h[s][0])
        chosen[h] = best
        for sname, (auc_val, p) in res_h.items():
            t = p[p["part"] == "test"]
            for q in COVER:
                sel = []
                for k in sorted(t["fold"].unique()):
                    if k == 1:
                        continue
                    thr = t[t["fold"] < k]["score"].abs().quantile(1 - q) if q < 1 else -1
                    cur = t[t["fold"] == k]
                    sel.append(cur[cur["score"].abs() >= thr])
                s = pd.concat(sel).copy()
                s["hit"] = (np.sign(s["score"]) == np.where(s["up"] == 1, 1, -1)).astype(float)
                s["y"] = s["up"]
                base = max(s["up"].mean(), 1 - s["up"].mean())
                (lo, hi), (dlo, dhi) = boot(s)
                rows.append({"h": h, "đặc trưng": sname, "chọn theo val": "✔" if sname == best else "",
                             "AUC val": auc_val, "AUC test": roc_auc_score(t["up"], t["score"]),
                             "độ phủ mục tiêu": f"{q:.0%}",
                             "độ phủ thực": len(s) / len(t[t["fold"] > 1]), "n": len(s),
                             "độ chính xác": s["hit"].mean(), "KTC95": f"[{lo:.3f}, {hi:.3f}]", "đoán mù": base,
                             "hơn đoán mù": s["hit"].mean() - base, "KTC95 phần hơn": f"[{dlo:+.3f}, {dhi:+.3f}]"})
            allp.append(p.assign(h=h, dac_trung=sname))
        print(f"h={h}: chọn theo validation -> {best} (AUC val: " +
              ", ".join(f"{s} {res_h[s][0]:.3f}" for s in res_h) + ")")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "reports" / "final_selective.csv", index=False, encoding="utf-8-sig")
    pd.concat(allp, ignore_index=True).to_parquet(OUT / "predictions" / "final_selective.parquet", index=False)
    with pd.option_context("display.width", 260, "display.max_rows", 100, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
