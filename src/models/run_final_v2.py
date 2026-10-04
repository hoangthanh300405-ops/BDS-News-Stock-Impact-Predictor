"""Vòng cải tiến CUỐI (khai báo trước, chọn bằng validation, sau đó ĐÓNG BĂNG).

Nền: cấu hình cuối v1 (Model A 3 lớp, bộ đặc trưng S2, LightGBM, chọn lọc theo ngưỡng từ các fold trước).
  C1  LightGBM lưới rộng (6 cấu hình)                  -> tiêu chí: AUC validation
  C2  Bagging 5 LightGBM khác seed                      -> tiêu chí: AUC validation
  C3  Chỉ báo khi A tự tin VÀ Model B (ngành) cùng chiều -> tiêu chí: độ chính xác có chọn lọc (~20%) trên validation
Test chỉ dùng để báo cáo, cho MỌI ứng viên. Dùng:  python -m src.models.run_final_v2
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import CLEAN, ROOT
from src.features.news_features import B_NEWS
from src.features.price_features import B_PRICE_CORE
from src.models.engine import LGBM, LGBMBag, LGBMWide, LogReg, run
from src.models.run_direction_v3 import SETS, boot, data
from src.models.run_updown import data as data_updown

OUT = ROOT / "outputs"
COVER = [1.0, 0.3, 0.2, 0.1]
FEATS = SETS["S2 giá + tin + đặc trưng mới"]


def prep(p, d, h):
    p = p.merge(d[["date", "ticker", f"r_{h}"]], on=["date", "ticker"])
    p["score"] = p["p_up"] - p["p_down"]
    p = p[p[f"r_{h}"] != 0].dropna(subset=[f"r_{h}"]).copy()
    p["up"] = (p[f"r_{h}"] > 0).astype(int)
    return p


def val_selective_acc(p, q=0.2, use_b=False):
    """Độ chính xác có chọn lọc TRÊN VALIDATION (ngưỡng = phân vị trong chính val của fold)."""
    hits = []
    for _, g in p[p["part"] == "val"].groupby("fold"):
        s = g[g["score"].abs() >= g["score"].abs().quantile(1 - q)]
        if use_b:
            s = s[np.sign(s["score"]) == np.sign(s["scoreB"])]
        hits.append((np.sign(s["score"]) == np.where(s["up"] == 1, 1, -1)).astype(float))
    return pd.concat(hits).mean()


def test_selective(p, use_b=False):
    t = p[p["part"] == "test"]
    rows = []
    for q in COVER:
        sel = []
        for k in sorted(t["fold"].unique()):
            if k == 1:
                continue
            thr = t[t["fold"] < k]["score"].abs().quantile(1 - q) if q < 1 else -1
            cur = t[t["fold"] == k]
            cur = cur[cur["score"].abs() >= thr]
            if use_b:
                cur = cur[np.sign(cur["score"]) == np.sign(cur["scoreB"])]
            sel.append(cur)
        s = pd.concat(sel).copy()
        s["hit"] = (np.sign(s["score"]) == np.where(s["up"] == 1, 1, -1)).astype(float)
        s["y"] = s["up"]
        base = max(s["up"].mean(), 1 - s["up"].mean())
        (lo, hi), (dlo, dhi) = boot(s)
        rows.append({"độ phủ mục tiêu": f"{q:.0%}", "độ phủ thực": len(s) / len(t[t["fold"] > 1]),
                     "độ chính xác": s["hit"].mean(), "KTC95": f"[{lo:.3f}, {hi:.3f}]", "đoán mù": base,
                     "hơn đoán mù": s["hit"].mean() - base, "KTC95 phần hơn": f"[{dlo:+.3f}, {dhi:+.3f}]"})
    return rows


def main(h=1):
    d = data().merge(pd.read_parquet(CLEAN / "target_A.parquet", columns=["date", "ticker", "y_1", "y_5"]),
                     on=["date", "ticker"])
    cands = {"nền (LightGBM)": LGBM, "C1 LightGBM lưới rộng": LGBMWide, "C2 bagging 5 LightGBM": LGBMBag}
    P, val_auc = {}, {}
    for name, algo in cands.items():
        P[name] = prep(run(d, FEATS, f"y_{h}", h, algo, name, return_val=True), d, h)
        v = P[name][P[name]["part"] == "val"]
        val_auc[name] = roc_auc_score(v["up"], v["score"])
        print(f"  {name}: AUC val {val_auc[name]:.4f}")
    best = max(val_auc, key=val_auc.get)

    # C3 — lọc theo đồng thuận với Model B, quyết định bằng validation
    _, dB = data_updown()
    pb = run(dB, B_PRICE_CORE + B_NEWS, f"yB_{h}", h, LogReg, "B", return_val=True)
    pb["scoreB"] = pb["p_up"] - pb["p_down"]
    for name in P:
        P[name] = P[name].merge(pb[["date", "fold", "part", "scoreB"]], on=["date", "fold", "part"], how="left")
    acc_plain, acc_b = val_selective_acc(P[best]), val_selective_acc(P[best], use_b=True)
    use_b = acc_b > acc_plain
    print(f"Chọn theo validation: mô hình = {best}; lọc đồng thuận B: {'CÓ' if use_b else 'KHÔNG'} "
          f"(độ chính xác có chọn lọc trên val: không lọc {acc_plain:.3f}, có lọc {acc_b:.3f})")

    rows = []
    for name, p in P.items():
        for ub in (False, True):
            chosen = (name == best) and (ub == use_b)
            for r in test_selective(p, use_b=ub):
                rows.append({"h": h, "ứng viên": name, "lọc đồng thuận B": "có" if ub else "không",
                             "AUC val": val_auc[name], "CHỌN": "✔" if chosen else "", **r})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "reports" / "final_v2.csv", index=False, encoding="utf-8-sig")
    pd.concat([p.assign(ung_vien=n) for n, p in P.items()]).to_parquet(OUT / "predictions" / "final_v2.parquet", index=False)
    with pd.option_context("display.width", 260, "display.max_rows", 100, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
