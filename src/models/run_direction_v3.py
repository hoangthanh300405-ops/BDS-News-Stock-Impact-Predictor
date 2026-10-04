"""v3 — Dự báo CHIỀU giá thực (tăng / giảm) của cổ phiếu, cải tiến mô hình & đặc trưng, KHÔNG nhìn trộm tập test.

Target   : up = 1 nếu lợi suất thực của cổ phiếu trong (t, t+h] > 0; 0 nếu < 0; bỏ phiên giá đứng yên.
Đặc trưng: S1 = giá + tin (như RQ1);  S2 = S1 + đặc trưng mới (chạm trần/sàn, thứ hạng chéo, thứ trong tuần,
           xu hướng riêng của mã, bối cảnh ngành) — xem price_features.features_A_extra.
Mô hình  : LogReg, LightGBM (lưới rộng), Vote = trung bình xác suất hai mô hình (§8.4).
Chọn     : cấu hình có AUC VALIDATION cao nhất (gộp val của 6 fold). Tập test chỉ để báo cáo — báo cáo MỌI cấu hình.
Có chọn lọc: chỉ báo tín hiệu khi |p − 0,5| vượt phân vị của chính fold đó trên VALIDATION -> độ phủ mục tiêu.
Dùng:  python -m src.models.run_direction_v3
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import CLEAN, HORIZONS, ROOT
from src.eval.metrics import BINARY
from src.features.news_features import A_NEWS, industry_news, ticker_news
from src.features.price_features import A_EXTRA, A_PRICE, features_A, features_A_extra, sample_A
from src.models.engine import LGBMWide, LogReg, run

OUT = ROOT / "outputs"
COVER = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
SETS = {"S1 giá + tin": A_PRICE + A_NEWS, "S2 giá + tin + đặc trưng mới": A_PRICE + A_NEWS + A_EXTRA}


def data():
    d = sample_A().merge(features_A(), on=["date", "ticker"])
    extra = features_A_extra().drop(columns=[c for c in ("rB_1d", "rB_5d") if c in features_A_extra().columns])
    d = d.merge(extra, on=["date", "ticker"], how="left")
    d = d.merge(ticker_news().drop(columns=["TONE_all", "INTz_all"]), on=["date", "ticker"], how="left")
    d = d.merge(industry_news()[["date", "TONE_all", "INTz_all"]], on="date", how="left")
    d = d.merge(pd.read_parquet(CLEAN / "target_A.parquet", columns=["date", "ticker", "r_1", "r_5"]), on=["date", "ticker"])
    for h in HORIZONS:
        r = d[f"r_{h}"]
        d[f"up_dir_{h}"] = np.where(r > 0, 1.0, np.where(r < 0, 0.0, np.nan))
    return d


def boot(t, block=20, n=2000, seed=0):
    s = t.groupby("date").agg(hit=("hit", "sum"), up=("y", "sum"), n=("hit", "size"))
    H, U, N = s["hit"].to_numpy(), s["up"].to_numpy(), s["n"].to_numpy()
    T = len(s); block = min(block, max(T // 3, 1))
    rng = np.random.default_rng(seed)
    st = rng.integers(0, T - block + 1, size=(n, int(np.ceil(T / block))))
    idx = (st[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    h, u, m = H[idx].sum(1), U[idx].sum(1), N[idx].sum(1)
    return np.percentile(h / m, [2.5, 97.5]), np.percentile(h / m - np.maximum(u / m, 1 - u / m), [2.5, 97.5])


def selective(p):
    """p: dự đoán có part ∈ {val, test}, cột p_1 và y. Ngưỡng độ tự tin học trên val của CÙNG fold."""
    rows = []
    for q in COVER:
        sel = []
        for k, g in p.groupby("fold"):
            v, t = g[g["part"] == "val"], g[g["part"] == "test"]
            thr = (v["p_1"] - 0.5).abs().quantile(1 - q) if q < 1 else -1
            sel.append(t[(t["p_1"] - 0.5).abs() >= thr])
        t = pd.concat(sel).copy()
        t["hit"] = ((t["p_1"] >= 0.5).astype(int) == t["y"]).astype(float)
        base = max(t["y"].mean(), 1 - t["y"].mean())
        (lo, hi), (dlo, dhi) = boot(t)
        rows.append({"độ phủ mục tiêu": f"{q:.0%}", "độ phủ thực": len(t) / (p["part"] == "test").sum(),
                     "độ chính xác": t["hit"].mean(), "KTC95": f"[{lo:.3f}, {hi:.3f}]", "đoán mù": base,
                     "hơn đoán mù": t["hit"].mean() - base, "KTC95 phần hơn": f"[{dlo:+.3f}, {dhi:+.3f}]"})
    return rows


def main():
    d = data()
    print(f"Mẫu: {len(d):,} cặp (mã, phiên) | đặc trưng mới: {A_EXTRA}")
    rows, allp = [], []
    for h in HORIZONS:
        tgt = f"up_dir_{h}"
        for sname, cols in SETS.items():
            P = {}
            for aname, algo in (("LogReg", LogReg), ("LightGBM", LGBMWide)):
                P[aname] = run(d, cols, tgt, h, algo, aname, classes=BINARY, return_val=True)
            key = ["date", "ticker", "fold", "part", "y"]
            vote = P["LogReg"][key + ["p_1"]].merge(P["LightGBM"][key + ["p_1"]], on=key, suffixes=("_lr", "_gb"))
            vote["p_1"] = (vote["p_1_lr"] + vote["p_1_gb"]) / 2
            P["Vote LogReg + LightGBM"] = vote[key + ["p_1"]]
            for aname, p in P.items():
                v, t = p[p["part"] == "val"], p[p["part"] == "test"]
                base_row = {"h": h, "đặc trưng": sname, "mô hình": aname,
                            "AUC val": roc_auc_score(v["y"], v["p_1"]), "AUC test": roc_auc_score(t["y"], t["p_1"])}
                for r in selective(p):
                    rows.append({**base_row, **r})
                allp.append(p.assign(h=h, dac_trung=sname, mo_hinh=aname))
            print(f"  xong h={h} {sname}")
    res = pd.DataFrame(rows)
    best = res.groupby("h").apply(lambda g: g.loc[g["AUC val"].idxmax(), ["đặc trưng", "mô hình"]].tolist(),
                                  include_groups=False)
    res["chọn theo val"] = [("✔" if [s, m] == best[h] else "") for h, s, m in zip(res["h"], res["đặc trưng"], res["mô hình"])]
    res.to_csv(OUT / "reports" / "direction_v3.csv", index=False, encoding="utf-8-sig")
    pd.concat(allp, ignore_index=True).to_parquet(OUT / "predictions" / "direction_v3.parquet", index=False)
    with pd.option_context("display.width", 260, "display.max_rows", 300, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
