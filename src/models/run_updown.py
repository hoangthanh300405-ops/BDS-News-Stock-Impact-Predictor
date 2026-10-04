"""Output cuối: "TĂNG / KHÔNG TĂNG" + "ĐỒNG THUẬN / MÂU THUẪN" giữa Model A và Model B.

  Model A (mã, LightGBM)  : lợi suất vượt ngành có tăng rõ không (z > +0,5σ)            target up_h
  Model B (ngành, LogReg) : chỉ số ngành có tăng rõ không (zB > +0,5σ)                   target upB_h
  Quyết định từng mô hình : "tăng" nếu P(tăng) > tỷ lệ tăng trong train (quy tắc Bayes cho balanced accuracy)
  Output cho (mã, phiên)  : trang_thai ∈ {đồng thuận tăng, đồng thuận không tăng, mâu thuẫn}
                            du_doan_cuoi = chiều chung nếu đồng thuận; theo A nếu mâu thuẫn
  Thước đo                : cổ phiếu có tăng rõ không theo lợi suất THỰC (up_raw_h)
So sánh hai cách chia: walk-forward 6 fold (chính) và một lần 70/10/20 theo thời gian.
Dùng:  python -m src.models.run_updown
"""
import numpy as np
import pandas as pd

from src.config import CLEAN, HORIZONS, ROOT
from src.eval.metrics import BINARY
from src.eval.walk_forward import make_folds, make_split_70_10_20
from src.features.news_features import A_NEWS, B_NEWS, industry_news, ticker_news
from src.features.price_features import A_PRICE, B_PRICE_CORE, features_A, features_B, sample_A
from src.models.engine import LGBM, LogReg, run

OUT = ROOT / "outputs"
SPLITS = {"walk-forward 6 fold": make_folds, "chia 70/10/20": make_split_70_10_20}


def data():
    dB = features_B().merge(industry_news(), on="date").merge(pd.read_parquet(CLEAN / "target_B.parquet"), on="date")
    dA = sample_A().merge(features_A(), on=["date", "ticker"])
    dA = dA.merge(ticker_news().drop(columns=["TONE_all", "INTz_all"]), on=["date", "ticker"], how="left")
    dA = dA.merge(industry_news()[["date", "TONE_all", "INTz_all"]], on="date", how="left")
    dA = dA.merge(pd.read_parquet(CLEAN / "target_A.parquet"), on=["date", "ticker"])
    return dA, dB


def decide(p, prior):
    return (p["p_1"] > prior).astype(int)


def boot_ci(d, col_hit, n=2000, block=20, seed=0):
    s = d.groupby("date")[col_hit].agg(["sum", "size"])
    H, N = s["sum"].to_numpy(), s["size"].to_numpy()
    T = len(s); block = min(block, max(T // 3, 1))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T - block + 1, size=(n, int(np.ceil(T / block))))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    return np.percentile(H[idx].sum(1) / N[idx].sum(1), [2.5, 97.5])


def evaluate(out, h, split):
    rows = []
    y = f"up_raw_{h}"
    base_up = out[y].mean()
    for name, d in [("TẤT CẢ (dự đoán cuối)", out)] + [(k, g) for k, g in out.groupby("trang_thai")]:
        d = d.dropna(subset=[y]).copy()
        d["hit"] = (d["du_doan_cuoi"] == d[y]).astype(float)
        tpr = d.loc[d[y] == 1, "hit"].mean()                       # đúng khi thực tế tăng
        tnr = d.loc[d[y] == 0, "hit"].mean()                       # đúng khi thực tế không tăng
        lo, hi = boot_ci(d, "hit")
        rows.append({"cách chia": split, "h": h, "trường hợp": name, "n": len(d), "tỷ trọng": len(d) / len(out),
                     "accuracy": d["hit"].mean(), "KTC95 acc": f"[{lo:.3f}, {hi:.3f}]",
                     "bal_acc": np.nanmean([tpr, tnr]), "tỷ lệ thực tế tăng": d[y].mean(),
                     "đoán mù (luôn 'không tăng')": 1 - d[y].mean(), "số phiên": d["date"].nunique()})
    return rows


def main():
    dA, dB = data()
    colsA, colsB = A_PRICE + A_NEWS, B_PRICE_CORE + B_NEWS
    rows, outs = [], []
    for split, splitter in SPLITS.items():
        for h in HORIZONS:
            pa = run(dA, colsA, f"up_{h}", h, LGBM, "A", classes=BINARY, splitter=splitter)
            pb = run(dB, colsB, f"upB_{h}", h, LogReg, "B", classes=BINARY, splitter=splitter)
            # ngưỡng quyết định = tỷ lệ "tăng" của phần dữ liệu dùng để fit (không nhìn test)
            priorA = {f.k: dA[dA["date"].isin(f.fit)][f"up_{h}"].mean() for f in splitter(h)}
            priorB = {f.k: dB[dB["date"].isin(f.fit)][f"upB_{h}"].mean() for f in splitter(h)}
            pa["A_tang"] = (pa["p_1"] > pa["fold"].map(priorA)).astype(int)
            pb["B_tang"] = (pb["p_1"] > pb["fold"].map(priorB)).astype(int)
            o = pa[["date", "ticker", "p_1", "A_tang"]].rename(columns={"p_1": "pA_tang"}).merge(
                pb[["date", "p_1", "B_tang"]].rename(columns={"p_1": "pB_tang"}), on="date")
            o["dong_thuan"] = o["A_tang"] == o["B_tang"]
            o["trang_thai"] = np.select([o["dong_thuan"] & (o["A_tang"] == 1), o["dong_thuan"]],
                                        ["đồng thuận TĂNG", "đồng thuận KHÔNG TĂNG"], "MÂU THUẪN")
            o["du_doan_cuoi"] = o["A_tang"]            # đồng thuận: chiều chung (= A); mâu thuẫn: theo A
            o = o.merge(dA[["date", "ticker", f"up_raw_{h}"]], on=["date", "ticker"], how="left")
            rows += evaluate(o, h, split)
            outs.append(o.assign(h=h, cach_chia=split))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "reports" / "updown_agreement.csv", index=False, encoding="utf-8-sig")
    pd.concat(outs, ignore_index=True).to_parquet(OUT / "predictions" / "updown_agreement.parquet", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
