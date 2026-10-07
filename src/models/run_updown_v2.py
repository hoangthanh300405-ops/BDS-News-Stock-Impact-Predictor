"""Cải thiện output "TĂNG / KHÔNG TĂNG" × "ĐỒNG THUẬN / MÂU THUẪN" — KHÔNG nhìn trộm tập test.

Nguyên tắc: mọi lựa chọn (biến thể, ngưỡng, mô hình kết hợp) học trên phần VALIDATION của từng fold; tập test chỉ
để báo cáo, và báo cáo MỌI biến thể. Biến thể "được chọn" = biến thể có AUC validation cao nhất.

Biến thể Model A (Model B giữ nguyên: ngành tăng rõ, LogReg):
  V0  A học "tăng vượt ngành" (như bản trước)
  V1  A học trực tiếp "cổ phiếu tăng thực" (đúng thước đo cuối)
  V2  V1 + chỉ mã thanh khoản tại phiên t (illiquid = 0)
Cách ra output (mỗi biến thể):
  quy tắc   : A, B quyết định theo ngưỡng = tỷ lệ tăng trên val; đồng thuận -> chiều chung, mâu thuẫn -> theo A
  stacking  : LogReg(logit pA, logit pB) học trên val của fold -> P(tăng) cuối; ngưỡng = tỷ lệ tăng trên val
  mạnh      : chỉ báo "TĂNG MẠNH" khi pA ≥ phân vị 80% của pA trên val VÀ B nói tăng  (có chọn lọc, độ phủ thấp)
Dùng:  python -m src.models.run_updown_v2
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import HORIZONS, ROOT
from src.eval.metrics import BINARY
from src.features.news_features import A_NEWS, B_NEWS
from src.features.price_features import A_PRICE, B_PRICE_CORE
from src.models.engine import LGBM, LogReg, run
from src.models.run_updown import data

OUT = ROOT / "outputs"


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def block_boot(d, fn, n=1500, block=20, seed=0):
    """KTC 95% của fn(DataFrame con) với bootstrap khối theo phiên (lấy mẫu chỉ số phiên)."""
    rng = np.random.default_rng(seed)
    dates = np.array(sorted(d["date"].unique()))
    groups = d.groupby("date").indices
    T = len(dates); block = min(block, max(T // 3, 1))
    vals = []
    for _ in range(n):
        st = rng.integers(0, T - block + 1, size=int(np.ceil(T / block)))
        pick = np.concatenate([dates[s:s + block] for s in st])[:T]
        idx = np.concatenate([groups[x] for x in pick])
        vals.append(fn(d.iloc[idx]))
    return np.nanpercentile(vals, [2.5, 97.5])


def bal_acc(y, yhat):
    return np.nanmean([(yhat[y == 1] == 1).mean(), (yhat[y == 0] == 0).mean()])


def build_outputs(pa, pb, y):
    """pa/pb: dự đoán có cột part ∈ {val, test}. Trả về bảng test với mọi cách ra output."""
    ab = pa.rename(columns={"p_1": "pA"})[["date", "ticker", "fold", "part", "pA", y]].merge(
        pb.rename(columns={"p_1": "pB"})[["date", "fold", "part", "pB"]], on=["date", "fold", "part"])
    outs = []
    for k, g in ab.groupby("fold"):
        v, t = g[g["part"] == "val"].dropna(subset=[y]), g[g["part"] == "test"].copy()
        base = v[y].mean()
        thrA, thrB = v["pA"].quantile(1 - base), v["pB"].quantile(1 - base)   # tỷ lệ "tăng" dự đoán ≈ tỷ lệ thực trên val
        t["A_tang"], t["B_tang"] = (t["pA"] >= thrA).astype(int), (t["pB"] >= thrB).astype(int)
        t["trang_thai"] = np.select([(t["A_tang"] == t["B_tang"]) & (t["A_tang"] == 1), t["A_tang"] == t["B_tang"]],
                                    ["đồng thuận TĂNG", "đồng thuận KHÔNG TĂNG"], "MÂU THUẪN")
        t["quy_tac"] = t["A_tang"]
        meta = LogisticRegression(C=1.0).fit(np.c_[logit(v["pA"]), logit(v["pB"])], v[y].astype(int))
        t["p_stack"] = meta.predict_proba(np.c_[logit(t["pA"]), logit(t["pB"])])[:, 1]
        pv = meta.predict_proba(np.c_[logit(v["pA"]), logit(v["pB"])])[:, 1]
        t["stack"] = (t["p_stack"] >= np.quantile(pv, 1 - base)).astype(int)
        t["manh"] = ((t["pA"] >= v["pA"].quantile(0.8)) & (t["B_tang"] == 1)).astype(int)
        t["val_auc_A"] = roc_auc_score(v[y], v["pA"]) if v[y].nunique() > 1 else np.nan
        outs.append(t)
    return pd.concat(outs, ignore_index=True)


def summarize(t, y, variant, h):
    t = t.dropna(subset=[y]).copy()
    base = t[y].mean()
    rows = []
    for method in ("quy_tac", "stack"):
        yy, yh = t[y].to_numpy(), t[method].to_numpy()
        prec = t.loc[t[method] == 1, y].mean()
        lo, hi = block_boot(t, lambda d, m=method: bal_acc(d[y].to_numpy(), d[m].to_numpy()))
        rows.append({"biến thể": variant, "h": h, "cách ra output": method, "độ phủ": 1.0,
                     "bal_acc": bal_acc(yy, yh), "KTC95 bal_acc": f"[{lo:.3f}, {hi:.3f}]",
                     "P(tăng | báo tăng)": prec, "tỷ lệ tăng chung": base, "lift": prec / base,
                     "AUC": roc_auc_score(yy, t["p_stack" if method == "stack" else "pA"])})
    s = t[t["manh"] == 1]
    lo, hi = block_boot(t, lambda d: d.loc[d["manh"] == 1, y].mean())
    rows.append({"biến thể": variant, "h": h, "cách ra output": "tín hiệu TĂNG MẠNH", "độ phủ": len(s) / len(t),
                 "P(tăng | báo tăng)": s[y].mean(), "KTC95 P(tăng)": f"[{lo:.3f}, {hi:.3f}]",
                 "tỷ lệ tăng chung": base, "lift": s[y].mean() / base})
    st = t.groupby("trang_thai")[y].agg(["mean", "size"])
    for k, r in st.iterrows():
        rows.append({"biến thể": variant, "h": h, "cách ra output": f"trạng thái: {k}", "độ phủ": r["size"] / len(t),
                     "P(tăng | báo tăng)": r["mean"], "tỷ lệ tăng chung": base, "lift": r["mean"] / base})
    return rows, t["val_auc_A"].mean()


def main():
    dA, dB = data()
    colsA, colsB = A_PRICE + A_NEWS, B_PRICE_CORE + B_NEWS
    rows, val_auc, tables = [], {}, []
    for h in HORIZONS:
        y = f"up_raw_{h}"
        pb = run(dB, colsB, f"upB_{h}", h, LogReg, "B", classes=BINARY, return_val=True)
        variants = {"V0 A: tăng vượt ngành": (dA, f"up_{h}"),
                    "V1 A: tăng thực": (dA, y),
                    "V2 A: tăng thực, chỉ mã thanh khoản": (dA[dA["illiquid"] == 0], y)}
        for name, (d, tgt) in variants.items():
            pa = run(d, colsA, tgt, h, LGBM, "A", classes=BINARY, return_val=True)
            pa = pa.merge(d[["date", "ticker", y]], on=["date", "ticker"], how="left") if tgt != y else pa.rename(columns={"y": y})
            t = build_outputs(pa, pb, y)
            r, va = summarize(t, y, name, h)
            rows += r; val_auc[(name, h)] = va
            tables.append(t.assign(bien_the=name, h=h))
            print(f"xong h={h} {name} (AUC validation của A: {va:.3f})")
    res = pd.DataFrame(rows)
    res["chọn theo validation"] = [("✔" if val_auc[(v, h)] == max(val_auc[(x, h)] for x in {k[0] for k in val_auc} if (x, h) in val_auc) else "")
                                   for v, h in zip(res["biến thể"], res["h"])]
    res.to_csv(OUT / "reports" / "updown_v2.csv", index=False, encoding="utf-8-sig")
    pd.concat(tables, ignore_index=True).to_parquet(OUT / "predictions" / "updown_v2.parquet", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 20, "display.max_rows", 200,
                           "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
