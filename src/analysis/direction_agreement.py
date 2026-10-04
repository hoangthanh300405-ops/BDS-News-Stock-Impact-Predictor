"""Độ chính xác theo CHIỀU (tăng / giảm) và quy tắc kết hợp A/B (§8.3) — dùng dự đoán NGOÀI MẪU đã lưu của RQ1.

Chiều dự đoán = dấu(p_up − p_down). Đo trên các quan sát có lợi suất thực khác 0.
  (1) Nhị phân: hit rate so với "luôn đoán chiều phổ biến hơn" (tỷ lệ nền), cả mẫu và 20% dự đoán tự tin nhất.
  (2) Kết hợp A (mã, AR) với B (ngành) cùng phiên: đo trên chiều lợi suất THỰC của cổ phiếu (AR + ngành),
      tách trường hợp đồng thuận / mâu thuẫn.
KTC 95% bằng bootstrap khối theo phiên.   Dùng:  python -m src.analysis.direction_agreement
"""
import numpy as np
import pandas as pd

from src.config import CLEAN, ROOT

OUT = ROOT / "outputs"
A_ALGO, B_ALGO = "LightGBM", "LogReg"       # thuật toán chính của mỗi cấp (§8.5)


def boot_ci(d, n=2000, block=20, seed=0):
    """KTC 95% của (hit rate − tỷ lệ nền), bootstrap khối theo phiên — vector hóa trên tổng theo phiên."""
    s = d.groupby("date").agg(hit=("hit", "sum"), pos=("pos", "sum"), n=("hit", "size"))
    H, P, N = s["hit"].to_numpy(), s["pos"].to_numpy(), s["n"].to_numpy()
    rng = np.random.default_rng(seed)
    T = len(s); nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    h, p, m = H[idx].sum(1), P[idx].sum(1), N[idx].sum(1)
    diff = h / m - np.maximum(p / m, 1 - p / m)
    return np.percentile(diff, [2.5, 97.5])


def hit_table(df, pred_col, true_col, label):
    d = df[(df[true_col] != 0) & df[true_col].notna() & (df[pred_col] != 0)].copy()
    d["hit"] = (np.sign(d[pred_col]) == np.sign(d[true_col])).astype(float)
    d["pos"] = (d[true_col] > 0).astype(float)
    base = max((d[true_col] > 0).mean(), (d[true_col] < 0).mean())
    lo, hi = boot_ci(d)
    return {"trường hợp": label, "n": len(d), "đúng chiều": d["hit"].mean(), "tỷ lệ nền": base,
            "chênh": d["hit"].mean() - base, "KTC95": f"[{lo:+.3f}, {hi:+.3f}]"}


def main():
    p = pd.read_parquet(OUT / "predictions" / "rq1.parquet")
    p["score"] = p["p_up"] - p["p_down"]
    ta = pd.read_parquet(CLEAN / "target_A.parquet", columns=["date", "ticker", "AR_1", "AR_5"])
    tb = pd.read_parquet(CLEAN / "target_B.parquet", columns=["date", "rB_1", "rB_5"])
    rows = []
    for h in (1, 5):
        for spec in ("B2 giá", "A1 giá + tin"):
            A = p[(p.level == "A mã") & (p.h == h) & (p.algo == A_ALGO) & (p.spec == spec)]
            B = p[(p.level == "B ngành") & (p.h == h) & (p.algo == B_ALGO) & (p.spec == spec)]
            A = A.merge(ta, on=["date", "ticker"]).merge(tb, on="date")
            A["r_stock"] = A[f"AR_{h}"] + A[f"rB_{h}"]                    # lợi suất thực của cổ phiếu
            B = B.merge(tb, on="date")
            tag = f"h={h} | {spec}"
            # (1) nhị phân từng mô hình trên chính target của nó
            rows.append(dict(phần="1 nhị phân", mô_hình=f"{tag} | Model A (vượt trội)", **hit_table(A, "score", f"AR_{h}", "tất cả")))
            conf = A[A["score"].abs() >= A["score"].abs().quantile(0.8)]
            rows.append(dict(phần="1 nhị phân", mô_hình=f"{tag} | Model A (vượt trội)", **hit_table(conf, "score", f"AR_{h}", "20% tự tin nhất")))
            rows.append(dict(phần="1 nhị phân", mô_hình=f"{tag} | Model B (ngành)", **hit_table(B, "score", f"rB_{h}", "tất cả")))
            # (2) kết hợp A/B trên chiều lợi suất thực của cổ phiếu
            AB = A.merge(B[["date", "score"]].rename(columns={"score": "scoreB"}), on="date")
            AB["dA"], AB["dB"] = np.sign(AB["score"]), np.sign(AB["scoreB"])
            agree, conflict = AB[AB["dA"] == AB["dB"]], AB[AB["dA"] != AB["dB"]]
            rows.append(dict(phần="2 kết hợp A/B", mô_hình=tag, **hit_table(AB, "dA", "r_stock", "chỉ theo A — mọi cặp")))
            rows.append(dict(phần="2 kết hợp A/B", mô_hình=tag, **hit_table(AB, "dB", "r_stock", "chỉ theo B — mọi cặp")))
            rows.append(dict(phần="2 kết hợp A/B", mô_hình=tag, **hit_table(agree, "dA", "r_stock",
                                                                               f"ĐỒNG THUẬN ({len(agree) / len(AB):.0%} số cặp)")))
            rows.append(dict(phần="2 kết hợp A/B", mô_hình=tag, **hit_table(conflict, "dA", "r_stock", "MÂU THUẪN — theo A")))
            rows.append(dict(phần="2 kết hợp A/B", mô_hình=tag, **hit_table(conflict, "dB", "r_stock", "MÂU THUẪN — theo B")))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "reports" / "direction_agreement.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 230, "display.max_rows", 100, "display.float_format", "{:.3f}".format):
        for part, g in res.groupby("phần"):
            print(f"\n===== {part} ====="); print(g.drop(columns="phần").to_string(index=False))


if __name__ == "__main__":
    main()
