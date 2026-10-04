"""Dự đoán CÓ CHỌN LỌC: chỉ đưa tín hiệu tăng/giảm khi mô hình đủ tự tin -> đổi độ phủ lấy độ chính xác.

Điểm = p_up − p_down (Model A, dự đoán ngoài mẫu của RQ1). Ngưỡng |điểm| cho độ phủ q được lấy từ các fold TRƯỚC
(chỉ quá khứ) — fold 1 không có quá khứ nên bị bỏ; báo cáo trên fold 2–6. Chiều dự đoán = dấu(điểm).
Mỗi dòng in kèm TỶ LỆ NỀN (luôn đoán chiều phổ biến hơn trong CÙNG tập được chọn) — độ chính xác chỉ có nghĩa khi
so với con số này.   Dùng:  python -m src.analysis.selective
"""
import numpy as np
import pandas as pd

from src.config import CLEAN, ROOT

OUT = ROOT / "outputs"
COVER = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]


def boot(d, block=20, n=2000, seed=0):
    s = d.groupby("date").agg(hit=("hit", "sum"), up=("up", "sum"), n=("hit", "size"))
    H, U, N = s["hit"].to_numpy(), s["up"].to_numpy(), s["n"].to_numpy()
    T = len(s); block = min(block, max(T // 3, 1))
    rng = np.random.default_rng(seed)
    st = rng.integers(0, T - block + 1, size=(n, int(np.ceil(T / block))))
    idx = (st[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    h, u, m = H[idx].sum(1), U[idx].sum(1), N[idx].sum(1)
    return np.percentile(h / m, [2.5, 97.5]), np.percentile(h / m - np.maximum(u / m, 1 - u / m), [2.5, 97.5])


def main():
    p = pd.read_parquet(OUT / "predictions" / "rq1.parquet")
    ta = pd.read_parquet(CLEAN / "target_A.parquet", columns=["date", "ticker", "AR_1", "r_1", "AR_5", "r_5"])
    rows = []
    for h in (1, 5):
        a = p[(p.level == "A mã") & (p.h == h) & (p.algo == "LightGBM") & (p.spec == "A1 giá + tin")]
        a = a.merge(ta, on=["date", "ticker"])
        a["score"] = a["p_up"] - a["p_down"]
        for target, lab in ((f"AR_{h}", "vượt ngành"), (f"r_{h}", "giá thực")):
            for q in COVER:
                sel = []
                for k in sorted(a["fold"].unique()):
                    if k == 1:
                        continue
                    past = a[a["fold"] < k]["score"].abs()
                    thr = past.quantile(1 - q) if q < 1 else -1
                    cur = a[a["fold"] == k]
                    sel.append(cur[cur["score"].abs() >= thr])
                d = pd.concat(sel)
                d = d[(d[target] != 0) & d[target].notna() & (d["score"] != 0)].copy()
                d["hit"] = (np.sign(d["score"]) == np.sign(d[target])).astype(float)
                d["up"] = (d[target] > 0).astype(float)
                base = max(d["up"].mean(), 1 - d["up"].mean())
                (lo, hi), (dlo, dhi) = boot(d)
                rows.append({"h": h, "đo trên": lab, "độ phủ mục tiêu": f"{q:.0%}", "n": len(d),
                             "số phiên": d["date"].nunique(), "độ chính xác": d["hit"].mean(),
                             "KTC95": f"[{lo:.3f}, {hi:.3f}]", "đoán mù (cùng tập)": base,
                             "hơn đoán mù": d["hit"].mean() - base, "KTC95 phần hơn": f"[{dlo:+.3f}, {dhi:+.3f}]"})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "reports" / "selective.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
