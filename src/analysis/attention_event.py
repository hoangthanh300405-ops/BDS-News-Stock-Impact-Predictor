"""Hướng B — khối lượng giao dịch bất thường quanh phiên có tin doanh nghiệp (event study cho SỰ CHÚ Ý).

Sự kiện = (mã, phiên có tin; đã loại nhãn chỉ thuật lại giá). Khối lượng bất thường tại phiên t+k:
  AV_k = log(giá trị GD_{t+k}) − trung bình log(giá trị GD) trên [t−25, t−6]  (mốc KHÔNG chạm cửa sổ sự kiện)
Tách theo sắc thái tin và theo "tin nổi bật" (mã được nhắc bất thường nhiều). Dùng:  python -m src.analysis.attention_event
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.reaction import build_events
from src.config import CLEAN, ROOT
from src.eval.p0_report import DIV_NEG, DIV_POS, INK2, MUTED, SERIES, _save

K = range(-5, 6)


def main():
    ev = build_events()
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet", columns=["date", "ticker", "value_bn"])
    v = px.pivot_table(index="date", columns="ticker", values="value_bn")
    lv = np.log(v.where(v > 0))
    L = lv.to_numpy(); pos = pd.Series(range(len(lv)), index=lv.index); col = {t: j for j, t in enumerate(lv.columns)}
    rows = []
    for r in ev.itertuples():
        if r.ticker not in col or r.date not in pos.index:
            continue
        i, j = pos[r.date], col[r.ticker]
        if i - 25 < 0 or i + 5 >= len(lv):
            continue
        base = np.nanmean(L[i - 25:i - 5, j])
        rows.append([r.nhom, r.n_bai] + [L[i + k, j] - base for k in K])
    A = pd.DataFrame(rows, columns=["nhom", "n_bai"] + list(K)).dropna(subset=[0])
    A["noi_bat"] = np.where(A["n_bai"] >= 3, "Nhiều bài (≥ 3) trong phiên", "1–2 bài")
    out = []
    for by in ("nhom", "noi_bat"):
        for g, d in A.groupby(by):
            m, se = 100 * d[list(K)].mean(), 100 * d[list(K)].std() / np.sqrt(len(d))
            out.append((by, g, len(d), m, se))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    cmap = {"Tin tốt": DIV_POS, "Tin xấu": DIV_NEG, "Trung lập": MUTED,
            "Nhiều bài (≥ 3) trong phiên": SERIES[1], "1–2 bài": SERIES[0]}
    for ax, by, title in ((axes[0], "nhom", "Theo sắc thái tin"), (axes[1], "noi_bat", "Theo mức độ nổi bật")):
        for b, g, n, m, se in out:
            if b != by:
                continue
            ax.fill_between(list(K), m - 1.96 * se, m + 1.96 * se, color=cmap[g], alpha=0.13, linewidth=0)
            ax.plot(list(K), m, color=cmap[g], linewidth=2, marker="o", markersize=3.5, label=f"{g} (n = {n:,})".replace(",", "."))
        ax.axvline(0, color=MUTED, linestyle="--", linewidth=1); ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(list(K)); ax.set_xlabel("Phiên so với phiên có tin"); ax.set_title(title)
        ax.legend(loc="lower left", fontsize=8.5)
    axes[0].set_ylabel("Giá trị GD bất thường (%, log)")
    fig.suptitle("Thị trường chú ý: giá trị giao dịch bất thường quanh phiên có tin doanh nghiệp", x=0.01, ha="left",
                 fontweight="semibold", fontsize=12)
    fig.tight_layout()
    _save(fig, "r5_khoi_luong_quanh_tin.png")
    tab = pd.DataFrame([{"nhóm": g, "n": n, **{f"k={k:+d}": m[k] for k in (-5, -1, 0, 1, 2, 5)}} for b, g, n, m, se in out])
    tab.to_csv(ROOT / "outputs" / "reports" / "attention_event.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(tab.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
