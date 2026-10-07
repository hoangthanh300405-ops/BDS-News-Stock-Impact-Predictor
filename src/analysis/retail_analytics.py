"""Tầng PHÂN TÍCH cho góc nhìn nhà đầu tư nhỏ lẻ.

A1  Báo trễ bao lâu so với giá? Đường lợi suất bất thường tích lũy (CAR, so với rổ BĐS thanh khoản) từ −10 đến +20
    phiên quanh PHIÊN ĐỌC TIN (k = 0: phiên có tin; nhà đầu tư giao dịch ở đóng cửa k = 0, nên chỉ hưởng từ k = 1).
    Tách: tin tốt / xấu × bình luận thị trường / sự kiện doanh nghiệp.
    AR CHÍNH = lợi suất − rổ BĐS − xu hướng riêng của mã (TB AR của mã trong phiên t−130..t−11; mean-adjusted,
    MacKinlay 1997). Lý do: mã hay có tin tốt vốn tăng hơn rổ ~0,06%/phiên dù có tin hay không — không trừ
    thì "đà tăng sau tin" bị phóng đại (chỉ trừ rổ: PROJECT +1,9%/20 phiên; trừ cả xu hướng: +0,5%).
A2  Loại tin nào còn "dư địa" sau khi đọc? Theo loại sự kiện × hướng: CAR[−5,−1], AR[0], CAR[+1,+5], CAR[+1,+20],
    phần biến động đã xảy ra TRƯỚC khi giao dịch được. Sai số chuẩn: lấy trung bình trong từng phiên rồi t-test theo phiên
    (tránh phóng đại do nhiều mã cùng phản ứng một cú sốc chung).
Mô tả trên toàn mẫu (không phải kiểm định dự báo).
Đầu ra: outputs/reports/retail_A1_car.csv, retail_A2_by_type.csv; outputs/figures/r8_car_path.png, r9_lag.png
Dùng:  python -m src.analysis.retail_analytics
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.config import CLEAN, ROOT
from src.retail.events import price_panels

REP, FIG = ROOT / "outputs" / "reports", ROOT / "outputs" / "figures"
K0, K1 = -10, 20


def ar_panel(adjust=True):
    """AR theo rổ BĐS thanh khoản; adjust=True trừ thêm xu hướng riêng của mã ước lượng TRƯỚC cửa sổ sự kiện."""
    close, _, illiq, _ = price_panels()
    r1 = np.log(close).diff()
    mkt = r1.where(illiq == 0).mean(axis=1)
    ar = r1.sub(mkt, axis=0)
    return ar - ar.rolling(120, min_periods=60).mean().shift(11) if adjust else ar


def paths(ev, ar):
    """Ma trận d × AR[t+k] cho mọi sự kiện, k = K0..K1 (đơn vị: log, theo hướng tin)."""
    pos = pd.Series(np.arange(len(ar)), index=ar.index)
    t = pos.reindex(ev["date"]).to_numpy()
    c = ar.columns.get_indexer(ev["ticker"])
    A = ar.to_numpy()
    ks = np.arange(K0, K1 + 1)
    M = np.full((len(ev), len(ks)), np.nan)
    for j, k in enumerate(ks):
        ok = (t + k >= 0) & (t + k < len(A))
        M[ok, j] = A[t[ok] + k, c[ok]]
    return pd.DataFrame(M * ev["d"].to_numpy()[:, None], columns=ks, index=ev.index)


def date_t(x, dates):
    """Trung bình theo phiên rồi t-test giữa các phiên."""
    g = pd.Series(x).groupby(dates.to_numpy()).mean().dropna()
    if len(g) < 5:
        return np.nan, np.nan
    return g.mean(), stats.ttest_1samp(g, 0).statistic


def groups(ev):
    loai = np.select([ev["frac_firm"] > 0.5, ev["frac_comm"] > 0.5], ["sự kiện DN", "bình luận TT"], "khác")
    huong = np.where(ev["d"] > 0, "tin tốt", "tin xấu")
    return pd.Series(huong, index=ev.index) + " · " + pd.Series(loai, index=ev.index)


def a1(ev, P):
    ev = ev.assign(nhom=groups(ev))
    rows = []
    for g, idx in ev.groupby("nhom").groups.items():
        if "khác" in g:
            continue
        sub = P.loc[idx]
        car = sub.cumsum(axis=1).sub(sub.loc[:, :-1].sum(axis=1) * 0, axis=0)   # CAR tích lũy từ K0
        for k in P.columns:
            m, t = date_t(sub[k], ev.loc[idx, "date"])
            rows.append({"nhóm": g, "k": k, "n": len(idx), "AR TB (theo hướng tin)": m, "t (theo phiên)": t,
                         "CAR từ −10": car[k].mean()})
    return pd.DataFrame(rows)


def a2(ev, P, P_raw):
    win = {"CAR[−5,−1]": range(-5, 0), "AR[0] (phiên đọc tin)": [0], "CAR[+1,+5]": range(1, 6),
           "CAR[+1,+20]": range(1, 21)}
    rows = []
    for (et, d), idx in ev.groupby(["event_type", "d"]).groups.items():
        if len(idx) < 40:
            continue
        r = {"loại sự kiện": et, "hướng": "tốt" if d > 0 else "xấu", "n": len(idx)}
        for name, ks in win.items():
            v = P.loc[idx, list(ks)].sum(axis=1, min_count=len(list(ks)))
            m, t = date_t(v, ev.loc[idx, "date"])
            r[name], r[f"t {name}"] = m, t
        r["CAR[+1,+20] chỉ trừ rổ (đối chiếu)"] = P_raw.loc[idx, list(range(1, 21))].sum(axis=1).mean()
        before = r["CAR[−5,−1]"] + r["AR[0] (phiên đọc tin)"]
        r["phần đã xảy ra trước khi giao dịch được"] = before / (before + r["CAR[+1,+5]"]) if (before + r["CAR[+1,+5]"]) else np.nan
        rows.append(r)
    out = pd.DataFrame(rows).sort_values(["hướng", "n"], ascending=[True, False])
    # hiệu chỉnh kiểm định bội (Benjamini–Hochberg) cho cột quan trọng nhất với nhà đầu tư: CAR[+1,+5]
    p = 2 * stats.t.sf(out["t CAR[+1,+5]"].abs(), df=100)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out["q (BH) CAR[+1,+5]"] = np.minimum(q, 1)
    return out


def figures(A1, ev, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"tin tốt · bình luận TT": "#1f6feb", "tin tốt · sự kiện DN": "#2da44e",
           "tin xấu · bình luận TT": "#f0883e", "tin xấu · sự kiện DN": "#cf222e"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for g, x in A1.groupby("nhóm"):
        ax.plot(x["k"], x["CAR từ −10"] * 100, marker="o", ms=3, lw=2, color=col.get(g), label=f"{g} (n={x['n'].iloc[0]:,})")
    ax.axvspan(-10, 0.5, color="#999", alpha=.08)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.text(0.7, ax.get_ylim()[1] * 0.92, "nhà đầu tư đọc tin\nvà giao dịch được từ đây →", fontsize=9)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("phiên so với phiên có tin (k = 0)")
    ax.set_ylabel("CAR theo hướng tin, %\n(trừ rổ BĐS và xu hướng riêng của mã)")
    ax.set_title("A1. Giá chạy TRƯỚC khi báo đưa tin — sau khi đọc còn lại bao nhiêu?")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "r8_car_path.png", dpi=150); plt.close(fig)
    # biểu đồ độ trễ: AR trung bình theo hướng tin tại từng k (mọi tin)
    m = P.mean() * 100
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(m.index, m.values, color=np.where(m.index <= 0, "#8c959f", "#1f6feb"))
    ax.axvline(0.5, color="k", ls="--", lw=1); ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("phiên so với phiên có tin"); ax.set_ylabel("AR TB theo hướng tin, %")
    ax.set_title("Lợi suất bất thường từng phiên theo hướng tin: phần lớn nằm ở k ≤ 0 (xám) — trước khi giao dịch được")
    fig.tight_layout(); fig.savefig(FIG / "r9_lag.png", dpi=150); plt.close(fig)


def main():
    ev = pd.read_parquet(CLEAN / "retail_events.parquet")
    P, P_raw = paths(ev, ar_panel()), paths(ev, ar_panel(adjust=False))
    A1 = a1(ev, P)
    A2 = a2(ev, P, P_raw)
    A1.to_csv(REP / "retail_A1_car.csv", index=False, encoding="utf-8-sig")
    A2.to_csv(REP / "retail_A2_by_type.csv", index=False, encoding="utf-8-sig")
    figures(A1, ev, P)
    summ = A1[A1["k"].isin([-10, -1, 0, 5, 20])].pivot(index="nhóm", columns="k", values="CAR từ −10") * 100
    tot = P.mean() * 100
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print("CAR (%) từ k=−10 đến k:"); print(summ); print()
        print(f"Mọi tin: tổng AR k∈[−10,−1] = {tot.loc[-10:-1].sum():.2f}%, k=0: {tot.loc[0]:.2f}%, "
              f"k∈[1,5] = {tot.loc[1:5].sum():.2f}%, k∈[1,20] = {tot.loc[1:20].sum():.2f}%")
        print(); print(A2.to_string(index=False))


if __name__ == "__main__":
    main()
