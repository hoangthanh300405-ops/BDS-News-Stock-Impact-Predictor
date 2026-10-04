"""Phản ứng của giá với tin doanh nghiệp (câu hỏi GIẢI THÍCH, bổ sung cho kết quả dự báo rỗng của RQ1).

Sự kiện = (mã i, phiên t) có ít nhất một bài nhắc mã i được gán vào phiên t (đăng trước 14:45 phiên t, hoặc
sau 14:45 phiên trước / ngày nghỉ). Phiên phản ứng k = 0 là phiên t: AR_0 = lợi suất close(t−1) -> close(t)
của mã trừ benchmark ngành. Loại nhãn chỉ thuật lại biến động giá (price_report) để tránh vòng lặp định nghĩa.

  (1) Đường CAR[−5, +5] theo nhóm tin tốt / trung lập / xấu (và theo loại tin, theo thanh khoản)
  (2) Hồi quy panel: AR_k = a_ngày + b⁺·tone⁺ + b⁻·tone⁻ + kiểm soát, sai số chuẩn gom cụm theo ngày
Dùng:  python -m src.analysis.reaction
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import CLEAN, ROOT
from src.eval.p0_report import DIV_MID, DIV_NEG, DIV_POS, GRID, INK2, MUTED, SURFACE, _save  # noqa: F401  (rcParams)

REP = ROOT / "outputs" / "reports"
K = range(-5, 6)
FUNDAMENTAL = {"tai_chinh_dn", "du_an_ha_tang", "phap_ly", "chinh_sach", "lai_suat_tin_dung"}


def build_events():
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet")
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session"])
    atl = atl[~atl["price_report"]].merge(art, on="article_id").dropna(subset=["session", "impact_score"])
    ev = atl.groupby(["ticker", "session"]).agg(
        tone=("impact_score", "mean"), n_bai=("article_id", "nunique"),
        fundamental=("nhom_tin", lambda s: s.isin(FUNDAMENTAL).mean())).reset_index()
    ev["nhom"] = np.select([ev["tone"] > 0, ev["tone"] < 0], ["Tin tốt", "Tin xấu"], "Trung lập")
    ev["loai"] = np.where(ev["fundamental"] >= 0.5, "Tin cơ bản", "Tin thị trường")
    return ev.rename(columns={"session": "date"})


def daily_ar():
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet")
    bench = pd.read_parquet(CLEAN / "benchmark_nganh.parquet")
    rB = bench[bench["method"] == "ew_liquid"].set_index("date")["log_return"]
    lr = px.pivot_table(index="date", columns="ticker", values="log_return").reindex(rB.index)
    ar = lr.sub(rB, axis=0)
    ill = px.pivot_table(index="date", columns="ticker", values="illiquid").reindex(rB.index)
    return ar, ill


def event_panel(ev, ar, ill):
    pos = pd.Series(range(len(ar)), index=ar.index)
    A = ar.to_numpy(); cols = {t: j for j, t in enumerate(ar.columns)}
    rows = []
    for r in ev.itertuples():
        if r.ticker not in cols or r.date not in pos.index:
            continue
        i, j = pos[r.date], cols[r.ticker]
        if i - 5 < 1 or i + 5 >= len(ar):
            continue
        rows.append([A[i + k, j] for k in K])
    ev = ev[[r.ticker in cols and r.date in pos.index and 6 <= pos[r.date] < len(ar) - 5 for r in ev.itertuples()]].copy()
    P = pd.DataFrame(rows, index=ev.index, columns=[f"AR{k:+d}" for k in K])
    ev = pd.concat([ev, P], axis=1)
    ev["illiquid"] = [ill.at[d, t] for d, t in zip(ev["date"], ev["ticker"])]
    return ev.dropna(subset=["AR+0"])


def car_table(ev, by):
    out = []
    for g, d in ev.groupby(by):
        C = d[[f"AR{k:+d}" for k in K]].fillna(0).cumsum(axis=1) - d[["AR-5"]].fillna(0).to_numpy() * 0
        # CAR tính từ k = −5; chuẩn hóa để CAR(−1) = 0 -> dễ đọc phản ứng từ phiên 0
        base = d[[f"AR{k:+d}" for k in range(-5, 0)]].fillna(0).sum(axis=1)
        for k in K:
            ck = d[[f"AR{j:+d}" for j in range(-5, k + 1)]].fillna(0).sum(axis=1) - base
            # sai số chuẩn gom cụm theo ngày: trung bình theo ngày rồi lấy SE giữa các ngày
            by_day = ck.groupby(d["date"]).mean()
            out.append({"nhom": g, "k": k, "CAR": ck.mean(), "se": by_day.std() / np.sqrt(len(by_day)), "n": len(d)})
    return pd.DataFrame(out)


def fig_car(tab, title, name, order=("Tin tốt", "Trung lập", "Tin xấu")):
    colors = {"Tin tốt": DIV_POS, "Trung lập": MUTED, "Tin xấu": DIV_NEG}
    fig, ax = plt.subplots(figsize=(9, 4))
    for g in order:
        d = tab[tab["nhom"] == g]
        if d.empty:
            continue
        ax.fill_between(d["k"], 100 * (d["CAR"] - 1.96 * d["se"]), 100 * (d["CAR"] + 1.96 * d["se"]),
                        color=colors[g], alpha=0.12, linewidth=0)
        ax.plot(d["k"], 100 * d["CAR"], color=colors[g], linewidth=2, marker="o", markersize=4,
                label=f"{g} (n = {int(d['n'].iloc[0]):,})".replace(",", "."))
        ax.annotate(f"{100 * d['CAR'].iloc[-1]:+.2f}%", (d["k"].iloc[-1], 100 * d["CAR"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--"); ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(list(K)); ax.set_xlabel("Phiên so với phiên có tin (0 = phiên phản ứng)")
    ax.set_ylabel("CAR (%, so với ngành; = 0 tại phiên −1)")
    ax.set_title(title); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
    _save(fig, name)


def regressions(ev):
    """AR_k theo tone⁺ và tone⁻ (bất đối xứng), hiệu ứng cố định theo ngày (khử trung bình theo ngày),
    kiểm soát AR 5 phiên trước tin; sai số chuẩn gom cụm theo ngày."""
    d = ev.copy()
    d["tone_pos"], d["tone_neg"] = d["tone"].clip(lower=0), d["tone"].clip(upper=0)
    d["pre5"] = d[[f"AR{k:+d}" for k in range(-5, 0)]].fillna(0).sum(axis=1)
    d["post_1_5"] = d[[f"AR{k:+d}" for k in range(1, 6)]].fillna(0).sum(axis=1)
    rows = []
    for y, lab in (("pre5", "trước tin [−5, −1]"), ("AR+0", "phiên phản ứng [0]"), ("AR+1", "phiên sau [+1]"),
                   ("post_1_5", "sau tin [+1, +5]")):
        x = ["tone_pos", "tone_neg"] + ([] if y == "pre5" else ["pre5"])
        dd = d[[y, "date"] + x].dropna()
        dm = dd[[y] + x] - dd.groupby("date")[[y] + x].transform("mean")        # hiệu ứng cố định theo ngày
        m = sm.OLS(dm[y] * 100, dm[x]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["date"])[0]})
        for v in ("tone_pos", "tone_neg"):
            rows.append({"kết quả (AR, %)": lab, "biến": v, "hệ số": m.params[v], "t": m.tvalues[v],
                         "p": m.pvalues[v], "n": int(m.nobs)})
    return pd.DataFrame(rows)


def main():
    ev = build_events()
    ar, ill = daily_ar()
    ev = event_panel(ev, ar, ill)
    print(f"Sự kiện (mã, phiên có tin; đã loại nhãn thuật lại giá): {len(ev):,} | nhóm: {ev['nhom'].value_counts().to_dict()}")
    t_all = car_table(ev, "nhom")
    fig_car(t_all, "Giá phản ứng với tin doanh nghiệp: CAR quanh phiên có tin (so với ngành)", "r1_car_tin_doanh_nghiep.png")
    for loai in ("Tin cơ bản", "Tin thị trường"):
        fig_car(car_table(ev[ev["loai"] == loai], "nhom"), f"CAR quanh phiên có tin — {loai.lower()}",
                f"r2_car_{'co_ban' if loai == 'Tin cơ bản' else 'thi_truong'}.png")
    liq = car_table(ev.assign(nhom=ev["nhom"] + np.where(ev["illiquid"] == 1, " · kém TK", " · thanh khoản")), "nhom")
    reg = regressions(ev)
    key = t_all[t_all["k"].isin([-1, 0, 1, 5])].pivot(index="nhom", columns="k", values="CAR") * 100
    pre = t_all[t_all["k"] == -1][["nhom"]]
    # CAR trước tin (−5 -> −1) theo nhóm, để kiểm tra "tin theo sau giá"
    pre_car = ev.groupby("nhom")[[f"AR{k:+d}" for k in range(-5, 0)]].apply(lambda d: 100 * d.fillna(0).sum(axis=1).mean())
    md = ["# Phản ứng của giá với tin doanh nghiệp", "",
          f"Sự kiện: {len(ev):,} cặp (mã, phiên có tin), đã loại nhãn chỉ thuật lại biến động giá. "
          "AR = lợi suất của mã − benchmark ngành (đồng trọng số, mã thanh khoản).", "",
          "## CAR (%), chuẩn hóa = 0 tại phiên −1", "", key.round(3).to_markdown(), "",
          "## CAR trước tin, phiên [−5, −1] (%)", "", pre_car.round(3).rename("CAR[−5,−1]").to_markdown(), "",
          "## Hồi quy panel (hiệu ứng cố định theo ngày, SE gom cụm theo ngày)", "", reg.round(4).to_markdown(index=False), "",
          "## CAR theo thanh khoản (%)", "",
          (liq[liq["k"].isin([0, 1, 5])].pivot(index="nhom", columns="k", values="CAR") * 100).round(3).to_markdown(), ""]
    (REP / "reaction.md").write_text("\n".join(md), encoding="utf-8")
    pd.set_option("display.width", 200)
    print("CAR (%) tại k = −1, 0, 1, 5:"); print(key.round(3))
    print("CAR trước tin [−5,−1] (%):"); print(pre_car.round(3))
    print(reg.round(4).to_string(index=False))
    print("Theo thanh khoản:"); print((liq[liq["k"].isin([0, 1, 5])].pivot(index="nhom", columns="k", values="CAR") * 100).round(3))


if __name__ == "__main__":
    main()
