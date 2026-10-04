"""P0 — Data contract & EDA (§6 P0). Sinh bảng kiểm kê + biểu đồ cho chương Dữ liệu của báo cáo.

Dùng:  python -m src.eval.p0_report      (cần chạy src.run_pipeline trước)
Đầu ra: outputs/reports/p0_data_contract.md, outputs/figures/p0_*.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src.config import CLEAN, NEWS_FILES, RAW, ROOT

FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"

# Bảng màu tham chiếu (dataviz skill, references/palette.md — chế độ sáng)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]            # 3 slot đầu: đạt kiểm tra mọi cặp (CVD)
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#d8d7d1", "#2a78d6"
SRC_NAME = {"cafef": "CafeF", "vietstock": "Vietstock", "kenh14": "Kenh14"}
NHOM_VI = {"du_an_ha_tang": "Dự án – hạ tầng", "chinh_sach": "Chính sách", "thi_truong": "Thị trường",
           "phap_ly": "Pháp lý", "tai_chinh_dn": "Tài chính DN", "lai_suat_tin_dung": "Lãi suất – tín dụng"}

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titleweight": "semibold", "axes.titlesize": 12,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "savefig.dpi": 200,
})
THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def fig_articles_by_month(art):
    m = art.assign(month=art["session"].dt.to_period("M")).groupby(["month", "src"]).size().unstack(fill_value=0)
    m = m[["cafef", "vietstock", "kenh14"]]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(m)); bottom = np.zeros(len(m))
    for col, color in zip(m.columns, SERIES):
        ax.bar(x, m[col], bottom=bottom, width=0.72, color=color, edgecolor=SURFACE, linewidth=1.5,
               label=SRC_NAME[col])
        bottom += m[col].values
    ax.set_xticks(x[::3], [str(p) for p in m.index[::3]], rotation=0)
    ax.yaxis.set_major_formatter(THOUSANDS); ax.grid(axis="x", visible=False)
    ax.set_title("Số bài báo mỗi tháng theo nguồn (47.272 bài, 01/2024 – 09/2026)")
    ax.set_ylabel("Số bài")
    ax.legend(loc="upper left", ncols=3)
    _save(fig, "p0_1_bai_theo_thang.png")


def fig_label_polarity(lab):
    d = lab[lab["typ"].isin(["keyword", "law"])].drop_duplicates("article_id")
    d = d[d["nhom_tin"].isin(NHOM_VI)]
    share = d.groupby("nhom_tin")["impact_score"].apply(
        lambda s: pd.Series({"Âm": (s < 0).mean(), "Trung lập": (s == 0).mean(), "Dương": (s > 0).mean()})).unstack()
    n = d.groupby("nhom_tin").size()
    share = share.loc[share["Âm"].sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(share)); left = np.zeros(len(share))
    for col, color in (("Âm", DIV_NEG), ("Trung lập", DIV_MID), ("Dương", DIV_POS)):
        ax.barh(y, share[col], left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=1.5, label=col)
        left += share[col].values
    for i, (g, row) in enumerate(share.iterrows()):
        ax.text(row["Âm"] + 0.01 if row["Âm"] < 0.08 else row["Âm"] / 2, i, f"{row['Âm']:.0%}", va="center",
                ha="left" if row["Âm"] < 0.08 else "center", color=INK if row["Âm"] < 0.08 else "white", fontsize=9)
        ax.text(1.01, i, f"n = {n[g]:,}".replace(",", "."), va="center", color=MUTED, fontsize=9)
    ax.set_yticks(y, [NHOM_VI[g] for g in share.index])
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Nhãn LLM lệch dương: tỷ lệ nhãn âm / trung lập / dương theo nhóm tin (cấp ngành)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncols=3)
    _save(fig, "p0_2_nhan_theo_nhom.png")


def fig_ticker_coverage(atl, art):
    pairs = atl.merge(art[["article_id", "session"]], on="article_id").drop_duplicates(["ticker", "session"])
    cnt = pairs.groupby("ticker").size().sort_values(ascending=False)
    top = cnt.head(25)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh(top.index, top.values, height=0.66, color=SERIES[0])
    for i, v in enumerate(top.values):
        ax.text(v + 6, i, str(v), va="center", color=INK2, fontsize=8.5)
    ax.grid(axis="y", visible=False); ax.set_xlabel("Số cặp (mã, phiên) có tin")
    ax.set_title(f"Phủ sóng tin theo mã — 25/83 mã nhiều tin nhất (trung vị toàn bộ: {int(cnt.median())} phiên có tin)")
    _save(fig, "p0_3_phu_song_theo_ma.png")


def fig_prices(px, bench):
    ew = bench[bench["method"] == "ew_liquid"].set_index("date")["value"]
    close = px.pivot_table(index="date", columns="ticker", values="close")
    vin = close[["VIC", "VHM"]]
    vin = 100 * vin / vin.iloc[0]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    lines = [(ew, "Chỉ số ngành BĐS (đồng trọng số, mã thanh khoản)", SERIES[0]),
             (vin["VIC"], "VIC", SERIES[1]), (vin["VHM"], "VHM", SERIES[2])]
    for s, lab, c in lines:
        ax.plot(s.index, s.values, color=c, linewidth=2, label=lab)
        ax.annotate(f"{lab.split(' (')[0]}: {s.iloc[-1]:.0f}", (s.index[-1], s.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_yscale("log"); ax.set_yticks([50, 100, 200, 400, 800, 1100], ["50", "100", "200", "400", "800", "1.100"])
    ax.axhline(100, color=AXIS, linewidth=1)
    ax.set_ylabel("Chỉ số (02/01/2024 = 100, thang log)")
    chg = ew.iloc[-1] / ew.iloc[0] - 1
    ax.set_title(f"Vì sao dùng benchmark đồng trọng số: VIC tăng hơn 10 lần, chỉ số ngành {chg:+.0%}".replace("-", "−"))
    ax.legend(loc="upper left")
    _save(fig, "p0_4_gia_va_benchmark.png")


def fig_illiquid(px):
    ill = px.pivot_table(index="date", columns="ticker", values="illiquid")
    q = ill.groupby(pd.PeriodIndex(ill.index, freq="Q")).mean().sum(axis=1).iloc[1:]
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.bar([str(p) for p in q.index], q.values, width=0.62, color=SERIES[0])
    for i, v in enumerate(q.values):
        ax.text(i, v + 0.6, f"{v:.0f}", ha="center", color=INK2, fontsize=9)
    ax.set_ylim(0, q.max() * 1.15)
    ax.grid(axis="x", visible=False); ax.set_ylabel("Số mã")
    ax.set_title("Số mã kém thanh khoản trung bình theo quý (GTGD trung vị < 0,5 tỷ/phiên hoặc > 30% phiên đứng giá)")
    _save(fig, "p0_5_kem_thanh_khoan.png")


def contract(art, lab, atl, px, cal, ta, tb, ls, ev):
    rows = []
    for (typ, src), rel in NEWS_FILES.items():
        df = pd.read_csv(RAW / rel, usecols=["url"], encoding="utf-8-sig")
        rows.append({"file": rel, "loại": typ, "nguồn": src, "số dòng": len(df), "URL duy nhất": df["url"].nunique()})
    files = pd.DataFrame(rows)
    live_a, live_b = ta[~ta["warmup"]], tb[~tb["warmup"]]
    md = ["# P0 — Data contract (sinh tự động bởi `src/eval/p0_report.py`)", "",
          "## Tệp nguồn (`data final/`, chỉ đọc)", "", files.to_markdown(index=False), "",
          "## Bảng Tầng 1 (`data/clean/`)", "",
          "| Bảng | Khóa | Số dòng | Ghi chú |", "|---|---|---:|---|",
          f"| articles_clean | article_id | {len(art):,} | {art['session'].notna().mean():.1%} bài gán được phiên; {int(art['short_content'].sum())} bài < 200 ký tự; {int(art['dup_text'].sum())} bài trùng nội dung khác URL |",
          f"| article_label | article_id, typ, label_run | {len(lab):,} | keyword {int((lab.typ == 'keyword').sum()):,} · law {int((lab.typ == 'law').sum()):,} · suy ra {int((lab.typ == 'suy_ra').sum()):,} |",
          f"| article_ticker_label | article_id, ticker | {len(atl):,} | {atl['ticker'].nunique()} mã; {atl['price_report'].mean():.1%} nhãn chỉ thuật lại biến động giá |",
          f"| legal_streams | article_id | {len(ls):,} | {ls['stream'].value_counts(normalize=True).round(3).to_dict()} |",
          f"| policy_events | date | {len(ev):,} | {ev['event_id'].nunique()} đợt (k = 5) |",
          f"| trading_calendar | date | {len(cal):,} | {cal['date'].min().date()} → {cal['date'].max().date()} |",
          f"| stock_prices_clean | ticker, date | {len(px):,} | {px['ticker'].nunique()} mã; {int(px['suspended'].sum())} ô tạm ngưng |",
          f"| target_B | date | {len(tb):,} | sau khởi động: {int(live_b['yB_1'].notna().sum())} phiên có target h = 1 |",
          f"| target_A | date, ticker | {len(ta):,} | sau khởi động: {int(live_a['y_1'].notna().sum()):,} cặp có target h = 1 |",
          "", "## Kiểm tra tự động", "", "Chạy `py -3.12 -m pytest -q`: chốt phiên khớp bảng trust 100%, múi giờ, khóa duy nhất, "
          "không rò rỉ tương lai (đặc trưng, target, fold, Đ2).", "",
          "## Biểu đồ", ""] + [f"![{p.stem}](../figures/{p.name})" for p in sorted(FIG.glob("p0_*.png"))]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "p0_data_contract.md").write_text("\n".join(md), encoding="utf-8")


def main():
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "src", "session", "short_content", "dup_text"])
    lab = pd.read_parquet(CLEAN / "article_label.parquet")
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet")
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet")
    bench = pd.read_parquet(CLEAN / "benchmark_nganh.parquet")
    cal = pd.read_parquet(CLEAN / "trading_calendar.parquet")
    ta, tb = pd.read_parquet(CLEAN / "target_A.parquet"), pd.read_parquet(CLEAN / "target_B.parquet")
    ls, ev = pd.read_parquet(CLEAN / "legal_streams.parquet"), pd.read_parquet(CLEAN / "policy_events.parquet")
    fig_articles_by_month(art); fig_label_polarity(lab); fig_ticker_coverage(atl, art)
    fig_prices(px, bench); fig_illiquid(px)
    contract(art, lab, atl, px, cal, ta, tb, ls, ev)
    print("Đã ghi:", sorted(p.name for p in FIG.glob("p0_*.png")), "+ outputs/reports/p0_data_contract.md")


if __name__ == "__main__":
    main()
