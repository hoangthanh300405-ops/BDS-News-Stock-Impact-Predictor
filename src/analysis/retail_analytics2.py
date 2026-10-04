"""Phân tích mở rộng A4–A9: tác động của tin khác nhau thế nào theo đặc điểm tin, mã và thị trường.

Cùng khung với A1–A2: AR = lợi suất − rổ BĐS − xu hướng riêng của mã (t−130…t−11); nhân với hướng tin (d);
t-thống kê theo phiên (trung bình trong phiên rồi t-test giữa các phiên). Ngoài CAR còn báo tỷ lệ "làm theo tin
thắng rổ BĐS" (follow_rel) và "có lãi sau phí" (follow_ok), giữ 5 phiên.
  A4  Câu chuyện: tin MỞ ĐẦU câu chuyện so với tin VIẾT LẠI (cần src.nlp.stories)
  A5  Nguồn báo chính của phiên: CafeF / Vietstock / Kenh14
  A6  Thời điểm đăng: toàn bộ trước giờ mở cửa / trong phiên / lẫn lộn
  A7  Chế độ thị trường: rổ BĐS 20 phiên trước tăng hay giảm × tin tốt / xấu
  A8  Thanh khoản của mã (tam phân vị giá trị giao dịch 20 phiên) và sàn
  A9  Sự chú ý: khối lượng đột biến ngày có tin (tam phân vị value_z) — Barber & Odean (2008)
Mô tả toàn mẫu (không phải kiểm định dự báo); có hiệu chỉnh kiểm định bội BH trên CAR[+1,+5].
Đầu ra: outputs/reports/retail_A4_A9.csv (+ .md); outputs/figures/r14_heterogeneity.png
Dùng:  python -m src.analysis.retail_analytics2
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.retail_analytics import ar_panel, date_t, paths
from src.config import CLEAN, ROOT

REP, FIG = ROOT / "outputs" / "reports", ROOT / "outputs" / "figures"
SRC_NAME = {"cafef": "CafeF", "vietstock": "Vietstock", "kenh14": "Kenh14"}


def event_attrs(ev):
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet", columns=["article_id", "ticker", "src", "event_type"])
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session"])
    x = atl[atl["event_type"] != "KHONG_LIEN_QUAN"].merge(art, on="article_id")
    x["date"] = pd.to_datetime(x["session"])
    src = x.groupby(["date", "ticker"])["src"].agg(lambda s: s.mode().iat[0]).rename("src_main")
    ev = ev.join(src, on=["date", "ticker"])
    st_path = CLEAN / "stories.parquet"
    if st_path.exists():
        st = pd.read_parquet(st_path).merge(x[["article_id", "ticker", "date"]], on=["article_id", "ticker"])
        new = st.groupby(["date", "ticker"])["story_rank"].agg(lambda r: (r == 1).any()).rename("has_new_story")
        ev = ev.join(new, on=["date", "ticker"])
    return ev


def groups(ev):
    g = {}
    if "has_new_story" in ev:
        g["A4 Câu chuyện"] = np.where(ev["has_new_story"].fillna(False), "có tin mở đầu câu chuyện", "chỉ tin viết lại / tiếp nối")
    g["A5 Nguồn chính"] = ev["src_main"].map(SRC_NAME).fillna("khác")
    g["A6 Thời điểm đăng"] = np.select([ev["frac_before_open"] >= 1, ev["frac_before_open"] <= 0],
                                       ["trước giờ mở cửa", "trong phiên"], "lẫn lộn")
    mkt_up = ev["d_mkt20"] * ev["d"] > 0
    g["A7 Thị trường × hướng tin"] = np.where(mkt_up, "rổ BĐS đang tăng", "rổ BĐS đang giảm") + np.where(
        ev["d"] > 0, " · tin tốt", " · tin xấu")
    g["A8 Thanh khoản"] = pd.qcut(ev["log_value"], 3, labels=["thanh khoản thấp", "trung bình", "cao"]).astype(str)
    g["A8 Sàn"] = np.where(ev["hose"] == 1, "HOSE", "HNX")
    g["A9 Khối lượng ngày có tin"] = pd.qcut(ev["value_z"].replace([np.inf, -np.inf], np.nan), 3,
                                             labels=["bình thường/thấp", "tăng vừa", "đột biến"]).astype(str)
    return g


def summarize(ev, P):
    rows = []
    for dim, lab in groups(ev).items():
        s = pd.Series(lab, index=ev.index)
        for k, idx in s.groupby(s).groups.items():
            if k in ("nan", "khác") or len(idx) < 60:
                continue
            r = {"chiều": dim, "nhóm": k, "n": len(idx)}
            for name, ks in (("CAR[−5,−1]", range(-5, 0)), ("AR[0]", [0]), ("CAR[+1,+5]", range(1, 6)), ("CAR[+1,+20]", range(1, 21))):
                v = P.loc[idx, list(ks)].sum(axis=1, min_count=len(list(ks)))
                m, t = date_t(v, ev.loc[idx, "date"])
                r[name + " (%)"], r["t " + name] = m * 100, t
            sub = ev.loc[idx]
            r["làm theo thắng rổ"] = sub["follow_rel"].mean()
            r["làm theo có lãi sau phí"] = sub["follow_ok"].mean()
            rows.append(r)
    out = pd.DataFrame(rows)
    p = 2 * stats.norm.sf(out["t CAR[+1,+5]"].abs())
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out["q (BH) CAR[+1,+5]"] = np.minimum(q, 1)
    return out


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o = out.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.34 * len(o) + 1.2))
    se = (o["CAR[+1,+5] (%)"] / o["t CAR[+1,+5]"]).abs()
    col = np.where(o["q (BH) CAR[+1,+5]"] < 0.1, "#2a78d6", "#8c959f")
    ax.errorbar(o["CAR[+1,+5] (%)"], range(len(o)), xerr=1.96 * se, fmt="none", ecolor="#b0b7bd", lw=1.2)
    ax.scatter(o["CAR[+1,+5] (%)"], range(len(o)), color=col, s=26, zorder=3)
    ax.axvline(0, color="k", lw=.7)
    ax.set_yticks(range(len(o)), [f"{c.split(' ', 1)[1]} · {g} (n={n:,})" for c, g, n in zip(o["chiều"], o["nhóm"], o["n"])], fontsize=8)
    ax.set_xlabel("CAR[+1,+5] theo hướng tin, % (sau khi nhà đầu tư giao dịch được) · KTC 95%")
    ax.set_title("A4–A9. Sau khi đọc tin còn 'dư địa' ở đâu?\n"
                 "(chấm xanh = có ý nghĩa sau hiệu chỉnh BH, q < 0,1; xám = không)", fontsize=11)
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(FIG / "r14_heterogeneity.png", dpi=150); plt.close(fig)


def main():
    ev = pd.read_parquet(CLEAN / "retail_events.parquet")
    ev = event_attrs(ev)
    P = paths(ev, ar_panel())
    out = summarize(ev, P)
    out.to_csv(REP / "retail_A4_A9.csv", index=False, encoding="utf-8-sig")
    (REP / "retail_A4_A9.md").write_text(out.round(3).to_markdown(index=False), encoding="utf-8")
    figure(out)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(out.drop(columns=[c for c in out if c.startswith("t CAR[−5") or c.startswith("t AR")]).to_string(index=False))


if __name__ == "__main__":
    main()
