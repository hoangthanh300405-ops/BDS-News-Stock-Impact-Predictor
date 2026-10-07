"""KIỂM CHỨNG NHÃN LLM NHIỀU CHIỀU (thay cho tập vàng gán tay khi không có thời gian gán).

  V1  Nhất quán nội bộ: cùng một bài được qwen gán ở luồng ngành (keyword/law) và luồng mã (company).
  V2  Hai LLM độc lập: qwen2.5:7b so với GPT-4.1-mini (file data/manual/gpt41mini_labels.csv,
      3.434 bài CafeF 10/2025 → 09/2026, nhãn cấp toàn bài).
  V3  Trọng tài thứ ba (Claude, gán MÙ — không thấy nhãn hai LLM kia) trên mẫu phân tầng bất đồng/đồng thuận:
      file data/manual/ref_sample.csv (xuất bởi `--export`) và data/manual/ref_labels.csv (nhãn trọng tài).
  V4  Soi bằng từ khóa rõ nghĩa trong tiêu đề.
  V5  Kiểm chứng bằng phản ứng giá: AR (trừ rổ BĐS + xu hướng riêng của mã) theo nhãn, cho các mã được nhắc.
  V6  Độ vững của kết luận "giá chạy trước tin": đường CAR theo hướng nhãn qwen / GPT / hai bên đồng thuận.
Đầu ra: outputs/reports/label_validation.md (+ .csv), outputs/figures/r13_label_car.png
Dùng:  python -m src.nlp.label_validation --export     (xuất mẫu mù cho trọng tài)
       python -m src.nlp.label_validation              (chạy toàn bộ; dùng nhãn trọng tài nếu đã có)
"""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.config import CLEAN, ROOT
from src.data.p1_news import article_id

GPT_FILE = ROOT / "data" / "manual" / "gpt41mini_labels.csv"
GOLD = ROOT / "data" / "manual"
REP, FIG = ROOT / "outputs" / "reports", ROOT / "outputs" / "figures"
SIGN = {"tich_cuc": 1, "trung_lap": 0, "tieu_cuc": -1}
RE_NEG = re.compile(r"khởi tố|bắt tạm giam|vỡ nợ|chậm trả|thua lỗ|lỗ ròng|lỗ nặng|cưỡng chế|đình chỉ|thu hồi dự án|"
                    r"sai phạm|vi phạm|phạt", re.I)
RE_POS = re.compile(r"lãi kỷ lục|lợi nhuận tăng|lãi tăng|vượt kế hoạch|được phê duyệt|được chấp thuận|trúng thầu", re.I)
N_DIS, N_AGR, SEED = 110, 40, 2026


def kappa_row(name, a, b, weights=None):
    return {"so sánh": name, "n": len(a), "tỷ lệ trùng": float(np.mean(np.asarray(a) == np.asarray(b))),
            "κ": cohen_kappa_score(a, b, weights=weights)}


def load_pair():
    """Bảng 3.434 bài: nhãn GPT (toàn bài) cạnh nhãn qwen cùng luồng keyword."""
    g = pd.read_csv(GPT_FILE)
    g["article_id"] = g["url"].map(article_id)
    lab = pd.read_parquet(CLEAN / "article_label.parquet")
    kw = lab[lab["typ"] == "keyword"].groupby("article_id").agg(et_q=("event_type", "first"), s_q=("impact_score", "mean"))
    m = g.join(kw, on="article_id", how="inner")
    m["sg"] = m["impact_co_phieu"].map(SIGN)
    m["sq"] = np.sign(m["s_q"].round(6)).astype(int)
    return m


def v1():
    lab = pd.read_parquet(CLEAN / "article_label.parquet")
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet")
    ind = lab[lab["typ"].isin(["keyword", "law"])].groupby("article_id").agg(a=("event_type", "first"), sa=("impact_score", "mean"))
    com = atl.groupby("article_id").agg(b=("event_type", lambda s: s.mode().iat[0]), sb=("impact_score", "mean"))
    x = ind.join(com, how="inner")
    return [kappa_row("V1 qwen luồng ngành – qwen luồng mã · loại sự kiện", x["a"], x["b"]),
            kappa_row("V1 qwen luồng ngành – qwen luồng mã · dấu", np.sign(x["sa"].round(6)), np.sign(x["sb"].round(6)))]


def v2(m):
    return [kappa_row("V2 GPT-4.1-mini – qwen · loại sự kiện", m["event_type"], m["et_q"]),
            kappa_row("V2 GPT-4.1-mini – qwen · dấu", m["sg"], m["sq"])], \
        pd.crosstab(m["sg"], m["sq"], rownames=["GPT"], colnames=["qwen"])


def export_sample(m):
    """Mẫu mù cho trọng tài: N_DIS bài bất đồng về dấu + N_AGR bài đồng thuận; không kèm nhãn nào."""
    rng = np.random.default_rng(SEED)
    dis, agr = m[m["sg"] != m["sq"]], m[m["sg"] == m["sq"]]
    pick = pd.concat([dis.iloc[rng.choice(len(dis), N_DIS, replace=False)].assign(nhom="bat_dong", pi=N_DIS / len(dis)),
                      agr.iloc[rng.choice(len(agr), N_AGR, replace=False)].assign(nhom="dong_thuan", pi=N_AGR / len(agr))])
    pick = pick.sample(frac=1, random_state=SEED).reset_index(drop=True)
    pick["tt_id"] = [f"T{i + 1:03d}" for i in range(len(pick))]
    GOLD.mkdir(parents=True, exist_ok=True)
    pick[["tt_id", "article_id", "nhom", "pi"]].to_csv(GOLD / "ref_key.csv", index=False, encoding="utf-8-sig")
    blind = pick[["tt_id", "published_at", "title", "description", "content"]].copy()
    blind["content"] = blind["content"].fillna("").str[:900]
    blind.to_csv(GOLD / "ref_sample.csv", index=False, encoding="utf-8-sig")
    print(f"Đã xuất {len(pick)} bài mù -> {GOLD / 'ref_sample.csv'} (bất đồng {N_DIS}, đồng thuận {N_AGR})")


def v3(m):
    f = GOLD / "ref_labels.csv"
    if not f.exists():
        return [], None
    key = pd.read_csv(GOLD / "ref_key.csv")
    t = pd.read_csv(f).merge(key, on="tt_id").merge(m[["article_id", "sg", "sq", "event_type", "et_q"]], on="article_id")
    rows, out = [], {}
    for grp, x in [("tất cả", t)] + list(t.groupby("nhom")):
        for who, col in (("GPT-4.1-mini", "sg"), ("qwen", "sq")):
            rows.append({"so sánh": f"V3 trọng tài – {who} · dấu · {grp}", "n": len(x),
                         "tỷ lệ trùng": float((x["tt_dau"] == x[col]).mean()),
                         "κ": cohen_kappa_score(x["tt_dau"], x[col]) if x[col].nunique() > 1 and x["tt_dau"].nunique() > 1 else np.nan})
    for who, col in (("GPT-4.1-mini", "event_type"), ("qwen", "et_q")):
        rows.append(kappa_row(f"V3 trọng tài – {who} · loại sự kiện", t["tt_loai"], t[col]))
    # ước lượng cho TOÀN BỘ 3.434 bài: trọng số 1/π theo nhóm
    w = 1 / t["pi"]
    for who, col in (("GPT-4.1-mini", "sg"), ("qwen", "sq")):
        out[who] = float(np.average(t["tt_dau"] == t[col], weights=w))
    d = t[t["nhom"] == "bat_dong"]
    out["bất đồng: trọng tài theo GPT"] = float((d["tt_dau"] == d["sg"]).mean())
    out["bất đồng: trọng tài theo qwen"] = float((d["tt_dau"] == d["sq"]).mean())
    out["bất đồng: trọng tài khác cả hai"] = float(((d["tt_dau"] != d["sg"]) & (d["tt_dau"] != d["sq"])).mean())
    a = t[t["nhom"] == "dong_thuan"]
    out["đồng thuận: trọng tài cũng đồng ý"] = float((a["tt_dau"] == a["sg"]).mean())
    # tách loại lỗi (ước lượng cho toàn bộ 3.434 bài, trọng số 1/π)
    for who, col in (("GPT-4.1-mini", "sg"), ("qwen", "sq")):
        out[f"{who}: ngược chiều trọng tài"] = float(np.average(t[col] * t["tt_dau"] == -1, weights=w))
        out[f"{who}: chấm tích cực khi trọng tài chấm 0"] = float(np.average((t[col] == 1) & (t["tt_dau"] == 0), weights=w))
        out[f"{who}: bỏ sót tin xấu (trong các bài trọng tài chấm âm)"] = float((t.loc[t["tt_dau"] == -1, col] != -1).mean())
    return rows, out


def v4(m):
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet")
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "title"])
    x = atl.merge(art, on="article_id")
    rows = []
    for name, rx, sg in (("tiêu đề rõ tiêu cực", RE_NEG, -1), ("tiêu đề rõ tích cực", RE_POS, 1)):
        s = x[x["title"].str.contains(rx)]
        rows.append({"kiểm tra": f"V4 qwen (nhãn theo mã, toàn kho) · {name}", "n": len(s),
                     "cùng dấu": float((np.sign(s["impact_score"]) == sg).mean()),
                     "chấm 0": float((s["impact_score"] == 0).mean()), "ngược dấu": float((np.sign(s["impact_score"]) == -sg).mean())})
        s = m[m["title"].str.contains(rx)]
        for who, col in (("GPT-4.1-mini", "sg"), ("qwen", "sq")):
            rows.append({"kiểm tra": f"V4 {who} (3.434 bài) · {name}", "n": len(s), "cùng dấu": float((s[col] == sg).mean()),
                         "chấm 0": float((s[col] == 0).mean()), "ngược dấu": float((s[col] == -sg).mean())})
    return rows


def ticker_pairs(m):
    """(bài, mã được nhắc) với phiên của bài — dùng cho V5, V6."""
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session"]).set_index("article_id")
    m = m.assign(session=pd.to_datetime(m["article_id"].map(art["session"]))).dropna(subset=["session"])
    rows = [(r.article_id, t.strip(), r.session, r.sg, r.sq) for r in m.itertuples()
            if isinstance(r.ma_co_phieu_lien_quan, str) for t in r.ma_co_phieu_lien_quan.split(",")]
    return pd.DataFrame(rows, columns=["article_id", "ticker", "date", "sg", "sq"])


def v5_v6(m):
    from src.analysis.retail_analytics import ar_panel, paths
    ar = ar_panel()
    p = ticker_pairs(m)
    p = p[p["ticker"].isin(ar.columns) & p["date"].isin(ar.index)].reset_index(drop=True)
    P = paths(p.assign(d=1), ar)                        # AR thô tại k = −10..+20 (chưa nhân hướng)
    rows = []
    for who, col in (("GPT-4.1-mini", "sg"), ("qwen", "sq")):
        for s, x in p.groupby(col):
            rows.append({"kiểm tra": f"V5 {who}", "nhãn": {1: "tích cực", 0: "trung lập", -1: "tiêu cực"}[s], "n": len(x),
                         "AR[−1,0] TB (%)": P.loc[x.index, [-1, 0]].sum(axis=1).mean() * 100,
                         "CAR[−5,0] TB (%)": P.loc[x.index, list(range(-5, 1))].sum(axis=1).mean() * 100,
                         "CAR[+1,+5] TB (%)": P.loc[x.index, list(range(1, 6))].sum(axis=1).mean() * 100})
    curves = {}
    for name, sel, sgn in (("theo nhãn qwen", p["sq"] != 0, p["sq"]), ("theo nhãn GPT-4.1-mini", p["sg"] != 0, p["sg"]),
                           ("hai LLM đồng thuận", (p["sq"] == p["sg"]) & (p["sq"] != 0), p["sq"])):
        x = P.loc[sel].mul(sgn[sel], axis=0)
        curves[name] = (int(sel.sum()), x.mean().cumsum() * 100)
    return rows, curves


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for (name, (n, c)), col in zip(curves.items(), ["#2a78d6", "#eb6834", "#1baf7a"]):
        ax.plot(c.index, c.values, marker="o", ms=3, lw=2, color=col, label=f"{name} (n={n:,})")
    ax.axvline(0.5, color="k", ls="--", lw=1); ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("phiên so với phiên có tin"); ax.set_ylabel("CAR theo hướng nhãn, %")
    ax.set_title("V6. Kết luận 'giá chạy trước tin' có phụ thuộc vào LLM gán nhãn không?")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "r13_label_car.png", dpi=150); plt.close(fig)


def main():
    m = load_pair()
    if "--export" in sys.argv:
        return export_sample(m)
    rows = v1()
    r2, cm = v2(m)
    rows += r2
    r3, adj = v3(m)
    rows += r3
    kap = pd.DataFrame(rows)
    chk = pd.DataFrame(v4(m))
    r5, curves = v5_v6(m)
    price = pd.DataFrame(r5)
    plot(curves)
    kap.to_csv(REP / "label_validation_kappa.csv", index=False, encoding="utf-8-sig")
    chk.to_csv(REP / "label_validation_keywords.csv", index=False, encoding="utf-8-sig")
    price.to_csv(REP / "label_validation_price.csv", index=False, encoding="utf-8-sig")
    fmt = lambda df: df.round(3).to_markdown(index=False)
    car_tab = pd.DataFrame({k: {"n": n, "CAR đến k=−1": c.loc[-1], "CAR đến k=0": c.loc[0], "CAR đến k=+5": c.loc[5],
                                "CAR đến k=+20": c.loc[20]} for k, (n, c) in curves.items()}).T
    md = ["# Kiểm chứng nhãn LLM nhiều chiều (không có nhãn người)\n",
          "## Độ đồng thuận (V1–V3)\n", fmt(kap), "\n\nMa trận dấu GPT-4.1-mini (hàng) × qwen (cột):\n", cm.to_markdown(),
          "\n\n## Trọng tài (V3)\n", "\n".join(f"- {k}: {v:.1%}" for k, v in (adj or {}).items()) or "- chưa có nhãn trọng tài",
          "\n\n## Soi bằng từ khóa tiêu đề (V4)\n", fmt(chk), "\n\n## Phản ứng giá theo nhãn (V5)\n", fmt(price),
          "\n\n## Độ vững của kết luận (V6)\n", car_tab.round(3).to_markdown()]
    (REP / "label_validation.md").write_text("\n".join(md), encoding="utf-8")
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(kap.to_string(index=False)); print(cm); print(adj); print(chk.to_string(index=False))
        print(price.to_string(index=False)); print(car_tab.round(3))


if __name__ == "__main__":
    main()
