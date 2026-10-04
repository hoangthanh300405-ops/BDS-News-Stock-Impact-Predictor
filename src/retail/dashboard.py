"""Dựng DASHBOARD nhà đầu tư nhỏ lẻ (1 file HTML, dữ liệu nhúng sẵn) từ các kết quả đã chạy.

Cần chạy trước: src.retail.run_final, src.retail.predict_latest, src.analysis.retail_analytics, src.retail.robustness,
src.sector.run_sector + src.sector.tune_sector (ô "Bối cảnh ngành", chỉ để tham khảo; thiếu file thì ô bị ẩn).
Đầu ra: outputs/dashboard/doc_tin_co_nen_mua.html
Dùng:  python -m src.retail.dashboard
"""
import json

import numpy as np
import pandas as pd

from src.config import CLEAN, ROOT
from src.retail.events import COST, price_panels
from src.retail.portfolio import CAPITAL, PRED, simulate

REP = ROOT / "outputs" / "reports"
OUT = ROOT / "outputs" / "dashboard"
TEMPLATE = ROOT / "src" / "retail" / "dashboard_template.html"


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def sector_context(n_last=5):
    """Bối cảnh ngành cho các phiên cuối: số liệu mô tả (tin ngành, rổ BĐS) + nhận định của mô hình phụ (tham khảo)."""
    f = REP / "sector_tune_moi_nhat.csv"
    if not f.exists():
        return None
    from src.sector.run_sector import build
    d = build().set_index("date")
    m = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    lab = {"MUA rổ": "Thuận lợi", "ĐỨNG NGOÀI": "Kém thuận lợi"}
    out = []
    for t in m.index[-n_last:]:
        x = d.loc[t]
        out.append({"t": t.strftime("%Y-%m-%d"), "n": int(round(np.expm1(x["n_cnt1"]))), "tone": r(x["n_tone1"], 2),
                    "tone20": r(x["n_tone20"], 2), "neg5": r(x["n_neg5"]), "legal5": int(round(np.expm1(x["n_legal_risk5"]))),
                    "r20": r(x["p_r20"]), "breadth": r(x["p_breadth1"]), "p": r(m.loc[t, "điểm 0–100"], 0),
                    "h": int(m.loc[t, "kỳ nắm giữ (phiên)"]), "v": lab.get(m.loc[t, "khuyến nghị"], "Trung tính")})
    return out


def main():
    res = pd.read_parquet(PRED / "final_tuyet_doi.parquet")
    cal = pd.DatetimeIndex(pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"])
    close, _, illiq, _ = price_panels()
    good = res[(res["d"] > 0) & res["ret_5"].notna()]

    # ---- đường vốn P4 (A = trung vị 200 lần chọn ngẫu nhiên, như portfolio.py)
    rng = np.random.default_rng(2026)
    runs = [simulate(good, cal, rng=rng)[0] for _ in range(200)]
    eqA = runs[int(np.argsort([e.iloc[-1] for e in runs])[100])]
    lr = np.log(close).diff().where(illiq == 0).mean(axis=1).loc[eqA.index[0]:eqA.index[-1]]
    eqB = CAPITAL * (1 - COST) * np.exp(lr.fillna(0).cumsum())
    eqC = simulate(good[good["top10|E3"]], cal, score="s|E3")[0]
    eqD = simulate(good[good["top20|CONS"]], cal, score="s|E3")[0]
    days = eqA.index
    curves = {"dates": [d.strftime("%Y-%m-%d") for d in days],
              "series": [{"key": k, "values": [r(v, 2) for v in e.reindex(days).ffill().bfill()]}
                         for k, e in (("C", eqC), ("D", eqD), ("A", eqA), ("B", eqB))]}

    # ---- đường CAR (A1)
    a1 = pd.read_csv(REP / "retail_A1_car.csv")
    car = {g: {"n": int(x["n"].iloc[0]), "k": x["k"].tolist(), "v": [r(v * 100, 3) for v in x["CAR từ −10"]]}
           for g, x in a1.groupby("nhóm")}

    # ---- độ vững (P5)
    rb = pd.read_csv(REP / "retail_robustness.csv")
    robust = [{"name": row["kịch bản"], "A": r(row["A · mọi tin tốt"]), "B": r(row["B · rổ BĐS"]),
               "C": r(row["C · tiêu chuẩn (top 10%)"]), "D": r(row["C · thận trọng (đồng thuận)"])} for _, row in rb.iterrows()]

    # ---- tin trên tập test + giá
    ev = res[res["ret_5"].notna()].copy()
    ev["loai"] = np.select([ev["frac_firm"] > 0.5, ev["frac_comm"] > 0.5], ["sự kiện DN", "bình luận TT"], "khác")
    events = [{"t": d.strftime("%Y-%m-%d"), "tk": tk, "ti": (ti or "")[:160], "et": et, "d": int(dd), "lo": lo,
               "p": r(p, 3), "t20": bool(t20), "t10": bool(t10), "cs": bool(cs), "r5": r(r5), "pn": r(pn), "mv": r(mv)}
              for d, tk, ti, et, dd, lo, p, t20, t10, cs, r5, pn, mv in zip(
                  ev["date"], ev["ticker"], ev["title"], ev["event_type"], ev["d"], ev["loai"], ev["s|E3"],
                  ev["top20|E3"], ev["top10|E3"], ev["top20|CONS"], ev["ret_5"], ev["pnl_5"], ev["d_r5"])]
    tick = sorted(ev["ticker"].unique(), key=lambda t: -int((ev["ticker"] == t).sum()))
    px = close.loc["2025-02-01":ev["date"].max() + pd.Timedelta(days=20), tick]
    prices = {"dates": [d.strftime("%Y-%m-%d") for d in px.index],
              "close": {t: [r(v, 2) for v in px[t]] for t in tick}}

    good_all = res[(res["d"] > 0) & res["ret_5"].notna()]
    kpi = {"n_test": int(len(ev)), "hitA": r(good_all["pnl_5"].gt(0).mean()),
           "hitC": r(good_all.loc[good_all["top10|E3"], "pnl_5"].gt(0).mean()),
           "hitD": r(good_all.loc[good_all["top20|CONS"], "pnl_5"].gt(0).mean()),
           "retA": r(eqA.iloc[-1] / CAPITAL - 1), "retB": r(eqB.iloc[-1] / CAPITAL - 1),
           "retC": r(eqC.iloc[-1] / CAPITAL - 1), "retD": r(eqD.iloc[-1] / CAPITAL - 1)}
    cov = res["hi10"] | res["lo10"]                        # chế độ tự tin hai phía (mọi tin, cả tốt lẫn xấu)
    kpi["acc2"] = r(((res["hi10"] & (res["follow_ok"] == 1)) | (res["lo10"] & (res["follow_ok"] == 0)))[cov].mean())
    kpi["cov2"] = r(cov.mean())
    lt = pd.read_csv(REP / "khuyen_nghi_moi_nhat.csv")
    latest = [{"t": str(x["phiên"])[:10], "tk": x["mã"], "ti": str(x["tiêu đề"])[:140], "d": int(x["hướng tin"]),
               "mv": r(x["giá đã chạy 5 phiên theo hướng tin"]), "p": r(x["điểm (0–100)"], 1),
               "std": x["khuyến nghị · tiêu chuẩn"], "cons": x["khuyến nghị · thận trọng"], "two": x["khuyến nghị · hai phía"]}
              for _, x in lt.iterrows()]
    data = {"sector": sector_context(), "kpi": kpi, "curves": curves, "car": car, "robust": robust, "events": events, "prices": prices,
            "tickers": tick, "latest": latest}
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data, ensure_ascii=False,
                                                                                separators=(",", ":")))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "doc_tin_co_nen_mua.html"
    path.write_text(html, encoding="utf-8")
    print(f"Đã ghi {path} ({path.stat().st_size / 1e6:.2f} MB) | {len(events):,} tin, {len(tick)} mã")


if __name__ == "__main__":
    main()
