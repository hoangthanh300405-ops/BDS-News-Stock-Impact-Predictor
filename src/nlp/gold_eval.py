"""Trụ cột 1(a) — CHẤM TẬP VÀNG sau khi nhóm gán xong data/gold/tap_vang_gan_nhan.xlsx.

  1. Độ tin cậy giữa người với người (100 bài trùng): Cohen's κ cho mức liên quan, loại sự kiện; κ có trọng số
     (bậc hai) cho sắc thái −2..2 — nếu người với người còn không thống nhất thì chính nhiệm vụ gán nhãn đã mơ hồ.
  2. LLM so với người (300 bài, người A làm chuẩn): độ chính xác, κ, ma trận nhầm lẫn — có trọng số 1/π để ước lượng
     cho TOÀN BỘ kho tin (mẫu được chọn phân tầng, không đều).
  3. Mô hình chưng cất (tf-idf) so với người — nếu đã chạy src.nlp.classifiers (cột dự đoán trong gold_key).
Dùng:  python -m src.nlp.gold_eval [đường_dẫn_xlsx]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.config import ROOT

GOLD = ROOT / "data" / "gold"
COLS = {"MỨC LIÊN QUAN": "relevance", "LOẠI SỰ KIỆN": "event_type", "SẮC THÁI (−2..2)": "sentiment"}


def read(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, dtype=str).rename(columns=COLS)
    df = df[["gold_id"] + list(COLS.values())].dropna(subset=list(COLS.values()), how="all")
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    return df


def wacc(y, p, w):
    ok = pd.notna(y) & pd.notna(p)
    return np.average((np.asarray(y)[ok] == np.asarray(p)[ok]).astype(float), weights=np.asarray(w)[ok])


def main(path=GOLD / "tap_vang_gan_nhan.xlsx"):
    A, B = read(path, "Nguoi_A"), read(path, "Nguoi_B")
    key = pd.read_parquet(GOLD / "gold_key.parquet")
    print(f"Đã gán: người A {len(A)}/300, người B {len(B)}/100")
    rows = []
    # ---- 1. người với người
    ab = A.merge(B, on="gold_id", suffixes=("_A", "_B"))
    for c in COLS.values():
        x = ab.dropna(subset=[f"{c}_A", f"{c}_B"])
        if len(x) < 10:
            continue
        kw = {"weights": "quadratic"} if c == "sentiment" else {}
        rows.append({"so sánh": "Người A – Người B", "nhãn": c, "n": len(x),
                     "đồng ý": (x[f"{c}_A"] == x[f"{c}_B"]).mean(),
                     "κ": cohen_kappa_score(x[f"{c}_A"].astype(str), x[f"{c}_B"].astype(str), **kw)
                     if c != "sentiment" else cohen_kappa_score(x[f"{c}_A"].astype(int), x[f"{c}_B"].astype(int), **kw)})
    # ---- 2. LLM với người A
    g = A.merge(key, on="gold_id")
    g["llm_sent"] = np.round(g["llm_score"]).clip(-2, 2)
    w = 1 / g["pi"]
    for c, llm in (("event_type", "llm_event_type"), ("sentiment", "llm_sent")):
        x = g.dropna(subset=[c, llm])
        if len(x) < 10:
            continue
        yt, yp = (x[c].astype(str), x[llm].astype(str)) if c == "event_type" else (x[c].astype(int), x[llm].astype(int))
        kw = {"weights": "quadratic"} if c == "sentiment" else {}
        rows.append({"so sánh": "LLM – Người A", "nhãn": c, "n": len(x), "đồng ý": (yt == yp).mean(),
                     "đồng ý (trọng số 1/π, ước lượng toàn kho)": wacc(yt, yp, w[x.index]),
                     "κ": cohen_kappa_score(yt, yp, **kw)})
        if c == "sentiment":                                   # dấu 3 lớp — thứ mô hình dự báo thực sự dùng
            s_t, s_p = np.sign(x[c]).astype(int), np.sign(x[llm]).astype(int)
            rows.append({"so sánh": "LLM – Người A", "nhãn": "dấu sắc thái (3 lớp)", "n": len(x),
                         "đồng ý": (s_t == s_p).mean(), "đồng ý (trọng số 1/π, ước lượng toàn kho)": wacc(s_t, s_p, w[x.index]),
                         "κ": cohen_kappa_score(s_t, s_p)})
            cm = pd.DataFrame(confusion_matrix(s_t, s_p, labels=[-1, 0, 1]),
                              index=["người: âm", "người: 0", "người: dương"], columns=["LLM: âm", "LLM: 0", "LLM: dương"])
    # ---- 3. mô hình chưng cất (nếu có)
    for c, col in (("event_type", "distil_event_type"), ("sentiment", "distil_sent")):
        if col in key:
            x = g.dropna(subset=[c, col])
            yt = x[c].astype(str) if c == "event_type" else np.sign(x[c]).astype(int).astype(str)
            rows.append({"so sánh": "Mô hình tf-idf – Người A", "nhãn": c if c == "event_type" else "dấu sắc thái (3 lớp)",
                         "n": len(x), "đồng ý": (yt == x[col].astype(str)).mean(),
                         "κ": cohen_kappa_score(yt, x[col].astype(str))})
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "reports" / "gold_eval.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))
        if "cm" in locals():
            print("\nMa trận nhầm lẫn dấu sắc thái (người A là chuẩn):"); print(cm)
    print("\nThang đọc κ (Landis & Koch, 1977): <0,2 kém · 0,21–0,40 khá thấp · 0,41–0,60 trung bình · 0,61–0,80 tốt · >0,80 rất tốt")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else GOLD / "tap_vang_gan_nhan.xlsx")
