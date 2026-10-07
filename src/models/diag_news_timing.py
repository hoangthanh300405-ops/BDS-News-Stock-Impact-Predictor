"""Chẩn đoán RQ1: tin đã phản ánh vào giá trong phiên t hay còn tác động sau đó?   Dùng: python -m src.models.diag_news_timing"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.models.run_rq1 import data_A, data_B

a = data_A(); a = a[a.date >= "2024-04-01"].copy()
a["ar_today"] = a["r_1d"] - a["rB_1d"]
b = data_B(); b = b[b.date >= "2024-04-01"].copy()


def rank_ic(df, x, y, min_n=5):
    ics = []
    for _, g in df.groupby("date"):
        g = g[[x, y]].dropna()
        if len(g) >= min_n and g[x].nunique() > 1:
            ics.append(spearmanr(g[x], g[y])[0])
    s = pd.Series(ics).dropna()
    return s.mean(), s.mean() / (s.std() / np.sqrt(len(s))), len(s)


print("#### CAP MA — Rank IC trung binh theo phien (t-stat)")
for sub_name, sub in (("tat ca cap trong mau", a), ("chi cap CO TIN HOM NAY", a[a.news_today == 1])):
    print(f"  [{sub_name}]  n = {len(sub)}")
    for x in ["TONE_i", "NEG_i", "TONE_gap"]:
        out = []
        for y, lab in (("ar_today", "AR phien t (cung phien)"), ("z_1", "z phien t+1"), ("z_2_5", "z phien t+2..t+5")):
            m, t, n = rank_ic(sub, x, y)
            out.append(f"{lab}: {m:+.3f} (t={t:+.1f})")
        print(f"    {x:<9} " + " | ".join(out))

print("\n#### CAP NGANH — tuong quan hang theo thoi gian")
for x in ["TONE_all", "breadth_net", "TONE_phap_ly"]:
    out = []
    for y, lab in (("rB_1d", "r nganh phien t"), ("zB_1", "zB t+1"), ("zB_2_5", "zB t+2..t+5")):
        g = b[[x, y]].dropna()
        r, p = spearmanr(g[x], g[y])
        out.append(f"{lab}: {r:+.3f} (p={p:.3f})")
    print(f"  {x:<13} " + " | ".join(out))

print("\n#### Tach giai doan: IC cua TONE_i voi z phien t+1, cap co tin hom nay")
s = a[a.news_today == 1]
for lo, hi in [("2024-04-01", "2025-03-31"), ("2025-04-01", "2025-12-31"), ("2026-01-01", "2026-09-30")]:
    m, t, n = rank_ic(s[(s.date >= lo) & (s.date <= hi)], "TONE_i", "z_1")
    print(f"  {lo} -> {hi}: IC {m:+.3f} (t={t:+.1f}, {n} phien)")
