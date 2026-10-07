"""Hướng B (T25) — đặc trưng cho bài toán "SỰ CHÚ Ý": mã nào sẽ giao dịch đột biến / biến động mạnh ở phiên sau?

Mẫu: MỌI (mã, phiên) — kể cả phiên không có tin (đặc trưng tin = 0), để so được mã có tin với mã không có tin.
Baseline "giá & khối lượng" được làm MẠNH có chủ đích (lịch sử khối lượng 5/20 phiên, tần suất đột biến quá khứ,
thứ hạng chéo, thanh khoản toàn ngành): khối lượng hôm nay tự nó đã dự báo khối lượng ngày mai rất tốt, nên nếu
baseline yếu thì tin tức sẽ "thắng" giả tạo. Mọi đặc trưng chỉ dùng dữ liệu đến hết phiên t.
"""
import numpy as np
import pandas as pd

from src.config import CLEAN
from src.features.news_features import industry_news, ticker_news
from src.features.price_features import features_A, features_A_extra, features_B

A_PRICE_ATTN = ["lv_z20", "lv_z5", "value_z20", "spike_rate60", "cs_rank_value", "abs_r1", "abs_r5", "sigma_ar60",
                "log_mcap", "illiquid", "log_amihud", "hit_ceiling", "hit_floor", "move_vs_limit", "r_1d", "r_5d",
                "rB_1d", "sector_lv_z20", "dow"]
A_NEWS_ATTN = ["news_today", "INT_i", "mention_z", "TONE_i", "NEG_i", "n_tickers_mean", "price_report_share",
               "INTz_all", "LRI_share", "PPI_share"]
B_PRICE_ATTN = ["rB_1d", "rB_5d", "sigmaB_20", "breadth", "sector_lv_z20", "sector_lv_z5", "sector_spike_rate60"]
B_NEWS_ATTN = ["INT_all", "INTz_all", "TONE_all", "breadth_net", "LRI_share", "PPI_share"]


def _volume_panel():
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet", columns=["date", "ticker", "value_bn", "log_return"])
    v = px.pivot_table(index="date", columns="ticker", values="value_bn")
    lv = np.log(v.where(v > 0))
    prev20m, prev20s = lv.rolling(20, min_periods=10).mean().shift(1), lv.rolling(20, min_periods=10).std().shift(1)
    z20 = (lv - prev20m) / prev20s                                    # khối lượng HÔM NAY so với 20 phiên trước
    z5 = (lv.rolling(5, min_periods=3).mean() - prev20m) / prev20s
    spike = (z20 > 1).astype(float).where(z20.notna())
    r = px.pivot_table(index="date", columns="ticker", values="log_return")
    parts = {"lv_z20": z20, "lv_z5": z5, "spike_rate60": spike.rolling(60, min_periods=20).mean(),
             "abs_r1": r.abs(), "abs_r5": r.abs().rolling(5, min_periods=3).mean()}
    f = pd.concat(parts, axis=1).stack(level=1, future_stack=True).reset_index()
    f = f.rename(columns={"level_0": "date", "level_1": "ticker"})
    tot = np.log(v.sum(axis=1, min_count=50))
    m, s = tot.rolling(20, min_periods=10).mean().shift(1), tot.rolling(20, min_periods=10).std().shift(1)
    zt = (tot - m) / s
    sec = pd.DataFrame({"sector_lv_z20": zt, "sector_lv_z5": (tot.rolling(5, min_periods=3).mean() - m) / s,
                        "sector_spike_rate60": (zt > 1).astype(float).rolling(60, min_periods=20).mean()})
    return f, sec


def panel_A():
    base = features_A()                                               # mọi (mã, phiên)
    extra = features_A_extra()
    extra = extra.drop(columns=[c for c in ("rB_1d", "rB_5d") if c in extra.columns])
    vol, sec = _volume_panel()
    d = base.merge(extra, on=["date", "ticker"], how="left").merge(vol, on=["date", "ticker"], how="left")
    d = d.merge(sec, left_on="date", right_index=True, how="left")
    tn = ticker_news().drop(columns=["TONE_all", "INTz_all", "TONE_gap"])
    d = d.merge(tn, on=["date", "ticker"], how="left")
    fill0 = ["news_today", "INT_i", "mention_z", "TONE_i", "NEG_i", "n_tickers_mean", "price_report_share"]
    d[fill0] = d[fill0].fillna(0.0)                                   # phiên / mã không có tin
    ind = industry_news()[["date", "INTz_all", "LRI_share", "PPI_share"]]
    d = d.merge(ind, on="date", how="left")
    ta = pd.read_parquet(CLEAN / "target_A.parquet")
    return d.merge(ta, on=["date", "ticker"])


def panel_B():
    _, sec = _volume_panel()
    d = features_B().merge(industry_news(), on="date").merge(sec, left_on="date", right_index=True, how="left")
    return d.merge(pd.read_parquet(CLEAN / "target_B.parquet"), on="date")
