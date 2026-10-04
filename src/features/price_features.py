"""F5 — Đặc trưng giá & thị trường (§7.6). Mọi đặc trưng tại phiên t chỉ dùng dữ liệu đến giá đóng cửa phiên t.

features_B(): 1 dòng / phiên      features_A(): 1 dòng / (mã, phiên)
"""
import numpy as np
import pandas as pd

from src.config import CLEAN

# Bộ lõi cho Model B (≤ 6 cột vì fold 1 chỉ có ~180 phiên train, quy tắc n/30 — §7.9)
B_PRICE_CORE = ["rB_1d", "rB_5d", "rB_20d", "sigmaB_20", "breadth", "distB_ma20"]
A_PRICE = ["r_1d", "r_5d", "r_20d", "ar_5d", "ar_20d", "sigma_ar60", "value_z20", "log_amihud",
           "illiquid", "log_mcap", "rB_1d", "rB_5d"]


def _load():
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet")
    bench = pd.read_parquet(CLEAN / "benchmark_nganh.parquet")
    rB = bench[bench["method"] == "ew_liquid"].set_index("date")["log_return"].sort_index()
    prof = pd.read_parquet(CLEAN / "company_profile.parquet").set_index("ticker")
    return px, rB, prof


def features_B() -> pd.DataFrame:
    px, rB, _ = _load()
    cal = rB.index
    logret = px.pivot_table(index="date", columns="ticker", values="log_return").reindex(cal)
    value = px.pivot_table(index="date", columns="ticker", values="value_bn").reindex(cal)
    level = rB.fillna(0).cumsum()
    f = pd.DataFrame(index=cal)
    f["rB_1d"] = rB
    f["rB_5d"] = level - level.shift(5)
    f["rB_20d"] = level - level.shift(20)
    f["sigmaB_20"] = rB.rolling(20).std()
    f["breadth"] = (logret > 0).sum(axis=1) / logret.notna().sum(axis=1).replace(0, np.nan)
    f["distB_ma20"] = level - level.rolling(20).mean()
    f["distB_ma60"] = level - level.rolling(60).mean()
    tot = value.sum(axis=1, min_count=10)
    f["value_z20"] = (tot - tot.rolling(20).mean()) / tot.rolling(20).std()
    f["limit_up_share"] = (logret > np.log(1.065)).mean(axis=1)       # ≈ chạm trần HOSE (7%)
    f["limit_down_share"] = (logret < np.log(1 - 0.065)).mean(axis=1)
    return f.reset_index().rename(columns={"index": "date"})


def features_A() -> pd.DataFrame:
    px, rB, prof = _load()
    cal = rB.index
    close = px.pivot_table(index="date", columns="ticker", values="close").reindex(cal)
    value = px.pivot_table(index="date", columns="ticker", values="value_bn").reindex(cal)
    logp = np.log(close)
    lr = logp.diff()
    ar = lr.sub(rB, axis=0)
    parts = {
        "r_1d": lr,
        "r_5d": logp - logp.shift(5),
        "r_20d": logp - logp.shift(20),
        "ar_5d": ar.rolling(5, min_periods=4).sum(),
        "ar_20d": ar.rolling(20, min_periods=15).sum(),
        "sigma_ar60": ar.rolling(60, min_periods=40).std(),
        "value_z20": (value - value.rolling(20, min_periods=10).mean()) / value.rolling(20, min_periods=10).std(),
        "log_amihud": np.log(px.pivot_table(index="date", columns="ticker", values="amihud").reindex(cal) + 1e-6),
        "illiquid": px.pivot_table(index="date", columns="ticker", values="illiquid").reindex(cal),
        # vốn hóa theo phiên = giá đóng cửa t × số CP lưu hành (T10: không dùng vốn hóa tĩnh 09/2026)
        "log_mcap": np.log(close.mul(prof["so_cp_luu_hanh"].reindex(close.columns), axis=1) * 1000 / 1e9),
    }
    f = pd.concat(parts, axis=1).stack(level=1, future_stack=True).reset_index()
    f = f.rename(columns={"level_0": "date", "level_1": "ticker"})
    lvl = rB.fillna(0).cumsum()
    bf = pd.DataFrame({"rB_1d": rB, "rB_5d": lvl - lvl.shift(5)})
    f = f.merge(bf, left_on="date", right_index=True, how="left")
    return f


A_EXTRA = ["hit_ceiling", "hit_floor", "move_vs_limit", "cs_rank_r1", "cs_rank_ar5", "cs_rank_value", "dow",
           "ticker_drift", "rB_20d", "breadth"]


def features_A_extra() -> pd.DataFrame:
    """Đặc trưng bổ sung cho Model A (v3) — chỉ dùng dữ liệu đến hết phiên t:
      hit_ceiling / hit_floor : phiên t đóng cửa chạm trần / sàn (HOSE ±7%, HNX ±10%) — đặc thù VN: dư mua trần
      move_vs_limit           : lợi suất phiên t chia cho biên độ sàn
      cs_rank_*               : thứ hạng chéo trong phiên (0..1) của lợi suất 1 phiên, AR 5 phiên, giá trị giao dịch
      dow                     : thứ trong tuần
      ticker_drift            : trung bình AR ngày của mã trên toàn bộ quá khứ đến t (expanding) — xu hướng riêng của mã
      rB_20d, breadth         : bối cảnh ngành
    """
    px, rB, prof = _load()
    cal = rB.index
    close = px.pivot_table(index="date", columns="ticker", values="close").reindex(cal)
    value = px.pivot_table(index="date", columns="ticker", values="value_bn").reindex(cal)
    lr = np.log(close).diff()
    simple = close.pct_change(fill_method=None)
    limit = prof["san"].map({"HOSE": 0.07, "HNX": 0.10}).fillna(0.15).reindex(close.columns)
    ar = lr.sub(rB, axis=0)
    level = rB.fillna(0).cumsum()
    parts = {
        "hit_ceiling": (simple >= limit - 0.003).astype(float).where(simple.notna()),
        "hit_floor": (simple <= -limit + 0.003).astype(float).where(simple.notna()),
        "move_vs_limit": simple / limit,
        "cs_rank_r1": lr.rank(axis=1, pct=True),
        "cs_rank_ar5": ar.rolling(5, min_periods=4).sum().rank(axis=1, pct=True),
        "cs_rank_value": value.rank(axis=1, pct=True),
        "ticker_drift": ar.expanding(min_periods=40).mean(),
    }
    f = pd.concat(parts, axis=1).stack(level=1, future_stack=True).reset_index()
    f = f.rename(columns={"level_0": "date", "level_1": "ticker"})
    ctx = pd.DataFrame({"rB_20d": level - level.shift(20),
                        "breadth": (lr > 0).sum(axis=1) / lr.notna().sum(axis=1).replace(0, np.nan),
                        "dow": pd.Series(cal.dayofweek, index=cal).astype(float)})
    return f.merge(ctx, left_on="date", right_index=True, how="left")


def sample_A(window: int = 5) -> pd.DataFrame:
    """Mẫu Model A: (mã, phiên t) có ít nhất 1 bài nhắc mã trong [t−window, t] (§8.2)."""
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet", columns=["article_id", "ticker"])
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session"])
    cal = pd.read_parquet(CLEAN / "trading_calendar.parquet").set_index("date")["trading_index"]
    pairs = atl.merge(art, on="article_id").dropna(subset=["session"]).drop_duplicates(["ticker", "session"])
    pairs["ti"] = pairs["session"].map(cal)
    idx2date = pd.Series(cal.index, index=cal.values)
    rows = [(t, ti + k) for t, ti in zip(pairs["ticker"], pairs["ti"]) for k in range(window + 1)]
    s = pd.DataFrame(rows, columns=["ticker", "ti"]).drop_duplicates()
    s = s[s["ti"] <= cal.max()]
    s["date"] = s["ti"].map(idx2date)
    return s[["date", "ticker"]].reset_index(drop=True)
