"""P6 — Target h = 1 và h = 5 (§8.1, §8.2, T24).

target_B (1 dòng / phiên):   rB_h, zB_h, yB_h, big_move_B_h  (+ rB_2_5, zB_2_5 cho RQ2 "kéo dài hay đảo chiều")
target_A (1 dòng / mã × phiên): AR_h, z_h, y_h, big_move_h   (+ AR_2_5, z_2_5)
Quy ước: y ∈ {-1: Giảm, 0: Đứng, 1: Tăng}. Mọi σ chỉ dùng dữ liệu đến hết phiên t.
Benchmark chính: ew_liquid; AR điều chỉnh theo ngành với β = 1 (T1).
"""
import numpy as np
import pandas as pd

from src.config import (ATTN_Z, BIG_MOVE_Z, CLEAN, HORIZONS, SIGMA_A_MIN, SIGMA_A_WINDOW, SIGMA_B_WINDOW,
                        WARMUP_END, Z_THRESHOLD)


def classes(z):
    """z -> {-1, 0, 1} theo ngưỡng ±Z_THRESHOLD; giữ NaN. Dùng được cho Series lẫn DataFrame."""
    return np.sign(z.where(z.abs() > Z_THRESHOLD, 0)).where(z.notna())


def forward(cum, a, b):
    """Tổng log-return từ sau phiên t+a đến hết t+b, dựa trên chuỗi tích lũy cum."""
    return cum.shift(-b) - cum.shift(-a)


def build(benchmark="ew_liquid", verbose=True):
    bench = pd.read_parquet(CLEAN / "benchmark_nganh.parquet")
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet")
    rB = bench[bench["method"] == benchmark].set_index("date")["log_return"].sort_index()
    cal = rB.index

    # ---------------- Model B ----------------
    cumB = rB.fillna(0).cumsum()
    sB = rB.rolling(SIGMA_B_WINDOW, min_periods=SIGMA_B_WINDOW).std()      # đến hết phiên t
    tb = pd.DataFrame(index=cal)
    for h in HORIZONS:
        r = forward(cumB, 0, h)
        z = r / (sB * np.sqrt(h))
        tb[f"rB_{h}"], tb[f"zB_{h}"] = r, z
        tb[f"yB_{h}"] = classes(z)
        tb[f"big_move_B_{h}"] = (z.abs() > BIG_MOVE_Z).astype(float).where(z.notna())
        tb[f"upB_{h}"] = (z > Z_THRESHOLD).astype(float).where(z.notna())          # ngành "tăng" / "không tăng"
    r25 = forward(cumB, 1, 5)
    tb["rB_2_5"], tb["zB_2_5"] = r25, r25 / (sB * np.sqrt(4))
    # Hướng B: thanh khoản toàn ngành đột biến (tổng giá trị GD của 83 mã)
    tot = px.pivot_table(index="date", columns="ticker", values="value_bn").reindex(cal).sum(axis=1, min_count=50)
    ltot = np.log(tot)
    mB, sBv = ltot.rolling(20, min_periods=10).mean(), ltot.rolling(20, min_periods=10).std()
    for h in HORIZONS:
        az = (sum(ltot.shift(-k) for k in range(1, h + 1)) / h - mB) / sBv
        tb[f"attnB_val_z_{h}"] = az.values
        tb[f"vol_spike_B_{h}"] = (az > ATTN_Z).astype(float).where(az.notna()).values
    tb["warmup"] = tb.index <= pd.Timestamp(WARMUP_END)
    tb = tb.reset_index().rename(columns={"index": "date"})

    # ---------------- Model A ----------------
    close = px.pivot_table(index="date", columns="ticker", values="close").reindex(cal)
    logp = np.log(close)
    ar_daily = logp.diff().sub(rB, axis=0)
    sA = ar_daily.rolling(SIGMA_A_WINDOW, min_periods=SIGMA_A_MIN).std()
    parts = {}
    for h in HORIZONS:
        ar = (logp.shift(-h) - logp).sub(forward(cumB, 0, h), axis=0)      # NaN nếu thiếu giá ở t hoặc t+h
        z = ar / (sA * np.sqrt(h))
        parts[f"AR_{h}"], parts[f"z_{h}"], parts[f"y_{h}"] = ar, z, classes(z)
        parts[f"big_move_{h}"] = (z.abs() > BIG_MOVE_Z).astype(float).where(z.notna())
    ar25 = (logp.shift(-5) - logp.shift(-1)).sub(forward(cumB, 1, 5), axis=0)
    parts["AR_2_5"], parts["z_2_5"] = ar25, ar25 / (sA * np.sqrt(4))
    # Output "tăng / không tăng": lợi suất THỰC của cổ phiếu (không trừ ngành), chuẩn hóa theo σ riêng của mã
    sR = logp.diff().rolling(SIGMA_A_WINDOW, min_periods=SIGMA_A_MIN).std()
    for h in HORIZONS:
        r = logp.shift(-h) - logp
        zr = r / (sR * np.sqrt(h))
        parts[f"r_{h}"], parts[f"zr_{h}"] = r, zr
        parts[f"up_raw_{h}"] = (zr > Z_THRESHOLD).astype(float).where(zr.notna())
        parts[f"up_{h}"] = (parts[f"z_{h}"] > Z_THRESHOLD).astype(float).where(parts[f"z_{h}"].notna())
    # ---------------- Hướng B: target "SỰ CHÚ Ý" của thị trường (T25) ----------------
    # Thanh khoản bất thường: log giá trị GD bình quân (t+1..t+h) so với trung bình/độ lệch chuẩn 20 phiên ĐẾN HẾT t.
    value = px.pivot_table(index="date", columns="ticker", values="value_bn").reindex(cal)
    lv = np.log(value.where(value > 0))
    m20, s20 = lv.rolling(20, min_periods=10).mean(), lv.rolling(20, min_periods=10).std()
    for h in HORIZONS:
        nxt = sum(lv.shift(-k) for k in range(1, h + 1)) / h
        az = (nxt - m20) / s20
        parts[f"attn_val_z_{h}"] = az
        parts[f"vol_spike_{h}"] = (az > ATTN_Z).astype(float).where(az.notna())            # thanh khoản đột biến
        parts[f"big_raw_{h}"] = (parts[f"zr_{h}"].abs() > BIG_MOVE_Z).astype(float).where(parts[f"zr_{h}"].notna())
    ta = pd.concat(parts, axis=1).stack(level=1, future_stack=True).reset_index()
    ta = ta.rename(columns={"level_0": "date", "level_1": "ticker"})
    ta = ta[ta["z_1"].notna() | ta["z_5"].notna()].reset_index(drop=True)
    ta["warmup"] = ta["date"] <= pd.Timestamp(WARMUP_END)

    tb.to_parquet(CLEAN / "target_B.parquet", index=False)
    ta.to_parquet(CLEAN / "target_A.parquet", index=False)

    if verbose:
        live = tb[~tb["warmup"]]
        for h in HORIZONS:
            d = live[f"yB_{h}"].value_counts(normalize=True).rename({-1: "Giảm", 0: "Đứng", 1: "Tăng"}).round(3).to_dict()
            print(f"[P6] target_B h={h}: {live[f'yB_{h}'].notna().sum()} phiên | {d} | big_move {live[f'big_move_B_{h}'].mean():.1%}")
        la = ta[~ta["warmup"]]
        for h in HORIZONS:
            d = la[f"y_{h}"].value_counts(normalize=True).rename({-1: "Giảm", 0: "Đứng", 1: "Tăng"}).round(3).to_dict()
            print(f"[P6] target_A h={h}: {la[f'y_{h}'].notna().sum()} cặp (mã, phiên) | {d} | big_move {la[f'big_move_{h}'].mean():.1%}")
    return tb, ta


if __name__ == "__main__":
    build()
