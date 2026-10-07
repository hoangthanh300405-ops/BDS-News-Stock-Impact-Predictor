"""P4 — Giá, lịch phiên, benchmark ngành và cờ thanh khoản.

Đầu vào : data/stock/Company_Profile_BDS_REAL_DATA__1_.xlsx
Đầu ra  : các file parquet trong data/clean/.
"""
import numpy as np
import pandas as pd

from src.config import (CLEAN, EXPOSURE_KEYWORDS, ILLIQ_MIN_VALUE_BN, ILLIQ_WINDOW, ILLIQ_ZERO_RET, PRICE_XLSX)
from src.data.xlsx_reader import read_xlsx, sheet_frame


def load_raw():
    x = read_xlsx(PRICE_XLSX, sheets={"Du_lieu_gia_hang_ngay", "Ho_so_cong_ty"})
    px = sheet_frame(x["Du_lieu_gia_hang_ngay"])
    px["date"] = pd.to_datetime(px["Ngay"], errors="coerce")
    px["close"] = pd.to_numeric(px["Gia_dong_cua_nghin_VND"], errors="coerce")
    px["volume"] = pd.to_numeric(px["Khoi_luong_GD"], errors="coerce")
    px = px.dropna(subset=["Ma_CK", "date", "close"]).rename(columns={"Ma_CK": "ticker"})
    prof = sheet_frame(x["Ho_so_cong_ty"]).dropna(subset=["Ma_CK"])
    return px[["ticker", "date", "close", "volume"]], prof


def build_company_profile(prof: pd.DataFrame) -> pd.DataFrame:
    p = prof[prof["Ma_CK"].str.fullmatch(r"[A-Z0-9]{3}")].copy()
    out = pd.DataFrame({
        "ticker": p["Ma_CK"],
        "ten_cong_ty": p.get("Ten_Cong_Ty"),
        "san": p["San_niem_yet"].str.upper(),
        "loai_hinh_bds": p["Loai_hinh_BDS"],
        "phan_khuc_gia": p["Phan_khuc_gia"],
        "khu_vuc_hoat_dong": p["Khu_vuc_hoat_dong"],
        "so_cp_luu_hanh": pd.to_numeric(p["So_luong_CP_luu_hanh"], errors="coerce"),
        "von_dieu_le": pd.to_numeric(p["Von_dieu_le_ty_VND"], errors="coerce"),
        "ngay_niem_yet": pd.to_datetime(p["Ngay_niem_yet"], format="%d/%m/%Y", errors="coerce"),
    })
    # KHÔNG lưu Von_hoa_thi_truong_ty_VND: tính bằng giá 11/09/2026 -> rò rỉ tương lai
    loai = out["loai_hinh_bds"].fillna("").str.lower()
    for seg, words in EXPOSURE_KEYWORDS.items():
        out[f"exposure_{seg}"] = loai.apply(lambda s: int(any(w in s for w in words)))
    return out.reset_index(drop=True)


def build(verbose=True):
    raw, prof = load_raw()
    stocks = raw[raw["ticker"].str.fullmatch(r"[A-Z0-9]{3}")]
    bds_index = raw[raw["ticker"] == "BDS_INDEX"].set_index("date")["close"].sort_index()

    # ---- lịch phiên: hợp các ngày có giá của 83 mã ----
    cal = pd.DatetimeIndex(sorted(stocks["date"].unique()), name="date")
    calendar = pd.DataFrame({"date": cal, "trading_index": np.arange(len(cal))})

    close = stocks.pivot_table(index="date", columns="ticker", values="close").reindex(cal)
    volume = stocks.pivot_table(index="date", columns="ticker", values="volume").reindex(cal)
    logret = np.log(close).diff()          # NaN nếu thiếu giá ở t hoặc t-1 -> không forward-fill

    # suspended: không có giá ở phiên nằm giữa phiên đầu và phiên cuối có giá của mã
    first, last = close.apply(pd.Series.first_valid_index), close.apply(pd.Series.last_valid_index)
    in_range = pd.DataFrame({t: (cal >= first[t]) & (cal <= last[t]) for t in close.columns}, index=cal)
    suspended = in_range & close.isna()

    value_bn = close * volume * 1000 / 1e9           # giá trị giao dịch (tỷ VND); giá tính bằng nghìn VND

    # illiquid(t): tính trên 60 phiên TRƯỚC t (shift 1) -> không rò rỉ
    zero_share = (logret == 0).astype(float).where(logret.notna()).rolling(ILLIQ_WINDOW, min_periods=20).mean().shift(1)
    med_value = value_bn.rolling(ILLIQ_WINDOW, min_periods=20).median().shift(1)
    illiquid = ((zero_share > ILLIQ_ZERO_RET) | (med_value < ILLIQ_MIN_VALUE_BN)).astype(int)
    illiquid = illiquid.where(zero_share.notna() | med_value.notna())

    # Amihud (2002): |r| / giá trị giao dịch (tỷ VND), trung bình 20 phiên
    amihud = (logret.abs() / value_bn.replace(0, np.nan)).rolling(20, min_periods=10).mean()

    long = pd.concat({
        "close": close, "volume": volume, "value_bn": value_bn, "log_return": logret,
        "suspended": suspended.astype(int), "illiquid": illiquid, "amihud": amihud,
    }, axis=1).stack(level=1, future_stack=True).reset_index().rename(columns={"level_1": "ticker"})
    long = long[long["close"].notna() | (long["suspended"] == 1)].reset_index(drop=True)

    # ---- benchmark ngành: đồng trọng số ----
    liquid_prev = illiquid.shift(1).eq(0)
    ew_liquid = logret.where(liquid_prev).mean(axis=1)
    ew_liquid = ew_liquid.where(logret.where(liquid_prev).notna().sum(axis=1) >= 10)   # đủ mã
    ew_all = logret.mean(axis=1)
    bds = np.log(bds_index.reindex(cal)).diff()
    bench = []
    for method, r in (("ew_liquid", ew_liquid), ("ew_all", ew_all), ("bds_index", bds)):
        r = r.copy()
        r.iloc[0] = 0.0
        val = 100 * np.exp(r.fillna(0).cumsum())
        bench.append(pd.DataFrame({"date": cal, "method": method, "value": val.values, "log_return": r.values}))
    bench = pd.concat(bench, ignore_index=True)

    profile = build_company_profile(prof)

    CLEAN.mkdir(parents=True, exist_ok=True)
    calendar.to_parquet(CLEAN / "trading_calendar.parquet", index=False)
    long.to_parquet(CLEAN / "stock_prices_clean.parquet", index=False)
    bench.to_parquet(CLEAN / "benchmark_nganh.parquet", index=False)
    profile.to_parquet(CLEAN / "company_profile.parquet", index=False)

    if verbose:
        wide = bench.pivot(index="date", columns="method", values="log_return").iloc[1:]
        print(f"[P4] lịch phiên: {len(cal)} phiên ({cal[0].date()} → {cal[-1].date()}) | {close.shape[1]} mã")
        print(f"[P4] stock_prices_clean: {len(long)} dòng | suspended: {int(suspended.values.sum())} ô | "
              f"illiquid (phiên cuối): {int(illiquid.iloc[-1].sum())} mã")
        print("[P4] tương quan benchmark:", wide.corr().round(3).to_dict())
        print(f"[P4] company_profile: {len(profile)} mã | phơi nhiễm:",
              {c: int(profile[c].sum()) for c in profile if c.startswith('exposure_')})
    return calendar, long, bench, profile


if __name__ == "__main__":
    build()
