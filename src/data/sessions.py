"""Ánh xạ thời điểm đăng tin -> phiên giao dịch.

Tin đăng TRƯỚC 14:45 của một ngày giao dịch -> phiên của chính ngày đó.
Tin đăng TỪ 14:45 trở đi, hoặc vào ngày không giao dịch -> phiên giao dịch kế tiếp.
"""
import numpy as np
import pandas as pd

from src.config import ATC_CUTOFF, TZ


def parse_published(s: pd.Series) -> pd.Series:
    """Chuỗi ISO 8601 (có hoặc không có +07:00) -> datetime giờ Việt Nam.

    Giờ không kèm múi giờ là giờ Việt Nam (đã kiểm chứng: khớp 100% trên >5.200 URL có ở cả hai dạng file).
    Mỗi file chỉ dùng một dạng, nên gọi hàm này theo từng file.
    """
    d = pd.to_datetime(s, format="ISO8601")
    return d.dt.tz_localize(TZ) if d.dt.tz is None else d.dt.tz_convert(TZ)


def to_session(published: pd.Series, calendar) -> pd.Series:
    """published: datetime có múi giờ VN. calendar: các ngày giao dịch (tăng dần).

    Trả về ngày phiên (datetime64, không múi giờ); NaT nếu tin nằm sau phiên cuối của lịch.
    """
    cal = np.asarray(pd.to_datetime(calendar), dtype="datetime64[ns]")
    local = published.dt.tz_localize(None)
    minutes = local.dt.hour * 60 + local.dt.minute
    after_cutoff = (minutes >= ATC_CUTOFF.hour * 60 + ATC_CUTOFF.minute).astype(int)
    key = (local.dt.normalize() + pd.to_timedelta(after_cutoff, unit="D")).to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(cal, key, side="left")
    out = np.full(len(idx), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = idx < len(cal)
    out[ok] = cal[idx[ok]]
    return pd.Series(out, index=published.index, name="session")
