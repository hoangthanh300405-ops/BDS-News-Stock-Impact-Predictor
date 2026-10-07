"""Hướng nhà đầu tư nhỏ lẻ: đặc trưng 'uy tín của tin' chỉ được dùng các tin đã BIẾT KẾT QUẢ trước phiên đọc tin."""
import numpy as np
import pandas as pd
import pytest

from src.config import CLEAN
from src.retail import events as E

PATH = CLEAN / "retail_events.parquet"


@pytest.mark.skipif(not PATH.exists(), reason="chưa chạy src.retail.events")
def test_credibility_ignores_unknown_outcomes():
    """Đảo ngược kết quả của mọi tin CHƯA đóng lệnh tại phiên t* -> đặc trưng của các tin đọc đến t* không đổi."""
    ev = pd.read_parquet(PATH)
    cal = pd.DatetimeIndex(pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"])
    pos = pd.Series(np.arange(len(cal)), index=cal)
    for tick in ev["ticker"].value_counts().index[:3]:
        sub = ev[ev["ticker"] == tick].sort_values("date").reset_index(drop=True)
        sub = sub[["date", "ticker", "event_type", "follow_ok", "follow_rel", "entry_shift"]].copy()
        p = pos.reindex(sub["date"]).to_numpy() + sub["entry_shift"].to_numpy()
        t_star = pos[sub["date"].iloc[len(sub) // 2]]
        a, b = sub.copy(), sub.copy()
        future = p + E.H > t_star
        b.loc[future, ["follow_ok", "follow_rel"]] = 1 - b.loc[future, ["follow_ok", "follow_rel"]]
        E.credibility(a, pos, p)
        E.credibility(b, pos, p)
        upto = pos.reindex(sub["date"]).to_numpy() <= t_star
        for c in ("cred_tick_ok", "cred_type_ok", "recent_ok", "cred_tick_rel", "cred_type_rel", "recent_rel"):
            np.testing.assert_allclose(a.loc[upto, c], b.loc[upto, c])
        assert (a.loc[~upto, "cred_tick_ok"] != b.loc[~upto, "cred_tick_ok"]).any()   # và test thực sự nhạy
