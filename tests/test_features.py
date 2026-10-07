"""Đặc trưng tin tức không được nhìn thấy tương lai: thay đổi tin SAU phiên t không làm đổi đặc trưng tại t."""
import numpy as np
import pandas as pd

from src.features.news_features import _decay


def test_decay_is_causal():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"v": rng.normal(size=50)})
    a = _decay(x, 50)
    x2 = x.copy(); x2.loc[30:, "v"] = 999.0            # sửa "tương lai" từ phiên 30
    b = _decay(x2, 50)
    assert np.allclose(a.loc[:29, "v"], b.loc[:29, "v"])
    assert not np.allclose(a.loc[30:, "v"], b.loc[30:, "v"])


def test_decay_weights():
    x = pd.DataFrame({"v": [1.0] + [0.0] * 20})
    d = _decay(x, 21, lam=3)["v"]
    assert d[0] == 1.0 and np.isclose(d[3], 0.5) and np.isclose(d[6], 0.25) and d[10] == 0.0   # cắt ở 3λ = 9
