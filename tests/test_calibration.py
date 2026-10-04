"""Đ2 không được nhìn thấy tương lai: xáo trộn mọi lợi suất SAU giai đoạn train không làm đổi điểm hiệu chỉnh."""
import numpy as np
import pandas as pd
import pytest

import src.features.calibration as cal
from src.eval.walk_forward import make_folds


def test_calibration_uses_train_only(monkeypatch):
    f = make_folds(1)[0]
    end = f.train[-1]
    ind1, tick1, _, _ = cal.fit_calibration(f.train)

    real = pd.read_parquet
    rng = np.random.default_rng(1)

    def scrambled(path, *a, **k):
        df = real(path, *a, **k)
        name = str(path)
        for col in ("zB_1", "z_1"):
            if col in df and ("target_B" in name or "target_A" in name):
                late = df["date"] > end
                df.loc[late, col] = rng.normal(size=late.sum()) * 5
        return df

    monkeypatch.setattr(cal.pd, "read_parquet", scrambled)
    ind2, tick2, _, _ = cal.fit_calibration(f.train)
    assert np.allclose(ind1.sort_index().values, ind2.sort_index().values, equal_nan=True)
    assert np.allclose(tick1.sort_index().values, tick2.sort_index().values, equal_nan=True)


def test_calibration_test_is_sensitive(monkeypatch):
    """Phép thử ngược: xáo trộn lợi suất TRONG train thì kết quả PHẢI đổi (nếu không, test trên vô nghĩa)."""
    f = make_folds(1)[0]
    end = f.train[-1]
    ind1, tick1, _, _ = cal.fit_calibration(f.train)
    real = pd.read_parquet
    rng = np.random.default_rng(2)

    def scrambled(path, *a, **k):
        df = real(path, *a, **k)
        for col in ("zB_1", "z_1"):
            if col in df:
                early = df["date"] <= end
                df.loc[early, col] = rng.normal(size=early.sum()) * 5
        return df

    monkeypatch.setattr(cal.pd, "read_parquet", scrambled)
    ind2, tick2, _, _ = cal.fit_calibration(f.train)
    assert not np.allclose(ind1.sort_index().values, ind2.sort_index().values, equal_nan=True)
    assert not np.allclose(tick1.sort_index().values, tick2.sort_index().values, equal_nan=True)
