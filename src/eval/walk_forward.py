"""Walk-forward 6 fold có purge & embargo (§11.1).

Test: 2025Q2 → 2026Q3 (mỗi fold 1 quý) | Val: quý liền trước | Train: từ 01/04/2024 đến trước val.
Purge  : bỏ mẫu có cửa sổ target (t, t+h] chạm sang đoạn sau.
Embargo: thêm khoảng trống EMBARGO phiên cuối mỗi đoạn train/val (không cắt vào tập test).
"""
from dataclasses import dataclass

import pandas as pd

from src.config import CLEAN

TEST_QUARTERS = ["2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2", "2026Q3"]
FIRST_SAMPLE = pd.Timestamp("2024-04-01")
LAST_NEWS = pd.Timestamp("2026-09-09")
EMBARGO = 5


@dataclass
class Fold:
    k: int
    test_q: str
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    fit: pd.DatetimeIndex      # train + val (purged) — dùng để fit lại trước khi dự đoán test
    test: pd.DatetimeIndex


def make_folds(h: int) -> list[Fold]:
    cal = pd.DatetimeIndex(pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"])
    pos = pd.Series(range(len(cal)), index=cal)
    q = cal.to_period("Q")
    folds = []
    for k, tq in enumerate(TEST_QUARTERS, 1):
        tq = pd.Period(tq, "Q")
        test = cal[(q == tq) & (cal <= LAST_NEWS)]
        val_all = cal[q == tq - 1]
        v0, t0 = pos[val_all[0]], pos[test[0]]
        ok_train = (cal >= FIRST_SAMPLE) & (pos.values + h + EMBARGO < v0)
        ok_val = (cal >= val_all[0]) & (pos.values + h + EMBARGO < t0)
        ok_fit = (cal >= FIRST_SAMPLE) & (pos.values + h + EMBARGO < t0)
        folds.append(Fold(k, str(tq), cal[ok_train], cal[ok_val], cal[ok_fit], test))
    return folds


def make_split_70_10_20(h: int) -> list[Fold]:
    """Một lần chia THEO THỜI GIAN: 70% đầu train, 10% val, 20% cuối test (tính trên các phiên từ 01/04/2024).
    Cùng quy tắc purge + embargo như walk-forward. Trả về list một phần tử để dùng chung engine.run."""
    cal = pd.DatetimeIndex(pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"])
    live = cal[(cal >= FIRST_SAMPLE) & (cal <= LAST_NEWS)]
    n = len(live)
    val_start, test_start = live[int(0.7 * n)], live[int(0.8 * n)]
    pos = pd.Series(range(len(cal)), index=cal)
    v0, t0 = pos[val_start], pos[test_start]
    ok_train = (cal >= FIRST_SAMPLE) & (pos.values + h + EMBARGO < v0)
    ok_val = (cal >= val_start) & (pos.values + h + EMBARGO < t0)
    ok_fit = (cal >= FIRST_SAMPLE) & (pos.values + h + EMBARGO < t0)
    test = live[live >= test_start]
    return [Fold(1, f"{test[0].date()}→{test[-1].date()}", cal[ok_train], cal[ok_val], cal[ok_fit], test)]


def describe(h: int):
    for f in make_folds(h):
        print(f"fold {f.k} test {f.test_q}: train {len(f.train)} ({f.train[0].date()}→{f.train[-1].date()}) | "
              f"val {len(f.val)} | fit {len(f.fit)} | test {len(f.test)}")


if __name__ == "__main__":
    for h in (1, 5):
        print(f"--- h = {h}")
        describe(h)
