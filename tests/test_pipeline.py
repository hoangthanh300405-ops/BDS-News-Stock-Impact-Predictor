"""Unit test cho Tầng 1. Chạy sau `python -m src.run_pipeline`:   python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.config import CLEAN, NEWS_FILES, RAW, SIGMA_B_WINDOW, TRUST_XLSX
from src.data.sessions import parse_published, to_session
from src.data.xlsx_reader import read_xlsx, sheet_frame


@pytest.fixture(scope="module")
def calendar():
    return pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"]


def test_session_rule_matches_trust_table(calendar):
    """Quy tắc 14:45 phải khớp 100% phiên tham chiếu do bảng trust tính độc lập (T16)."""
    t = sheet_frame(read_xlsx(TRUST_XLSX, sheets={"Du_lieu_tho"})["Du_lieu_tho"]).dropna(subset=["STT"])
    published = pd.to_datetime(t["Ngay_dang"] + " " + t["Gio_dang_bai"]).dt.tz_localize("Asia/Ho_Chi_Minh")
    ours = to_session(published, calendar)
    theirs = pd.to_datetime(t["Ngay_gia_ngay_dang"])
    mismatch = (ours.values != theirs.values).sum()
    assert mismatch == 0, f"{mismatch} dòng lệch phiên"


def test_cutoff_boundaries(calendar):
    cal = pd.DatetimeIndex(calendar)
    d = cal[100]
    nxt = cal[101]
    ts = pd.Series(pd.to_datetime([f"{d.date()} 14:44", f"{d.date()} 14:45", f"{d.date()} 08:00"]).tz_localize("Asia/Ho_Chi_Minh"))
    s = to_session(ts, calendar)
    assert s[0] == d and s[1] == nxt and s[2] == d


def test_timezone_naive_equals_aware(calendar):
    """Cùng URL ở file không múi giờ (keyword/cafef) và file +07:00 (company/cafef) -> cùng phiên."""
    a = pd.read_csv(RAW / NEWS_FILES[("keyword", "cafef")], usecols=["url", "published_at"], encoding="utf-8-sig")
    b = pd.read_csv(RAW / NEWS_FILES[("company", "cafef")], usecols=["url", "published_at"], encoding="utf-8-sig").drop_duplicates("url")
    a["s"] = to_session(parse_published(a["published_at"]), calendar)
    b["s"] = to_session(parse_published(b["published_at"]), calendar)
    m = a.merge(b, on="url")
    assert len(m) > 1000
    assert (m["s_x"] == m["s_y"]).all()


def test_articles_unique_and_mapped():
    art = pd.read_parquet(CLEAN / "articles_clean.parquet")
    assert art["article_id"].is_unique
    assert art["session"].notna().mean() > 0.999


def test_one_label_per_article_run():
    lab = pd.read_parquet(CLEAN / "article_label.parquet")
    assert not lab.duplicated(["article_id", "typ", "label_run"]).any()
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet")
    assert not atl.duplicated(["article_id", "ticker"]).any()


def test_target_B_no_lookahead():
    """σB tại t chỉ dùng log-return đến hết t; zB_1 tính lại thủ công phải khớp."""
    bench = pd.read_parquet(CLEAN / "benchmark_nganh.parquet")
    r = bench[bench["method"] == "ew_liquid"].set_index("date")["log_return"]
    tb = pd.read_parquet(CLEAN / "target_B.parquet").set_index("date")
    i = 200
    t = r.index[i]
    sigma = r.iloc[i - SIGMA_B_WINDOW + 1: i + 1].std()
    z = r.iloc[i + 1] / sigma
    assert np.isclose(tb.loc[t, "zB_1"], z)


def test_targets_classes():
    ta = pd.read_parquet(CLEAN / "target_A.parquet")
    assert set(ta["y_1"].dropna().unique()) <= {-1.0, 0.0, 1.0}
    live = ta[~ta["warmup"]]
    assert live["y_1"].value_counts(normalize=True).min() >= 0.20      # §8.5: lớp nhỏ nhất ≥ 20%


@pytest.mark.parametrize("h", [1, 5])
def test_folds_purged(h, calendar):
    """Cửa sổ target (t, t+h] của mọi mẫu train/fit phải kết thúc trước đoạn sau (purge + embargo)."""
    from src.eval.walk_forward import EMBARGO, make_folds
    pos = pd.Series(range(len(calendar)), index=pd.DatetimeIndex(calendar))
    for f in make_folds(h):
        assert pos[f.train].max() + h + EMBARGO < pos[f.val].min()
        assert pos[f.fit].max() + h + EMBARGO < pos[f.test].min()
        assert pos[f.val].max() + h + EMBARGO < pos[f.test].min()
        assert not set(f.test) & set(f.fit)


def test_raw_folder_untouched():
    """data final/ là chỉ đọc: pipeline không được tạo file mới trong đó."""
    produced = [p for p in RAW.rglob("*.parquet")]
    assert not produced
