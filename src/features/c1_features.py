"""C1 — sau một biến động MẠNH, tin tức có quyết định giá ĐI TIẾP hay ĐẢO CHIỀU? (Chan, 2003)

Sự kiện : (mã i, phiên t) với |z_t| > Z_EVENT, z_t = AR_t / độ lệch chuẩn AR 60 phiên TRƯỚC t (AR so với ngành).
Target  : cont_1 = 1 nếu AR_{t+1} cùng dấu AR_t (đi tiếp), 0 nếu ngược dấu (đảo chiều); bỏ AR_{t+1} = 0
          cont_5 = như trên với tổng AR_{t+1..t+5}
Đặc trưng chỉ dùng thông tin đến giờ đóng cửa phiên t:
  giá    : độ lớn và dấu biến động, lợi suất / AR trước đó, biến động, thanh khoản, khối lượng hôm nay, trần/sàn, quy mô
  TIN    : có tin không, có tin CƠ BẢN không, số bài, sắc thái (chung / tin cơ bản), sắc thái CÙNG CHIỀU biến động không,
           loại sự kiện, tỷ trọng tin trước giờ mở cửa, tỷ trọng tin chỉ thuật lại giá, mức được nhắc bất thường
"""
import numpy as np
import pandas as pd

from src.config import CLEAN
from src.features.attention_features import _volume_panel
from src.features.price_features import features_A, features_A_extra

Z_EVENT = 1.5
FUND = {"tai_chinh_dn", "phap_ly", "chinh_sach", "lai_suat_tin_dung"}
EVENT_FLAGS = {"EARNINGS": "ev_kqkd", "LEGAL": "ev_phap_ly", "CAPITAL_RAISE": "ev_von", "M_AND_A": "ev_mna",
               "DEBT_BOND": "ev_no", "MARKET": "ev_thi_truong", "PROJECT": "ev_du_an", "POLICY": "ev_chinh_sach"}
C1_PRICE = ["abs_z", "move_sign", "ar_t", "ar_5d_prev", "r_5d", "r_20d", "sigma_ar60", "illiquid", "log_amihud",
            "lv_z20", "log_mcap", "hit_ceiling", "hit_floor", "move_vs_limit", "rB_1d", "dow"]
C1_NEWS = ["has_news", "has_fund", "n_bai", "tone", "tone_fund", "tone_x_move", "agree_move", "share_preopen",
           "share_price_report", "mention_z"] + list(EVENT_FLAGS.values())


def _news_by_session():
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet")
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session", "published_at"])
    c = atl.merge(art, on="article_id").dropna(subset=["session", "impact_score"])
    loc = c["published_at"].dt.tz_localize(None)
    c["preopen"] = ((loc.dt.normalize() != c["session"]) | ((loc.dt.hour * 60 + loc.dt.minute) < 540)).astype(float)
    c["fund"] = c["nhom_tin"].isin(FUND)
    c["s_fund"] = c["impact_score"].where(c["fund"])
    for et, col in EVENT_FLAGS.items():
        c[col] = (c["event_type"] == et).astype(float)
    g = c.groupby(["session", "ticker"])
    out = g.agg(n_bai=("article_id", "nunique"), tone=("impact_score", "mean"), tone_fund=("s_fund", "mean"),
                has_fund=("fund", "max"), share_preopen=("preopen", "mean"),
                share_price_report=("price_report", "mean"), **{col: (col, "max") for col in EVENT_FLAGS.values()})
    out = out.reset_index().rename(columns={"session": "date"})
    out["has_news"] = 1.0
    out["has_fund"] = out["has_fund"].astype(float)
    return out


def build_c1():
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet", columns=["date", "ticker", "log_return"])
    b = pd.read_parquet(CLEAN / "benchmark_nganh.parquet")
    rB = b[b["method"] == "ew_liquid"].set_index("date")["log_return"]
    ar = px.pivot_table(index="date", columns="ticker", values="log_return").reindex(rB.index).sub(rB, axis=0)
    sd = ar.rolling(60, min_periods=40).std().shift(1)
    parts = {"ar_t": ar, "z_t": ar / sd, "ar_5d_prev": ar.shift(1).rolling(5, min_periods=4).sum(),
             "ar_next1": ar.shift(-1), "ar_next5": sum(ar.shift(-k) for k in range(1, 6))}
    d = pd.concat(parts, axis=1).stack(level=1, future_stack=True).reset_index()
    d = d.rename(columns={"level_1": "ticker"})
    d = d[(d["z_t"].abs() > Z_EVENT) & (d["date"] >= "2024-04-01")].copy()
    d["abs_z"], d["move_sign"] = d["z_t"].abs(), np.sign(d["z_t"])
    for h, col in ((1, "ar_next1"), (5, "ar_next5")):
        same = np.sign(d[col]) == d["move_sign"]
        d[f"cont_{h}"] = np.where(d[col].isna() | (d[col] == 0), np.nan, same.astype(float))
        d[f"cont_ret_{h}"] = d["move_sign"] * d[col]                     # lợi suất theo chiều biến động
    # đặc trưng giá & khối lượng
    fa = features_A()[["date", "ticker", "r_5d", "r_20d", "sigma_ar60", "illiquid", "log_amihud", "log_mcap", "rB_1d"]]
    fx = features_A_extra()[["date", "ticker", "hit_ceiling", "hit_floor", "move_vs_limit", "dow"]]
    vol, _ = _volume_panel()
    d = d.merge(fa, on=["date", "ticker"], how="left").merge(fx, on=["date", "ticker"], how="left")
    d = d.merge(vol[["date", "ticker", "lv_z20"]], on=["date", "ticker"], how="left")
    # đặc trưng tin tại phiên t
    n = _news_by_session()
    d = d.merge(n, on=["date", "ticker"], how="left")
    zero = ["has_news", "has_fund", "n_bai", "share_preopen", "share_price_report"] + list(EVENT_FLAGS.values())
    d[zero] = d[zero].fillna(0.0)
    d[["tone", "tone_fund"]] = d[["tone", "tone_fund"]].fillna(0.0)
    d["tone_x_move"] = d["tone"] * d["move_sign"]
    d["agree_move"] = (np.sign(d["tone"]) == d["move_sign"]).astype(float) * d["has_news"]
    from src.features.news_features import ticker_news
    tn = ticker_news()[["date", "ticker", "mention_z"]]
    d = d.merge(tn, on=["date", "ticker"], how="left")
    d["mention_z"] = d["mention_z"].fillna(0.0)
    d["nhom"] = np.select([d["has_fund"] == 1, d["has_news"] == 1], ["có tin cơ bản", "có tin khác"], "không có tin")
    return d.reset_index(drop=True)
