"""Hướng NHÀ ĐẦU TƯ NHỎ LẺ — bảng sự kiện "đọc tin xong, có nên làm theo không?".

Một dòng = (mã, phiên t) có ít nhất một bài báo nói về mã với hướng tác động khác 0 (nhãn LLM cấp mã).
  • Nhà đầu tư đọc tin; sớm nhất giao dịch được ở GIÁ ĐÓNG CỬA phiên t (phiên t = phiên của bài: đăng trước 14:45
    -> cùng phiên, sau 14:45 / ngày nghỉ -> phiên kế tiếp).
  • Hành động "làm theo tin": tin tốt (d = +1) -> MUA; tin xấu (d = −1) -> BÁN nếu đang giữ (mua lại sau H phiên).
    VN không cho nhà đầu tư cá nhân bán khống, nên với tin xấu lãi/lỗ là phần lỗ TRÁNH ĐƯỢC so với cứ giữ.
  • Chạm trần (tin tốt) / chạm sàn (tin xấu) ở phiên t: thực tế không khớp được lệnh -> vào lệnh ở đóng cửa t+1.
  • Lãi/lỗ = d × lợi suất giá (thô, không trừ thị trường — đúng thứ nhà đầu tư nhận) trong H phiên − chi phí.
  • Target: follow_ok = 1 nếu làm theo tin có lãi sau phí.
Mọi đặc trưng chỉ dùng thông tin đến hết phiên t (tin đăng trước 14:45 phiên t, giá đến đóng cửa t).
Đầu ra: data/clean/retail_events.parquet
Dùng:  python -m src.retail.events
"""
import numpy as np
import pandas as pd

from src.config import CLEAN

H = 5                      # nắm giữ 5 phiên (~1 tuần)
COST = 0.004               # phí mua 0,15% + phí bán 0,15% + thuế bán 0,1%
LIMIT_TOL = 0.003
NHOM = ["tai_chinh_dn", "du_an_ha_tang", "phap_ly", "chinh_sach", "lai_suat_tin_dung", "thi_truong"]


def price_panels():
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet")
    px = px[px["ticker"].str.len() == 3]
    wide = lambda c: px.pivot(index="date", columns="ticker", values=c).sort_index()
    close, value, illiq = wide("close"), wide("value_bn"), wide("illiquid")
    prof = pd.read_parquet(CLEAN / "company_profile.parquet").set_index("ticker")
    limit = prof["san"].map({"HOSE": 0.07, "HNX": 0.10}).fillna(0.15).reindex(close.columns)
    return close, value, illiq, limit


def build():
    close, value, illiq, limit = price_panels()
    L = np.log(close)
    r1 = L.diff()
    simple = close.pct_change(fill_method=None)
    mkt = r1.where(illiq == 0).mean(axis=1)
    ceil_, floor_ = simple >= limit - LIMIT_TOL, simple <= -limit + LIMIT_TOL
    dates = close.index

    # ---- tin tức cấp (mã, phiên)
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session", "published_at", "title"])
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet")
    nlp = pd.read_parquet(CLEAN / "article_nlp.parquet", columns=["article_id", "pred_relevance"])
    e = atl.merge(art, on="article_id").merge(nlp, on="article_id", how="left")
    e = e[(e["event_type"] != "KHONG_LIEN_QUAN") & e["ticker"].isin(close.columns)].dropna(subset=["session"])
    e["session"] = pd.to_datetime(e["session"])
    pub = pd.to_datetime(e["published_at"], utc=True).dt.tz_convert("Asia/Ho_Chi_Minh")
    e["read_before_open"] = (pub.dt.tz_localize(None).dt.normalize() < e["session"]) | (pub.dt.hour < 9)
    e["abs_score"] = e["impact_score"].abs()
    for k, v in (("comm", "BINH_LUAN_TT"), ("firm", "SU_KIEN_DN"), ("policy", "CHINH_SACH_NGANH"), ("promo", "QUANG_BA")):
        e[f"f_{k}"] = (e["pred_relevance"] == v).astype(float)
    for n in NHOM:
        e[f"g_{n}"] = (e["nhom_tin"] == n).astype(float)
    e["lon"] = (e["muc_do_ma"] == "lon").astype(float)
    e = e.sort_values("abs_score", ascending=False)
    ev = e.groupby(["ticker", "session"]).agg(
        score=("impact_score", "mean"), score_absmax=("abs_score", "max"), n_art=("article_id", "nunique"),
        frac_comm=("f_comm", "mean"), frac_firm=("f_firm", "mean"), frac_policy=("f_policy", "mean"),
        frac_promo=("f_promo", "mean"), frac_lon=("lon", "mean"), frac_price_report=("price_report", "mean"),
        n_tick_in_art=("n_tickers_in_article", "mean"), frac_before_open=("read_before_open", "mean"),
        frac_cafef=("src", lambda s: (s == "cafef").mean()),
        **{f"g_{n}": (f"g_{n}", "mean") for n in NHOM},
        event_type=("event_type", "first"), title=("title", "first"), article_id=("article_id", "first"),
        titles=("title", lambda x: " | ".join(x.dropna().unique()[:8])),
    ).reset_index().rename(columns={"session": "date"})
    ev = ev[ev["date"].isin(dates)]
    ev["d"] = np.sign(ev["score"].round(6)).astype(int)
    n_all = len(ev)
    ev = ev[ev["d"] != 0].copy()

    # độ "mới" của tin: số phiên có tin về mã trong 5 / 20 phiên trước, số phiên từ lần có tin trước
    has = pd.DataFrame(0.0, index=dates, columns=close.columns)
    for t, g in e.groupby("ticker"):
        s = g["session"][g["session"].isin(dates)].unique()
        has.loc[s, t] = 1.0
    prev5, prev20 = has.shift(1).rolling(5).sum(), has.shift(1).rolling(20).sum()
    pos = pd.Series(np.arange(len(dates)), index=dates)
    last = has.mul(pos, axis=0).replace(0, np.nan).shift(1).ffill()
    since = (-last).add(pos, axis=0)

    # ---- đặc trưng giá tại đóng cửa t
    vz = np.log((value + 0.01) / (value.shift(1).rolling(20).mean() + 0.01))
    feats = {
        "r1": r1, "r5": r1.rolling(5).sum(), "r20": r1.rolling(20).sum(),
        "ar5": r1.sub(mkt, axis=0).rolling(5).sum(), "vol20": r1.rolling(20).std(), "value_z": vz,
        "dist_high20": L - L.rolling(20).max(), "dist_low20": L - L.rolling(20).min(),
        "ceil": ceil_.astype(float), "floor": floor_.astype(float), "illiquid": illiq,
        "news_prev5": prev5, "news_prev20": prev20, "since_last_news": since,
        "log_value": np.log1p(value.rolling(20).mean()),
    }
    r5 = r1.rolling(5).sum()
    feats["cs_rank_r5"] = r5.rank(axis=1, pct=True) - 0.5
    feats["cs_rank_value_z"] = vz.rank(axis=1, pct=True) - 0.5
    breadth5 = (r5 > 0).where(illiq == 0).mean(axis=1)
    mk = {"mkt1": mkt, "mkt5": mkt.rolling(5).sum(), "mkt20": mkt.rolling(20).sum(), "mkt_vol20": mkt.rolling(20).std(),
          "breadth5": breadth5 - 0.5}
    idx = pd.MultiIndex.from_frame(ev[["date", "ticker"]])
    for k, f in feats.items():
        ev[k] = f.stack(future_stack=True).reindex(idx).to_numpy()
    for k, s in mk.items():
        ev[k] = s.reindex(ev["date"]).to_numpy()

    # ---- hướng theo tin (d × ...) : "giá đã chạy bao nhiêu THEO hướng tin" — thứ người đọc báo thấy
    for k in ("r1", "r5", "r20", "ar5", "mkt1", "mkt5", "mkt20", "breadth5", "cs_rank_r5", "dist_high20", "dist_low20"):
        ev[f"d_{k}"] = ev["d"] * ev[k]
    ev["limit_in_dir"] = np.where(ev["d"] > 0, ev["ceil"], ev["floor"])
    ev["limit_against"] = np.where(ev["d"] > 0, ev["floor"], ev["ceil"])

    # ---- kết quả khi làm theo tin (vào lệnh ở t, hoặc t+1 nếu kẹt trần/sàn)
    off = ev["limit_in_dir"].fillna(0).astype(int).to_numpy()
    p = pos.reindex(ev["date"]).to_numpy() + off
    col = close.columns.get_indexer(ev["ticker"])
    C = close.to_numpy()

    def ret(hh):
        ok = p + hh < len(dates)
        out = np.full(len(ev), np.nan)
        out[ok] = C[p[ok] + hh, col[ok]] / C[p[ok], col[ok]] - 1
        return out
    for hh in (1, H, 10):
        ev[f"ret_{hh}"] = ret(hh)
        ev[f"pnl_{hh}"] = ev["d"] * ev[f"ret_{hh}"] - COST
    # "không đọc báo": lợi suất H phiên của rổ BĐS thanh khoản (trung bình cộng) từ cùng điểm vào lệnh
    ew = np.exp(mkt.fillna(0).cumsum()).to_numpy()
    okH = p + H < len(dates)
    ev["mkt_ret_H"] = np.nan
    ev.loc[okH, "mkt_ret_H"] = ew[p[okH] + H] / ew[p[okH]] - 1
    ev["entry_shift"] = off
    ev["follow_ok"] = (ev[f"pnl_{H}"] > 0).astype(float).where(ev[f"pnl_{H}"].notna())
    # so với rổ BĐS mua cùng lúc (cả hai cách đều mất phí nên không trừ phí)
    ev["rel_pnl"] = ev["d"] * (ev[f"ret_{H}"] - ev["mkt_ret_H"])
    ev["follow_rel"] = (ev["rel_pnl"] > 0).astype(float).where(ev["rel_pnl"].notna())
    credibility(ev, pos, p)
    context_v3(ev, e, close)
    ev = ev[ev["date"] > "2024-03-29"].reset_index(drop=True)
    print(f"(mã, phiên) có tin: {n_all:,} | có hướng ≠ 0 và sau khởi động: {len(ev):,} "
          f"(tin tốt {np.mean(ev['d'] > 0):.0%}) | có kết quả {H} phiên: {ev['follow_ok'].notna().sum():,}")
    print(f"Tỷ lệ 'làm theo tin có lãi sau phí' ({H} phiên): {ev['follow_ok'].mean():.3f} | "
          f"thắng rổ BĐS: {ev['follow_rel'].mean():.3f} | kẹt trần/sàn phải vào lệnh phiên sau: {ev['entry_shift'].mean():.1%}")
    ev.to_parquet(CLEAN / "retail_events.parquet", index=False)
    return ev


def credibility(ev, pos, p, k=10):
    """"Uy tín" của tin, chỉ dùng các tin ĐÃ BIẾT KẾT QUẢ trước thời điểm đọc (vị trí thoát lệnh ≤ phiên t):
    • cred_tick_*: trước đây làm theo tin về MÃ NÀY đúng bao nhiêu (co về 0,5 với trọng số k tin)
    • cred_type_*: ... về LOẠI SỰ KIỆN này
    • recent_*   : 20 phiên gần nhất, làm theo tin (mọi mã) đúng bao nhiêu — bắt 'chế độ thị trường'."""
    t_pos = pos.reindex(ev["date"]).to_numpy()
    known = p + H                                           # phiên đóng lệnh -> biết kết quả sau đóng cửa phiên này
    for y, tag in (("follow_ok", "ok"), ("follow_rel", "rel")):
        hist = pd.DataFrame({"known": known, "y": ev[y].to_numpy(), "ticker": ev["ticker"].to_numpy(),
                             "et": ev["event_type"].to_numpy()}).dropna(subset=["y"])
        q = pd.DataFrame({"t": t_pos, "ticker": ev["ticker"].to_numpy(), "et": ev["event_type"].to_numpy(),
                          "i": np.arange(len(ev))})
        for key, col in (("ticker", f"cred_tick_{tag}"), ("et", f"cred_type_{tag}")):
            h = hist.sort_values("known").copy()
            h["cs"], h["cn"] = h.groupby(key)["y"].cumsum(), h.groupby(key).cumcount() + 1
            h = h.groupby([key, "known"]).last().reset_index()[[key, "known", "cs", "cn"]]
            m = pd.merge_asof(q.sort_values("t"), h.sort_values("known"), left_on="t", right_on="known", by=key,
                              allow_exact_matches=True).set_index("i").reindex(q["i"])
            cs, cn = m["cs"].fillna(0).to_numpy(), m["cn"].fillna(0).to_numpy()
            ev[col] = (cs + 0.5 * k) / (cn + k) - 0.5
        g = hist.groupby("known")["y"].agg(["sum", "count"]).reindex(np.arange(len(pos)), fill_value=0)
        roll = g.rolling(20, min_periods=1).sum()
        rate = (roll["sum"] / roll["count"].replace(0, np.nan)).to_numpy()
        ev[f"recent_{tag}"] = rate[np.clip(t_pos, 0, len(rate) - 1).astype(int)] - 0.5


def context_v3(ev, e, close):
    """v3: hồ sơ doanh nghiệp + 'nhiệt độ tin tức' của phiên (chỉ dùng tin đã đăng đến phiên t)."""
    prof = pd.read_parquet(CLEAN / "company_profile.parquet").set_index("ticker")
    ev["hose"] = ev["ticker"].map(prof["san"].eq("HOSE").astype(float))
    cap = np.log(close.stack(future_stack=True).rename("c").reset_index().merge(
        prof["so_cp_luu_hanh"], left_on="ticker", right_index=True).assign(v=lambda x: x["c"] * x["so_cp_luu_hanh"])
        .set_index(["date", "ticker"])["v"])
    ev["log_mcap"] = cap.reindex(pd.MultiIndex.from_frame(ev[["date", "ticker"]])).to_numpy()
    for c in [c for c in prof if c.startswith("exposure_")]:
        ev[c] = ev["ticker"].map(prof[c]).astype(float)
    day = e.groupby("session").agg(day_n_art=("article_id", "nunique"), day_n_tick=("ticker", "nunique"),
                                   day_pos=("impact_score", lambda x: (x > 0).mean()))
    day["day_n_art"] = np.log1p(day["day_n_art"])
    for c in day:
        ev[c] = ev["date"].map(day[c]).to_numpy()
    ev["share_of_day"] = ev["n_art"] / np.expm1(ev["day_n_art"]).clip(lower=1)
    ev["d_day_pos"] = ev["d"] * (ev["day_pos"] - 0.5)


FEATURES = ["d", "score", "score_absmax", "n_art", "frac_comm", "frac_firm", "frac_policy", "frac_promo", "frac_lon",
            "frac_price_report", "n_tick_in_art", "frac_before_open", "frac_cafef"] + [f"g_{n}" for n in NHOM] + [
            "news_prev5", "news_prev20", "since_last_news",
            "d_r1", "d_r5", "d_r20", "d_ar5", "d_mkt1", "d_mkt5", "d_dist_high20", "d_dist_low20",
            "limit_in_dir", "limit_against", "vol20", "value_z", "illiquid", "log_value"]
NEWS_ONLY = FEATURES[:FEATURES.index("since_last_news") + 1]      # chỉ thông tin từ bài báo
# v2: thêm bối cảnh thị trường, thứ hạng trong ngày, "uy tín" của tin (chỉ dùng kết quả đã biết)
EXTRA_V2 = ["d_mkt20", "mkt_vol20", "d_breadth5", "d_cs_rank_r5", "cs_rank_value_z",
            "cred_tick_ok", "cred_type_ok", "recent_ok", "cred_tick_rel", "cred_type_rel", "recent_rel"]
FEATURES_V2 = FEATURES + EXTRA_V2
NEWS_ONLY_V2 = NEWS_ONLY + ["cred_tick_ok", "cred_type_ok", "cred_tick_rel", "cred_type_rel"]
# v3: hồ sơ doanh nghiệp + nhiệt độ tin tức của phiên. KHÔNG dùng log_mcap: số CP lưu hành là số hiện tại (nhìn trước)
EXTRA_V3 = ["hose", "exposure_can_ho", "exposure_dat_nen", "exposure_thap_tang", "exposure_kcn",
            "exposure_nghi_duong", "day_n_art", "day_n_tick", "d_day_pos", "share_of_day"]
FEATURES_V3 = FEATURES_V2 + EXTRA_V3

if __name__ == "__main__":
    build()
