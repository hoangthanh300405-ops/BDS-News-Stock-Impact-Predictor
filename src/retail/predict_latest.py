"""SẢN PHẨM — khuyến nghị cho những tin MỚI NHẤT (chưa biết kết quả) bằng mô hình FINAL.

Cách làm (giống hệt quy trình đánh giá, chỉ dời mốc thời gian đến dữ liệu mới nhất):
  1. "Val" = quý gần nhất có kết quả; train = mọi tin có kết quả trước đó (có embargo) → điểm trên val làm thang
     phân vị và ngưỡng (top 10%, top 20% từng mô hình, top/bottom 15%).
  2. Fit lại trên TOÀN BỘ tin đã có kết quả → chấm các tin chưa có kết quả (phiên cuối dữ liệu).
Đầu ra: outputs/reports/khuyen_nghi_moi_nhat.csv
Dùng:  python -m src.retail.predict_latest      (sau src.retail.events)
"""
import numpy as np
import pandas as pd

from src.config import CLEAN, ROOT
from src.eval.walk_forward import EMBARGO
from src.retail.events import FEATURES_V3, H
from src.retail.run_retail_v4 import CONTS, fit_score, pct_vs, weights

REP = ROOT / "outputs" / "reports"
COMPS = ["LR", "RF", "REG"]
Y, CONT = "follow_ok", CONTS["tuyet_doi"]


def main():
    ev = pd.read_parquet(CLEAN / "retail_events.parquet")
    lab = ev[ev[Y].notna()]
    last = lab["date"].max()
    new = ev[ev[Y].isna() & (ev["date"] > last)].copy()     # tin sau phiên có kết quả cuối (bỏ mã ngừng GD cũ)
    cal = pd.DatetimeIndex(pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"])
    val_start = (last.to_period("Q") - 0).start_time if (last - last.to_period("Q").start_time).days > 45 else \
        (last.to_period("Q") - 1).start_time
    pos = pd.Series(np.arange(len(cal)), index=cal)
    cut = cal[max(pos[cal[cal >= val_start][0]] - H - 1 - EMBARGO, 0)]
    tr, va = lab[lab["date"] < cut], lab[lab["date"] >= val_start]
    ref = {}
    for k in COMPS:
        ref[k] = fit_score(k, tr, va, Y, CONT, FEATURES_V3, weights(tr["date"], tr["date"].max()))
    e_va = np.mean([pct_vs(ref[k], ref[k]) for k in COMPS], axis=0)
    w_all = weights(lab["date"], lab["date"].max())
    sc = {k: fit_score(k, lab, new, Y, CONT, FEATURES_V3, w_all) for k in COMPS}
    new["điểm (0–100)"] = np.mean([pct_vs(ref[k], sc[k]) for k in COMPS], axis=0) * 100
    s = new["điểm (0–100)"] / 100
    std = s >= np.quantile(e_va, 0.90)
    cons = np.all([sc[k] >= np.quantile(ref[k], 0.80) for k in COMPS], axis=0)
    hi, lo = s >= np.quantile(e_va, 0.85), s <= np.quantile(e_va, 0.15)
    act = lambda d, yes: np.where(yes, np.where(d > 0, "MUA", "BÁN nếu đang giữ"),
                                  np.where(d > 0, "Không mua", "Giữ, không bán tháo"))
    new["khuyến nghị · tiêu chuẩn"] = act(new["d"], std)
    new["khuyến nghị · thận trọng"] = np.where(cons, act(new["d"], True), "—")
    new["khuyến nghị · hai phía"] = np.where(hi, act(new["d"], True),
                                             np.where(lo, np.where(new["d"] > 0, "ĐỪNG mua", "ĐỪNG bán tháo"), "Không đủ tự tin"))
    out = new[["date", "ticker", "title", "event_type", "d", "frac_comm", "frac_firm", "d_r5", "d_mkt20",
               "điểm (0–100)", "khuyến nghị · tiêu chuẩn", "khuyến nghị · thận trọng", "khuyến nghị · hai phía"]].rename(
        columns={"date": "phiên", "ticker": "mã", "title": "tiêu đề", "event_type": "loại sự kiện", "d": "hướng tin",
                 "frac_comm": "tỷ lệ bài bình luận", "frac_firm": "tỷ lệ bài sự kiện DN",
                 "d_r5": "giá đã chạy 5 phiên theo hướng tin", "d_mkt20": "rổ BĐS 20 phiên theo hướng tin"})
    out = out.sort_values(["phiên", "điểm (0–100)"], ascending=[False, False])
    out.to_csv(REP / "khuyen_nghi_moi_nhat.csv", index=False, encoding="utf-8-sig")
    print(f"Train đến {lab['date'].max().date()} ({len(lab):,} tin có kết quả) | thang điểm từ quý bắt đầu {val_start.date()} "
          f"({len(va):,} tin) | chấm {len(new):,} tin mới ({new['date'].min().date()} → {new['date'].max().date()})")
    print(out["khuyến nghị · tiêu chuẩn"].value_counts().to_dict(), "| thận trọng:",
          int((out["khuyến nghị · thận trọng"] != "—").sum()), "| hai phía:", out["khuyến nghị · hai phía"].value_counts().to_dict())
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(out.head(10)[["phiên", "mã", "tiêu đề", "điểm (0–100)", "khuyến nghị · tiêu chuẩn", "khuyến nghị · hai phía"]])


if __name__ == "__main__":
    main()
