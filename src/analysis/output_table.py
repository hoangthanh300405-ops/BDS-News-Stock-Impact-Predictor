"""Bảng OUTPUT hợp nhất của hệ thống (giai đoạn test, h = 1) — mỗi dòng = (mã, phiên), dự đoán cạnh kết quả thực tế.

  Chiều giá  : từ cấu hình cuối (§2f: Model A 3 lớp, bộ S2) — dấu(p_up − p_down); "tín hiệu tự tin" = |điểm| thuộc
               top 20% theo ngưỡng từ các fold trước (fold 1 chưa có ngưỡng -> để trống)
  Đồng thuận : trạng thái A/B từ §2d (biến thể được chọn theo validation)
  Chú ý      : xác suất thanh khoản đột biến phiên sau (§3b, giá & khối lượng + tin) và cờ top 10% mã trong phiên
  Thực tế    : lợi suất phiên sau, chiều thực tế, có đột biến thanh khoản thực tế không
Đầu ra: outputs/reports/bang_output.csv (mở bằng Excel).   Dùng:  python -m src.analysis.output_table
"""
import numpy as np
import pandas as pd

from src.config import CLEAN, ROOT

OUT = ROOT / "outputs"


def main():
    # --- chiều giá: cấu hình cuối
    fs = pd.read_parquet(OUT / "predictions" / "final_selective.parquet")
    fs = fs[(fs["h"] == 1) & (fs["part"] == "test") & (fs["dac_trung"].str.startswith("S2"))].copy()
    fs["thr"] = np.nan
    for k in sorted(fs["fold"].unique()):
        if k > 1:
            fs.loc[fs["fold"] == k, "thr"] = fs[fs["fold"] < k]["score"].abs().quantile(0.8)
    dir_ = fs[["date", "ticker", "score", "thr"]]

    # --- đồng thuận A/B
    ud = pd.read_parquet(OUT / "predictions" / "updown_v2.parquet")
    ud = ud[(ud["h"] == 1) & (ud["bien_the"].str.startswith("V1"))][["date", "ticker", "trang_thai"]]

    # --- chú ý
    at = pd.read_parquet(OUT / "predictions" / "attention.parquet")
    at = at[(at["level"] == "A mã") & (at["target"] == "vol_spike_1") & (at["spec"] == "giá & khối lượng + tin")]
    at = at[["date", "ticker", "p_1"]].copy()
    at["rank_pct"] = at.groupby("date")["p_1"].rank(ascending=False, pct=True)

    # --- thực tế
    ta = pd.read_parquet(CLEAN / "target_A.parquet", columns=["date", "ticker", "r_1", "vol_spike_1"])

    t = at.merge(dir_, on=["date", "ticker"], how="left").merge(ud, on=["date", "ticker"], how="left")
    t = t.merge(ta, on=["date", "ticker"], how="left")
    out = pd.DataFrame({
        "Phiên": t["date"].dt.date,
        "Mã": t["ticker"],
        "Dự báo chiều": np.select([t["score"] > 0, t["score"] < 0], ["Tăng", "Giảm"], ""),
        "Điểm tự tin |p_up − p_down|": t["score"].abs().round(3),
        "Tín hiệu tự tin (top 20%)": np.where(t["thr"].isna() | t["score"].isna(), "",
                                              np.where(t["score"].abs() >= t["thr"], "CÓ", "không")),
        "Đồng thuận A/B": t["trang_thai"].fillna(""),
        "Xác suất thanh khoản đột biến": t["p_1"].round(3),
        "Top 10% chú ý trong phiên": np.where(t["rank_pct"] <= 0.10, "CÓ", "không"),
        "Thực tế: lợi suất phiên sau (%)": (100 * t["r_1"]).round(2),
        "Thực tế: chiều": np.select([t["r_1"] > 0, t["r_1"] < 0], ["Tăng", "Giảm"], "Đứng"),
        "Thực tế: thanh khoản đột biến": np.where(t["vol_spike_1"] == 1, "CÓ", np.where(t["vol_spike_1"] == 0, "không", "")),
    })
    ok = (out["Dự báo chiều"] != "") & (out["Thực tế: chiều"] != "Đứng")
    out["Đúng chiều?"] = np.where(ok, np.where(out["Dự báo chiều"] == out["Thực tế: chiều"], "ĐÚNG", "sai"), "")
    out = out.sort_values(["Phiên", "Điểm tự tin |p_up − p_down|"], ascending=[True, False])
    out.to_csv(OUT / "reports" / "bang_output.csv", index=False, encoding="utf-8-sig")

    # --- tóm tắt kiểm tra (khớp với KET_QUA_TONG_HOP)
    conf = out[(out["Tín hiệu tự tin (top 20%)"] == "CÓ") & (out["Đúng chiều?"] != "")]
    top = out[(out["Top 10% chú ý trong phiên"] == "CÓ") & (out["Thực tế: thanh khoản đột biến"] != "")]
    allv = out[out["Thực tế: thanh khoản đột biến"] != ""]
    print(f"Bảng: {len(out):,} dòng, {out['Phiên'].nunique()} phiên, {out['Mã'].nunique()} mã -> outputs/reports/bang_output.csv")
    print(f"Tín hiệu tự tin: {len(conf):,} lần, đúng chiều {(conf['Đúng chiều?'] == 'ĐÚNG').mean():.1%}")
    print(f"Top 10% chú ý: thanh khoản đột biến thực tế {(top['Thực tế: thanh khoản đột biến'] == 'CÓ').mean():.1%} "
          f"(mức chung {(allv['Thực tế: thanh khoản đột biến'] == 'CÓ').mean():.1%})")


if __name__ == "__main__":
    main()
