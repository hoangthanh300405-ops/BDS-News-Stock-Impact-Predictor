"""Demo 100 triệu đồng với MÔ HÌNH TOÀN NGÀNH (đọc outputs/predictions/sector.parquet của src.sector.run_sector).

Quy tắc: 5 phiên một lần, xem khuyến nghị của phiên đó → dồn toàn bộ vốn vào rổ BĐS (EW liquid) ở giá đóng cửa và giữ
5 phiên, hoặc để tiền mặt (lãi 0%). Phí 0,4% (mua + bán) mỗi lần vào rổ; đang giữ thì không mất thêm phí.
Có 5 cách chọn phiên bắt đầu (lệch 0–4 phiên) → báo cáo đường vốn của cách 0 và khoảng thấp–cao của cả 5 cách.
Đầu ra: outputs/reports/sector_demo_100tr.csv, outputs/figures/r21_sector_100tr.png
Dùng:  python -m src.sector.demo_100tr
"""
import numpy as np
import pandas as pd

from src.config import ROOT

VON, H, COST = 100_000_000, 5, 0.004
REP, FIG = ROOT / "outputs" / "reports", ROOT / "outputs" / "figures"


def strategies(r):
    return {"Giữ rổ BĐS suốt kỳ": pd.Series(True, index=r.index),
            "Mô hình ngành: giữ khi điểm > trung vị": r["s|ALL"] > 0.5,
            "Mô hình ngành: chỉ giữ khi MUA (top 30%)": r["hi30"].astype(bool),
            "Quy tắc động lượng (rổ tăng 20 phiên)": r["p_r20"] > 0 if "p_r20" in r else r["mom_rule"].astype(bool)}


def path(r, hold):
    enter = hold & ~hold.shift(1, fill_value=False)
    step = np.where(hold, np.exp(r["fwd"]), 1.0) * np.where(enter, 1 - COST, 1.0)
    # vốn tại ngày QUYẾT ĐỊNH của mỗi kỳ (trước khi kỳ đó diễn ra); điểm cuối = sau kỳ cuối cùng
    ends = list(r["date"]) + [r["date"].iloc[-1] + pd.offsets.BDay(H)]
    return pd.Series(VON * np.concatenate([[1.0], np.cumprod(step)]), index=pd.DatetimeIndex(ends)), int(enter.sum())


def main():
    res = pd.read_parquet(ROOT / "outputs" / "predictions" / "sector.parquet").sort_values("date").reset_index(drop=True)
    rows, curves = [], {}
    for off in range(H):
        r = res.iloc[off::H].reset_index(drop=True)
        for name, hold in strategies(r).items():
            eq, n_in = path(r, hold)
            wins = (r.loc[hold, "fwd"] > np.log(1 + COST)).mean()
            peak = eq.cummax()
            rows.append({"chiến lược": name, "offset": off, "vốn cuối": eq.iloc[-1], "số lần vào rổ": n_in,
                         "kỳ cầm rổ": int(hold.sum()), "tỷ lệ kỳ cầm rổ có lãi": wins,
                         "sụt giảm lớn nhất": (eq / peak - 1).min()})
            if off == 0:
                curves[name] = eq
    d = pd.DataFrame(rows)
    main_ = d[d.offset == 0].drop(columns="offset").set_index("chiến lược")
    rng = d.groupby("chiến lược", sort=False)["vốn cuối"].agg(["min", "max"])
    main_["lãi/lỗ (triệu)"] = (main_["vốn cuối"] - VON) / 1e6
    main_["lợi suất"] = main_["vốn cuối"] / VON - 1
    main_["lợi suất 5 cách bắt đầu (thấp–cao)"] = [f"{a / VON - 1:+.1%} → {b / VON - 1:+.1%}" for a, b in rng.loc[main_.index].to_numpy()]
    out = main_.reset_index()
    out.to_csv(REP / "sector_demo_100tr.csv", index=False, encoding="utf-8-sig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, eq in curves.items():
        ax.plot(eq.index, eq / 1e6, lw=2 if "Mô hình" in name else 1.4, label=name)
    ax.axhline(100, color="grey", lw=1, ls="--")
    ax.set_ylabel("Vốn (triệu đồng)")
    ax.set_title("Demo 100 triệu — mô hình toàn ngành, rổ BĐS, 04/2025 → 09/2026 (phí 0,4%)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "r21_sector_100tr.png", dpi=150)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(out.to_string(index=False, formatters={"vốn cuối": "{:,.0f}".format, "lãi/lỗ (triệu)": "{:+.1f}".format,
                                                     "lợi suất": "{:+.1%}".format, "tỷ lệ kỳ cầm rổ có lãi": "{:.1%}".format,
                                                     "sụt giảm lớn nhất": "{:.1%}".format}))
        print("Kỳ test:", res.date.min().date(), "→", res.date.max().date())


if __name__ == "__main__":
    main()
