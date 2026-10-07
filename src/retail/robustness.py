"""P5 — KIỂM TRA ĐỘ VỮNG của mô phỏng danh mục (P4): kết luận có đổi khi thay giả định không?

Kịch bản (mỗi lần đổi MỘT giả định so với mặc định: phí 0,4%, 10 vị thế, giữ 5 phiên, vào lệnh ở đóng cửa phiên đọc tin):
  • Phí + thuế: 0,2% / 0,6% / 0,8% mỗi vòng (0,8% ≈ có thêm trượt giá)
  • Số vị thế tối đa: 5 / 20
  • Giữ 1 / 10 phiên (tín hiệu vẫn từ mô hình học với 5 phiên)
  • Vào lệnh TRỄ 1 phiên (đọc tin buổi tối, hôm sau mới mua)
  • Nửa đầu (2025Q2–Q4) / nửa sau (2026Q1–Q3) của giai đoạn test
Đầu ra: outputs/reports/retail_robustness.csv; outputs/figures/r12_robustness.png
Dùng:  python -m src.retail.robustness   (sau src.retail.run_final)
"""
import numpy as np
import pandas as pd

from src.config import CLEAN, ROOT
from src.retail.events import COST, H, price_panels
from src.retail.portfolio import CAPITAL, PRED, SLOTS, simulate, stats_of

REP, FIG = ROOT / "outputs" / "reports", ROOT / "outputs" / "figures"
N_RANDOM = 50
STRATS = {"C · tiêu chuẩn (top 10%)": "top10|E3", "C · top 20%": "top20|E3", "C · thận trọng (đồng thuận)": "top20|CONS"}  # điểm = FINAL


def delayed_ret(sig, close, hold, delay):
    pos = pd.Series(np.arange(len(close)), index=close.index)
    p = pos.reindex(sig["date"]).to_numpy() + sig["entry_shift"].to_numpy() + delay
    c = close.columns.get_indexer(sig["ticker"])
    C = close.to_numpy()
    ok = p + hold < len(C)
    out = np.full(len(sig), np.nan)
    out[ok] = C[p[ok] + hold, c[ok]] / C[p[ok], c[ok]] - 1
    return out


def basket(close, illiq, start, end, cost):
    r1 = np.log(close).diff().where(illiq == 0).mean(axis=1).loc[start:end]
    return CAPITAL * (1 - cost) * np.exp(r1.fillna(0).cumsum())


def run_scenario(name, good, cal, close, illiq, cost=COST, slots=SLOTS, hold=H, delay=0):
    good = good.copy()
    good["_ret"] = delayed_ret(good, close, hold, delay)
    good = good[good["_ret"].notna()]
    kw = dict(cost=cost, slots=slots, hold=hold, ret_col="_ret", delay=delay)
    rng = np.random.default_rng(2026)
    finals = [simulate(good, cal, rng=rng, **kw)[0] for _ in range(N_RANDOM)]
    eqA = finals[int(np.argsort([e.iloc[-1] for e in finals])[N_RANDOM // 2])]
    row = {"kịch bản": name, "A · mọi tin tốt": stats_of(eqA)["lợi nhuận cả kỳ"],
           "B · rổ BĐS": stats_of(basket(close, illiq, eqA.index[0], eqA.index[-1], cost))["lợi nhuận cả kỳ"]}
    for k, col in STRATS.items():
        eq, st = simulate(good[good[col]], cal, score="s|E3", **kw)
        row[k] = stats_of(eq)["lợi nhuận cả kỳ"]
        row[f"{k} · tỷ lệ lệnh lãi"] = st["tỷ lệ lệnh lãi"]
    return row


def main():
    res = pd.read_parquet(PRED / "final_tuyet_doi.parquet")
    cal = pd.DatetimeIndex(pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"])
    close, _, illiq, _ = price_panels()
    good = res[res["d"] > 0].copy()
    sc = [("Mặc định (phí 0,4%, 10 vị thế, giữ 5, vào lệnh ngay)", {})]
    sc += [(f"Phí {c * 100:.1f}%".replace(".", ","), {"cost": c}) for c in (0.002, 0.006, 0.008)]
    sc += [(f"{n} vị thế tối đa", {"slots": n}) for n in (5, 20)]
    sc += [(f"Giữ {h} phiên", {"hold": h}) for h in (1, 10)]
    sc += [("Vào lệnh trễ 1 phiên", {"delay": 1})]
    rows = [run_scenario(n, good, cal, close, illiq, **kw) for n, kw in sc]
    for label, qs in (("Chỉ nửa đầu (2025Q2–Q4)", ["2025Q2", "2025Q3", "2025Q4"]),
                      ("Chỉ nửa sau (2026Q1–Q3)", ["2026Q1", "2026Q2", "2026Q3"])):
        rows.append(run_scenario(label, good[good["quy_test"].isin(qs)], cal, close, illiq))
    out = pd.DataFrame(rows)
    out.to_csv(REP / "retail_robustness.csv", index=False, encoding="utf-8-sig")
    main_cols = ["A · mọi tin tốt", "B · rổ BĐS"] + list(STRATS)
    beat = (out[list(STRATS)].min(axis=1) > out[["A · mọi tin tốt", "B · rổ BĐS"]].max(axis=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(out))
    colors = ["#cf222e", "#8c959f", "#0a3069", "#1f6feb", "#2da44e"]
    for i, (c, col) in enumerate(zip(main_cols, colors)):
        ax.scatter(out[c] * 100, y + (i - 2) * 0.12, color=col, label=c, s=28, zorder=3)
    ax.axvline(0, color="k", lw=.6)
    ax.set_yticks(y, out["kịch bản"], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("lợi nhuận cả kỳ sau phí (%)"); ax.grid(alpha=.3, axis="x")
    ax.set_title(f"P5. Độ vững: mô hình (xanh) hơn cả A và B ở {beat.sum()}/{len(out)} kịch bản")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "r12_robustness.png", dpi=150); plt.close(fig)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(out[["kịch bản"] + main_cols].to_string(index=False))
        print(f"\nMọi chiến lược C hơn cả A và B ở {beat.sum()}/{len(out)} kịch bản")
        print(out[["kịch bản"] + [c for c in out if "tỷ lệ" in c]].to_string(index=False))


if __name__ == "__main__":
    main()
