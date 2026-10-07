"""P4 — Mô phỏng DANH MỤC của một nhà đầu tư nhỏ lẻ thật (tập test 2025Q2 → 2026Q3).

Giả định:
  • Vốn 100 triệu đồng, chia 10 "ô" 10 triệu; tối đa 10 vị thế cùng lúc; ô trống không sinh lãi.
  • Chỉ MUA theo tin tốt (không bán khống); giữ 5 phiên; vào lệnh ở đóng cửa phiên đọc tin (phiên sau nếu kịch trần).
  • Phí + thuế 0,4% mỗi vòng mua–bán. Mỗi phiên, nếu có nhiều tín hiệu hơn số ô trống: chiến lược có mô hình ưu tiên
    điểm cao hơn; chiến lược A (làm theo mọi tin) chọn ngẫu nhiên — lặp 200 lần, báo trung vị và khoảng 5–95%.
  • B: mua rổ BĐS (trung bình cộng các mã thanh khoản) đầu kỳ, giữ đến cuối kỳ (một lần phí).
Chiến lược C dùng dự đoán của mô hình FINAL (src.retail.run_final): top 10% = chế độ tiêu chuẩn, CONS = thận trọng.
Đầu ra: outputs/reports/retail_portfolio.csv; outputs/figures/r11_portfolio.png
Dùng:  python -m src.retail.portfolio   (sau src.retail.run_final)
"""
import numpy as np
import pandas as pd

from src.config import CLEAN, ROOT
from src.retail.events import COST, H, price_panels

REP, FIG, PRED = ROOT / "outputs" / "reports", ROOT / "outputs" / "figures", ROOT / "outputs" / "predictions"
CAPITAL, SLOTS = 100.0, 10          # triệu đồng
N_RANDOM = 200


def simulate(sig, cal, score=None, rng=None, cost=COST, slots=SLOTS, hold=H, ret_col=f"ret_{H}", delay=0):
    """sig: các tín hiệu mua (date, ret_col, entry_shift). Trả về chuỗi vốn theo phiên + thống kê lệnh.
    delay: số phiên chờ thêm trước khi vào lệnh (ret_col phải được tính tương ứng)."""
    pos = pd.Series(np.arange(len(cal)), index=cal)
    sig = sig.assign(t=pos.reindex(sig["date"]).to_numpy())
    sig["key"] = -sig[score] if score else rng.random(len(sig))
    by_t = {t: g.sort_values("key") for t, g in sig.groupby("t")}
    slot_val = np.full(slots, CAPITAL / slots)
    busy_until = np.full(slots, -1)
    pending = {}                                       # ô -> (phiên đóng lệnh, lợi suất)
    equity, trades = [], []
    t0, t1 = int(sig["t"].min()), int(sig["t"].max()) + hold + delay + 2
    for t in range(t0, t1 + 1):
        for s, (t_exit, r) in list(pending.items()):   # đóng lệnh đến hạn
            if t_exit <= t:
                slot_val[s] *= (1 + r - cost)
                trades.append(r - cost)
                del pending[s]
        for _, row in by_t.get(t, pd.DataFrame()).iterrows():
            free = [s for s in range(slots) if busy_until[s] < t and s not in pending]
            if not free:
                break
            s = free[0]
            t_exit = t + int(row["entry_shift"]) + delay + hold
            pending[s] = (t_exit, row[ret_col])
            busy_until[s] = t_exit
        equity.append((cal[min(t, len(cal) - 1)], slot_val.sum()))
    eq = pd.Series(dict(equity))
    tr = np.array(trades)
    return eq, {"số lệnh": len(tr), "tỷ lệ lệnh lãi": (tr > 0).mean() if len(tr) else np.nan,
                "lãi TB/lệnh": tr.mean() if len(tr) else np.nan}


def stats_of(eq):
    ret = eq.iloc[-1] / CAPITAL - 1
    dd = (eq / eq.cummax() - 1).min()
    daily = eq.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(250) if daily.std() > 0 else np.nan
    return {"lợi nhuận cả kỳ": ret, "sụt giảm lớn nhất": dd, "Sharpe (năm hóa)": sharpe}


def main():
    res = pd.read_parquet(PRED / "final_tuyet_doi.parquet")
    cal = pd.DatetimeIndex(pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"])
    good = res[(res["d"] > 0) & res[f"ret_{H}"].notna()].copy()
    rows, curves = [], {}
    # A: làm theo mọi tin tốt (chọn ngẫu nhiên khi thiếu ô)
    rng = np.random.default_rng(2026)
    runs = [simulate(good, cal, rng=rng) for _ in range(N_RANDOM)]
    finals = np.array([e.iloc[-1] for e, _ in runs])
    med = int(np.argsort(finals)[len(finals) // 2])
    eqA, stA = runs[med]
    rows.append({"nhà đầu tư": "A · Đọc báo, mua theo mọi tin tốt (trung vị 200 lần)", **stA, **stats_of(eqA),
                 "khoảng 5–95% lợi nhuận": f"[{np.percentile(finals, 5) / CAPITAL - 1:+.1%}, {np.percentile(finals, 95) / CAPITAL - 1:+.1%}]"})
    curves["A · mua theo mọi tin tốt"] = eqA
    # B: mua rổ BĐS và giữ
    close, _, illiq, _ = price_panels()
    r1 = np.log(close).diff().where(illiq == 0).mean(axis=1)
    idx = r1.loc[eqA.index[0]:eqA.index[-1]]
    eqB = CAPITAL * (1 - COST) * np.exp(idx.fillna(0).cumsum())
    rows.append({"nhà đầu tư": "B · Không đọc báo, mua rổ BĐS và giữ", "số lệnh": 1, **stats_of(eqB)})
    curves["B · mua rổ BĐS và giữ"] = eqB
    # C: có mô hình (ưu tiên điểm cao)
    for name, col, score in (("C · FINAL top 20%", "top20|E3", "s|E3"), ("C · FINAL tiêu chuẩn (top 10%)", "top10|E3", "s|E3"),
                             ("C · FINAL thận trọng (3 mô hình đồng thuận)", "top20|CONS", "s|E3")):
        eq, st = simulate(good[good[col]], cal, score=score)
        rows.append({"nhà đầu tư": name, **st, **stats_of(eq)})
        curves[name] = eq
    out = pd.DataFrame(rows)
    out.to_csv(REP / "retail_portfolio.csv", index=False, encoding="utf-8-sig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#cf222e", "#8c959f", "#1f6feb", "#0a3069", "#2da44e"]
    for (k, e), c in zip(curves.items(), colors):
        ax.plot(e.index, e.values, label=k, color=c, lw=2 if "A" in k or "B" in k else 1.6)
    ax.axhline(CAPITAL, color="k", lw=.6)
    ax.set_ylabel("giá trị danh mục (triệu đồng)")
    ax.set_title("P4. 100 triệu đồng, tối đa 10 vị thế, giữ 5 phiên, đã trừ phí — tập test 2025Q2–2026Q3")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "r11_portfolio.png", dpi=150); plt.close(fig)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
