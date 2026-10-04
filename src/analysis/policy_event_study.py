"""Mô-đun chính sách P2 — event study quanh đợt chính sách (§9.2).

(1) Cấp ngành: CAR của benchmark quanh 'đợt lớn' (số bài chính sách bất thường, z ≥ 2,5 so với 60 phiên trước),
    mô hình lợi suất trung bình ước lượng trên [t−60, t−11]; cửa sổ [−5, +10]; tách theo nhóm giai đoạn.
    Kiểm định: t chéo các đợt + bootstrap; giới hạn: không tách được cú sốc chung của cả thị trường (không có VN-Index).
(2) Hồi quy chéo: CAR_i,e[0,+3] (AR so với ngành) = α_e + b·exposure_i + b3·(exposure_i × nhóm giai đoạn_e),
    trên 107 đợt (k = 5), SE gom cụm theo đợt. α_e hấp thụ mọi cú sốc chung của đợt.
Dùng:  python -m src.analysis.policy_event_study
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import CLEAN, ROOT
from src.eval.p0_report import INK2, MUTED, SERIES, _save

REP = ROOT / "outputs" / "reports"
W = range(-5, 11)
EXPO = ["exposure_can_ho", "exposure_dat_nen", "exposure_thap_tang", "exposure_kcn", "exposure_nghi_duong"]
STAGES = ("ban_hanh", "hieu_luc")


def major_events(z_thr=2.5, min_gap=10):
    cal = pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"]
    ls = pd.read_parquet(CLEAN / "legal_streams.parquet")
    pol = ls[ls["stream"] == "chinh_sach"]
    cnt = pol.groupby("session").size().reindex(cal, fill_value=0)
    z = (cnt - cnt.rolling(60, min_periods=20).mean().shift(1)) / cnt.rolling(60, min_periods=20).std().shift(1)
    days = list(z[z >= z_thr].index)
    ev, last = [], None
    for d in days:                                        # gộp các ngày gần nhau: giữ ngày đầu của mỗi cụm
        i = cal[cal == d].index[0]
        if last is None or i - last > min_gap:
            ev.append(d)
        last = i
    stage = pol.assign(g=pol["nhom_giai_doan"].replace("", "khong_ro")).groupby("session")["g"] \
        .agg(lambda s: s[s != "khong_ro"].mode().iat[0] if (s != "khong_ro").any() else "khong_ro")
    return pd.DataFrame({"date": ev, "nhom_giai_doan": [stage.get(d, "khong_ro") for d in ev]})


def industry_car(ev):
    bench = pd.read_parquet(CLEAN / "benchmark_nganh.parquet")
    r = bench[bench["method"] == "ew_liquid"].set_index("date")["log_return"]
    pos = pd.Series(range(len(r)), index=r.index)
    rows = []
    for e in ev.itertuples():
        i = pos[e.date]
        if i - 60 < 1 or i + 10 >= len(r):
            continue
        mu = r.iloc[i - 60:i - 10].mean()                   # mô hình lợi suất trung bình
        ar = [r.iloc[i + k] - mu for k in W]
        rows.append({"date": e.date, "nhom_giai_doan": e.nhom_giai_doan, **{k: v for k, v in zip(W, ar)}})
    A = pd.DataFrame(rows)
    C = A[list(W)].cumsum(axis=1)
    C = C.sub(C[-1], axis=0)                                # CAR = 0 tại phiên −1
    return A, C


def cross_section(k=5):
    """CAR_i,e[0,+3] của từng mã (so với ngành) quanh 107 đợt (k = 5 bài/phiên)."""
    ev = pd.read_parquet(CLEAN / "policy_events.parquet").sort_values("date").groupby("event_id").first().reset_index()
    ls = pd.read_parquet(CLEAN / "legal_streams.parquet")
    st = ls[ls["stream"] == "chinh_sach"].groupby("session")["nhom_giai_doan"] \
        .agg(lambda s: s[s != ""].mode().iat[0] if (s != "").any() else "khong_ro")
    ev["nhom_giai_doan"] = ev["date"].map(st).fillna("khong_ro")
    px = pd.read_parquet(CLEAN / "stock_prices_clean.parquet")
    bench = pd.read_parquet(CLEAN / "benchmark_nganh.parquet")
    rB = bench[bench["method"] == "ew_liquid"].set_index("date")["log_return"]
    lr = px.pivot_table(index="date", columns="ticker", values="log_return").reindex(rB.index)
    ill = px.pivot_table(index="date", columns="ticker", values="illiquid").reindex(rB.index)
    ar = lr.sub(rB, axis=0)
    pos = pd.Series(range(len(ar)), index=ar.index)
    prof = pd.read_parquet(CLEAN / "company_profile.parquet").set_index("ticker")
    rows = []
    for e in ev.itertuples():
        i = pos[e.date]
        if i + 3 >= len(ar):
            continue
        car = ar.iloc[i:i + 4].sum(min_count=3)
        liquid = ill.iloc[i].eq(0)
        for t in car.index[car.notna() & liquid.reindex(car.index).fillna(False)]:
            rows.append({"event_id": e.event_id, "stage": e.nhom_giai_doan, "ticker": t, "car": car[t] * 100,
                         **prof.loc[t, EXPO].to_dict()})
    d = pd.DataFrame(rows)
    # Chỉ giữ giai đoạn có đủ đợt để SE gom cụm đáng tin: 'de_xuat' chỉ có 4 đợt -> loại (ghi rõ trong báo cáo).
    # Tham số hóa: một độ dốc phơi nhiễm RIÊNG cho mỗi giai đoạn, không có hiệu ứng chính — bản đầu có cả hiệu ứng
    # chính lẫn đủ 3 tương tác nên ma trận thiết kế suy biến (hạng 13/15) và Wald chung ra F = 80 giả tạo.
    d = d[d["stage"].isin(STAGES)].reset_index(drop=True)
    X = pd.DataFrame({f"{c}×{s}": d[c].astype(float) * (d["stage"] == s) for s in STAGES for c in EXPO})
    # hiệu ứng cố định theo đợt: khử trung bình trong đợt (tổng AR so với ngành ≈ 0 theo cấu trúc)
    y = d["car"] - d.groupby("event_id")["car"].transform("mean")
    Xd = X - X.groupby(d["event_id"]).transform("mean")
    assert np.linalg.matrix_rank(Xd.to_numpy()) == Xd.shape[1], "ma trận thiết kế suy biến"
    m = sm.OLS(y, Xd).fit(cov_type="cluster", cov_kwds={"groups": d["event_id"]})
    return d, m


def main():
    ev = major_events()
    A, C = industry_car(ev)
    print(f"Đợt chính sách lớn (z ≥ 2,5, cách nhau > 10 phiên): {len(C)} | giai đoạn: {A['nhom_giai_doan'].value_counts().to_dict()}")
    rng = np.random.default_rng(0)
    rows = []
    for win in [(-5, -1), (0, 0), (0, 1), (0, 5), (0, 10)]:
        v = A[list(range(win[0], win[1] + 1))].sum(axis=1) * 100          # CAR = tổng AR trong cửa sổ
        boots = [rng.choice(v, len(v)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"cửa sổ": f"[{win[0]:+d}, {win[1]:+d}]", "CAR TB (%)": v.mean(), "t": v.mean() / (v.std() / np.sqrt(len(v))),
                     "CI95 bootstrap": f"[{lo:+.2f}, {hi:+.2f}]", "n đợt": len(v)})
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    m, se = C.mean() * 100, C.std() * 100 / np.sqrt(len(C))
    ax.fill_between(list(W), m - 1.96 * se, m + 1.96 * se, color=SERIES[0], alpha=0.14, linewidth=0)
    ax.plot(list(W), m, color=SERIES[0], linewidth=2, marker="o", markersize=4, label=f"Trung bình {len(C)} đợt lớn")
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1); ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.annotate(f"{m.iloc[-1]:+.2f}%", (10, m.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                color=INK2, fontsize=9)
    ax.set_xticks(list(W)); ax.set_xlabel("Phiên so với ngày đầu đợt chính sách")
    ax.set_ylabel("CAR ngành (%), = 0 tại phiên −1")
    ax.set_title("Phản ứng của ngành BĐS quanh các đợt tin chính sách lớn (khoảng tin cậy 95%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _save(fig, "r3_car_chinh_sach.png")

    d, model = cross_section()
    from statsmodels.stats.multitest import multipletests
    coefs = pd.DataFrame({"hệ số (điểm %)": model.params, "t": model.tvalues, "p": model.pvalues})
    # 20 hệ số -> hiệu chỉnh kiểm định bội (Benjamini–Hochberg) + kiểm định Wald chung cho các tương tác
    coefs["q (BH)"] = multipletests(coefs["p"], method="fdr_bh")[1]
    coefs = coefs.round(3)
    names = list(model.params.index)
    wald = model.wald_test(np.eye(len(names)), scalar=True)                    # mọi độ dốc = 0
    R = np.zeros((len(EXPO), len(names)))                                       # độ dốc bằng nhau giữa 2 giai đoạn
    for r_, c in enumerate(EXPO):
        R[r_, names.index(f"{c}×ban_hanh")], R[r_, names.index(f"{c}×hieu_luc")] = 1, -1
    wald_eq = model.wald_test(R, scalar=True)
    print(f"Wald — mọi độ dốc phơi nhiễm = 0: χ² = {float(wald.statistic):.2f}, p = {float(wald.pvalue):.3f} | "
          f"độ dốc giống nhau giữa 'ban hành' và 'có hiệu lực': χ² = {float(wald_eq.statistic):.2f}, p = {float(wald_eq.pvalue):.3f}")
    print(tab.round(3).to_string(index=False))
    print(f"\nHồi quy chéo: {len(d):,} quan sát (mã × đợt), {d['event_id'].nunique()} đợt, SE gom cụm theo đợt")
    print(coefs.to_string())
    md = ["# Mô-đun chính sách — event study", "",
          f"## (1) CAR ngành quanh {len(C)} đợt tin chính sách lớn", "",
          "Mô hình lợi suất trung bình (ước lượng [t−60, t−11]); benchmark đồng trọng số mã thanh khoản. "
          "Giới hạn: không có VN-Index nên không tách được cú sốc chung của thị trường.", "",
          tab.round(3).to_markdown(index=False), "", "![](../figures/r3_car_chinh_sach.png)", "",
          f"## (2) Hồi quy chéo CAR[0,+3] theo phơi nhiễm phân khúc ({len(d):,} quan sát, {d['event_id'].nunique()} đợt)", "",
          "Hiệu ứng cố định theo đợt; SE gom cụm theo đợt; chỉ mã thanh khoản. q = p hiệu chỉnh Benjamini–Hochberg.", "",
          "Giai đoạn 'đề xuất' bị loại vì chỉ có 4 đợt (không đủ cụm cho SE gom cụm).", "",
          f"Wald: mọi độ dốc = 0 -> χ²(10) = {float(wald.statistic):.2f}, p = {float(wald.pvalue):.3f}; "
          f"độ dốc giống nhau giữa hai giai đoạn -> χ²(5) = {float(wald_eq.statistic):.2f}, p = {float(wald_eq.pvalue):.3f}", "",
          coefs.to_markdown(), ""]
    (REP / "policy_event_study.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
