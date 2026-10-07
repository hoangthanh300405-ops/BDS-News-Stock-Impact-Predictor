"""A3 — Mô hình E3 (nhà đầu tư nhỏ lẻ) dựa vào NHÓM THÔNG TIN nào để chọn tin?

Permutation importance theo NHÓM đặc trưng: trong từng fold, fit LR3/RF3/REG3 trên train+val, chấm AUC của E3
(trung bình hạng của 3 mô hình) trên test; xáo trộn đồng thời mọi cột của một nhóm (3 lần) và đo AUC giảm bao nhiêu.
Đây là phân tích GIẢI THÍCH trên mô hình đã chốt (không dùng để chọn gì).
Đầu ra: outputs/reports/retail_A3_importance.csv; outputs/figures/r10_importance_groups.png
Dùng:  python -m src.analysis.retail_importance
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.config import CLEAN, ROOT
from src.eval.walk_forward import make_folds
from src.retail.events import FEATURES_V3, H
from src.retail.run_retail import TARGETS, logreg, prep
from src.retail.run_retail_v3 import reg, rf

REP, FIG = ROOT / "outputs" / "reports", ROOT / "outputs" / "figures"
GROUPS = {
    "Nội dung tin (nhãn LLM: hướng, mức độ, nhóm tin)": ["d", "score", "score_absmax", "n_art", "frac_lon", "frac_price_report",
                                                         "g_tai_chinh_dn", "g_du_an_ha_tang", "g_phap_ly", "g_chinh_sach",
                                                         "g_lai_suat_tin_dung", "g_thi_truong"],
    "Loại bài (NLP: sự kiện DN / bình luận / quảng bá…)": ["frac_comm", "frac_firm", "frac_policy", "frac_promo",
                                                          "n_tick_in_art", "frac_before_open", "frac_cafef"],
    "Độ mới & uy tín của tin": ["news_prev5", "news_prev20", "since_last_news", "cred_tick_ok", "cred_type_ok",
                                "cred_tick_rel", "cred_type_rel"],
    "Giá đã chạy trước khi đọc": ["d_r1", "d_r5", "d_r20", "d_ar5", "d_dist_high20", "d_dist_low20", "d_cs_rank_r5",
                                  "limit_in_dir", "limit_against"],
    "Bối cảnh thị trường & nhiệt độ tin": ["d_mkt1", "d_mkt5", "d_mkt20", "mkt_vol20", "d_breadth5", "recent_ok",
                                           "recent_rel", "day_n_art", "day_n_tick", "d_day_pos", "share_of_day"],
    "Rủi ro & thanh khoản": ["vol20", "value_z", "illiquid", "log_value", "cs_rank_value_z"],
    "Hồ sơ doanh nghiệp": ["hose", "exposure_can_ho", "exposure_dat_nen", "exposure_thap_tang", "exposure_kcn",
                           "exposure_nghi_duong"],
}
REPS = 3


def e3_score(models, X):
    return np.mean([rankdata(s) / len(s) for s in
                    (models["LR3"].predict_proba(X)[:, 1], models["RF3"].predict_proba(X)[:, 1], models["REG3"].predict(X))],
                   axis=0)


def main():
    assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURES_V3), "mỗi đặc trưng phải thuộc đúng một nhóm"
    ev = pd.read_parquet(CLEAN / "retail_events.parquet")
    conts = {"tuyet_doi": f"pnl_{H}", "tuong_doi": "rel_pnl"}
    rng = np.random.default_rng(0)
    rows = []
    for key, (y, _) in TARGETS.items():
        d = ev[ev[y].notna()]
        for f in make_folds(H + 1):
            fit, te = d[d["date"].isin(f.fit)], d[d["date"].isin(f.test)]
            med = fit[FEATURES_V3].replace([np.inf, -np.inf], np.nan).median()
            Xf, Xt = prep(fit, FEATURES_V3, med), prep(te, FEATURES_V3, med)
            lo, hi = fit[conts[key]].quantile([0.01, 0.99])
            models = {"LR3": logreg().fit(Xf, fit[y].astype(int)), "RF3": rf().fit(Xf, fit[y].astype(int)),
                      "REG3": reg().fit(Xf, fit[conts[key]].clip(lo, hi))}
            base = roc_auc_score(te[y], e3_score(models, Xt))
            for g, cols in GROUPS.items():
                drops = []
                for _ in range(REPS):
                    Xp = Xt.copy()
                    perm = rng.permutation(len(Xp))
                    Xp[cols] = Xp[cols].to_numpy()[perm]
                    drops.append(base - roc_auc_score(te[y], e3_score(models, Xp)))
                rows.append({"target": key, "quý": f.test_q, "nhóm": g, "AUC gốc": base, "AUC giảm": np.mean(drops)})
            print(f"  {key} {f.test_q}: AUC E3 = {base:.3f}")
    out = pd.DataFrame(rows)
    summ = out.groupby(["target", "nhóm"]).agg(auc_giam_tb=("AUC giảm", "mean"), auc_giam_sd=("AUC giảm", "std"),
                                               so_quy_duong=("AUC giảm", lambda s: int((s > 0).sum()))).reset_index()
    summ["phần của tín hiệu"] = summ["auc_giam_tb"].clip(lower=0) / summ.groupby("target")["auc_giam_tb"].transform(
        lambda s: s.clip(lower=0).sum())
    summ = summ.sort_values(["target", "auc_giam_tb"], ascending=[True, False])
    summ.to_csv(REP / "retail_A3_importance.csv", index=False, encoding="utf-8-sig")
    out.to_csv(REP / "retail_A3_importance_by_quarter.csv", index=False, encoding="utf-8-sig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    order = summ[summ["target"] == "tuyet_doi"].sort_values("auc_giam_tb")["nhóm"]
    for ax, (key, title) in zip(axes, (("tuyet_doi", "Target: làm theo có lãi sau phí"),
                                       ("tuong_doi", "Target: làm theo thắng rổ BĐS"))):
        s = summ[summ["target"] == key].set_index("nhóm").reindex(order)
        ax.barh(s.index, s["auc_giam_tb"], xerr=s["auc_giam_sd"], color="#1f6feb", alpha=.85)
        ax.axvline(0, color="k", lw=.6); ax.set_title(title); ax.set_xlabel("AUC giảm khi xáo trộn nhóm (TB 6 quý)")
        for i, (v, n) in enumerate(zip(s["auc_giam_tb"], s["so_quy_duong"])):
            ax.text(max(v, 0) + 0.001, i, f"{n}/6 quý", va="center", fontsize=8)
    fig.suptitle("A3. Mô hình E3 dựa vào nhóm thông tin nào để chọn tin đáng làm theo?")
    fig.tight_layout(); fig.savefig(FIG / "r10_importance_groups.png", dpi=150); plt.close(fig)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
