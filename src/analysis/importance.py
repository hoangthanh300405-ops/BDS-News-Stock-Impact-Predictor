"""Mô hình dựa vào đặc trưng nào? — permutation importance trên VALIDATION của từng fold (Model A, h = 1, bộ S2).

Mỗi fold: huấn luyện LightGBM trên train, đo AUC (điểm p_up − p_down so với chiều giá thực) trên val; xáo trộn từng
đặc trưng (và từng NHÓM đặc trưng) rồi đo AUC giảm bao nhiêu. Giảm càng nhiều = mô hình càng dựa vào đặc trưng đó.
Dùng:  python -m src.analysis.importance
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import CLEAN, ROOT
from src.eval.p0_report import INK2, SERIES, _save
from src.eval.walk_forward import make_folds
from src.features.news_features import A_NEWS
from src.features.price_features import A_EXTRA, A_PRICE
from src.models.engine import LGBM
from src.models.run_direction_v3 import data

GROUPS = {"Giá (F5)": A_PRICE, "Tin tức (F1, F6, F7)": A_NEWS, "Đặc trưng mới (trần/sàn, thứ hạng, …)": A_EXTRA}
FEATS = A_PRICE + A_NEWS + A_EXTRA
N_REPEAT = 3


def auc(m, X, up):
    P = m.predict_proba(X)
    return roc_auc_score(up, P[:, 2] - P[:, 0])


def main(h=1):
    d = data().merge(pd.read_parquet(CLEAN / "target_A.parquet", columns=["date", "ticker", f"y_{h}"]), on=["date", "ticker"])
    d = d.dropna(subset=FEATS + [f"y_{h}"])
    rng = np.random.default_rng(0)
    rows = []
    for f in make_folds(h):
        tr, va = d[d["date"].isin(f.train)], d[d["date"].isin(f.val)]
        va = va[va[f"r_{h}"] != 0]
        m = LGBM(); m.classes = [-1, 0, 1]
        m.fit(tr[FEATS].to_numpy(), tr[f"y_{h}"].to_numpy(), va[FEATS].to_numpy(), va[f"y_{h}"].to_numpy())
        X, up = va[FEATS].to_numpy().copy(), (va[f"r_{h}"] > 0).astype(int).to_numpy()
        base = auc(m, X, up)
        units = {c: [c] for c in FEATS} | {f"[NHÓM] {g}": cols for g, cols in GROUPS.items()}
        for name, cols in units.items():
            idx = [FEATS.index(c) for c in cols]
            drops = []
            for _ in range(N_REPEAT):
                Xp = X.copy()
                perm = rng.permutation(len(Xp))
                Xp[:, idx] = X[perm][:, idx]                   # xáo trộn cả nhóm cùng lúc (giữ tương quan trong nhóm)
                drops.append(base - auc(m, Xp, up))
            rows.append({"fold": f.k, "đặc trưng": name, "AUC giảm": np.mean(drops), "AUC gốc": base})
    r = pd.DataFrame(rows)
    s = r.groupby("đặc trưng")["AUC giảm"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    s["se"] = s["std"] / np.sqrt(r["fold"].nunique())
    s.to_csv(ROOT / "outputs" / "reports" / "importance.csv", encoding="utf-8-sig")
    grp = s[s.index.str.startswith("[NHÓM]")]
    ind = s[~s.index.str.startswith("[NHÓM]")].head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    col = [SERIES[1] if c in A_NEWS else SERIES[2] if c in A_EXTRA else SERIES[0] for c in ind.index]
    ax.barh(ind.index, 1000 * ind["mean"], xerr=1000 * 1.96 * ind["se"], color=col, height=0.62,
            error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2})
    ax.axvline(0, color=INK2, linewidth=0.8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC giảm khi xáo trộn đặc trưng (×1000; trung bình 6 fold, thanh = KTC 95%)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label="Giá"), Patch(color=SERIES[1], label="Tin tức"),
                       Patch(color=SERIES[2], label="Đặc trưng mới")], loc="lower right")
    ax.set_title("Mô hình dựa vào đặc trưng nào? (Model A, h = 1, permutation importance trên validation)")
    _save(fig, "r4_importance.png")
    pd.set_option("display.width", 200)
    print("THEO NHÓM (AUC giảm, ×1000):"); print((grp[["mean", "se"]] * 1000).round(2))
    print("\nTOP 15 ĐẶC TRƯNG (AUC giảm, ×1000):"); print((s[~s.index.str.startswith('[NHÓM]')].head(15)[["mean", "se"]] * 1000).round(2))
    print(f"\nAUC gốc trên val (trung bình fold): {r.groupby('fold')['AUC gốc'].first().mean():.3f}")


if __name__ == "__main__":
    main()
