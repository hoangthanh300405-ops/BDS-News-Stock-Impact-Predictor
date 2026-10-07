"""Nhà đầu tư nhỏ lẻ — SO SÁNH THUẬT TOÁN trên cùng bài toán, cùng đặc trưng (v1), cùng walk-forward và ngưỡng từ val.

Siêu tham số đặt trước theo mức mặc định có điều chuẩn (không tinh chỉnh theo test). Đặc trưng: FEATURES (tin + giá).
Đầu ra: outputs/reports/retail_models_compare.csv (+ .md)
Dùng:  python -m src.retail.run_models_compare
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import src.retail.run_retail as R
from src.config import CLEAN, ROOT
from src.retail.events import FEATURES

REP = ROOT / "outputs" / "reports"
S = R.SEED


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=S)


def et():
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=S)


def hgb():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=80,
                                          l2_regularization=1.0, random_state=S)


def svm():
    return make_pipeline(StandardScaler(), SVC(C=0.5, kernel="rbf", probability=True, random_state=S))


def knn():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=150))


def nb():
    return make_pipeline(StandardScaler(), GaussianNB())


def mlp():
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, early_stopping=True,
                                                         max_iter=500, random_state=S))


def vote():
    return VotingClassifier([("lr", R.logreg()), ("lgbm", R.lgbm()), ("rf", rf()), ("hgb", hgb()), ("mlp", mlp())],
                            voting="soft")


MODELS = {
    "Mốc: luôn làm theo tin": None,
    "Logistic Regression": (R.logreg, FEATURES),
    "LightGBM": (R.lgbm, FEATURES),
    "Random Forest": (rf, FEATURES),
    "Extra Trees": (et, FEATURES),
    "HistGradientBoosting": (hgb, FEATURES),
    "SVM (RBF)": (svm, FEATURES),
    "k-NN (k=150)": (knn, FEATURES),
    "Naive Bayes": (nb, FEATURES),
    "MLP (mạng nơ-ron 32-16)": (mlp, FEATURES),
    "Kết hợp mềm (LR+LGBM+RF+HGB+MLP)": (vote, FEATURES),
}


def main():
    R.MODELS = MODELS
    R.MAIN = "LightGBM"
    ev = pd.read_parquet(CLEAN / "retail_events.parquet")
    rows = []
    for key, (y, pnl) in R.TARGETS.items():
        res = R.run(ev, y)
        acc = R.accuracy_f1(res, key, y)
        _, inv = R.evaluate(res, key, y, pnl)
        for name, spec in MODELS.items():
            if spec is None:
                continue
            p = res[f"p|{name}"]
            auc = R.roc_auc_score(res[y], p)
            per_q = [R.roc_auc_score(g[y], g[f"p|{name}"]) for _, g in res.groupby("fold")]
            for rule in ("p > tỷ lệ gốc train", "top 10% (ngưỡng từ val)", "top 30% (ngưỡng từ val)"):
                a = acc[(acc["mô hình"] == name) & (acc["quy tắc quyết định"] == rule)].iloc[0]
                tag = {"top 10% (ngưỡng từ val)": "chọn lọc top 10%", "top 30% (ngưỡng từ val)": "chọn lọc top 30%"}.get(rule)
                iv = inv[inv["nhà đầu tư"] == f"C · {name} · {tag}"] if tag else pd.DataFrame()
                rows.append({"target": key, "mô hình": name, "quy tắc": rule.replace(" (ngưỡng từ val)", ""),
                             "AUC": auc, "AUC thấp nhất / cao nhất theo quý": f"{min(per_q):.3f} / {max(per_q):.3f}",
                             "khuyên làm theo": a["tỷ lệ khuyên làm theo"], "accuracy": a["accuracy"],
                             "balanced acc": a["balanced acc"], "precision": a["precision (làm theo)"],
                             "recall": a["recall (làm theo)"], "F1 làm theo": a["F1 (làm theo)"], "macro-F1": a["macro-F1"],
                             "lãi TB/lệnh": iv["lãi TB/lệnh"].iloc[0] if len(iv) else np.nan,
                             "CI95 precision − A": iv["CI95 chênh (khối phiên)"].iloc[0] if len(iv) else "",
                             "số quý thắng A": iv["số quý thắng A (/6)"].iloc[0] if len(iv) else np.nan})
        base = acc[acc["mô hình"].str.startswith("Mốc")]
        for _, b in base.iterrows():
            rows.append({"target": key, "mô hình": b["mô hình"], "quy tắc": "—", "khuyên làm theo": b["tỷ lệ khuyên làm theo"],
                         "accuracy": b["accuracy"], "balanced acc": b["balanced acc"], "precision": b["precision (làm theo)"],
                         "recall": b["recall (làm theo)"], "F1 làm theo": b["F1 (làm theo)"], "macro-F1": b["macro-F1"]})
        print(f"xong target {key}")
    out = pd.DataFrame(rows)
    out.to_csv(REP / "retail_models_compare.csv", index=False, encoding="utf-8-sig")
    (REP / "retail_models_compare.md").write_text(out.round(3).to_markdown(index=False), encoding="utf-8")
    with pd.option_context("display.width", 300, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
