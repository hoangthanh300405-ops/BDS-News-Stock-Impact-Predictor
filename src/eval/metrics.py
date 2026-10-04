"""Chỉ số đánh giá (§11.2) và kiểm định thống kê (§11.3). Hỗ trợ target 3 lớp và nhị phân."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (balanced_accuracy_score, brier_score_loss, f1_score, log_loss, matthews_corrcoef,
                             recall_score, roc_auc_score)

CLASSES = [-1, 0, 1]          # Giảm, Đứng, Tăng
BINARY = [0, 1]               # big_move: không / có
PROBA_COLS = ["p_down", "p_flat", "p_up"]


def proba_cols(classes=CLASSES):
    return PROBA_COLS if list(classes) == CLASSES else [f"p_{c}" for c in classes]


def _classes_of(df):
    return CLASSES if "p_down" in df else BINARY


def _P(df, classes):
    P = np.clip(df[proba_cols(classes)].to_numpy(), 1e-6, 1)
    return P / P.sum(axis=1, keepdims=True)


def summarize(df: pd.DataFrame, classes=None) -> dict:
    """df: cột y (thật), y_hat và các cột xác suất."""
    classes = classes or _classes_of(df)
    y, yh, P = df["y"].astype(int), df["y_hat"].astype(int), _P(df, classes)
    out = {"n": len(df), "bal_acc": balanced_accuracy_score(y, yh), "mcc": matthews_corrcoef(y, yh),
           "log_loss": log_loss(y, P, labels=list(classes))}
    if list(classes) == CLASSES:
        out.update(macro_f1=f1_score(y, yh, labels=CLASSES, average="macro", zero_division=0),
                   recall_giam=recall_score(y, yh, labels=[-1], average="macro", zero_division=0))
    else:
        out.update(auc=roc_auc_score(y, P[:, 1]) if y.nunique() > 1 else np.nan,
                   brier=brier_score_loss(y, P[:, 1]), base_rate=y.mean())
    return out


def row_logloss(df: pd.DataFrame, classes=None) -> pd.Series:
    classes = classes or _classes_of(df)
    P = _P(df, classes)
    idx = df["y"].astype(int).map({c: i for i, c in enumerate(classes)}).to_numpy()
    return pd.Series(-np.log(P[np.arange(len(df)), idx]), index=df.index)


def score(df: pd.DataFrame) -> pd.Series:
    """Điểm liên tục của dự đoán: E[y] = p_up − p_down (3 lớp) hoặc p(big_move) (nhị phân)."""
    return df["p_up"] - df["p_down"] if "p_down" in df else df["p_1"]


def rank_ic(df: pd.DataFrame, outcome: str, h: int = 1) -> dict:
    """Model A: trung bình theo phiên của tương quan hạng (điểm dự đoán, kết quả liên tục).
    Model B (1 dòng/phiên): tương quan hạng theo thời gian. t-stat chia √h để tính chồng lấn."""
    s = score(df)
    if "ticker" in df and df.groupby("date").size().max() > 1:
        g = pd.DataFrame({"date": df["date"], "s": s, "o": df[outcome]}).dropna()
        ics = g.groupby("date").apply(lambda x: stats.spearmanr(x["s"], x["o"])[0] if len(x) >= 5 else np.nan,
                                      include_groups=False).dropna()
        t = ics.mean() / (ics.std() / np.sqrt(len(ics))) / np.sqrt(h)
        return {"IC": ics.mean(), "IC_t": t, "IC_n": len(ics)}
    ok = df[outcome].notna()
    r, p = stats.spearmanr(s[ok], df.loc[ok, outcome])
    return {"IC": r, "IC_t": r * np.sqrt((ok.sum() / h - 2) / (1 - r ** 2)), "IC_n": int(ok.sum())}


def dm_test(loss_a: pd.Series, loss_b: pd.Series, h: int) -> tuple[float, float]:
    """Diebold–Mariano + hiệu chỉnh mẫu nhỏ HLN (1997); phương sai HAC Bartlett, độ trễ h−1.
    loss_*: chuỗi tổn thất THEO PHIÊN (đã lấy trung bình trong phiên). Âm = mô hình a tốt hơn b."""
    d = (loss_a - loss_b).dropna().to_numpy()
    T = len(d)
    dc = d - d.mean()
    var = dc @ dc / T
    for lag in range(1, h):
        var += 2 * (1 - lag / h) * (dc[lag:] @ dc[:-lag]) / T
    dm = d.mean() / np.sqrt(var / T)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * stats.t.sf(abs(hln), df=T - 1)
    return hln, p


def block_bootstrap_diff(pred_a: pd.DataFrame, pred_b: pd.DataFrame, metric="bal_acc",
                         block=20, n_boot=1000, seed=0) -> tuple[float, float, float]:
    """Chênh lệch metric(a) − metric(b), khoảng tin cậy 95% bằng bootstrap khối theo phiên."""
    rng = np.random.default_rng(seed)
    dates = np.array(sorted(pred_a["date"].unique()))
    ga, gb = dict(tuple(pred_a.groupby("date"))), dict(tuple(pred_b.groupby("date")))
    base = summarize(pred_a)[metric] - summarize(pred_b)[metric]
    n, diffs = len(dates), []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        pick = np.concatenate([dates[s:s + block] for s in starts])[:n]
        a = pd.concat([ga[d] for d in pick]); b = pd.concat([gb[d] for d in pick])
        diffs.append(summarize(a)[metric] - summarize(b)[metric])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return base, lo, hi
