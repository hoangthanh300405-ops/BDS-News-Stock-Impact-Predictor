"""Chạy một mô hình qua 6 fold walk-forward: chọn siêu tham số trên val, fit lại trên train+val, dự đoán test.

Hỗ trợ target 3 lớp (y ∈ {-1, 0, 1}) và nhị phân (big_move ∈ {0, 1}).
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import CLASSES, proba_cols
from src.eval.walk_forward import make_folds

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

# Xác suất được huấn luyện KHÔNG trọng số lớp (giữ hiệu chỉnh tốt -> log-loss, DM có nghĩa).
# Quyết định lớp = argmax(P / tỷ lệ lớp trong train): quy tắc Bayes tối ưu cho balanced accuracy,
# thay cho class_weight='balanced' (làm méo xác suất: log-loss B2 tệ hơn cả B0 ở bản đầu).


class _Base:
    classes = CLASSES               # run() gán lại theo target

    def _align(self, P, seen):
        out = np.zeros((len(P), len(self.classes)))
        for j, c in enumerate(seen):
            out[:, list(self.classes).index(int(c))] = P[:, j]
        return out


class Majority(_Base):
    """B0 — luôn dự đoán phân bố lớp của train."""
    grid = [{}]

    def __init__(self, **_):
        pass

    def fit(self, X, y):
        self.p = np.array([(y == c).mean() for c in self.classes]); return self

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))


class Momentum(_Base):
    """B1 — dự đoán theo z của lợi suất 5 phiên vừa qua (cột đầu tiên của X); xác suất = phân bố y
    có điều kiện trên lớp dự đoán, ước lượng từ train (làm trơn Laplace). Chỉ dùng cho target 3 lớp."""
    grid = [{}]

    def __init__(self, **_):
        pass

    @staticmethod
    def _cls(x):
        return np.select([x < -0.5, x > 0.5], [-1, 1], 0)

    def fit(self, X, y):
        c = self._cls(X[:, 0])
        self.table = {k: np.array([((c == k) & (y == v)).sum() + 1 for v in self.classes], float) for k in CLASSES}
        for k in self.table:
            self.table[k] /= self.table[k].sum()
        return self

    def predict_proba(self, X):
        return np.vstack([self.table[k] for k in self._cls(X[:, 0])])


class LogReg(_Base):
    grid = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]

    def __init__(self, C=1.0):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

    def fit(self, X, y):
        self.m.fit(X, y); return self

    def predict_proba(self, X):
        return self._align(self.m.predict_proba(X), self.m.classes_)

    def coef(self):
        return self.m[-1].coef_


class LGBM(_Base):
    grid = [{"num_leaves": nl, "learning_rate": 0.03} for nl in (7, 15)]

    def __init__(self, num_leaves=7, learning_rate=0.03, n_estimators=300, min_child_samples=20, random_state=0):
        self.params = dict(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators,
                           min_child_samples=min_child_samples, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=1.0, verbose=-1, n_jobs=8, random_state=random_state)
        self.best_iter = n_estimators

    def fit(self, X, y, X_val=None, y_val=None):
        m = lgb.LGBMClassifier(**self.params)
        if X_val is not None:
            m.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
            self.best_iter = m.best_iteration_ or self.params["n_estimators"]
        else:
            m.fit(X, y)
        self.m = m; return self

    def refit(self, X, y):
        p = dict(self.params, n_estimators=max(self.best_iter, 20))
        self.m = lgb.LGBMClassifier(**p).fit(X, y); return self

    def predict_proba(self, X):
        return self._align(self.m.predict_proba(X), self.m.classes_)


class LGBMWide(LGBM):
    """LightGBM với lưới rộng hơn (v3): số lá × số mẫu tối thiểu mỗi lá; n_estimators chọn bằng early stopping."""
    grid = [{"num_leaves": nl, "learning_rate": 0.03, "n_estimators": 600, "min_child_samples": mc}
            for nl in (7, 15, 31) for mc in (20, 100)]


class LGBMBag(LGBM):
    """Trung bình xác suất của N LightGBM khác seed (bagging) — giảm phương sai trên dữ liệu nhiễu."""
    SEEDS = (0, 1, 2, 3, 4)

    def fit(self, X, y, X_val=None, y_val=None):
        self.members = [LGBM(**{k: v for k, v in self.params.items()
                                if k in ("num_leaves", "learning_rate", "n_estimators", "min_child_samples")},
                             random_state=s) for s in self.SEEDS]
        for m in self.members:
            m.classes = self.classes
            m.fit(X, y, X_val, y_val)
        return self

    def refit(self, X, y):
        for m in self.members:
            m.refit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)


# ---------------- vòng walk-forward ----------------
def run(data: pd.DataFrame, features: list[str], target: str, h: int, model_cls, name: str,
        classes=CLASSES, fold_data=None, splitter=make_folds, return_val=False) -> pd.DataFrame:
    """data: có cột date (+ ticker nếu Model A), các cột features và target.
    fold_data: tùy chọn — hàm fold -> DataFrame, khi đặc trưng phụ thuộc fold (vd. hiệu chỉnh Đ2 tính trên train).
    splitter: make_folds (walk-forward 6 fold, mặc định) hoặc make_split_70_10_20.
    Dòng thiếu đặc trưng/target bị bỏ. Trả về dự đoán test gộp mọi fold."""
    cols = proba_cols(classes)
    preds = []
    for f in splitter(h):
        d = (fold_data(f) if fold_data else data).dropna(subset=features + [target])
        tr, va = d[d["date"].isin(f.train)], d[d["date"].isin(f.val)]
        fit, te = d[d["date"].isin(f.fit)], d[d["date"].isin(f.test)]
        Xtr, ytr, Xva, yva = tr[features].to_numpy(), tr[target].to_numpy(), va[features].to_numpy(), va[target].to_numpy()
        best, best_loss = None, np.inf
        for params in model_cls.grid:                       # chọn siêu tham số trên val
            m = model_cls(**params)
            m.classes = classes
            m.fit(Xtr, ytr, Xva, yva) if isinstance(m, LGBM) else m.fit(Xtr, ytr)
            loss = log_loss(yva, np.clip(m.predict_proba(Xva), 1e-6, 1), labels=list(classes))
            if loss < best_loss:
                best, best_loss = m, loss
        if return_val:                                      # dự đoán val của mô hình CHỈ học trên train
            Pv = best.predict_proba(Xva)
            ov = va[[c for c in ("date", "ticker") if c in va]].copy()
            ov["y"], ov[cols], ov["fold"], ov["model"], ov["part"] = yva, Pv, f.k, name, "val"
            preds.append(ov)
        # fit lại trên train + val (đã purge) với siêu tham số tốt nhất
        if isinstance(best, LGBM):
            best.refit(fit[features].to_numpy(), fit[target].to_numpy())
        else:
            best.fit(fit[features].to_numpy(), fit[target].to_numpy())
        P = best.predict_proba(te[features].to_numpy())
        prior = np.array([(fit[target] == c).mean() for c in classes])
        out = te[[c for c in ("date", "ticker") if c in te]].copy()
        out["y"] = te[target].to_numpy()
        out[cols] = P
        out["y_hat"] = np.array(classes)[(P / np.maximum(prior, 1e-6)).argmax(axis=1)]
        out["fold"], out["model"] = f.k, name
        if return_val:
            out["part"] = "test"
        preds.append(out)
    return pd.concat(preds, ignore_index=True)
