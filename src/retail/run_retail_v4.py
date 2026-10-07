"""Nhà đầu tư nhỏ lẻ — VÒNG 4. Cấu hình chính thức CHỐT TRƯỚC khi chạy (30/09/2026): E4.

E4 = trung bình PHÂN VỊ (so với phân phối trên val) của 5 mô hình khác loại, cùng đặc trưng v4, có trọng số thời gian:
  LR    Logistic Regression
  RF    Random Forest
  REG   LightGBM hồi quy Huber trên MỨC lãi (winsorize 1–99%)
  RANK  LightGBM lambdarank — xếp hạng các tin TRONG CÙNG PHIÊN (nhãn = ngũ phân vị mức lãi của train)
  CAT   CatBoost
Đặc trưng v4 = v3 + ngữ nghĩa bài báo (TB vector e5 của các bài trong phiên → PCA 16 chiều, PCA học lại trong
từng fold chỉ trên train) + câu chuyện (có tin mở đầu câu chuyện, số câu chuyện, bài thứ mấy của câu chuyện).
Trọng số thời gian: w = 0,5^(tuổi / 250 phiên) tính đến phiên cuối của train (ưu tiên dữ liệu gần).
Quy tắc CONS4: làm theo khi ≥ 4/5 mô hình cùng xếp tin vào top q (ngưỡng từ val).
Bóc tách (không phải cấu hình chính): E3 (vòng 3, đối chứng cùng mẫu), E4 bỏ ngữ nghĩa, E4 bỏ trọng số thời gian,
từng thành phần riêng. Không chọn gì theo test.
Đầu ra: outputs/reports/retail_v4.csv, retail_v4_quarters.csv; outputs/predictions/retail_v4_{target}.parquet
Dùng:  python -m src.retail.run_retail_v4
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMRanker, LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.config import CLEAN, ROOT
from src.eval.walk_forward import make_folds
from src.retail.events import FEATURES_V3, H
from src.retail.run_retail import SEED, TARGETS, block_boot, cls_metrics, logreg, prep

REP, PRED = ROOT / "outputs" / "reports", ROOT / "outputs" / "predictions"
QS = (0.1, 0.2, 0.3)
N_PCA, HALF_LIFE = 16, 250
COMPS = ["LR", "RF", "REG", "RANK", "CAT"]
STORY = ["has_new_story", "n_stories", "min_story_rank"]
EMB = [f"emb{i}" for i in range(N_PCA)]
CONTS = {"tuyet_doi": f"pnl_{H}", "tuong_doi": "rel_pnl"}


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=SEED)


def reg():
    return LGBMRegressor(objective="huber", alpha=0.02, num_leaves=15, learning_rate=0.03, n_estimators=300,
                         min_child_samples=80, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
                         random_state=SEED, verbose=-1)


def ranker():
    return LGBMRanker(objective="lambdarank", num_leaves=15, learning_rate=0.03, n_estimators=300, min_child_samples=50,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, random_state=SEED, verbose=-1)


def cat():
    return CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=5, random_seed=SEED, verbose=0,
                              thread_count=-1, allow_writing_files=False)


# ---------------------------------------------------------------- dữ liệu
def event_text(ev):
    """Vector ngữ nghĩa TB của các bài trong (phiên, mã) + đặc trưng câu chuyện."""
    E = np.load(CLEAN / "article_emb.npy").astype(np.float32)
    ids = pd.read_parquet(CLEAN / "article_emb_ids.parquet")["article_id"]
    pos = pd.Series(np.arange(len(ids)), index=ids)
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet", columns=["article_id", "ticker", "event_type"])
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session"])
    x = atl[atl["event_type"] != "KHONG_LIEN_QUAN"].merge(art, on="article_id").dropna(subset=["session"])
    x["date"] = pd.to_datetime(x["session"])
    x = x.merge(pd.read_parquet(CLEAN / "stories.parquet"), on=["article_id", "ticker"], how="left")
    x["row"] = x["article_id"].map(pos)
    x = x.dropna(subset=["row"])
    key = pd.MultiIndex.from_frame(ev[["date", "ticker"]])
    g = x.groupby(["date", "ticker"])
    M = np.zeros((len(ev), E.shape[1]), np.float32)
    rows_by = g["row"].apply(lambda r: r.astype(int).to_numpy())
    for i, k in enumerate(key):
        r = rows_by.get(k)
        if r is not None:
            v = E[r].mean(axis=0)
            M[i] = v / (np.linalg.norm(v) + 1e-9)
    st = g.agg(has_new_story=("story_rank", lambda s: float((s == 1).any())), n_stories=("story_id", "nunique"),
               min_story_rank=("story_rank", "min"))
    ev = ev.join(st, on=["date", "ticker"])
    return ev, M


# ---------------------------------------------------------------- mô hình
def weights(dates, end):
    cal = pd.DatetimeIndex(pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"])
    pos = pd.Series(np.arange(len(cal)), index=cal)
    age = pos[end] - pos.reindex(dates).to_numpy()
    return 0.5 ** (age / HALF_LIFE)


def fit_score(kind, tr, te, y, cont, feats, w):
    med = tr[feats].replace([np.inf, -np.inf], np.nan).median()
    Xtr, Xte = prep(tr, feats, med), prep(te, feats, med)
    yb = tr[y].astype(int).to_numpy()
    if kind == "LR":
        return logreg().fit(Xtr, yb, logisticregression__sample_weight=w).predict_proba(Xte)[:, 1]
    if kind == "RF":
        return rf().fit(Xtr, yb, sample_weight=w).predict_proba(Xte)[:, 1]
    if kind == "CAT":
        return cat().fit(Xtr, yb, sample_weight=w).predict_proba(Xte)[:, 1]
    lo, hi = tr[cont].quantile([0.01, 0.99])
    yc = tr[cont].clip(lo, hi).to_numpy()
    if kind == "REG":
        return reg().fit(Xtr, yc, sample_weight=w).predict(Xte)
    if kind == "RANK":
        order = np.argsort(tr["date"].to_numpy(), kind="stable")
        rel = pd.qcut(pd.Series(yc), 5, labels=False, duplicates="drop").to_numpy()
        grp = tr["date"].iloc[order].value_counts(sort=False).reindex(tr["date"].iloc[order].unique()).to_numpy()
        m = ranker().fit(Xtr.iloc[order], rel[order], group=grp, sample_weight=None)
        return m.predict(Xte)
    raise ValueError(kind)


def pct_vs(ref, x):
    r = np.sort(ref)
    return np.searchsorted(r, x, side="right") / len(r)


def add_pca(tr, others, M_tr, M_others):
    p = PCA(N_PCA, random_state=SEED).fit(M_tr)
    tr = tr.copy(); tr[EMB] = p.transform(M_tr)
    outs = []
    for o, M in zip(others, M_others):
        o = o.copy(); o[EMB] = p.transform(M); outs.append(o)
    return tr, outs


def run(ev, M, y, cont):
    keep = ev[y].notna().to_numpy()
    ev, M = ev[keep].reset_index(drop=True), M[keep]
    feats_full = FEATURES_V3 + STORY + EMB
    feats_notext = FEATURES_V3 + STORY
    out = []
    for f in make_folds(H + 1):
        m_tr, m_va = ev["date"].isin(f.train).to_numpy(), ev["date"].isin(f.val).to_numpy()
        m_fit, m_te = ev["date"].isin(f.fit).to_numpy(), ev["date"].isin(f.test).to_numpy()
        tr, (va,) = add_pca(ev[m_tr], [ev[m_va]], M[m_tr], [M[m_va]])
        fit, (te,) = add_pca(ev[m_fit], [ev[m_te]], M[m_fit], [M[m_te]])
        w_tr, w_fit = weights(tr["date"], tr["date"].max()), weights(fit["date"], fit["date"].max())
        one_tr, one_fit = np.ones(len(tr)), np.ones(len(fit))
        configs = {"": (feats_full, w_tr, w_fit), "notext_": (feats_notext, w_tr, w_fit),
                   "noweight_": (feats_full, one_tr, one_fit), "v3_": (FEATURES_V3, one_tr, one_fit)}
        for pre, (feats, wt, wf) in configs.items():
            comps = COMPS if pre != "v3_" else ["LR", "RF", "REG"]
            for k in comps:
                va[f"s|{pre}{k}"] = fit_score(k, tr, va, y, cont, feats, wt)
                te[f"s|{pre}{k}"] = fit_score(k, fit, te, y, cont, feats, wf)
            ens = f"s|{pre}E" if pre != "v3_" else "s|E3"
            for d in (va, te):
                d[ens] = np.mean([pct_vs(va[f"s|{pre}{k}"], d[f"s|{pre}{k}"]) for k in comps], axis=0)
        scored = [c for c in te if c.startswith("s|")]
        for c in scored:
            for q in QS:
                te[f"top{int(q * 100)}|{c[2:]}"] = te[c] >= np.quantile(va[c], 1 - q)
        for q in QS:
            t = int(q * 100)
            te[f"top{t}|CONS4"] = te[[f"top{t}|{k}" for k in COMPS]].sum(axis=1) >= 4
        te["fold"], te["quy_test"] = f.k, f.test_q
        out.append(te)
        print(f"   {f.test_q}: AUC E4 = {roc_auc_score(te[y], te['s|E']):.3f} | E3 = {roc_auc_score(te[y], te['s|E3']):.3f}")
    return pd.concat(out, ignore_index=True)


NAMES = {"E": "★ E4 (chính thức)", "CONS4": "CONS4 đồng thuận ≥4/5", "E3": "E3 (vòng 3, đối chứng)",
         "notext_E": "E4 bỏ ngữ nghĩa bài báo", "noweight_E": "E4 bỏ trọng số thời gian",
         "LR": "thành phần LR", "RF": "thành phần RF", "REG": "thành phần REG", "RANK": "thành phần RANK (xếp hạng)",
         "CAT": "thành phần CatBoost"}


def table(res, key, y, pnl):
    rows = []
    A = res[y].mean()
    for k, name in NAMES.items():
        auc = roc_auc_score(res[y], res[f"s|{k}"]) if f"s|{k}" in res else np.nan
        per_q = [roc_auc_score(g[y], g[f"s|{k}"]) for _, g in res.groupby("fold")] if f"s|{k}" in res else []
        for q in QS:
            col = f"top{int(q * 100)}|{k}"
            dec = res[col].astype(bool)
            if dec.sum() < 30:
                continue
            ci = block_boot(res["date"], res[y].where(dec), res[y])
            wins = sum(res.loc[dec & (res["fold"] == j), y].mean() > res.loc[res["fold"] == j, y].mean()
                       for j in res["fold"].unique())
            rows.append({"target": key, "mô hình": name, "quy tắc": f"top {q:.0%}", "AUC": auc,
                         "AUC quý thấp/cao": f"{min(per_q):.3f}/{max(per_q):.3f}" if per_q else "",
                         "khuyên làm theo": dec.mean(), **cls_metrics(res[y], dec), "lãi TB/lệnh": res.loc[dec, pnl].mean(),
                         "precision − A": res.loc[dec, y].mean() - A, "CI95 (precision − A)": f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                         "số quý thắng A": int(wins)})
    return rows


def main():
    ev = pd.read_parquet(CLEAN / "retail_events.parquet")
    ev, M = event_text(ev)
    PRED.mkdir(parents=True, exist_ok=True)
    rows, pq = [], []
    for key, (y, pnl) in TARGETS.items():
        print(f"== {key}")
        res = run(ev, M, y, CONTS[key])
        rows += table(res, key, y, pnl)
        keep = ["date", "ticker", "quy_test", "fold", "d", "title", "event_type", "entry_shift", f"ret_{H}", f"pnl_{H}",
                "rel_pnl", "follow_ok", "follow_rel", "mkt_ret_H", "d_r5", "frac_comm", "frac_firm"]
        res[keep + [c for c in res if c.startswith(("s|", "top"))]].to_parquet(PRED / f"retail_v4_{key}.parquet", index=False)
        for qt, g in res.groupby("quy_test"):
            pq.append({"target": key, "quý": qt, "tỷ lệ gốc": g[y].mean(), "AUC E4": roc_auc_score(g[y], g["s|E"]),
                       "AUC E3": roc_auc_score(g[y], g["s|E3"]), "precision E4 top 10%": g.loc[g["top10|E"], y].mean(),
                       "precision E3 top 10%": g.loc[g["top10|E3"], y].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(REP / "retail_v4.csv", index=False, encoding="utf-8-sig")
    (REP / "retail_v4.md").write_text(out.round(3).to_markdown(index=False), encoding="utf-8")
    pd.DataFrame(pq).to_csv(REP / "retail_v4_quarters.csv", index=False, encoding="utf-8-sig")
    cols = ["target", "mô hình", "quy tắc", "AUC", "AUC quý thấp/cao", "khuyên làm theo", "accuracy", "precision (làm theo)",
            "macro-F1", "lãi TB/lệnh", "CI95 (precision − A)", "số quý thắng A"]
    with pd.option_context("display.width", 260, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(out[cols].to_string(index=False)); print(pd.DataFrame(pq).to_string(index=False))


if __name__ == "__main__":
    main()
