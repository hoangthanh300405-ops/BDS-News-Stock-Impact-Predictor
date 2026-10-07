"""Nhà đầu tư nhỏ lẻ — VÒNG CẢI TIẾN (v2). Mọi lựa chọn làm trên VAL, không nhìn test.

Ứng viên (cho mỗi target):
  V1  LightGBM, đặc trưng v1                          (mô hình chính của vòng 1 — mốc so sánh)
  L2  LightGBM, đặc trưng v2 (+ bối cảnh thị trường, thứ hạng trong ngày, uy tín của tin)
  R2  Logistic, đặc trưng v2
  E2  Trung bình L2 + R2
  T2  E2 + đặc trưng NỘI DUNG BÀI (tf-idf tiêu đề -> Logistic, cross-fitting theo thời gian trong train)
  D2  LightGBM v2, train bỏ các tin làm theo lãi/lỗ quá nhỏ (|lãi| < 1%) — giảm nhiễu nhãn; test vẫn đủ mọi tin
  N2  CHỈ TIN: LightGBM trên đặc trưng bài báo + uy tín + nội dung bài (không dùng giá)
Chọn theo val tích lũy: ở quý test k, cấu hình được chọn là cấu hình có AUC cao nhất trên val của các fold 1..k
(mọi val này đều nằm TRƯỚC quý test k); quy tắc quyết định (top q% hoặc p > tỷ lệ gốc) chọn theo macro-F1 trên cùng val.
Đầu ra: outputs/reports/retail_v2_candidates.csv, retail_v2_selected.csv, retail_v2_choices.csv; retail_output_v2.csv
Dùng:  python -m src.retail.run_retail_v2
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import CLEAN, ROOT
from src.eval.walk_forward import make_folds
from src.retail.events import FEATURES, FEATURES_V2, H, NEWS_ONLY_V2
from src.retail.run_retail import SEED, TARGETS, block_boot, cls_metrics, logreg, prep, reasons

REP = ROOT / "outputs" / "reports"
QS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
NOISE = 0.01
CANDS = ["V1", "L2", "R2", "E2", "T2", "D2", "N2"]
NAMES = {"V1": "V1 LightGBM v1 (vòng 1)", "L2": "L2 LightGBM v2", "R2": "R2 Logistic v2", "E2": "E2 Kết hợp L2+R2",
         "T2": "T2 Kết hợp + nội dung bài", "D2": "D2 LightGBM v2, train bỏ nhiễu", "N2": "N2 Chỉ tin + nội dung bài"}


def lgbm():
    return LGBMClassifier(num_leaves=15, learning_rate=0.03, n_estimators=300, min_child_samples=80,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
                          random_state=SEED, verbose=-1)


def text_model():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True), \
        LogisticRegression(C=0.5, max_iter=2000)


def text_prob(tr, others, y, n_blocks=5):
    """P(y | tiêu đề): cross-fitting theo khối thời gian cho tr (không tự chấm chính mình), fit toàn tr cho others."""
    txt = lambda d: (d["titles"].fillna("") + " __d" + d["d"].astype(str)).str.lower()
    oof = np.full(len(tr), np.nan)
    blocks = np.array_split(np.argsort(tr["date"].to_numpy(), kind="stable"), n_blocks)
    for b in blocks:
        mask = np.ones(len(tr), bool); mask[b] = False
        vec, m = text_model()
        m.fit(vec.fit_transform(txt(tr[mask])), tr[y].to_numpy()[mask].astype(int))
        oof[b] = m.predict_proba(vec.transform(txt(tr.iloc[b])))[:, 1]
    vec, m = text_model()
    m.fit(vec.fit_transform(txt(tr)), tr[y].astype(int))
    return oof - 0.5, [m.predict_proba(vec.transform(txt(o)))[:, 1] - 0.5 for o in others]


def fit_pred(kind, tr, te, y, pnl):
    """Trả P(y=1) cho te; kèm (model, med, feats) nếu là LightGBM (để giải thích)."""
    if kind == "V1":
        feats, make, trn = FEATURES, lgbm, tr
    elif kind == "L2":
        feats, make, trn = FEATURES_V2, lgbm, tr
    elif kind == "R2":
        feats, make, trn = FEATURES_V2, logreg, tr
    elif kind == "D2":
        feats, make, trn = FEATURES_V2, lgbm, tr[(tr[pnl].abs() >= NOISE)]
    elif kind == "N2":
        feats, make, trn = NEWS_ONLY_V2 + ["text_p"], lgbm, tr
    med = trn[feats].replace([np.inf, -np.inf], np.nan).median()
    m = make().fit(prep(trn, feats, med), trn[y].astype(int))
    return m.predict_proba(prep(te, feats, med))[:, 1], (m, med, feats)


def predict_all(tr, te, y, pnl):
    """Mọi ứng viên: fit trên tr, dự đoán te. Đặc trưng nội dung bài được cross-fit trong tr."""
    tr, te = tr.copy(), te.copy()
    tr["text_p"], (te["text_p"],) = text_prob(tr, [te], y)
    P, keep = {}, {}
    for k in ("V1", "L2", "R2", "D2", "N2"):
        P[k], keep[k] = fit_pred(k, tr, te, y, pnl)
    P["E2"] = (P["L2"] + P["R2"]) / 2
    # T2: logistic xếp chồng trên (điểm E2, điểm nội dung) — hệ số học trên tr bằng dự đoán cross-fit
    oofE = np.full(len(tr), np.nan)
    for b in np.array_split(np.argsort(tr["date"].to_numpy(), kind="stable"), 5):
        mask = np.ones(len(tr), bool); mask[b] = False
        oofE[b] = (fit_pred("L2", tr[mask], tr.iloc[b], y, pnl)[0] + fit_pred("R2", tr[mask], tr.iloc[b], y, pnl)[0]) / 2
    Z = lambda e, t: np.column_stack([np.log(e / (1 - e)), t])
    st = LogisticRegression(C=1.0).fit(Z(np.clip(oofE, 1e-4, 1 - 1e-4), tr["text_p"]), tr[y].astype(int))
    P["T2"] = st.predict_proba(Z(np.clip(P["E2"], 1e-4, 1 - 1e-4), te["text_p"]))[:, 1]
    return P, keep


def decide(p, rule, thr_src, base):
    if rule == "p>base":
        return p > base
    return p >= np.quantile(thr_src, 1 - rule)


def run_target(ev, key, y, pnl):
    ev = ev[ev[y].notna()].copy()
    folds = make_folds(H + 1)
    val_store, out, choices = [], [], []
    for f in folds:
        tr, va = ev[ev["date"].isin(f.train)], ev[ev["date"].isin(f.val)].copy()
        fit, te = ev[ev["date"].isin(f.fit)], ev[ev["date"].isin(f.test)].copy()
        Pv, _ = predict_all(tr, va, y, pnl)
        Pt, keep = predict_all(fit, te, y, pnl)
        base = fit[y].mean()
        for k in CANDS:
            va[f"p|{k}"], te[f"p|{k}"] = Pv[k], Pt[k]
        va["base"], te["base"], va["fold_val"] = tr[y].mean(), base, f.k
        val_store.append(va)
        V = pd.concat(val_store)                          # val của các fold 1..k — đều trước quý test k
        auc = {k: roc_auc_score(V[y], V[f"p|{k}"]) for k in CANDS}
        best = max(auc, key=auc.get)
        # quy tắc quyết định: chọn theo macro-F1 trên val tích lũy (ngưỡng quantile tính trong từng val)
        scores = {}
        for rule in list(QS) + ["p>base"]:
            dec = np.concatenate([decide(g[f"p|{best}"].to_numpy(), rule, g[f"p|{best}"].to_numpy(), g["base"].iloc[0])
                                  for _, g in V.groupby("fold_val")])
            yy = np.concatenate([g[y].to_numpy() for _, g in V.groupby("fold_val")])
            scores[rule] = cls_metrics(yy, dec)["macro-F1"]
        rule = max(scores, key=scores.get)
        te["chosen"] = best
        te["rule"] = str(rule)
        te["follow"] = decide(te[f"p|{best}"].to_numpy(), rule, va[f"p|{best}"].to_numpy(), base)
        for k in CANDS:
            for q in (0.1, 0.3):
                te[f"top{int(q * 100)}|{k}"] = te[f"p|{k}"] >= np.quantile(va[f"p|{k}"], 1 - q)
        if key == "tuyet_doi":
            m, med, feats = keep["L2"]
            X = prep(te, feats, med)
            contrib = m.booster_.predict(X, pred_contrib=True)[:, :-1]
            te["_reasons"] = [reasons(r, c, feats) for r, c in zip(X.to_numpy(), contrib)]
        te["fold"], te["quy_test"] = f.k, f.test_q
        choices.append({"target": key, "quý test": f.test_q, "cấu hình chọn": NAMES[best], "quy tắc": str(rule),
                        **{f"AUC val {k}": round(v, 3) for k, v in auc.items()}})
        out.append(te)
        print(f"  [{key}] {f.test_q}: chọn {best} ({auc[best]:.3f}), quy tắc {rule}")
    return pd.concat(out, ignore_index=True), pd.DataFrame(choices)


def summarize(res, key, y, pnl):
    rows = []
    A_hit = res[y].mean()
    for k in CANDS:
        for tag, dec in (("top 10%", res[f"top10|{k}"]), ("top 30%", res[f"top30|{k}"]), ("p > tỷ lệ gốc", res[f"p|{k}"] > res["base"])):
            rows.append({"target": key, "mô hình": NAMES[k], "quy tắc": tag, "AUC": roc_auc_score(res[y], res[f"p|{k}"]),
                         "khuyên làm theo": dec.mean(), **cls_metrics(res[y], dec),
                         "lãi TB khi làm theo": res.loc[dec.astype(bool), pnl].mean()})
    sel = res["follow"].astype(bool)
    p_sel = res.apply(lambda r: r[f"p|{r['chosen']}"], axis=1)
    ci = block_boot(res["date"], res[y].where(sel), res[y])
    selected = {"target": key, "mô hình": "★ CHỌN THEO VAL (mô hình + quy tắc)", "quy tắc": "theo từng quý",
                "AUC": roc_auc_score(res[y], p_sel), "khuyên làm theo": sel.mean(), **cls_metrics(res[y], sel),
                "lãi TB khi làm theo": res.loc[sel, pnl].mean(), "precision − A": res.loc[sel, y].mean() - A_hit,
                "CI95 (precision − A)": f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"}
    for tag, dec in (("Mốc: luôn làm theo", np.ones(len(res))), ("Mốc: không bao giờ làm theo", np.zeros(len(res)))):
        rows.append({"target": key, "mô hình": tag, "quy tắc": "—", "khuyên làm theo": dec.mean(), **cls_metrics(res[y], dec),
                     "lãi TB khi làm theo": res[pnl].mean() if dec.mean() else np.nan})
    per_q = res.groupby("quy_test").apply(lambda g: pd.Series({
        "target": key, "n": len(g), "tỷ lệ gốc": g[y].mean(), "cấu hình": g["chosen"].iloc[0], "quy tắc": g["rule"].iloc[0],
        **{k: v for k, v in cls_metrics(g[y], g["follow"]).items() if k in ("accuracy", "balanced acc", "precision (làm theo)",
                                                                            "F1 (làm theo)", "macro-F1")},
        "khuyên làm theo": g["follow"].mean()}), include_groups=False).reset_index()
    return pd.DataFrame(rows), pd.DataFrame([selected]), per_q


def main():
    ev = pd.read_parquet(CLEAN / "retail_events.parquet")
    cands, sels, chs, pqs, res_by = [], [], [], [], {}
    for key, (y, pnl) in TARGETS.items():
        res, ch = run_target(ev, key, y, pnl)
        c, s, pq = summarize(res, key, y, pnl)
        cands.append(c); sels.append(s); chs.append(ch); pqs.append(pq); res_by[key] = res
    cand, sel, ch, pq = (pd.concat(x, ignore_index=True) for x in (cands, sels, chs, pqs))
    cand.to_csv(REP / "retail_v2_candidates.csv", index=False, encoding="utf-8-sig")
    sel.to_csv(REP / "retail_v2_selected.csv", index=False, encoding="utf-8-sig")
    ch.to_csv(REP / "retail_v2_choices.csv", index=False, encoding="utf-8-sig")
    pq.to_csv(REP / "retail_v2_quarters.csv", index=False, encoding="utf-8-sig")
    r = res_by["tuyet_doi"]
    o = r[["date", "quy_test", "ticker", "titles", "event_type", "d", "d_r5", "chosen", "rule", "follow", "_reasons",
           f"ret_{H}", f"pnl_{H}", "follow_ok", "rel_pnl"]].copy()
    o["P(làm theo có lãi)"] = r.apply(lambda x: x[f"p|{x['chosen']}"], axis=1)
    o["khuyến nghị"] = np.where(o["follow"], np.where(o["d"] > 0, "MUA", "BÁN (nếu đang giữ)"),
                                np.where(o["d"] > 0, "KHÔNG MUA / không đuổi giá", "GIỮ, không bán tháo"))
    o["khuyến nghị đúng?"] = np.where(o["follow"], o["follow_ok"] == 1, o["follow_ok"] == 0)
    o.to_csv(REP / "retail_output_v2.csv", index=False, encoding="utf-8-sig")
    cols = ["target", "mô hình", "quy tắc", "AUC", "khuyên làm theo", "accuracy", "balanced acc", "precision (làm theo)",
            "recall (làm theo)", "F1 (làm theo)", "F1 (không làm theo)", "macro-F1", "lãi TB khi làm theo"]
    with pd.option_context("display.width", 280, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(cand[cols].to_string(index=False)); print()
        print(sel[cols + ["precision − A", "CI95 (precision − A)"]].to_string(index=False)); print()
        print(pq.to_string(index=False)); print()
        print(ch.to_string(index=False))


if __name__ == "__main__":
    main()
