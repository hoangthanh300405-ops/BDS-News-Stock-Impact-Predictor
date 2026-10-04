"""Hướng NHÀ ĐẦU TƯ NHỎ LẺ — mô hình "đọc tin xong, có nên làm theo không?" + so sánh 3 kiểu nhà đầu tư.

Walk-forward 6 fold (test 2025Q2 → 2026Q3), purge + embargo; mô hình fit trên train+val, ngưỡng chọn lọc lấy từ val.
Hai target (chốt trước):
  TUYỆT ĐỐI  follow_ok  = làm theo tin 5 phiên có lãi sau phí (thứ nhà đầu tư thực nhận)
  TƯƠNG ĐỐI  follow_rel = làm theo tin thắng việc mua rổ BĐS cùng lúc (bỏ ảnh hưởng xu hướng chung của thị trường)
Mô hình (chốt trước khi nhìn test):
  M0 Luôn làm theo tin          — nhà đầu tư A (đọc báo, làm theo mọi tin)
  M1 Quy tắc kinh nghiệm        — chỉ làm theo TIN SỰ KIỆN DN khi giá CHƯA chạy theo hướng tin trong 5 phiên
  M2 LightGBM — chỉ thông tin bài báo
  M3 Logistic — tin + giá
  M4 LightGBM — tin + giá        (mô hình chính)
Khuyến nghị: "làm theo" khi P > 0,5; bản CHỌN LỌC: chỉ top q% điểm (q = 10/20/30/50%, ngưỡng lấy từ val).
Nhà đầu tư B = không đọc báo, mua rổ BĐS thanh khoản (trung bình cộng) cùng thời điểm, giữ cùng số phiên.
Đầu ra: outputs/reports/retail_models.csv, retail_accuracy_f1.csv, retail_investors.csv, retail_folds.csv, retail_by_type.csv, retail_output.csv;
        outputs/figures/r6_retail_cum.png, r7_retail_move.png
Dùng:  python -m src.retail.run_retail
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CLEAN, ROOT
from src.eval.walk_forward import make_folds
from src.retail.events import COST, FEATURES, H, NEWS_ONLY

REP, FIG = ROOT / "outputs" / "reports", ROOT / "outputs" / "figures"
TOPS = (0.10, 0.20, 0.30, 0.50)    # đường cong độ phủ <-> độ chính xác (ngưỡng đều lấy từ val)
SEED = 2026
TARGETS = {"tuyet_doi": ("follow_ok", f"pnl_{H}"), "tuong_doi": ("follow_rel", "rel_pnl")}
MAIN = "M4 LightGBM — tin + giá"


def lgbm():
    return LGBMClassifier(num_leaves=15, learning_rate=0.03, n_estimators=300, min_child_samples=80,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
                          random_state=SEED, verbose=-1)


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def rule(x):
    return ((x["frac_firm"] > 0.5) & (x["d_r5"].fillna(0) <= 0)).astype(float).to_numpy()


MODELS = {
    "M0 Luôn làm theo tin": None,
    "M1 Quy tắc kinh nghiệm": "rule",
    "M2 LightGBM — chỉ tin": (lgbm, NEWS_ONLY),
    "M3 Logistic — tin + giá": (logreg, FEATURES),
    MAIN: (lgbm, FEATURES),
}


def prep(x, feats, med):
    return x[feats].replace([np.inf, -np.inf], np.nan).fillna(med)


def fit_predict(spec, tr, te, y):
    make, feats = spec
    med = tr[feats].replace([np.inf, -np.inf], np.nan).median()
    m = make().fit(prep(tr, feats, med), tr[y].astype(int))
    return m.predict_proba(prep(te, feats, med))[:, 1], m, med


def run(ev, y):
    folds = make_folds(H + 1)             # +1: có thể vào lệnh ở t+1 khi kẹt trần/sàn
    ev = ev[ev[y].notna()].copy()
    out = []
    for f in folds:
        tr, va = ev[ev["date"].isin(f.train)], ev[ev["date"].isin(f.val)]
        fit, te = ev[ev["date"].isin(f.fit)], ev[ev["date"].isin(f.test)].copy()
        te["fold"], te["quy_test"] = f.k, f.test_q
        for name, spec in MODELS.items():
            if spec is None:
                p = np.ones(len(te))
            elif spec == "rule":
                p = rule(te)
            else:
                p_va = fit_predict(spec, tr, va, y)[0]
                thrs = {q: np.quantile(p_va, 1 - q) for q in TOPS}      # ngưỡng lấy từ VAL
                p, m, med = fit_predict(spec, fit, te, y)
                for q, t in thrs.items():
                    te[f"top{int(q * 100)}|{name}"] = p >= t
                if name == MAIN:
                    X = prep(te, spec[1], med)
                    contrib = m.booster_.predict(X, pred_contrib=True)[:, :-1]
                    te["_reasons"] = [reasons(r, c, spec[1]) for r, c in zip(X.to_numpy(), contrib)]
            te[f"p|{name}"] = p
        te["base_rate_train"] = fit[y].mean()
        out.append(te)
    return pd.concat(out, ignore_index=True)


LABEL = {
    "d_r5": "giá đã chạy theo hướng tin 5 phiên", "d_r1": "giá phiên đọc tin theo hướng tin",
    "d_r20": "giá đã chạy theo hướng tin 20 phiên", "d_ar5": "vượt/kém rổ BĐS 5 phiên (theo hướng tin)",
    "d_mkt1": "cả rổ BĐS trong phiên (theo hướng tin)", "d_mkt5": "cả rổ BĐS 5 phiên (theo hướng tin)",
    "d_dist_high20": "khoảng cách tới đỉnh 20 phiên", "d_dist_low20": "khoảng cách tới đáy 20 phiên",
    "frac_comm": "tỷ lệ bài bình luận thị trường", "frac_firm": "tỷ lệ bài sự kiện doanh nghiệp",
    "d": "hướng tin (1 = tốt, −1 = xấu)", "score": "điểm tác động LLM", "n_art": "số bài trong phiên",
    "news_prev5": "số phiên có tin trong 5 phiên trước", "news_prev20": "số phiên có tin trong 20 phiên trước",
    "since_last_news": "số phiên từ lần có tin trước", "vol20": "độ biến động 20 phiên",
    "value_z": "giá trị giao dịch so với TB 20 phiên (log)", "log_value": "quy mô thanh khoản",
    "limit_in_dir": "kịch trần/sàn theo hướng tin", "n_tick_in_art": "số mã được nhắc trong bài",
}
PCT = {"d_r5", "d_r1", "d_r20", "d_ar5", "d_mkt1", "d_mkt5", "d_dist_high20", "d_dist_low20", "vol20"}


def reasons(vals, contrib, feats, k=2):
    idx = np.argsort(-np.abs(contrib))[:k]
    parts = []
    for i in idx:
        f, v = feats[i], vals[i]
        vs = f"{v * 100:+.1f}%" if f in PCT else (f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}")
        parts.append(f"{LABEL.get(f, f)} = {vs} ({'ủng hộ' if contrib[i] > 0 else 'phản đối'} làm theo)")
    return "; ".join(parts)


def block_boot(dates, a, b, n=2000, block=5):
    """CI 95% cho mean(a) − mean(b) (bỏ NaN), lấy mẫu lại theo khối phiên liên tiếp."""
    df = pd.DataFrame({"date": dates, "a": a, "b": b})
    g = df.groupby("date").agg(sa=("a", "sum"), na=("a", "count"), sb=("b", "sum"), nb=("b", "count"))
    arr = g.to_numpy()
    T = len(arr)
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, T - block, size=(n, T // block + 1))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    S = arr[idx]
    d = S[..., 0].sum(1) / S[..., 1].sum(1) - S[..., 2].sum(1) / S[..., 3].sum(1)
    return np.nanpercentile(d, [2.5, 97.5])


def evaluate(res, key, y, pnl):
    """Bảng mô hình + bảng 'nhà đầu tư' (A, B, C ở nhiều mức chọn lọc)."""
    rows, inv = [], []
    yy = res[y].astype(int)
    A_hit, A_pnl = res[y].mean(), res[pnl].mean()
    for name, spec in MODELS.items():
        p = res[f"p|{name}"]
        fixed = spec in (None, "rule")
        yhat = (p > 0.5) if fixed else (p > res["base_rate_train"])
        rows.append({"target": key, "mô hình": name, "n test": len(res), "tỷ lệ gốc test": A_hit,
                     "AUC": np.nan if fixed else roc_auc_score(yy, p),
                     "balanced acc": balanced_accuracy_score(yy, yhat.astype(int)),
                     "accuracy khuyến nghị (p>0,5)": np.mean((p > 0.5).astype(int) == yy),
                     "accuracy 'không bao giờ làm theo'": 1 - A_hit})
        if spec is None:
            continue
        sels = [("khuyên làm theo (p>0,5)", p > 0.5)]
        if not fixed:
            sels += [(f"chọn lọc top {q:.0%}", res[f"top{int(q * 100)}|{name}"]) for q in TOPS]
        for tag, sel in sels:
            sel = sel.astype(bool)
            s_ = res[sel]
            if len(s_) < 30:
                continue
            ci = block_boot(res["date"], res[y].where(sel), res[y])
            cip = block_boot(res["date"], res[pnl].where(sel), res[pnl])
            wins = sum(res.loc[sel & (res["fold"] == k), y].mean() > res.loc[res["fold"] == k, y].mean()
                       for k in res["fold"].unique())
            inv.append({"target": key, "nhà đầu tư": f"C · {name} · {tag}", "số lệnh": len(s_), "độ phủ": len(s_) / len(res),
                        "tỷ lệ đúng": s_[y].mean(), "chênh so với A": s_[y].mean() - A_hit,
                        "CI95 chênh (khối phiên)": f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                        "lãi TB/lệnh": s_[pnl].mean(), "chênh lãi so với A": s_[pnl].mean() - A_pnl,
                        "CI95 chênh lãi": f"[{cip[0]:+.4f}, {cip[1]:+.4f}]", "số quý thắng A (/6)": int(wins)})
    base = [{"target": key, "nhà đầu tư": "A · Đọc báo, làm theo mọi tin", "số lệnh": len(res), "độ phủ": 1.0,
             "tỷ lệ đúng": A_hit, "lãi TB/lệnh": A_pnl}]
    if key == "tuyet_doi":
        B = res["mkt_ret_H"] - COST
        base.append({"target": key, "nhà đầu tư": "B · Không đọc báo, mua rổ BĐS cùng thời điểm", "số lệnh": len(res),
                     "tỷ lệ đúng": np.mean(B > 0), "lãi TB/lệnh": B.mean()})
    return pd.DataFrame(rows), pd.DataFrame(base + inv)


def cls_metrics(y, yhat):
    """Accuracy, precision/recall/F1 cho lớp 'làm theo' (1) và 'không làm theo' (0), macro-F1."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    out = {"accuracy": accuracy_score(y, yhat), "balanced acc": balanced_accuracy_score(y, yhat)}
    for c, tag in ((1, "làm theo"), (0, "không làm theo")):
        out[f"precision ({tag})"] = precision_score(y, yhat, pos_label=c, zero_division=0)
        out[f"recall ({tag})"] = recall_score(y, yhat, pos_label=c, zero_division=0)
        out[f"F1 ({tag})"] = f1_score(y, yhat, pos_label=c, zero_division=0)
    out["macro-F1"] = f1_score(y, yhat, average="macro", zero_division=0)
    return out


def accuracy_f1(res, key, y):
    """Bảng accuracy / F1 của MỌI quyết định "làm theo / không làm theo" trên TOÀN BỘ tin trong tập test.
    Quyết định: p > 0,5 | p > tỷ lệ gốc của train | top q% (ngưỡng từ val) -> làm theo, còn lại -> không làm theo."""
    yy = res[y].astype(int)
    rows = [{"target": key, "mô hình": "Mốc: luôn làm theo tin", "quy tắc quyết định": "—", "tỷ lệ khuyên làm theo": 1.0,
             **cls_metrics(yy, np.ones(len(yy)))},
            {"target": key, "mô hình": "Mốc: không bao giờ làm theo", "quy tắc quyết định": "—", "tỷ lệ khuyên làm theo": 0.0,
             **cls_metrics(yy, np.zeros(len(yy)))}]
    for name, spec in MODELS.items():
        if spec is None:
            continue
        p = res[f"p|{name}"]
        rules = [("p > 0,5", p > 0.5)]
        if spec != "rule":
            rules += [("p > tỷ lệ gốc train", p > res["base_rate_train"])]
            rules += [(f"top {q:.0%} (ngưỡng từ val)", res[f"top{int(q * 100)}|{name}"]) for q in TOPS]
        for tag, dec in rules:
            dec = dec.astype(int)
            rows.append({"target": key, "mô hình": name, "quy tắc quyết định": tag, "tỷ lệ khuyên làm theo": dec.mean(),
                         **cls_metrics(yy, dec)})
    return pd.DataFrame(rows)


def by_type(ev):
    """Mô tả (toàn mẫu có kết quả): làm theo loại tin nào thì có lãi / thắng rổ BĐS."""
    ev = ev[ev["follow_ok"].notna()].copy()
    g_cols = [c for c in ev if c.startswith("g_")]
    ev["_nhom"] = np.where(ev[g_cols].max(axis=1) > 0, ev[g_cols].idxmax(axis=1).str[2:], "khac")
    ev["_loai"] = np.select([ev["frac_firm"] > 0.5, ev["frac_comm"] > 0.5], ["sự kiện DN", "bình luận TT"], "khác")
    ev["_huong"] = np.where(ev["d"] > 0, "tin tốt", "tin xấu")
    ev["_move"] = pd.cut(ev["d_r5"] * 100, [-np.inf, -5, -1, 1, 5, 10, np.inf],
                         labels=["ngược hướng >5%", "ngược 1–5%", "±1%", "theo hướng 1–5%", "theo hướng 5–10%", "theo hướng >10%"])
    ev["_moi"] = pd.cut(ev["news_prev5"], [-1, 0, 2, 5], labels=["tin mới (0 phiên có tin trước)", "1–2 phiên", "3–5 phiên"])
    ev["_tran"] = np.where(ev["limit_in_dir"] == 1, "kịch trần/sàn theo hướng tin", "không")
    rows = []
    for dim, col in (("hướng tin", "_huong"), ("loại bài", "_loai"), ("nhóm tin", "_nhom"),
                     ("giá đã chạy 5 phiên (theo hướng tin)", "_move"), ("độ mới của tin", "_moi"), ("trần/sàn", "_tran")):
        for k, g in ev.groupby(col, observed=True):
            if len(g) < 20:
                continue
            rows.append({"chiều": dim, "nhóm": k, "n": len(g), "làm theo có lãi (sau phí)": g["follow_ok"].mean(),
                         "làm theo thắng rổ BĐS": g["follow_rel"].mean(), "lãi TB/lệnh sau phí": g[f"pnl_{H}"].mean(),
                         "hơn rổ BĐS TB": g["rel_pnl"].mean(), "lãi TB 1 phiên": g["pnl_1"].mean(),
                         "lãi TB 10 phiên": g["pnl_10"].mean()})
    return pd.DataFrame(rows)


def figures(res, bt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    daily = pd.DataFrame({"date": res["date"],
                          "A: đọc báo, làm theo mọi tin": res[f"pnl_{H}"],
                          "B: không đọc báo, mua rổ BĐS": res["mkt_ret_H"] - COST,
                          "C: chỉ làm theo khi mô hình khuyên (top 30%)": res[f"pnl_{H}"].where(res[f"top30|{MAIN}"])})
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for c, col in zip(daily.columns[1:], ["#c0392b", "#7f8c8d", "#1f6feb"]):
        s = daily.groupby("date")[c].mean().fillna(0)
        ax.plot(s.index, s.cumsum() * 100 / H, label=c, color=col, lw=2)
    ax.axhline(0, color="k", lw=.6)
    ax.set_title("Tập test 2025Q2–2026Q3: lãi/lỗ cộng dồn sau phí (%)\n"
                 "(mỗi phiên chia đều vốn cho các lệnh mở trong phiên, giữ 5 phiên)")
    ax.set_ylabel("%"); ax.legend(frameon=False); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "r6_retail_cum.png", dpi=150); plt.close(fig)
    m = bt[bt["chiều"] == "giá đã chạy 5 phiên (theo hướng tin)"]
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - .2, m["làm theo có lãi (sau phí)"] * 100, .4, color="#1f6feb", label="làm theo có lãi sau phí")
    ax.bar(x + .2, m["làm theo thắng rổ BĐS"] * 100, .4, color="#f0883e", label="làm theo thắng rổ BĐS")
    ax.axhline(50, color="k", ls="--", lw=.8, label="tung đồng xu")
    ax.set_xticks(x, [f"{k}\nn={n}" for k, n in zip(m["nhóm"].astype(str), m["n"])], fontsize=8)
    ax.set_ylim(30, 60); ax.set_ylabel("%")
    ax.set_title("Giá đã chạy bao nhiêu trước khi bạn đọc tin — làm theo còn đáng không? (giữ 5 phiên)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "r7_retail_move.png", dpi=150); plt.close(fig)


def output_table(ra, rr):
    """Bảng output từng tin: mô hình chính M4, target tuyệt đối, top 30%; kèm xác suất thắng rổ BĐS."""
    o = ra[["date", "quy_test", "ticker", "title", "event_type", "d", "frac_comm", "frac_firm", "d_r1", "d_r5", "entry_shift",
            f"p|{MAIN}", f"top30|{MAIN}", "_reasons", f"ret_{H}", f"pnl_{H}", "follow_ok", "rel_pnl"]].copy()
    rr = rr.set_index(["date", "ticker"])
    o["P(thắng rổ BĐS)"] = rr[f"p|{MAIN}"].reindex(pd.MultiIndex.from_frame(ra[["date", "ticker"]])).to_numpy()
    follow = o.pop(f"top30|{MAIN}")
    o["khuyến nghị"] = np.where(follow, np.where(o["d"] > 0, "MUA", "BÁN (nếu đang giữ)"),
                                np.where(o["d"] > 0, "KHÔNG MUA / không đuổi giá", "GIỮ, không bán tháo"))
    o["khuyến nghị đúng?"] = np.where(follow, o["follow_ok"] == 1, o["follow_ok"] == 0)
    return o.rename(columns={
        "title": "tiêu đề", "d": "hướng tin", "d_r1": "giá phiên đọc tin theo hướng tin",
        "d_r5": "giá đã chạy 5 phiên theo hướng tin", f"p|{MAIN}": "P(làm theo có lãi)", "_reasons": "lý do chính",
        f"ret_{H}": f"lợi suất {H} phiên", f"pnl_{H}": "lãi nếu làm theo (sau phí)", "follow_ok": "làm theo có lãi",
        "rel_pnl": "hơn rổ BĐS nếu làm theo"})


def main():
    ev = pd.read_parquet(CLEAN / "retail_events.parquet")
    mods, invs, folds, accs, res_by = [], [], [], [], {}
    for key, (y, pnl) in TARGETS.items():
        res = run(ev, y)
        mod, inv = evaluate(res, key, y, pnl)
        mods.append(mod); invs.append(inv); res_by[key] = res
        accs.append(accuracy_f1(res, key, y))
        pf = res.groupby(["fold", "quy_test"]).apply(lambda g: pd.Series({
            "n": len(g), "tỷ lệ gốc": g[y].mean(), "AUC M4": roc_auc_score(g[y], g[f"p|{MAIN}"]),
            "tỷ lệ đúng top 30%": g.loc[g[f"top30|{MAIN}"], y].mean(), "độ phủ": g[f"top30|{MAIN}"].mean(),
            **{f"{k} (M4 top 30%)": v for k, v in cls_metrics(g[y], g[f"top30|{MAIN}"]).items()
               if k in ("accuracy", "F1 (làm theo)", "macro-F1")}}),
            include_groups=False).reset_index()
        folds.append(pf.assign(target=key))
    mod, inv, per_fold = pd.concat(mods), pd.concat(invs), pd.concat(folds)
    bt = by_type(ev)
    out = output_table(res_by["tuyet_doi"], res_by["tuong_doi"])
    out.to_csv(REP / "retail_output.csv", index=False, encoding="utf-8-sig")
    mod.to_csv(REP / "retail_models.csv", index=False, encoding="utf-8-sig")
    inv.to_csv(REP / "retail_investors.csv", index=False, encoding="utf-8-sig")
    bt.to_csv(REP / "retail_by_type.csv", index=False, encoding="utf-8-sig")
    per_fold.to_csv(REP / "retail_folds.csv", index=False, encoding="utf-8-sig")
    acc = pd.concat(accs)
    acc.to_csv(REP / "retail_accuracy_f1.csv", index=False, encoding="utf-8-sig")
    figures(res_by["tuyet_doi"], bt)
    with pd.option_context("display.width", 260, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(mod.to_string(index=False)); print()
        print(inv.drop(columns=["CI95 chênh lãi"]).to_string(index=False)); print()
        print(per_fold.to_string(index=False)); print(); print(bt.to_string(index=False)); print()
        print(acc.to_string(index=False))
    print(f"\nKhuyến nghị M4 top 30% (tuyệt đối) đúng: {out['khuyến nghị đúng?'].mean():.3f}")


if __name__ == "__main__":
    main()
