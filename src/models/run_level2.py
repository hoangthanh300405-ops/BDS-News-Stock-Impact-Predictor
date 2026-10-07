"""Mức 2 — các thí nghiệm sau RQ1.   Dùng:  python -m src.models.run_level2

E1  Rank IC của các mô hình RQ1 (không huấn luyện lại)
E2  Model A chỉ trên cặp CÓ TIN TRONG NGÀY (nơi tín hiệu tập trung — chẩn đoán RQ1)
E3  Target phụ big_move (RQ9): tin dự báo ĐỘ LỚN biến động tốt hơn CHIỀU?
E4  Thêm F3 — LRI/PPI (luồng rủi ro pháp lý, chính sách)  (RQ7)
E5  Đ2 — hiệu chỉnh nhãn trong fold (RQ3)
Thuật toán chính: Model B -> LogReg (mẫu nhỏ), Model A -> LightGBM (§8.5).
"""
import time

import numpy as np
import pandas as pd

from src.config import CLEAN, HORIZONS, ROOT
from src.eval.metrics import BINARY, CLASSES, block_bootstrap_diff, dm_test, rank_ic, row_logloss, summarize
from src.features.calibration import fit_calibration
from src.features.news_features import A_NEWS, B_NEWS, LEGAL, industry_news, ticker_news
from src.features.price_features import A_PRICE, B_PRICE_CORE, features_A, features_B, sample_A
from src.models.engine import LGBM, LogReg, run

OUT = ROOT / "outputs"
ALGO = {"B": LogReg, "A": LGBM}
N_BOOT = {"B": 1000, "A": 300}


# ---------------- dữ liệu ----------------
def base_B():
    return features_B().merge(pd.read_parquet(CLEAN / "target_B.parquet"), on="date")


def base_A():
    d = sample_A().merge(features_A(), on=["date", "ticker"])
    return d.merge(pd.read_parquet(CLEAN / "target_A.parquet"), on=["date", "ticker"])


def news_B(ind=None, tick=None):
    return industry_news(ind, tick)


def news_A(ind=None, tick=None):
    t = ticker_news(ind, tick)
    return t.merge(industry_news(ind, tick)[["date"] + LEGAL], on="date", how="left")


def add_abs(d, prefix):
    for h in HORIZONS:
        d[f"abs_{prefix}{h}"] = d[f"{prefix}{h}"].abs()
    return d


# ---------------- so sánh ----------------
def compare(level, h, specs, target, ref, classes=CLASSES, ic_outcome=None, fold_data=None):
    """specs: {tên: (data, cột)}; ref: tên mô hình tham chiếu cho DM & bootstrap."""
    algo, metric = ALGO[level[0]], ("bal_acc" if list(classes) == CLASSES else "auc")
    preds = {}
    for name, (data, cols) in specs.items():
        fd = fold_data.get(name) if fold_data else None
        preds[name] = run(data, cols, target, h, algo, name, classes=classes, fold_data=fd)
    # đánh giá trên phần GIAO của mẫu test (các spec có thể rơi dòng khác nhau)
    keys = ["date", "ticker"] if "ticker" in preds[ref] else ["date"]
    common = None
    for p in preds.values():
        k = p[keys].drop_duplicates()
        common = k if common is None else common.merge(k, on=keys)
    for n in preds:
        preds[n] = preds[n].merge(common, on=keys)
    outcome = ic_outcome or target
    rows = []
    r = preds[ref]
    r_daily = row_logloss(r).groupby(r["date"]).mean()
    for name, p in preds.items():
        s = summarize(p)
        oc = specs[name][0].set_index(keys)[outcome]
        p_oc = p.join(oc, on=keys)
        s.update(rank_ic(p_oc, outcome, h))
        s.update(level=level, h=h, model=name)
        if name != ref:
            daily = row_logloss(p).groupby(p["date"]).mean()
            s["DM_vs_ref"], s["p_DM"] = dm_test(daily, r_daily, h)
            d, lo, hi = block_bootstrap_diff(p, r, metric=metric, n_boot=N_BOOT[level[0]])
            s[f"d{metric}_vs_ref"], s["CI95"] = d, f"[{lo:+.3f}, {hi:+.3f}]"
            s["ref"] = ref
        rows.append(s)
    return rows


def main():
    t_all = time.time()
    bB, bA = add_abs(base_B(), "zB_"), add_abs(base_A(), "z_")
    nB, nA = news_B(), news_A()
    dB = bB.merge(nB, on="date")
    dA = bA.merge(nA, on=["date", "ticker"], how="left")
    rows = []

    # E1 — Rank IC của RQ1 (dự đoán đã lưu)
    rq1 = pd.read_parquet(OUT / "predictions" / "rq1.parquet")
    for (lvl, h, algo, spec), p in rq1.groupby(["level", "h", "algo", "spec"]):
        oc = (dB.set_index("date")[f"zB_{h}"] if lvl.startswith("B") else dA.set_index(["date", "ticker"])[f"z_{h}"])
        keys = ["date"] if lvl.startswith("B") else ["date", "ticker"]
        q = p.join(oc.rename("oc"), on=keys)
        rows.append(dict(exp="E1 IC (RQ1)", level=lvl, h=h, model=f"{spec} — {algo}", **rank_ic(q, "oc", h)))
    print(f"E1 xong ({time.time() - t_all:.0f}s)")

    for h in HORIZONS:
        # E2 — chỉ cặp có tin trong ngày
        t0 = time.time()
        fresh = dA[dA["news_today"] == 1]
        for s in compare("A mã", h, {"giá": (fresh, A_PRICE), "giá + tin": (fresh, A_PRICE + A_NEWS)},
                         f"y_{h}", "giá", ic_outcome=f"z_{h}"):
            rows.append(dict(exp="E2 chỉ cặp có tin trong ngày", **s))
        # E3 — big_move
        for lvl, d, pc, nc, tg, oc in (("B ngành", dB, B_PRICE_CORE, B_NEWS + LEGAL, f"big_move_B_{h}", f"abs_zB_{h}"),
                                       ("A mã", dA, A_PRICE, A_NEWS + LEGAL, f"big_move_{h}", f"abs_z_{h}")):
            for s in compare(lvl, h, {"giá": (d, pc), "giá + tin": (d, pc + nc)}, tg, "giá", classes=BINARY, ic_outcome=oc):
                rows.append(dict(exp="E3 big_move (RQ9)", **s))
        # E4 — thêm LRI/PPI
        for lvl, d, pc, nc, tg in (("B ngành", dB, B_PRICE_CORE, B_NEWS, f"yB_{h}"), ("A mã", dA, A_PRICE, A_NEWS, f"y_{h}")):
            specs = {"giá": (d, pc), "giá + tin": (d, pc + nc), "giá + tin + LRI/PPI": (d, pc + nc + LEGAL)}
            for s in compare(lvl, h, specs, tg, "giá + tin", ic_outcome=tg.replace("y", "z")):
                rows.append(dict(exp="E4 + LRI/PPI (RQ7)", **s))
        print(f"h={h}: E2–E4 xong ({time.time() - t0:.0f}s)")

        # E5 — Đ2: điểm hiệu chỉnh học trên train của từng fold
        t0 = time.time()
        cache = {}

        def cal_data(level):
            def fn(f):
                key = (level, f.k)
                if key not in cache:
                    ind, tick, _, _ = fit_calibration(f.train)
                    cache[key] = (bB.merge(news_B(ind, tick), on="date") if level == "B"
                                  else bA.merge(news_A(ind, tick), on=["date", "ticker"], how="left"))
                return cache[key]
            return fn
        for lvl, d, pc, nc, tg in (("B ngành", dB, B_PRICE_CORE, B_NEWS, f"yB_{h}"), ("A mã", dA, A_PRICE, A_NEWS, f"y_{h}")):
            specs = {"giá + tin (nhãn thô)": (d, pc + nc), "giá + tin (Đ2)": (d, pc + nc)}
            fd = {"giá + tin (Đ2)": cal_data(lvl[0])}
            for s in compare(lvl, h, specs, tg, "giá + tin (nhãn thô)", ic_outcome=tg.replace("y", "z"), fold_data=fd):
                rows.append(dict(exp="E5 Đ2 hiệu chỉnh (RQ3)", **s))
        print(f"h={h}: E5 xong ({time.time() - t0:.0f}s)")

    res = pd.DataFrame(rows)
    cols = [c for c in ["exp", "level", "h", "model", "n", "bal_acc", "mcc", "auc", "base_rate", "log_loss", "IC", "IC_t",
                        "ref", "DM_vs_ref", "p_DM", "dbal_acc_vs_ref", "dauc_vs_ref", "CI95"] if c in res]
    res = res[cols]
    res.to_csv(OUT / "reports" / "level2.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 260, "display.max_columns", 40, "display.float_format", "{:.3f}".format,
                           "display.max_rows", 200):
        print(res.to_string(index=False))
    print(f"Tổng thời gian: {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
