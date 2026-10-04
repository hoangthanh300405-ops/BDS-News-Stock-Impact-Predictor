"""Hướng B (T25) — Tin tức có dự báo SỰ CHÚ Ý của thị trường ở phiên sau không?

Target  : vol_spike_h (thanh khoản đột biến, z log giá trị GD > 1) và big_raw_h (|lợi suất thực| > 1σ của mã)
Cấp     : A (mọi mã × phiên, LightGBM) và B (toàn ngành, LogReg)
So sánh : "giá & khối lượng" (baseline mạnh) so với "giá & khối lượng + tin", cùng mẫu, cùng 6 fold walk-forward
Chỉ số  : AUC (chính), ΔAUC có KTC bootstrap khối theo phiên, DM (HLN) trên log-loss theo phiên,
          precision trong top 10% mã mỗi phiên (cấp A) — "danh sách mã sẽ được chú ý", Rank IC với z liên tục
Dùng:  python -m src.models.run_attention
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.config import HORIZONS, ROOT
from src.eval.metrics import BINARY, block_bootstrap_diff, dm_test, row_logloss
from src.features.attention_features import (A_NEWS_ATTN, A_PRICE_ATTN, B_NEWS_ATTN, B_PRICE_ATTN, panel_A,
                                             panel_B)
from src.models.engine import LGBM, LogReg, run

OUT = ROOT / "outputs"


def topk_precision(p, q=0.10):
    """Mỗi phiên chọn q% mã có xác suất cao nhất -> tỷ lệ thực sự xảy ra, so với tỷ lệ chung."""
    sel = p.groupby("date", group_keys=False).apply(
        lambda g: g.nlargest(max(1, int(round(q * len(g)))), "p_1"), include_groups=False)
    return sel["y"].mean(), p["y"].mean()


def rank_ic_A(p, cont):
    ics = p.groupby("date").apply(lambda g: spearmanr(g["p_1"], g[cont])[0] if len(g) >= 10 else np.nan,
                                  include_groups=False).dropna()
    return ics.mean(), ics.mean() / (ics.std() / np.sqrt(len(ics)))


def evaluate(level, h, target, cont, data, price, news, algo, n_boot):
    d = data.dropna(subset=price + news + [target])
    P = {"giá & khối lượng": run(d, price, target, h, algo, "gia", classes=BINARY),
         "giá & khối lượng + tin": run(d, price + news, target, h, algo, "gia_tin", classes=BINARY),
         "chỉ tin": run(d, news, target, h, algo, "tin", classes=BINARY)}
    ref = P["giá & khối lượng"]
    ref_daily = row_logloss(ref).groupby(ref["date"]).mean()
    keys = ["date", "ticker"] if "ticker" in ref else ["date"]
    rows = []
    for name, p in P.items():
        p = p.merge(d[keys + [cont]], on=keys, how="left")
        r = {"level": level, "h": h, "target": target, "đặc trưng": name, "n": len(p),
             "tỷ lệ xảy ra": p["y"].mean(), "AUC": roc_auc_score(p["y"], p["p_1"])}
        if level.startswith("A"):
            prec, base = topk_precision(p)
            r["precision top10%/phiên"], r["lift top10%"] = prec, prec / base
            r["IC"], r["IC_t"] = rank_ic_A(p, cont)
        else:
            ok = p[cont].notna()
            r["IC"] = spearmanr(p.loc[ok, "p_1"], p.loc[ok, cont])[0]
        if name != "giá & khối lượng":
            daily = row_logloss(p).groupby(p["date"]).mean()
            r["DM vs giá"], r["p_DM"] = dm_test(daily, ref_daily, h)
            dd, lo, hi = block_bootstrap_diff(p, ref.merge(p[keys], on=keys), metric="auc", n_boot=n_boot)
            r["ΔAUC vs giá"], r["KTC95 ΔAUC"] = dd, f"[{lo:+.4f}, {hi:+.4f}]"
        rows.append(r)
    return rows, {k: v.assign(level=level, h=h, target=target, spec=k) for k, v in P.items()}


def main():
    t0 = time.time()
    dA, dB = panel_A(), panel_B()
    print(f"Mẫu A: {len(dA):,} (mã, phiên) | mẫu B: {len(dB)} phiên ({time.time() - t0:.0f}s)")
    rows, preds = [], []
    for h in HORIZONS:
        for tgt_a, tgt_b, cont_a, cont_b in ((f"vol_spike_{h}", f"vol_spike_B_{h}", f"attn_val_z_{h}", f"attnB_val_z_{h}"),
                                             (f"big_raw_{h}", f"big_move_B_{h}", f"zr_{h}", f"zB_{h}")):
            for level, data, pr, nw, tgt, cont, algo, nb in (
                    ("A mã", dA, A_PRICE_ATTN, A_NEWS_ATTN, tgt_a, cont_a, LGBM, 200),
                    ("B ngành", dB, B_PRICE_ATTN, B_NEWS_ATTN, tgt_b, cont_b, LogReg, 1000)):
                if cont.startswith("z"):                              # biến động: dùng |z| làm biến liên tục
                    data = data.assign(**{f"abs_{cont}": data[cont].abs()}); cont = f"abs_{cont}"
                t1 = time.time()
                r, p = evaluate(level, h, tgt, cont, data, pr, nw, algo, nb)
                rows += r; preds += list(p.values())
                print(f"  xong {level} {tgt} ({time.time() - t1:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "reports" / "attention.csv", index=False, encoding="utf-8-sig")
    pd.concat(preds, ignore_index=True).to_parquet(OUT / "predictions" / "attention.parquet", index=False)
    with pd.option_context("display.width", 280, "display.max_columns", 30, "display.max_rows", 100,
                           "display.float_format", "{:.4f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
