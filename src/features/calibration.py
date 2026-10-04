"""Đ2 — Hiệu chỉnh nhãn LLM theo thị trường (§7.3, v2.1). CHỈ ước lượng trên các phiên train của một fold.

Với mỗi ô o và dấu nhãn σ ∈ {+, −}:
  hit_o,σ = tỷ lệ (có trọng số, mỗi (ô, [mã,] phiên) một phiếu) dấu(z₁ − z̄₁_train) trùng σ
  b_σ     = tỷ lệ nền của dấu σ trên các phiên train
  co ngót phân cấp:  ô -> loại dữ liệu -> tỷ lệ nền  (m = 20)
  c_o,σ   = clip(2·(ĥit_o,σ − b_σ), −1, 1);    điểm hiệu chỉnh s' = s × c_{o, dấu(s)}
Ô cấp ngành: (loại dữ liệu × nhóm tin), đối chiếu zB_1.   Ô cấp mã: (nguồn × nhóm tin), đối chiếu z_1 của AR.
"""
import numpy as np
import pandas as pd

from src.config import CLEAN

M_SHRINK = 20


def _table(df, cell, parent, outcome_sign, base):
    """df: dòng nhãn khác 0 với cột sigma (±1), w (trọng số), cột ô và cột cha. Trả về c theo (ô, σ)."""
    df = df.assign(hit=(outcome_sign == df["sigma"]).astype(float) * df["w"])
    par = df.groupby([parent, "sigma"]).agg(k=("w", "sum"), h=("hit", "sum"))
    par["b"] = [base[s] for s in par.index.get_level_values("sigma")]
    par["hat"] = (par["h"] + M_SHRINK * par["b"]) / (par["k"] + M_SHRINK)
    cel = df.groupby([cell, parent, "sigma"]).agg(k=("w", "sum"), h=("hit", "sum")).reset_index()
    cel = cel.merge(par["hat"].rename("hat_par").reset_index(), on=[parent, "sigma"])
    cel["hat"] = (cel["h"] + M_SHRINK * cel["hat_par"]) / (cel["k"] + M_SHRINK)
    cel["b"] = cel["sigma"].map(base)
    cel["c"] = np.clip(2 * (cel["hat"] - cel["b"]), -1, 1)
    return cel


def fit_calibration(train_dates):
    """Trả về (ind_scores, tick_scores, bảng_ngành, bảng_mã) — điểm đã hiệu chỉnh cho MỌI bài,
    nhưng hệ số c chỉ học từ các bài có phiên thuộc train_dates."""
    train_dates = pd.DatetimeIndex(train_dates)
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "session"])
    tb = pd.read_parquet(CLEAN / "target_B.parquet").set_index("date")["zB_1"]
    ta = pd.read_parquet(CLEAN / "target_A.parquet", columns=["date", "ticker", "z_1"]).set_index(["date", "ticker"])["z_1"]

    # ---------------- cấp ngành ----------------
    lab = pd.read_parquet(CLEAN / "article_label.parquet")
    lab = lab[lab["typ"].isin(["keyword", "law"])].merge(art, on="article_id")
    lab["cell"] = lab["typ"] + "|" + lab["nhom_tin"]
    zb = tb[tb.index.isin(train_dates)].dropna()
    zb_c = np.sign(zb - zb.mean())
    base_b = {s: (zb_c == s).mean() for s in (-1.0, 1.0)}
    tr = lab[lab["session"].isin(zb.index) & (lab["impact_score"] != 0)].copy()
    tr["sigma"] = np.sign(tr["impact_score"])
    tr["w"] = 1 / tr.groupby(["cell", "session"])["article_id"].transform("size")
    cel_b = _table(tr, "cell", "typ", tr["session"].map(zb_c).to_numpy(), base_b)
    cmap = cel_b.set_index(["cell", "sigma"])["c"]
    lab["sigma"] = np.sign(lab["impact_score"])
    key = pd.MultiIndex.from_arrays([lab["cell"], lab["sigma"]])
    lab["s_adj"] = lab["impact_score"] * key.map(cmap).fillna(0).to_numpy()   # ô không có trong train -> 0
    lab.loc[lab["impact_score"] == 0, "s_adj"] = 0.0
    ind_scores = lab.groupby("article_id")["s_adj"].mean()

    # ---------------- cấp mã ----------------
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet",
                          columns=["article_id", "ticker", "src", "nhom_tin", "impact_score"]).merge(art, on="article_id")
    atl["cell"] = atl["src"] + "|" + atl["nhom_tin"]
    atl["grp"] = "company"
    za = ta[ta.index.get_level_values("date").isin(train_dates)].dropna()
    za_c = np.sign(za - za.mean())
    base_a = {s: (za_c == s).mean() for s in (-1.0, 1.0)}
    tr = atl[atl["session"].isin(train_dates) & (atl["impact_score"] != 0)].copy()
    tr["o"] = pd.MultiIndex.from_arrays([tr["session"], tr["ticker"]]).map(za_c)
    tr = tr.dropna(subset=["o"])
    tr["sigma"] = np.sign(tr["impact_score"])
    tr["w"] = 1 / tr.groupby(["cell", "ticker", "session"])["article_id"].transform("size")
    cel_a = _table(tr, "cell", "grp", tr["o"].to_numpy(), base_a)
    cmap = cel_a.set_index(["cell", "sigma"])["c"]
    atl["sigma"] = np.sign(atl["impact_score"])
    key = pd.MultiIndex.from_arrays([atl["cell"], atl["sigma"]])
    atl["s_adj"] = atl["impact_score"] * key.map(cmap).fillna(0).to_numpy()
    atl.loc[atl["impact_score"] == 0, "s_adj"] = 0.0
    tick_scores = atl.set_index(["article_id", "ticker"])["s_adj"]
    return ind_scores, tick_scores, cel_b, cel_a


if __name__ == "__main__":
    from src.eval.walk_forward import make_folds
    f = make_folds(1)[0]
    _, _, cb, ca = fit_calibration(f.train)
    pd.set_option("display.width", 200)
    print(f"Bảng hiệu chỉnh fold 1 (train đến {f.train[-1].date()})")
    print("CẤP NGÀNH:"); print(cb[["cell", "sigma", "k", "hat", "b", "c"]].round(3).to_string(index=False))
    print("CẤP MÃ:"); print(ca[["cell", "sigma", "k", "hat", "b", "c"]].round(3).to_string(index=False))
