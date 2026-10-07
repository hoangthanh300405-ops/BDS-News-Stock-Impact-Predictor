"""Gộp mọi kết quả của hướng nhà đầu tư nhỏ lẻ (vòng 1, đã chốt trước) thành MỘT bảng đầy đủ.

Nguồn: retail_accuracy_f1.csv (accuracy/F1), retail_models.csv (AUC), retail_investors.csv (lãi, CI, số quý thắng).
Đầu ra: outputs/reports/BANG_KET_QUA_DAY_DU.csv + .md
Dùng:  python -m src.retail.full_table   (sau src.retail.run_retail)
"""
import re

import pandas as pd

from src.config import ROOT

REP = ROOT / "outputs" / "reports"


def rule_key(s):
    m = re.search(r"top (\d+)%", s)
    return f"top {m.group(1)}%" if m else ("p > 0,5" if "0,5" in s else s)


def build():
    acc = pd.read_csv(REP / "retail_accuracy_f1.csv")
    auc = pd.read_csv(REP / "retail_models.csv")[["target", "mô hình", "AUC"]]
    inv = pd.read_csv(REP / "retail_investors.csv")
    inv = inv[inv["nhà đầu tư"].str.startswith("C · ")].copy()
    inv["mô hình"] = inv["nhà đầu tư"].str.split(" · ").str[1]
    inv["rk"] = inv["nhà đầu tư"].str.split(" · ").str[2].map(rule_key)
    acc["rk"] = acc["quy tắc quyết định"].map(rule_key)
    base_pnl = pd.read_csv(REP / "retail_investors.csv").query("`nhà đầu tư`.str.startswith('A')", engine="python")
    t = acc.merge(auc, on=["target", "mô hình"], how="left").merge(
        inv[["target", "mô hình", "rk", "lãi TB/lệnh", "CI95 chênh (khối phiên)", "số quý thắng A (/6)"]],
        on=["target", "mô hình", "rk"], how="left")
    for tgt, pnl in base_pnl.set_index("target")["lãi TB/lệnh"].items():
        t.loc[(t["target"] == tgt) & (t["mô hình"] == "Mốc: luôn làm theo tin"), "lãi TB/lệnh"] = pnl
    t["target"] = t["target"].map({"tuyet_doi": "Có lãi sau phí", "tuong_doi": "Thắng rổ BĐS"})
    t = t.rename(columns={"quy tắc quyết định": "quy tắc", "tỷ lệ khuyên làm theo": "khuyên làm theo",
                          "precision (làm theo)": "precision", "recall (làm theo)": "recall",
                          "F1 (làm theo)": "F1 làm theo", "F1 (không làm theo)": "F1 không làm theo",
                          "lãi TB/lệnh": "lãi TB/lệnh làm theo", "CI95 chênh (khối phiên)": "CI95 precision − A",
                          "số quý thắng A (/6)": "số quý thắng A"})
    cols = ["target", "mô hình", "quy tắc", "khuyên làm theo", "AUC", "accuracy", "balanced acc", "precision", "recall",
            "F1 làm theo", "F1 không làm theo", "macro-F1", "lãi TB/lệnh làm theo", "CI95 precision − A", "số quý thắng A"]
    t = t[cols]
    t.to_csv(REP / "BANG_KET_QUA_DAY_DU.csv", index=False, encoding="utf-8-sig")
    f = t.copy()
    pct = ["khuyên làm theo", "accuracy", "balanced acc", "precision", "recall"]
    for c in pct:
        f[c] = (f[c] * 100).map(lambda v: f"{v:.1f}%".replace(".", ","))
    for c in ["AUC", "F1 làm theo", "F1 không làm theo", "macro-F1"]:
        f[c] = f[c].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}".replace(".", ","))
    f["lãi TB/lệnh làm theo"] = f["lãi TB/lệnh làm theo"].map(lambda v: "—" if pd.isna(v) else f"{v * 100:+.2f}%".replace(".", ","))
    f["số quý thắng A"] = f["số quý thắng A"].map(lambda v: "—" if pd.isna(v) else f"{int(v)}/6")
    f = f.fillna("—")
    (REP / "BANG_KET_QUA_DAY_DU.md").write_text(f.to_markdown(index=False), encoding="utf-8")
    return f


if __name__ == "__main__":
    print(build().to_markdown(index=False))
