"""P1 + P2 — Hợp nhất, làm sạch tin; chuẩn hóa nhãn (§6 P1, P2; quy tắc định tuyến §3.3).

Đầu vào : 9 file *_classified.csv trong data final/ (keyword, law, company × cafef, kenh14, vietstock)
Đầu ra  (data/clean/, parquet):
  articles_clean        1 dòng / bài            article_id, url, source, published_at, session, title, ..., origin_files
  article_label         nhãn CẤP NGÀNH          article_id, typ ∈ {keyword, law, suy_ra}, label_run, event_type, nhom_tin, impact_score
  article_ticker_label  nhãn CẤP MÃ             article_id, ticker, label_run, event_type, nhom_tin, impact_score, muc_do_ma, ly_do_ma
Cần chạy P4 trước (dùng trading_calendar để gán phiên).
"""
import hashlib
import json
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

from src.config import CLEAN, KEYWORD_GROUPS, MIN_CONTENT_CHARS, NEWS_FILES, NHOM_TIN, RAW
from src.data.sessions import parse_published, to_session

PRICE_REPORT_RE = re.compile(r"(?:cổ phiếu|giá|mã)[^.]{0,40}(?:tăng|giảm|lên|xuống|trần|sàn)[^.]{0,20}\d+[,.]?\d*\s*%",
                             re.IGNORECASE)

NEWS_COLS = ["url", "source", "category", "published_at", "title", "description", "content",
             "event_type", "impact_score", "label_model", "label_run", "su_kien_chinh"]


def canonical_url(u: str) -> str:
    p = urlsplit(str(u).strip())
    path = p.path.rstrip("/") or "/"
    return urlunsplit((p.scheme.lower() or "https", p.netloc.lower().removeprefix("www."), path, "", ""))


def article_id(u: str) -> str:
    return hashlib.sha1(canonical_url(u).encode("utf-8")).hexdigest()[:16]


def nfc(s):
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else s


def text_hash(s) -> str:
    s = re.sub(r"\s+", " ", nfc(s or "")).strip().lower()[:2000]
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16] if s else ""


def load_file(typ, src, rel):
    extra = ["ticker", "match_keyword", "impact_theo_ma"] if typ == "company" else ["keyword_groups"]
    df = pd.read_csv(RAW / rel, usecols=NEWS_COLS + extra, dtype=str, encoding="utf-8-sig", low_memory=False)
    df["typ"], df["src"] = typ, src
    df["published_at"] = parse_published(df["published_at"])       # theo từng file (T: múi giờ)
    df["impact_score"] = pd.to_numeric(df["impact_score"], errors="coerce")
    df["article_id"] = df["url"].map(article_id)
    return df


def clean_groups(s) -> str:
    toks = [t.strip() for t in str(s).split(",")] if isinstance(s, str) else []
    return ",".join(sorted({t for t in toks if t in KEYWORD_GROUPS}))


def parse_theo_ma(s):
    try:
        d = json.loads(s) if isinstance(s, str) and s.strip() else {}
    except json.JSONDecodeError:
        return {}
    return d if isinstance(d, dict) else {}


def build(verbose=True):
    calendar = pd.read_parquet(CLEAN / "trading_calendar.parquet")["date"]
    frames = {k: load_file(*k, rel) for k, rel in NEWS_FILES.items()}
    report = {}

    # ---------------- nhãn cấp ngành (keyword/, law/) ----------------
    ind = pd.concat([f for (typ, _), f in frames.items() if typ != "company"], ignore_index=True)
    before = len(ind)
    ind = ind.drop_duplicates(["article_id", "typ", "label_run"])        # vd. 724 URL lặp trong law/cafef
    report["trùng trong cùng file (cấp ngành)"] = before - len(ind)

    # ---------------- nhãn cấp mã (company/) ----------------
    com = pd.concat([f for (typ, _), f in frames.items() if typ == "company"], ignore_index=True)
    before = len(com)
    com = com.drop_duplicates(["article_id", "ticker"])
    report["trùng (bài × mã)"] = before - len(com)
    theo_ma = com["impact_theo_ma"].map(parse_theo_ma)

    def from_json(row_map, ticker, key):
        v = row_map.get(ticker)
        return v.get(key) if isinstance(v, dict) else (v if key == "impact_score" and v is not None else None)

    js_score = [from_json(m, t, "impact_score") for m, t in zip(theo_ma, com["ticker"])]
    js_score = pd.to_numeric(pd.Series(js_score, index=com.index), errors="coerce")
    has_js = js_score.notna()
    report["điểm dòng khớp impact_theo_ma[mã]"] = f"{(js_score[has_js] == com.loc[has_js, 'impact_score']).mean():.1%} (trên {has_js.sum()} dòng có JSON)"

    atl = pd.DataFrame({
        "article_id": com["article_id"], "ticker": com["ticker"], "src": com["src"],
        "label_model": com["label_model"], "label_run": com["label_run"],
        "event_type": com["event_type"], "nhom_tin": com["event_type"].map(NHOM_TIN).fillna("khac"),
        "impact_score": com["impact_score"],
        "muc_do_ma": [from_json(m, t, "muc_do_anh_huong") for m, t in zip(theo_ma, com["ticker"])],
        "ly_do_ma": [from_json(m, t, "ly_do") for m, t in zip(theo_ma, com["ticker"])],
        "match_keyword": com["match_keyword"],
    })
    n_tick = atl.groupby("article_id")["ticker"].transform("nunique")
    atl["n_tickers_in_article"] = n_tick
    # Nhãn chỉ thuật lại biến động giá đã xảy ra ("Cổ phiếu X tăng 6,9%"): phản ánh lợi suất quá khứ,
    # không phải thông tin mới -> gắn cờ làm biến kiểm soát / ablation (khoảng 3% số dòng)
    atl["price_report"] = atl["ly_do_ma"].fillna("").str.contains(PRICE_REPORT_RE)
    report["nhãn cấp mã chỉ thuật lại biến động giá"] = f"{atl['price_report'].mean():.1%}"

    # ---------------- articles_clean: 1 dòng / bài ----------------
    allrows = pd.concat([ind, com], ignore_index=True)
    allrows["content_len"] = allrows["content"].fillna("").str.len()
    allrows = allrows.sort_values("content_len", ascending=False)          # giữ bản content dài nhất
    origin = allrows.groupby("article_id")["typ"].agg(lambda s: ",".join(sorted(set(s))))
    groups = ind.groupby("article_id")["keyword_groups"].agg(lambda s: clean_groups(",".join(s.dropna())))
    art = allrows.drop_duplicates("article_id").set_index("article_id")
    art = art[["url", "src", "category", "published_at", "title", "description", "content", "content_len",
               "su_kien_chinh", "event_type"]].copy()
    for c in ("title", "description", "content", "su_kien_chinh"):
        art[c] = art[c].map(nfc)
    art["canonical_url"] = art["url"].map(canonical_url)
    art["origin_files"] = origin.reindex(art.index)
    art["keyword_groups"] = groups.reindex(art.index).fillna("")
    art["text_hash"] = art["content"].map(text_hash)
    art["dup_text"] = art.duplicated("text_hash", keep=False) & art["text_hash"].ne("")
    art["short_content"] = art["content_len"] < MIN_CONTENT_CHARS
    art["session"] = to_session(art["published_at"], calendar)
    art = art.reset_index()

    # ---------------- article_label: nhãn cấp ngành + nhãn suy ra (§3.3) ----------------
    lab = pd.DataFrame({
        "article_id": ind["article_id"], "typ": ind["typ"], "src": ind["src"],
        "label_model": ind["label_model"], "label_run": ind["label_run"],
        "event_type": ind["event_type"], "nhom_tin": ind["event_type"].map(NHOM_TIN).fillna("khac"),
        "impact_score": ind["impact_score"],
    })
    only_company = set(atl["article_id"]) - set(lab["article_id"])
    inferred = (atl[atl["article_id"].isin(only_company)]
                .groupby("article_id")
                .agg(src=("src", "first"), label_model=("label_model", "first"), label_run=("label_run", "first"),
                     event_type=("event_type", "first"), nhom_tin=("nhom_tin", "first"),
                     impact_score=("impact_score", "mean"))
                .reset_index().assign(typ="suy_ra"))
    lab = pd.concat([lab, inferred[lab.columns]], ignore_index=True)
    lab["nhan_suy_ra"] = lab["typ"].eq("suy_ra")
    lab["impact_3lop"] = np.sign(lab["impact_score"]).astype("Int64")

    # loại bài quá ngắn khỏi mọi bảng nhãn (giữ trong articles_clean với cờ để truy vết)
    short = set(art.loc[art["short_content"], "article_id"])
    lab = lab[~lab["article_id"].isin(short)].reset_index(drop=True)
    atl = atl[~atl["article_id"].isin(short)].reset_index(drop=True)

    # báo cáo đồng thuận keyword vs law trên cùng một bài
    both = lab[lab["typ"].isin(["keyword", "law"])].pivot_table(index="article_id", columns="typ", values="impact_score")
    both = both.dropna()
    report["bài có nhãn ở cả keyword và law"] = f"{len(both)} | cùng dấu: {(np.sign(both['keyword']) == np.sign(both['law'])).mean():.1%}"

    CLEAN.mkdir(parents=True, exist_ok=True)
    art.to_parquet(CLEAN / "articles_clean.parquet", index=False)
    lab.to_parquet(CLEAN / "article_label.parquet", index=False)
    atl.to_parquet(CLEAN / "article_ticker_label.parquet", index=False)

    if verbose:
        print(f"[P1] articles_clean: {len(art)} bài | nguồn: {art['src'].value_counts(normalize=True).round(3).to_dict()}")
        print(f"[P1] bài ngắn < {MIN_CONTENT_CHARS} ký tự: {int(art['short_content'].sum())} | trùng nội dung khác URL: {int(art['dup_text'].sum())} "
              f"| không gán được phiên: {int(art['session'].isna().sum())}")
        print(f"[P1] article_label: {len(lab)} dòng | theo loại: {lab['typ'].value_counts().to_dict()}")
        print(f"[P1] article_ticker_label: {len(atl)} dòng (bài × mã) | {atl['ticker'].nunique()} mã")
        for k, v in report.items():
            print(f"[P1] {k}: {v}")
    return art, lab, atl


if __name__ == "__main__":
    build()
