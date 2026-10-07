"""Gom bài báo thành CÂU CHUYỆN theo từng mã (trụ cột 1d).

Với mỗi mã, xét các bài theo thời gian đăng; một bài thuộc câu chuyện cũ nếu độ tương đồng ngữ nghĩa (cosine của vector
e5) với một bài cùng mã trong 3 ngày trước ≥ THR; ngược lại mở câu chuyện mới. Chỉ dùng văn bản và thời gian đăng
(không dùng giá/nhãn) -> không rò rỉ; mỗi bài chỉ so với bài ĐĂNG TRƯỚC nó.
Ngưỡng THR chọn bằng cách đọc các cặp quanh ngưỡng (xem --inspect), không dựa vào kết quả dự báo.
Đầu ra: data/clean/stories.parquet (article_id, ticker, story_id, story_rank, story_age_h)
Dùng:  python -m src.nlp.stories [--inspect]
"""
import sys

import numpy as np
import pandas as pd

from src.config import CLEAN

THR, WINDOW_H = 0.92, 72


def load():
    E = np.load(CLEAN / "article_emb.npy").astype(np.float32)
    ids = pd.read_parquet(CLEAN / "article_emb_ids.parquet")["article_id"]
    pos = pd.Series(np.arange(len(ids)), index=ids)
    atl = pd.read_parquet(CLEAN / "article_ticker_label.parquet", columns=["article_id", "ticker"]).drop_duplicates()
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "published_at", "title"])
    art["t"] = pd.to_datetime(art["published_at"], utc=True)
    d = atl.merge(art[["article_id", "t", "title"]], on="article_id").dropna(subset=["t"])
    d["row"] = d["article_id"].map(pos)
    return E, d.dropna(subset=["row"]).astype({"row": int}).sort_values(["ticker", "t"]).reset_index(drop=True)


def build(E, d, thr=THR, collect=None):
    sid, rank, age = np.zeros(len(d), int), np.zeros(len(d), int), np.zeros(len(d))
    next_id = 0
    for _, g in d.groupby("ticker", sort=False):
        idx, ts, rows = g.index.to_numpy(), g["t"].to_numpy(), g["row"].to_numpy()
        start = {}
        for k, i in enumerate(idx):
            lo = np.searchsorted(ts, ts[k] - np.timedelta64(WINDOW_H, "h"))
            prev = np.arange(lo, k)
            best, sim = -1, -1.0
            if len(prev):
                sims = E[rows[prev]] @ E[rows[k]]
                j = int(np.argmax(sims)); best, sim = prev[j], float(sims[j])
                if collect is not None:
                    collect.append((sim, g["title"].iat[k], g["title"].iat[best]))
            if sim >= thr:
                s = sid[idx[best]]
                sid[i], rank[i] = s, rank[idx[best]] + 1
            else:
                s = next_id; next_id += 1
                sid[i], rank[i] = s, 1
                start[s] = ts[k]
            age[i] = (ts[k] - start.get(sid[i], ts[k])) / np.timedelta64(1, "h") if sid[i] in start else 0
    return d.assign(story_id=sid, story_rank=rank, story_age_h=age)


def main():
    E, d = load()
    if "--inspect" in sys.argv:
        pairs = []
        build(E, d.head(20000), thr=9, collect=pairs)
        p = pd.DataFrame(pairs, columns=["sim", "bài sau", "bài trước"])
        for lo, hi in ((0.84, 0.86), (0.87, 0.89), (0.89, 0.91), (0.92, 0.94)):
            print(f"\n=== cosine {lo}–{hi}")
            for r in p[(p.sim >= lo) & (p.sim < hi)].sample(6, random_state=0).itertuples():
                print(f"  {r.sim:.3f} | {r._2[:80]}  <->  {r._3[:80]}")
        return
    s = build(E, d)
    s[["article_id", "ticker", "story_id", "story_rank", "story_age_h"]].to_parquet(CLEAN / "stories.parquet", index=False)
    n_story = s["story_id"].nunique()
    print(f"{len(s):,} cặp (bài, mã) -> {n_story:,} câu chuyện | TB {len(s) / n_story:.2f} bài/câu chuyện | "
          f"bài mở đầu câu chuyện: {(s['story_rank'] == 1).mean():.1%}")


if __name__ == "__main__":
    main()
