"""Mã hóa ngữ nghĩa bài báo bằng multilingual-e5-small (đa ngôn ngữ, có tiếng Việt, chạy CPU).

Đầu vào: tiêu đề + mô tả (≤ 300 ký tự). Vector chuẩn hóa L2, lưu float16.
Không dùng giá hay nhãn -> không có rò rỉ; dùng chung cho mọi fold (mô hình ngôn ngữ huấn luyện sẵn, cố định).
Đầu ra: data/clean/article_emb.npy + data/clean/article_emb_ids.parquet
Dùng:  python -m src.nlp.embed
"""
import time

import numpy as np
import pandas as pd

from src.config import CLEAN

MODEL = "intfloat/multilingual-e5-small"


def main():
    from sentence_transformers import SentenceTransformer
    art = pd.read_parquet(CLEAN / "articles_clean.parquet", columns=["article_id", "title", "description"])
    text = ("query: " + art["title"].fillna("") + ". " + art["description"].fillna("").str[:300]).tolist()
    m = SentenceTransformer(MODEL, device="cpu")
    t0 = time.time()
    E = m.encode(text, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    print(f"{len(text):,} bài, {E.shape[1]} chiều, {time.time() - t0:.0f} giây")
    np.save(CLEAN / "article_emb.npy", E.astype(np.float16))
    art[["article_id"]].to_parquet(CLEAN / "article_emb_ids.parquet", index=False)


if __name__ == "__main__":
    main()
