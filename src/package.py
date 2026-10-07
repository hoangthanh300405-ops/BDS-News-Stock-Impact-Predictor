"""Đóng gói project để nộp / chia sẻ.

Tạo hai file zip trong dist/:

1. Doc_tin_co_nen_mua_code_v1.0.zip
   - source code
   - tests
   - README
   - documentation
   - requirements
   - data/manual dùng cho kiểm chứng nhãn

2. Doc_tin_co_nen_mua_v1.0.zip
   - toàn bộ nội dung của bản code
   - các kết quả đã được sinh trong outputs/ nếu có

data/clean/ và outputs/ là dữ liệu/kết quả được sinh bởi pipeline
và không phải dữ liệu đầu vào bắt buộc phải commit vào repository.

Dùng:
    python -m src.package
"""
import zipfile
from pathlib import Path

from src.config import ROOT

VERSION = "v1.0"

FILES = [
    "README.md",
    "requirements.txt",
]

CODE_DIRS = [
    "src",
    "tests",
    "docs",
    "data/manual",
]

FULL_DIRS = CODE_DIRS + [
    "data/classified",
    "data/stock",
    "data/crawl",
    "outputs/reports",
    "outputs/figures",
    "outputs/predictions",
    "outputs/dashboard",
]

SKIP = {
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
}

def collect(dirs):
    paths = [ROOT / f for f in FILES]
    for d in dirs:
        paths += sorted(p for p in (ROOT / d).rglob("*") if p.is_file() and not SKIP & set(p.parts)
                        and not p.name.startswith("~$") and not p.name.startswith("_preview") and p.suffix != ".pyc")
    return paths


def build(name, dirs):
    out = ROOT / "dist" / f"{name}.zip"
    out.parent.mkdir(exist_ok=True)
    paths = collect(dirs)
    size = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, Path(name) / p.relative_to(ROOT))
            size += p.stat().st_size
    print(f"{out.name}: {len(paths)} file ({size / 1e6:.1f} MB trước nén) -> {out.stat().st_size / 1e6:.1f} MB")
    return out


def main():
    build(f"Doc_tin_co_nen_mua_code_{VERSION}", CODE_DIRS)
    build(f"Doc_tin_co_nen_mua_{VERSION}", FULL_DIRS)


if __name__ == "__main__":
    main()
