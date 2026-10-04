"""Đóng gói sản phẩm để nộp / chia sẻ. Tạo hai file zip trong dist/:

1. Doc_tin_co_nen_mua_code_v1.0.zip — CHỈ CODE: src/, tests/, README, MODEL_CARD, HƯỚNG DẪN, requirements, tài liệu thiết kế,
   data/gold/ (nhãn trọng tài dùng để kiểm chứng nhãn LLM).
   Người nhận đặt dữ liệu gốc vào `data final/` rồi chạy `python -m src.run_all` để sinh lại mọi kết quả.
2. Doc_tin_co_nen_mua_v1.0.zip — ĐẦY ĐỦ: gói code + báo cáo (bao_cao/), kết quả (reports/, figures/, dashboard/),
   công cụ gán nhãn (crawl_data/gan_nhan).
Cả hai KHÔNG gồm dữ liệu gốc (data final/) và dữ liệu trung gian (data/clean/): dữ liệu gốc do nhóm giữ;
data/clean/ và outputs/ sinh lại bằng src.run_all (thư mục được tạo tự động, xem src/config.py).
Dùng:  python -m src.package
"""
import zipfile
from pathlib import Path

from src.config import ROOT

VERSION = "v1.0"
FILES = ["README.md", "MODEL_CARD.md", "HUONG_DAN_SU_DUNG.md", "requirements.txt"]
CODE_DIRS = ["src", "tests", "pipeline_thiet_ke_du_an", "data/gold"]   # data/gold: nhãn trọng tài + tập mẫu (đầu vào, không sinh lại được)
FULL_DIRS = CODE_DIRS + ["bao_cao", "outputs/reports", "outputs/figures", "outputs/dashboard", "crawl_data/gan_nhan"]
SKIP = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
DATA_README = """Đặt dữ liệu gốc vào thư mục này, giữ nguyên cấu trúc:
  keyword/  law/  company/  (các file *_classified.csv đã gán nhãn)
  Stock/Company_Profile_BDS_REAL_DATA__1_.xlsx  và  Stock/Bang_Trust_Score_CafeF_Kenh14.xlsx
Danh sách file chính xác: NEWS_FILES trong src/config.py. Sau đó chạy:  py -3.12 -m src.run_all
"""


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
        z.writestr(f"{name}/data final/DAT_DU_LIEU_VAO_DAY.txt", DATA_README)
    print(f"{out.name}: {len(paths)} file ({size / 1e6:.1f} MB trước nén) -> {out.stat().st_size / 1e6:.1f} MB")
    return out


def main():
    build(f"Doc_tin_co_nen_mua_code_{VERSION}", CODE_DIRS)
    build(f"Doc_tin_co_nen_mua_{VERSION}", FULL_DIRS)


if __name__ == "__main__":
    main()
