"""Cấu hình chung của pipeline: đường dẫn, hằng số và quy ước."""
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"

RAW = DATA / "classified"
CLEAN = DATA / "clean"
MODEL_INPUT = DATA / "model_input"

for _d in (
    CLEAN,
    DATA / "manual",
    *(ROOT / "outputs" / s
      for s in ("reports", "figures", "predictions", "dashboard"))
):
    _d.mkdir(parents=True, exist_ok=True)

NEWS_FILES = {
    ("keyword", "cafef"): "cafef_keyword_classified.csv",
    ("keyword", "kenh14"): "kenh14_keywords_classified.csv",
    ("keyword", "vietstock"): "vietstock_keyword_classified.csv",

    ("law", "cafef"): "cafef_law_classified.csv",
    ("law", "kenh14"): "kenh14_law_classified.csv",
    ("law", "vietstock"): "vietstock_law_classified.csv",

    ("company", "cafef"): "cafef_company_classified.csv",
    ("company", "kenh14"): "kenh14_company_classified.csv",
    ("company", "vietstock"): "vietstock_company_classified.csv",
}

PRICE_XLSX = (
    DATA / "stock" /
    "Company_Profile_BDS_REAL_DATA__1_.xlsx"
)

TRUST_XLSX = (
    DATA / "stock" /
    "Bang_Trust_Score.xlsx"
)

TZ = "Asia/Ho_Chi_Minh"
ATC_CUTOFF = time(14, 45)          # tin đăng TỪ 14:45 trở đi -> phiên kế tiếp
HORIZONS = (1, 5)                  # h = 1 chính, h = 5 = một tuần giao dịch
WARMUP_END = "2024-03-29"          # 2024Q1 chỉ dùng để khởi động; mẫu đầu tiên 01/04/2024
NEWS_END = "2026-09-09"            # ngày cuối có tin

Z_THRESHOLD = 0.5                  # ngưỡng 3 lớp ±0.5σ
BIG_MOVE_Z = 1.0                   # target phụ big_move 
ATTN_Z = 1.0                       # hướng B: thanh khoản đột biến = log giá trị GD vượt TB 20 phiên hơn 1σ
SIGMA_B_WINDOW = 20                # σ của benchmark ngành
SIGMA_A_WINDOW, SIGMA_A_MIN = 60, 40
# Kém thanh khoản: >30% phiên giá đứng yên HOẶC giá trị giao dịch trung vị < 0,5 tỷ VND/phiên (60 phiên trước t).
# Dùng giá trị (tỷ VND) chứ không dùng số cổ phiếu: 100k CP giá 5k đồng khác xa 100k CP giá 100k đồng.
ILLIQ_WINDOW, ILLIQ_ZERO_RET, ILLIQ_MIN_VALUE_BN = 60, 0.30, 0.5
MIN_CONTENT_CHARS = 200

# event_type (14 loại) -> nhom_tin (6 nhóm + khac).
NHOM_TIN = {
    "MARKET": "thi_truong", "ANALYST": "thi_truong",
    "PROJECT": "du_an_ha_tang",
    "POLICY": "chinh_sach",
    "LEGAL": "phap_ly",
    "EARNINGS": "tai_chinh_dn", "CAPITAL_RAISE": "tai_chinh_dn", "M_AND_A": "tai_chinh_dn",
    "DEBT_BOND": "tai_chinh_dn", "MANAGEMENT": "tai_chinh_dn", "INSIDER_TRADING": "tai_chinh_dn",
    "INTEREST_RATE": "lai_suat_tin_dung", "BANKING_CREDIT": "lai_suat_tin_dung",
    "OTHER": "khac",
}

# Từ vựng keyword_groups hợp lệ; token khác là rác và bị bỏ ở P1
KEYWORD_GROUPS = {
    "bds_chung", "qd_ban_hanh", "bds_nha", "qd_thu_tuc", "can_ho", "luat_van_ban_co_quan", "khu_dan_cu",
    "nha_o", "bds_dat", "gia", "luat_van_ban", "nguon_cung", "mua_ban", "nhu_cau", "qd_tai_chinh",
    "luat_thu_tuc", "dat_nen", "mo_ban", "biet_thu", "nha_pho", "giao_dich", "cho_thue", "shophouse",
    "bds_giay_to", "lien_ke", "nghi_duong", "ban_giao", "nha_xuong",
}

# Vector phơi nhiễm phân khúc từ Loai_hinh_BDS (bước 2)
EXPOSURE_KEYWORDS = {
    "can_ho": ["căn hộ", "chung cư"],
    "dat_nen": ["đất nền"],
    "thap_tang": ["thấp tầng", "nhà phố", "biệt thự"],
    "kcn": ["khu công nghiệp", "kcn", "nhà xưởng"],
    "nghi_duong": ["nghỉ dưỡng"],
}
