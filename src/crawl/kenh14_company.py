# -*- coding: utf-8 -*-
# KENH14 COMPANY — ĐỘC LẬP, NHÚNG SẴN 83 CÔNG TY
# Dán toàn bộ script vào một cell Google Colab rồi chạy.

import importlib.util
import subprocess
import sys

for module, package in [
    ("requests", "requests"),
    ("bs4", "beautifulsoup4"),
    ("dateutil", "python-dateutil"),
]:
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", package
        ])

import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import unicodedata

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from dateutil.parser import isoparse


# ============================================================
# CẤU HÌNH
# ============================================================

BASE = "https://kenh14.vn"

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/crawl data/kenh14_company"
)

VN_TZ = timezone(timedelta(hours=7))
DATE_START = datetime(2024, 1, 1, tzinfo=VN_TZ)
DATE_END_EXCLUSIVE = datetime(2026, 9, 10, tzinfo=VN_TZ)

MAX_PAGES = 1000                  # Mỗi chuyên mục / mỗi lượt
MAX_ARTICLE_ATTEMPTS = 3          # Tổng cộng, kể cả chạy lại
REQUEST_DELAY = 1.5
MAX_HTML_BYTES = 8 * 1024 * 1024
MIN_FREE_MB = 100

SELECTED_COMPANY = None           # None = tất cả; hoặc "NVL", "VHM"...
SELECTED_CATEGORY = "all"         # all hoặc một tên trong CATEGORIES

SITE = 'kenh14'
OLD_PAGE_STREAK = 3
CATEGORIES = {'xa-hoi': 215142, 'money14': 215238}

PROFILE_FIELDS = [
    "Ma_CK",
    "San_niem_yet",
    "Nganh",
    "Loai_hinh_BDS",
    "Phan_khuc_gia",
    "Von_dieu_le_ty_VND",
    "So_luong_CP_luu_hanh",
    "Menh_gia_VND",
    "Ngay_niem_yet",
    "CEO",
    "Dia_chi",
    "Website",
    "Mo_ta_nganh_nghe",
    "Gia_dong_cua_gan_nhat_nghin_VND",
    "Von_hoa_thi_truong_ty_VND",
    "Nha_thau_xay_dung",
    "Cong_ty_lien_quan",
    "Ten_Cong_Ty",
    "Khu_vuc_hoat_dong",
]


# ============================================================
# DỮ LIỆU NHÚNG SẴN — 83 DÒNG × 19 CỘT
# Mỗi dòng dùng ký tự | để phân cách.
# Giữ số liệu dưới dạng chuỗi như bảng gốc, không tự đổi đơn vị.
# ============================================================

COMPANY_DATA = r"""
AGG|HOSE|Bất động sản|Căn hộ chung cư / Khu đô thị|Trung cấp - Cao cấp|1625|162528081|10000|09/01/2020|Mr. Nguyễn Bá Sáng|Số 60 Nguyễn Đình Chiểu - P. Tân Định - Tp. Hồ Chí Minh|https://angia.com.vn/|- Kinh doanh Bất động sản. - Dịch vụ môi giới Bất động sản, dịch vụ quản lý Bất động sản, sàn Bất động sản. - Tư vấn quản lý, quảng cáo, nghiên cứu thị trường và thăm dò dư luận, tổ chức giới thiệu và xúc tiến thương mại Bất động sản. - Xây dựng nhà các loại, xây dựng công trình kỹ thuật dân dụng khác, xây dựng công trình đường sắt và đường bộ, xây dựng công trình công ích, phá dỡ, san lấp mặt bằng.|10,30|1.674|Coteccons, Ricons, Newtecons|Westgate, An Gia Investment, MT Eastmark City|CTCP Phát triển Bất động sản An Gia|TP. Hồ Chí Minh & các tỉnh lân cận (Long An, Bình Dương)
API|HNX|Bất động sản|BĐS nghỉ dưỡng / Nhà ở đô thị|Trung cấp|841|84083976|10000|13/09/2010|Ms. Nguyễn Phương Dung|Tầng 3 Tòa nhà Grand Plaza - Số 117 Trần Duy Hưng - P. Yên Hòa - Tp. Hà Nội|https://apeci.com.vn/|- Dịch vụ tư vấn đầu tư trong và ngoài nước - Tư vấn môi giới, kinh doanh bất động sản, tư vấn quản lý kinh doanh, chiến lược kinh doanh - Xây dựng dân dụng, giao thông, thủy lợi...|5,50|462|Apec Construction, Vinaconex|Apec Group, IDJ, APS|CTCP Đầu tư Châu Á - Thái Bình Dương|Hà Nội, Bắc Ninh, Bắc Giang, Huế, Phú Thọ
BAX|HNX|Bất động sản|Khu công nghiệp / Hạ tầng KCN|Không áp dụng (BĐS KCN)|82|8200000|10000|24/03/2017|Mr. Trần Trung Tuấn|Đường số 2A - KCN Bàu Xéo - P. Trảng Bom - Tp. Đồng Nai|http://www.bauxeo.com.vn|- Đầu tư xây dựng và kinh doanh cơ sở hạ tầng khu công nghiệp. - Đầu tư xây dựng và kinh doanh khu dân cư, khu dịch vụ. - Xây dựng các hạng mục công trình trong khu công nghiệp, khu dân cư và khu dịch vụ. - Dịch vụ tư vấn đầu tư và các dịch vụ kinh doanh khác phục vụ khách hàng đầu tư. - Kinh doanh nước sạch... - Bán buôn nhiên liệu rắn, lỏng, khí và các sản phẩm liên quan.|30,00|246|Tổng Công ty Xây dựng số 1 (CC1)|Sonadezi (SNZ), Bàu Xéo|CTCP Đầu tư Xây dựng và Cơ sở Hạ tầng Bàu Xéo|Đồng Nai (Khu công nghiệp Bàu Xéo)
BCE|HOSE|Bất động sản|Xây dựng hạ tầng / Khu dân cư|Trung cấp|420|41999846|10000|28/06/2010|Mr. Nguyễn Kim Tiên|Tầng 7 - Tòa nhà WTC Tower - Số 1 đường Hùng Vương - P. Bình Dương - Tp. Hồ Chí Minh|https://becamexbce.com.vn|- Thi công xây dựng, tư vấn thiết kế các công trình: dân dụng, công nghiệp, giao thông, san lấp mặt bằng, điện trung thế, hạ thế, trạm biến áp, điện chiếu sáng, trang trí nội, ngoại thất. - Sản xuất và lắp đặt cấu kiện bằng thép, khung nhà tiền chế, cấu kiện bê tông cốt thép đúc sẵn. - Sản xuất kinh doanh vật liệu xây dựng.|5,62|236|Becamex BCE, các nhà thầu khu vực Bình Dương|Becamex IDC (BCM)|CTCP Xây dựng và Giao thông Bình Dương|Bình Dương & các tỉnh Đông Nam Bộ
BCM|HOSE|Bất động sản|Khu công nghiệp / Hạ tầng đô thị|Không áp dụng (BĐS KCN)|10350|1035000000|10000|31/08/2020|Mr. Nguyễn Văn Hùng|Tầng 10 - Tòa nhà mPlaza Saigon - Số 39 Lê Duẩn - P. Sài Gòn - Tp. Hồ Chí Minh|https://becamex.com.vn/|- Phát triển khu công nghiệp, BĐS dân cư - thương mại. - Hạ tầng dịch vụ. - Cung cấp dịch vụ y tế, giáo dục. - Hạ tầng kỹ thuật. - Hoạt động xây dựng. - Lắp đặt và kinh doanh điện. - Khai thác cảng. - Hạ tầng xã hội.|38,85|40.210|Becamex Tokyu Construction, các nhà thầu nội bộ|Becamex Tokyu, IJC, TDC, UDJ, Warburg Pincus (BW Industrial)|Tổng Công ty Đầu tư và Phát triển Công nghiệp - CTCP (Becamex IDC)|Bình Dương, Bình Phước, Vùng Kinh tế trọng điểm phía Nam
CCL|HOSE|Bất động sản|Khu đô thị / Đất nền nhà ở|Bình dân - Trung cấp|655|65538283|10000|03/03/2011|Mr. Dương Thế Nghiêm|Số 02 - Lô KTM-06 - Đường số 6 - Khu đô thị 5A - P. Phú Lợi - Tp. Cần Thơ|https://beta.dothi5a.com/|- Hoạt động kiến trúc và tư vấn kỹ thuật có liên quan. - Tư vấn đầu tư, thiết kế, quản lý dự án. - Đầu tư các dự án khu dân cư. - Tư vấn, kinh doanh bất động sản. - Xây dựng, lắp đặt, sửa chữa các công trình dân dụng chuyên nghiệp, hạ tầng khu đô thị và công nghiệp. - Hoàn thiện công trình xây dựng. - Cho thuê showroom, mặt bằng kinh doanh để tổ chức các sự kiện. - Sản xuất, mua bán vật liệu xây dựng. - Mua bán các thiết bị lắp đặt khác trong xây dựng.|3,49|229|Nhà thầu địa phương Miền Tây, CCL Cons|Đầu tư và Phát triển Đô thị Tây Nam|CTCP Đầu tư và Phát triển Đô thị Tây Nam|Cần Thơ & các tỉnh Đồng bằng sông Cửu Long
CDC|HOSE|Bất động sản|Xây dựng hạ tầng / BĐS dân dụng|Trung cấp|1055|105545322|10000|13/09/2010|Mr. Nguyễn Ngọc Bền|Số 328 Võ Văn Kiệt - P. Cầu Ông Lãnh - Tp. Hồ Chí Minh|http://www.chuongduongcorp.vn|- Xây dựng các công trình dân dụng, công nghiệp. - Xây dựng các công trình giao thông: cầu, đường, bến cảng, sân bay. - Xây dựng các công trình đường dây và trạm biến thế điện từ 0.4kv đến 110kv….|19,30|2.037|Chương Dương (In-house contractor)|Chương Dương Corp|CTCP Chương Dương|TP. Hồ Chí Minh & các tỉnh Nam Bộ
CEO|HNX|Bất động sản|BĐS nghỉ dưỡng / Khu đô thị|Trung cấp - Cao cấp|5958|595775755|10000|29/09/2014|Mr. Trần Trung Kết|Tầng 5 - Tháp CEO - HH2-1 - Đô thị mới Mễ Trì Hạ - Đường Phạm Hùng - P. Từ Liêm - Tp. Hà Nội|https://ceogroup.com.vn/|- Bất động sản. - Xây dựng. - Dịch vụ.|11,90|7.090|CEO Construction, Coteccons|C.E.O International, Sonasea Vân Đồn, C.E.O Construction|CTCP Tập đoàn C.E.O|Hà Nội, Phú Quốc (Kiên Giang), Hà Nam, Quảng Ninh
CIG|HOSE|Bất động sản|Nhà ở / Khu đô thị / Xây lắp|Trung cấp|510|51039947|10000|19/07/2011|Mr. Nguyễn Trọng Hiền|Tầng 1 Tòa nhà Westa - Số 108 Trần Phú - P. Hà Đông - Tp. Hà Nội|https://coma18.vn|- Dịch vụ tư vấn, môi giới, quản lý, quảng cáo và sàn giao dịch bất động sản. - Đào tạo dạy nghề: điện, điện tử, tin học. - Khai thác kinh doanh khoáng sản; Sản xuất, kinh doanh vật liệu xây dựng;….|7,26|371|COMA, Vinaconex|Coma 18, Ha Long Star|CTCP COMA 18|Hà Nội & các tỉnh phía Bắc
CKG|HOSE|Bất động sản|Khu đô thị / Nhà ở / San lấp|Bình dân - Trung cấp|1618|161807526|10000|25/03/2020|Mr. Nguyễn Xuân Dũng|Số 34 Trần Phú - P. Rạch Giá - T. An Giang|https://cicgroups.com|- Kinh doanh bất động sản. - Thi công xây dựng. - Tư vấn, thiết kế và giám sát xây dựng. - Kinh doanh nhà hàng. - Đầu tư tài chính. - Kinh doanh vật liệu xây dựng.|5,60|906|CIC Group (In-house)|Kiên Giang Construction (CIC Group)|CTCP Tập đoàn Tư vấn Đầu tư Xây dựng Kiên Giang|Kiên Giang (Rạch Giá, Phú Quốc) & Tây Nam Bộ
CRE|HOSE|Bất động sản|Môi giới BĐS / Đầu tư thứ cấp|Trung cấp - Cao cấp|4637|463678426|10000|05/09/2018|Mr. Phạm Đức Hùng|Tầng 1 - Tòa B Sky City - Số 88 Láng Hạ - P. Láng - Tp. Hà Nội|https://www.cenland.vn/|- Tư vấn, môi giới, đấu giá bất động sản, đấu giá quyền sử dụng đất.|6,10|2.828|Đối tác liên kết phát triển dự án|Cen Land, Cen Group, Cen Invest|CTCP Tập đoàn Thế kỷ (Cen Land)|Toàn quốc (Hà Nội, TP. Hồ Chí Minh, Đà Nẵng)
CRV|HOSE|Bất động sản|Bất động sản thương mại / Nhà ở|Cao cấp|6874|687399875|10000|10/10/2025|Ms. Phạm Thị Thu Huyền|Tầng 4 - Số 183 phố Bà Triệu - P. Hai Bà Trưng - Tp. Hà Nội|https://www.crvgroup.vn/|- Kinh doanh bất động sản, quyền sử dụng đất thuộc chủ sở hữu, chủ sử dụng hoặc đi thuê. - Xây dựng nhà ở. - Xây dựng nhà không để ở. - Hoàn thiện công trình xây dựng.|23,30|16.016|Hoàng Huy Construction, Coteccons|Tập đoàn Hoàng Huy (TCH)|CTCP Đầu tư Bất động sản Hoàng Huy|Hải Phòng & các tỉnh phía Bắc
CSC|HNX|Bất động sản|Khu đô thị / Nghỉ dưỡng|Trung cấp - Cao cấp|453|45263582|10000|04/11/2009|Mr. Lê Văn Thành|Lô CC5A Bán đảo Linh Đàm - P. Hoàng Liệt - Tp. Hà Nội|https://www.cotanagroup.vn|- Xây dựng các công trình dân dụng, công nghiệp, giao thông, thủy lợi. - Lắp đặt điện nước, điện lạnh, trang trí nội ngoại thất công trình. - Xây lắp đường dây và trạm biến áp đến 35KV. - Sản xuất buôn bán vật liệu xây dựng. - Đầu tư bất động sản.|8,90|403|Cotana, Vinaconex|Cotana Group, Ecopark (liên quan)|CTCP Tập đoàn Cotana|Hà Nội, Hưng Yên (Ecopark), Bắc Ninh, Bình Định
D11|HNX|Bất động sản|Nhà ở đô thị / Xây lắp|Trung cấp|82|8218456|10000|25/02/2011|Mr. Trần Thị Kim Huệ|Số 205 Lạc Long Quân - P. Bình Thới - Tp. Hồ Chí Minh|https://www.diaoc11.com.vn|- Kinh doanh nhà, xây dựng các công trình công nghiệp, công trình công cộng, nhà ở. - Trang trí nội thất, sản xuất và kinh doanh vật liệu xây dựng. - Thiết kế các công trình dân dụng, công nghiệp. - Cho thuê kho bãi, cửa hàng...|11,20|92|Resco, Tổng thầu khu vực TP.HCM|Địa ốc 11 (Resco)|CTCP Địa ốc 11|TP. Hồ Chí Minh
D2D|HOSE|Bất động sản|Khu công nghiệp / Khu dân cư|Trung cấp|303|30259742|10000|14/08/2009|Mr. Nguyễn Văn Tuấn|Số 47 - Đường D9 - Khu dân cư Đường Võ Thị Sáu - Kp. Vinh Thạnh - P. Trấn Biên - Tp. Đồng Nai|https://www.d2d.com.vn/|- Xây dựng công trình dân dụng, công nghiệp, giao thông , thủy lợi, công trình kỹ thuật khác (xây dựng cơ sở hạ tầng KCN, khu dân cư). - Xây dựng nhà ở. - Đầu tư kinh doanh hạ tầng KCN, khu dân cư, khu đô thị, khu du lịch, khu kinh doanh chợ, trung tâm thương mại, cao ốc văn phòng cho thuê.|29,95|906|D2D Construction, các nhà thầu Đồng Nai|Tập đoàn Công nghiệp Cao su Việt Nam (VRG)|CTCP Phát triển Đô thị Công nghiệp Số 2|Đồng Nai (Biên Hoà), Bà Rịa - Vũng Tàu
DIG|HOSE|Bất động sản|Khu đô thị / Nhà ở / Nghỉ dưỡng|Trung cấp - Cao cấp|7964|796431191|10000|19/08/2009|Mr. Nguyễn Hùng Cường|Số 15 Thi Sách - P. Vũng Tàu - Tp. Hồ Chí Minh|https://www.dic.vn|- Kinh doanh bất động sản, quyền sử dụng đất chủ sở hữu, chủ sử dụng hoặc đi thuê. - Tư vấn, môi giới, đấu giá bất động sản, đấu giá quyền sử dụng đất. - Dịch vụ lưu trú ngắn ngày. - Xây dựng nhà để ở. - Hoạt động tư vấn quản lý. - Hoạt động kiến trúc và tư vấn kỹ thuật có liên quan. - Xây dựng công trình cấp thoát nước.|10,05|8.004|DIC Cons, Ricons, Hoa Binh|DIC Corp, DIC Cons, DIC No.4, DIC Hospitality|Tổng Công ty cổ phần Đầu tư Phát triển Xây dựng (DIC Corp)|Bà Rịa - Vũng Tàu, Đồng Nai, Vĩnh Phúc, Hậu Giang
DRH|HOSE|Bất động sản|Căn hộ chung cư / Khu đô thị|Trung cấp|1244|123707866|10000|26/07/2010|Mr. Phan Tấn Đạt|67 Hàm Nghi - P. Sài Gòn - Tp. Hồ Chí Minh|https://drh.vn|- XD dân dụng, công nghiệp, CSHT, XD nhà để bán và cho thuê - Kinh doanh nhà, môi giới bất động sản, kinh doanh khách sạn - Tư vấn đầu tư, đào tạo nghề - DV cung cấp thông tin lên mạng Internet - DV sàn giao dịch BĐS, định giá BĐS...|1,67|207|An Phong, Phước Thành|DRH Holdings, KSB (Khoáng sản và Xây dựng Bình Dương)|CTCP DRH Holdings|TP. Hồ Chí Minh & Bình Dương
DTA|HOSE|Bất động sản|Khu đô thị / Nhà xưởng|Bình dân - Trung cấp|195|19504499|10000|16/07/2010|Ms. Phạm Thị Kim Xuân|2/6–2/8 Núi Thành - P. Tân Bình - Tp. Hồ Chí Minh|http://www.detamland.com|- Đầu tư, kinh doanh, môi giới, đấu giá ,dịch vụ định giá BĐS. - Xây dựng công trình dân dụng, công nghiệp, san lấp mặt bằng. - Trang trí nội ngoại thất công trình, nhà cửa. - Dịch vụ quản lý nhà đất, văn phòng cho thuê. - Khai thác khoáng sản, khai thác, cung cấp nước sạch. - Đầu tư tài chính. - Mua bán, cung cấp vật liệu xây dựng, hàng trang trí nội - ngoại thất...|3,41|67|Đệ Thất Construction|Đệ Thất (DTA)|CTCP Đệ Thất|Long An, TP. Hồ Chí Minh
DTD|HNX|Bất động sản|Khu công nghiệp / BĐS dân cư|Trung cấp|734|73353788|10000|16/11/2017|Mr. Nguyễn Huy Cương|Đường Nguyễn Thị Định - P. Phủ Lý - T. Ninh Bình|http://thanhdathanam.vn/|- Xây dựng dân dụng các công trình công nghiệp, công trình đường giao thông. - Sản xuất và cung cấp bê tông thương phẩm. - Khai thác và cung cấp nước sạch. - Khai thác khoáng sản. - Dịch vụ bến xe.|10,60|778|Nhà thầu địa phương Hà Nam|Đầu tư Phát triển Thành Danh (Hà Nam)|CTCP Đầu tư Phát triển Thành Danh|Hà Nam & các tỉnh phía Bắc
DXG|HOSE|Bất động sản|Căn hộ / Khu đô thị / Môi giới BĐS|Trung cấp - Cao cấp|12699|1268104965|10000|22/12/2009|Mr. Nguyễn Trường Sơn|Số 2W Ung Văn Khiêm - P. Thạnh Mỹ Tây - Tp. Hồ Chí Minh|https://bluemarq.com/|- Kinh doanh địa ốc, đầu tư phát triển các dự án bất động sản, trung tâm thương mại, văn phòng cho thuê, khách sạn, nhà hàng. - Phân phối và Marketing các dự án Bất động sản; Đầu tư tài chính, Môi giới chứng khoán...|10,75|13.632|Phước Thành, Hòa Bình, Ricons, Central|Đất Xanh Group, DXS (Đất Xanh Services), Gem Sky World|CTCP Tập đoàn Đất Xanh|TP. Hồ Chí Minh, Đồng Nai, Bình Dương, Miền Trung
DXS|HOSE|Bất động sản|Dịch vụ môi giới & Phân phối BĐS|Trung cấp - Cao cấp|5791|579103124|10000|15/07/2021|Mr. Trần Quốc Thịnh|Số 2W - Đường Ung Văn Khiêm - P. Thạnh Mỹ Tây - Tp. Hồ Chí Minh|https://datxanhservices.vn/|- Kinh doanh và Môi giới Bất động sản. - Dịch vụ Tài chính Bất động sản. - Quản lý bất động sản. - Lĩnh vực phụ trợ.|5,40|3.127|Không trực tiếp thi công (đơn vị dịch vụ)|Tập đoàn Đất Xanh (DXG)|CTCP Dịch vụ Bất động sản Đất Xanh|Toàn quốc (Hệ thống môi giới rộng khắp 3 miền)
EVG|HOSE|Bất động sản|BĐS nghỉ dưỡng / Đất nền|Trung cấp - Cao cấp|2260|226011737|10000|08/06/2017|Mr. Lê Đình Vinh|Tầng 3 Tòa nhà 97-99 Láng Hạ - P. Đống Đa - Tp. Hà Nội|https://www.everland.vn/|- Xây dựng các công trình dân dụng, công nghiệp, công trình đường sắt, đường bộ - Lập dự án, quản lý dự án đầu tư xây dựng công trình - Khai thác, xuất nhập khẩu và kinh doanh thương mại vật liệu xây dựng - Bán buôn máy móc thiết bị, hàng nông lâm sản, sản xuất đồ gỗ - Tư vấn thiết kế kiến trúc, mặt bằng tổng quan, cảnh quan...|4,09|924|Delta, Vinaconex|EverLand Group|CTCP Tập đoàn Đầu tư Địa ốc NoVa / EverLand|Quảng Ninh, Phú Quốc, Khánh Hòa, Hà Nội
FDC|HOSE|Bất động sản|Khu công nghiệp / Kho bãi / Nhà ở|Trung cấp|386|38629988|10000|18/01/2010|Mr. Tạ Chí Cường|Số 28 Phùng Khắc Khoan - P. Tân Định - Tp. Hồ Chí Minh|https://fideco.com.vn/|- Liên doanh hợp tác đầu tư, xây dựng trong lĩnh vực nuôi trồng thủy sản. - Chế biến hàng xuất khẩu: nông sản, phương tiện vận tải các loại. - Xây dựng dân dụng, kinh doanh bất động sản. - Đào tạo, phát triển các dự án giáo dục.|22,65|875|Fideco Cons|Fideco|CTCP Ngoại thương và Phát triển Đầu tư TP.HCM (Fideco)|TP. Hồ Chí Minh
FIR|HOSE|Bất động sản|Bất động sản thương mại / Khu dân cư|Trung cấp|707|70669620|10000|18/10/2018|Mr. Trương Thế Tùng|Tầng 5 Khu văn phòng - Khu phức hợp Khách sạn Bạch Đằng - Số 50 Đường Bạch Đằng - P. Hải Châu - Tp. Đà Nẵng|https://fir.vn|- Buôn bán vật liệu, thiết bị lắp đặt khác trong xây dựng. - Hoạt động xây dựng, cung cấp vật tư, thiết bị các công trình điện dân dụng, công nghiệp đến cấp điện áp 500 KV. - Bán buôn máy móc thiết bị và phụ tùng máy khác (thang máy). - Buôn bán tư liệu sản xuất (chủ yếu là hàng vật liệu xây dựng, thiết bị điện và vật liệu điện). - Xây dựng công trình kỹ thuật dân dụng khác (xây dựng các công trình công nghiệp). - Hoạt động xây dựng chuyên dụng khác (xây dựng các công trình dân dụng). - Kinh doanh|3,04|215|Central, An Phong|Địa ốc First Real|CTCP Địa ốc First Real|Đà Nẵng, Quảng Nam & miền Trung
HAR|HOSE|Bất động sản|Bất động sản thương mại / Khách sạn|Trung cấp - Cao cấp|1014|95684090|10000|17/01/2013|Mr. Nguyễn Nhân Bảo|Số 2 Ngô Đức Kế - P. Sài Gòn - Tp. Hồ Chí Minh|https://adtdgroup.com/|- Kinh doanh bất động sản. - Cung cấp dịch vụ cho thuê căn hộ, khách sạn (trên các dự án đã hoàn thành). Các lĩnh vực kinh doanh khác. - Giáo dục, mua bán các mặt hàng trang trí nội thất, đồ gỗ, vật liệu xây dựng, mua bán nông sản. - Công ty đã dần chuyển sang mô hình Holding sau khi sáp nhập các công ty con có đa ngành nghề như: xây dựng kho bạc, sản xuất két sắt, xe chở tiền, cho thuê BĐS, sản xuất - kinh doanh chất tẩy rửa, hóa chất, kinh doanh trung tâm thương mại,…|2,51|240|Hòa Bình, Coteccons|An Gia Real Estate / Hoà Bình (liên quan)|CTCP Đầu tư Thương mại Realm|TP. Hồ Chí Minh & các tỉnh phía Nam
HDC|HOSE|Bất động sản|Khu đô thị / Nhà ở / Nghỉ dưỡng|Trung cấp - Cao cấp|2297|229712443|10000|08/10/2007|Mr. Đoàn Hữu Thuận|Tầng 3 - Tòa nhà Hodeco Plaza - Số 36 Nguyễn Thái Học - P. Tam Thắng - Tp. Hồ Chí Minh|https://hodeco.vn/|- Cung cấp các dịch vụ. - Phát triển và kinh doanh bất động sản. - Sản xuất bê tông và sản phẩm từ xi măng và vữa. - Xây dựng các công trình dân dụng, công nghiệp, giao thông.|11,35|2.607|Hodeco In-house, Ricons|Hodeco (Phát triển Nhà Bà Rịa - Vũng Tàu)|CTCP Phát triển Nhà Bà Rịa - Vũng Tàu (Hodeco)|Bà Rịa - Vũng Tàu & TP. Hồ Chí Minh
HDG|HOSE|Bất động sản|Nhà ở đô thị + Năng lượng (Thủy điện/Điện gió)|Trung cấp - Cao cấp|4070|406957333|10000|02/02/2010|Mr. Nguyễn Trọng Minh|Số 8 Láng Hạ - P. Giảng Võ - Tp. Hà Nội|https://hado.com.vn|- Bất động sản: Đầu tư và kinh doanh bất động sản: Bao gồm khu đô thị; văn phòng; khách sạn; xây dựng và kinh doanh cơ sở hạ tầng khu công nghiệp; khai thác, xử lý và cung cấp nước sạch…; Tư vấn phân phối các sản phẩm Bất động sản. M&A các dự án bất động sản; Tư vấn thiết kế, tư vấn đầu tư xây dựng; Quản lý vận hành khai thác bất động sản sau đầu tư. - Phát triển năng lượng: Đầu tư, thi công, lắp đặt các dự án thủy điện, các dự án năng lượng tái tạo: điện gió, điện mặt trời; M&A các dự án năng l|15,95|6.491|Hà Đô Z756, Coteccons, Phước Thành|Hà Đô Group, Ha Do Charmvilla, các nhà máy điện Hà Đô|CTCP Tập đoàn Hà Đô|Hà Nội, TP. Hồ Chí Minh, Quảng Nam, Miền Trung (Thủy điện)
HLD|HNX|Bất động sản|Khu đô thị / Nhà ở thấp tầng|Trung cấp|550|54999961|10000|26/03/2013|Mr. Phạm Cao Sơn|Tầng 12 - Tòa nhà HUDLAND TOWER - Lô ACC7 - Khu dịch vụ tổng hợp Linh Đàm - P. Định Công - Tp. Hà Nội|http://www.hudland.com.vn/|- Đầu tư phát triển khu dân cư, khu đô thị mới, khu kinh tế, KCN. - Đầu tư xây dựng các khu trung tâm thương mại, siêu thị, dịch vụ... - Tư vấn đầu tư xây dựng nhà và công trình kỹ thuật hạ tầng đô thị. - Thi công xây lắp các công trình dân dụng, công nghiệp, giao thông, thủy lợi, bưu chính viễn thông, công trình kỹ thuật hạ tầng KĐT, KCN, công trình đường dây và trạm điện dưới 35KV.|13,70|753|HUD Cons, Vinaconex|HUD Land (Tổng Công ty Đầu tư Phát triển Nhà và Đô thị)|CTCP Đầu tư và Phát triển Nhà HUDland|Hà Nội & các tỉnh phía Bắc
HPX|HOSE|Bất động sản|Căn hộ chung cư / Khu đô thị|Trung cấp|3042|304168581|10000|24/07/2018|Mr. Đỗ Quý Hải|Tầng 5 - Tòa CT3 - The Pride - Khu ĐTM An Hưng - P. Hà Đông - Tp. Hà Nội|https://haiphat.com.vn|- Kinh doanh bất động sản. - Xây dựng các công trình dân dụng, công nghiệp. - Hoạt động tư vấn, quản lý dự án. - Kinh doanh vật liệu xây dựng. - Tư vấn môi giới bất động sản.|3,99|1.214|Hải Phát Cons, Delta, Conteccons|Hải Phát Invest (Hải Phát Group)|CTCP Đầu tư Hải Phát|Hà Nội, Quảng Ninh, Bắc Giang, Bình Thuận
HQC|HOSE|Bất động sản|Nhà ở xã hội / Khu đô thị|Bình dân (Nhà ở xã hội)|6266|626599274|10000|20/10/2010|Mr. Trương Anh Tuấn|Số 15 Nguyễn Lương Bằng - P. Tân Mỹ - Tp. Hồ Chí Minh|https://hoangquan.com.vn|- Kinh doanh bất động sản, quyền sử dụng đất thuộc chủ sở hữu, chủ sử dụng hoặc đi thuê. - Tư vấn, môi giới, đấu giá bất động sản, đấu giá quyền sử dụng đất. - Xây dựng nhà các loại; Chuẩn bị mặt bằng. - Bán buôn đồ dùng khác cho gia đình, Bán buôn thực phẩm. - Hoạt động chuyên môn, khoa học và công nghệ khác chưa được phân vào đâu... - Đầu tư tài chính. - Giáo dục.|1,95|1.222|Hoàng Quân Con, Hòa Bình|Địa ốc Hoàng Quân, HQC Mekong|CTCP Tư vấn - Thương mại - Dịch vụ Địa ốc Hoàng Quân|TP. Hồ Chí Minh, Tây Ninh, Cần Thơ, Vĩnh Long, Khánh Hòa
HU1|HOSE|Bất động sản|Xây lắp / Nhà ở đô thị|Trung cấp|250|25000000|10000|03/11/2011|Mr. Dương Tất Khiêm|Số 168 Giải Phóng - P. Phương Liệt - Tp. Hà Nội|http://www.hud1.com.vn|- Đầu tư, thi công xây lắp các loại công trình dân dụng, công nghiệp, giao thông thủy lợi, bưu chính viễn thông, đường dây và trạm biến thế điện, công trình kỹ thuật hạ tầng. - Thi công lắp đặt thiết bị kỹ thuật công trình. - Trang trí nội ngoại thất các công trình xây dựng. - Sản xuất và kinh doanh vật tư, thiết bị vật liệu xây dựng..|5,70|143|HUD1 In-house|HUD1 (Đầu tư và Xây dựng HUD1)|CTCP Đầu tư và Xây dựng HUD1|Hà Nội & các tỉnh phía Bắc
ICG|HNX|Bất động sản|Khu đô thị / Xây dựng hạ tầng|Trung cấp|200|17572000|10000|21/04/2009|Ms. Phạm Quỳnh Trang|Số 164 Lò Đúc - P. Hai Bà Trưng - Tp. Hà Nội|https://incomex.com.vn|- Sản xuất, kinh doanh các mặt hàng: đồng, thau, nhôm, dây và cáp điện. - Đại lý bán hàng và dịch vụ thương mại. - Kinh doanh xuất nhập khẩu vật tư, máy móc thiết bị và phụ tùng….|10,00|176|Sông Hồng Construction|Sông Hồng (ICG)|CTCP Xây dựng Sông Hồng|Hà Nội & miền Bắc
IDJ|HNX|Bất động sản|BĐS nghỉ dưỡng / Khu đô thị|Trung cấp|1735|173490193|10000|13/09/2010|Mr. Nguyễn Mạnh Cường|Tầng 3 - Tòa nhà TTTM Grand Plaza - Số 117 Trần Duy Hưng - P. Yên Hòa - Tp. Hà Nội|http://www.idjf.vn|- KInh doanh BĐS dân dụng, nghỉ dưỡng, thương mại. - Thu gon, xử lý và tiêu hủy rác thải không độc hại. - Tái chế phế liệu, xử lý ô nhiễm. - Cho thuê văn phòng thương mại. - Điều hành, quản lý nhà và đất không để ở.|3,50|607|Apec Construction|Apec Group, API|CTCP Đầu tư IDJ Việt Nam|Hà Nội, Bắc Ninh, Hoà Bình, Phú Thọ
IDV|HNX|Bất động sản|Khu công nghiệp / Hạ tầng KCN|Không áp dụng (BĐS KCN)|474|47421881|10000|01/06/2010|Ms. Nguyễn Ngọc Lan|KCN Khai Quang - P. Vĩnh Phúc - T. Phú Thọ|https://www.vpid.vn|- Đầu tư kinh doanh hạ tầng Khu công nghiệp (KCN) và các dịch vụ trong KCN (xử lý nước thải,...).|21,20|1.005|IDV Construction|Đầu tư Phát triển Vĩnh Phúc|CTCP Phát triển Hạ tầng Vĩnh Phúc|Vĩnh Phúc, Phú Thọ & các tỉnh miền Bắc
IJC|HOSE|Bất động sản|Hạ tầng giao thông / BĐS đô thị|Trung cấp|6296|629580640|10000|19/04/2010|Mr. Trịnh Thanh Hùng|Tầng 5 Becamex Tower - 230 Đại lộ Bình Dương - P. Phú Lợi - Tp. Hồ Chí Minh|https://becamexijc.com|- Kinh doanh bất động sản. - Thi công xây dựng. - Thương mại dịch vụ. - Nhà hàng khách sạn. - Thu phí giao thông. - Hợp tác kinh doanh.|6,77|4.262|Becamex, các nhà thầu hạ tầng Bình Dương|Becamex IDC (BCM)|CTCP Phát triển Hạ tầng kỹ thuật (Becamex IJC)|Bình Dương & Vùng kinh tế trọng điểm phía Nam
ITC|HOSE|Bất động sản|Căn hộ chung cư / Khu đô thị|Trung cấp - Cao cấp|964|95935049|10000|19/10/2009|Mr. Trương Minh Thuận|18 Nguyễn Bỉnh Khiêm - P. Tân Định - Tp. Hồ Chí Minh|http://www.intresco.com.vn|- Đầu tư kinh doanh bất động sản. - Thiết kế và xây dựng. - Dịch vụ bất động sản. - Dịch vụ nhà hàng, khách sạn.|8,87|851|Intresco Cons, Hòa Bình|Intresco|CTCP Đầu tư và Kinh doanh Nhà (Intresco)|TP. Hồ Chí Minh & các tỉnh phía Nam
KBC|HOSE|Bất động sản|Khu công nghiệp / Đô thị vệ tinh KCN|Không áp dụng (BĐS KCN)|9418|941754759|10000|18/12/2009|Mr. Đặng Thành Tâm|Lô 7B KCN Quế Võ - P. Phương Liễu - T. Bắc Ninh|https://kinhbaccity.vn/|- Kinh doanh bất động sản, quyền sử dụng đất thuộc chủ sở hữu, chủ sử dụng hoặc đi thuê. - Xây dựng nhà để ở và nhà không để ở. - Xây dựng công trình kỹ thuật dân dụng khác.|26,50|24.957|KBC In-house, Coteccons, VINACONEX|Kinh Bắc City, Sài Gòn - Bắc Giang, Trang cát|Tổng Công ty Phát triển Đô thị Kinh Bắc - CTCP|Bắc Ninh, Bắc Giang, Hải Phòng, Quảng Ninh, Long An
KDH|HOSE|Bất động sản|Nhà ở thấp tầng (Biệt thự/Nhà phố) / Căn hộ|Cao cấp - Hạng sang|10111|1122214899|10000|01/02/2010|Mr. Vương Văn Minh|Phòng 1 và 2 Lầu 11 SaiGon Centre - 67 Lê Lợi - P. Sài Gòn - Tp. Hồ Chí Minh|http://www.khangdien.com.vn|- Đầu tư và kinh doanh bất động sản, nhận quyền sử dụng đất để đầu tư và kinh doanh nhà ở. - Đầu tư xây dựng cơ sở hạ tầng theo quy hoạch, xây dựng nhà để ở, chuyển quyền sử dụng đất. - Giáo dục mầm non, tiểu học. - Hoạt động kiến trúc, tư vấn kỹ thuật...|16,30|18.292|An Phong, Ricons, Coteccons|Khang Điền House, các dự án The Classia, The Privia, Clarita|CTCP Đầu tư và Kinh doanh Nhà Khang Điền|TP. Hồ Chí Minh (Khu Đông & Khu Nam)
KHG|HOSE|Bất động sản|Môi giới / Bất động sản thương mại|Trung cấp - Cao cấp|4494|449435205|10000|19/07/2021|Ms. Đinh Thị Nhật Hạnh|Số 5-7-9-11 - Đường Nội Khu Hưng Gia 4 - Khu đô thị Phú Mỹ Hưng - P. Tân Hưng - Tp. Hồ Chí Minh|https://khaihoanland.vn/|- Tư vấn, môi giới và kinh doanh bất động sản.|4,68|2.103|Đối tác xây dựng liên kết|Khai Hoan Land|CTCP Tập đoàn Khải Hoàn Land|TP. Hồ Chí Minh & các tỉnh miền Nam
KOS|HOSE|Bất động sản|Khu đô thị / Năng lượng (Điện gió/Mặt trời)|Trung cấp|2165|216481335|10000|08/12/2017|Mr. Nguyễn Việt Cường|Tầng 24 - Tòa nhà Rox Tower - Số 136 đường Hồ Tùng Mậu - P. Phú Diễn - Tp. Hà Nội|https://kosy.vn|- Kinh doanh bất động sản. Tư vấn, môi giới, đấu giá... - Xây dựng công trình công ích. - Hoạt động kinh doanh dịch vụ hỗ trợ khai thác mỏ và quặng khác.|33,25|7.198|Kosy Cons, các tổng thầu năng lượng|Kossmo, KOSY Group|CTCP KOSY|Hà Nội, Long An, Lào Cai, Nghệ An, Đắk Lắk
KSF|HNX|Bất động sản|Bất động sản thương mại / Khu đô thị|Cao cấp - Hạng sang|8998|899787308|10000|06/10/2021|Ms. Nguyễn Thị Phương Loan|Tầng 12 - Tòa nhà Sunshine Center - Số 16 - Đường Phạm Hùng - P. Từ Liêm - Tp. Hà Nội|https://sunshinegroup.vn/|- Đầu tư và xây dựng căn hộ chung cư để bán. - Kinh doanh bất động sản. - Xây nhà các loại, xây dựng công trình kỹ thuật dân dụng. - Quản lý, khai thác các tài sản sau đầu tư. - Dịch vụ lưu trú ngắn ngày. - Nhà hàng, quán ăn, hàng ăn uống. - Lắp đặt hệ thống điện, Lắp đặt hệ thống xây dựng khác. - Hoàn thiện công trình xây dựng. - Hoạt động xây dựng chuyên dụng khác.|80,00|71.983|SCG Construction|Sunshine Group (hệ sinh thái liên quan)|CTCP Tập đoàn Sunshine Homes|Hà Nội & các tỉnh phía Bắc
L14|HNX|Bất động sản|Khu đô thị / Đất nền|Trung cấp|309|30859315|10000|13/09/2011|Mr. Lại Xuân Hùng|Số 2068 Đại lộ Hùng Vương - P. Nông Trang - T. Phú Thọ|https://licogi14.vn|- Xây dựng nhà để ở. - San lấp mặt bằng, đóng ép cọc, xử lý nền móng công trình, dịch vụ sửa chữa, lắp đặt thiết bị máy móc, cho thuê thiết bị máy công trình. - Xây dựng công trình thủy điện, nhiệt điện, dân dụng và công nghiệp, đường dây trạm biến áp đến 35kv. - Đầu tư kinh doanh nhà ở, đô thị mới, các dự án thủy điện vừa và nhỏ..|19,40|599|Licogi 14 In-house|Licogi 14|CTCP Licogi 14|Phú Thọ, Hà Nội & miền Bắc
LDG|HOSE|Bất động sản|Khu đô thị / Căn hộ chung cư|Bình dân - Trung cấp|2570|255615849|10000|12/08/2015|Mr. Ngô Văn Minh|Lô E9 đường D2 - Khu dân cư - Dịch vụ Giang Điền (Khu A) - X. Trảng Bom - T. Đồng Nai|https://ldggroup.com.vn/|- Đầu tư các dự án bất động sản. - Kinh doanh và phân phối sản phẩm bất động sản. - Đầu tư kinh doanh và phát triển dịch vụ du lịch nghỉ dưỡng. - Đầu tư tài chính.|2,52|644|Hòa Bình, Phước Thành|LDG Investment (từng thuộc Đất Xanh)|CTCP Đầu tư LDG|Đồng Nai, TP. Hồ Chí Minh, Bình Dương
LGL|HOSE|Bất động sản|Khu đô thị / Nhà ở|Trung cấp|515|51497100|10000|08/10/2009|Mr. Lê Hà Giang|173 Xuân Thủy - P. Cầu Giấy - Tp. Hà Nội|https://www.longgiangland.com.vn|- Xây dựng dân dụng, xây dựng công nghiệp, xây dựng các công trình giao thông. - Xây dựng và kinh doanh hạ tầng kỹ thuật các khu đô thị và khu công nghiệp. - Kinh doanh nhà và bất động sản. - Kinh doanh máy móc thiết bị và vật tư ngành xây dựng...|4,60|237|Long Đức Cons|Long Đức Investment (LGL)|CTCP Đầu tư và Phát triển Nhà Long Đức|TP. Hồ Chí Minh, Gia Lai & miền Trung
LHG|HOSE|Bất động sản|Khu công nghiệp / Nhà xưởng cho thuê|Không áp dụng (BĐS KCN)|500|50012010|10000|23/03/2010|Mr. Trần Hồng Sơn|KCN Long Hậu - X. Cần Giuộc - T. Tây Ninh|https://longhau.com.vn|- Mua bán nhà ở, chung cư, nhà xưởng, kho, bến bãi, bãi đỗ xe - Mua bán vật liệu xây dựng - DV tư vấn quản lý chất lượng và môi trường; quan trắc môi trường - SX, kinh doanh nước sạch, nước tinh khiết, ...|26,75|1.338|Long Hậu Cons, An Phong|Long Hậu, IPC|CTCP Long Hậu|Long An & TP. Hồ Chí Minh
NBB|HOSE|Bất động sản|Căn hộ chung cư / Khu đô thị|Trung cấp|1005|100475656|10000|18/02/2009|Mr. Nguyễn Bá Lân|Tòa nhà CII TOWER - Số 152 Điện Biên Phủ - P. Thạnh Mỹ Tây - Tp. Hồ Chí Minh|http://www.nbb.com.vn|- Đầu tư kinh doanh BĐS, đầu tư tài chính. - Hoạt động xây dựng dân dụng và công nghiệp, xây dựng công trình giao thông. - SX điện; Lắp đặt trang thiết bị cho công trình xây dựng. - Trang trí nội, ngoại thất công trình. - Kinh doanh lưu trú du lịch (không hoạt động tại trụ sở). - Các hoạt động kinh doanh khác phù hợp với Pháp luật.|16,55|1.663|CII E&C, Coteccons|Năm Bảy Năm (CII liên quan)|CTCP Đầu tư Năm Bảy Năm|TP. Hồ Chí Minh & Quảng Ngãi
NDN|HNX|Bất động sản|Đất nền / Căn hộ thương mại Đà Nẵng|Trung cấp|717|71657936|10000|21/04/2011|Mr. Nguyễn Quang Trung|Số 38 Nguyễn Chí Thanh - P. Hải Châu - Tp. Đà Nẵng|https://ndn.com.vn|- Kinh doanh bất động sản, quyền sử dụng đất thuộc chủ sở hữu, chủ sử dụng hoặc đi thuê. - Tư vấn, môi giới, đấu giá bất động sản, đấu giá quyền sử dụng đất. - Cho thuê máy móc, thiết bị và đồ dùng hữu hình khác. - Hoạt động kiến trúc và tư vấn kỹ thuật có liên quan. - Xây dựng nhà các loại, các công trình đường bộ, hệ thống cấp/thoát nước.|8,50|609|Nhà thầu miền Trung|Đầu tư Phát triển Nhà Đà Nẵng|CTCP Đầu tư Phát triển Nhà Đà Nẵng|Đà Nẵng & miền Trung
NHA|HOSE|Bất động sản|Khu đô thị / Nhà ở Hà Nam|Bình dân - Trung cấp|713|71266847|10000|21/01/2021|Mr. Nguyễn Đắc Long|Cụm Công nghiệp Cầu Giát - P. Duy Tiên - T. Ninh Bình|https://www.namhanoi.com.vn|- Xây dựng công trình. - Kinh doanh bất động sản. - Nhà xưởng công nghiệp. - Hạ tầng kỹ thuật. - Nhà hàng khách sạn.|7,21|514|Nam Định Cons|Đầu tư và Xây dựng Nam Định|CTCP Đầu tư và Xây dựng Nam Định|Hà Nam & miền Bắc
NLG|HOSE|Bất động sản|Khu đô thị / Nhà ở vừa túi tiền (Flora, Valora, EHome)|Bình dân - Trung cấp|4851|485097383|10000|08/04/2013|Mr. Nguyễn Xuân Quang|Số 6 Nguyễn Khắc Viện - P. Tân Mỹ - Tp. Hồ Chí Minh|https://www.namlongvn.com/|- Xây dựng công nghiệp và dân dụng. - Sửa chữa nhà ở và trang trí nội thất. - Kinh doanh nhà ở (xây dựng, sửa chữa nhà để bán hoặc cho thuê). - Xây dựng cầu đường bến cảng. - San lấp mặt bằng. - Thi công xây dựng hệ thống cấp thoát nước. - Lắp đặt và sửa chữa hệ thống điện dưới 35KV. - Dịch vụ môi giới nhà đất.|24,70|11.982|Newtecons, Hòa Bình, Central|Nam Long Group, Mizuki Park, Akari City, Waterpoint|CTCP Đầu tư Nam Long|TP. Hồ Chí Minh, Long An, Đồng Nai, Cần Thơ, Hải Phòng
NRC|HNX|Bất động sản|BĐS nghỉ dưỡng / Đất nền|Trung cấp|926|92597762|10000|05/04/2018|Mr. Lê Thống Nhất|Số 3 - Đường Trần Nhật Duật - P. Tân Định - Tp. Hồ Chí Minh|https://nrc.com.vn/|- Kinh doanh bất động sản. - Dược phẩm và vật tư y tế. - Lương thực và Nông nghiệp công nghệ cao.|4,60|426|Phước Thành, An Phong|Danh Khải (Netland / NRC)|CTCP Tập đoàn Danh Khải (Netland)|Bình Định, Khánh Hòa, TP. Hồ Chí Minh
NTC|HOSE|Bất động sản|Khu công nghiệp Tân Uyên|Không áp dụng (BĐS KCN)|240|23999980|10000|28/10/2025|Mr. Trần Quốc Thái|Đường ĐT 747B - Kp. Long Bình - P. Tân Hiệp - Tp. Hồ Chí Minh|http://namtanuyen.com.vn|- Đầu tư xây dựng và kinh doanh kết cấu hạ tầng kỹ thuật khu công nghiệp. - Thi công xây dựng công trình công nghiệp, dân dụng, thủy lợi, giao thông, cầu đường. - San lấp mặt bằng. - Kinh doanh nhà ở, cho thuê văn phòng, nhà xưởng, nhà kho, bến bãi,... - Trồng, khai thác, chế biến, kinh doanh nguyên liệu, sản phẩm cao su, gỗ rừng trồng...|129,10|3.098|Nhà thầu KCN Bình Dương|Khu công nghiệp Nam Tân Uyên (Tập đoàn Cao su)|CTCP Khu công nghiệp Nam Tân Uyên|Bình Dương
NTL|HOSE|Bất động sản|Khu đô thị / Nhà ở thấp tầng|Trung cấp - Cao cấp|1220|121979900|10000|21/12/2007|Mr. Lê Minh Tuân|Số 08 đường Hoàng Tăng Bí - P. Đông Ngạc - Tp. Hà Nội|http://www.lideco.vn|- Đầu tư phát triển và kinh doanh khai thác các dự án khu đô thị mới, khu nhà ở và KCN. - XD các công trình dân dụng, giao thông, thủy lợi, công nghiệp. - Cung cấp các DV tư vấn đầu tư xây dựng, Quản lý DA, lập DA đầu tư, thiết kế xây dựng, kiểm định chất lượng công trình và thiết bị XD... - Kinh doanh bất động sản. - Khai thác vật liệu xây dựng.|12,75|1.555|Ladeco In-house, Vinaconex|Ladeco (Phát triển Đô thị Từ Liêm)|CTCP Phát triển Đô thị Từ Liêm (Ladeco)|Hà Nội (Mỹ Đình), Quảng Ninh
NVL|HOSE|Bất động sản|Khu đô thị quy mô lớn / BĐS nghỉ dưỡng (NovaWorld)|Trung cấp - Cao cấp - Hạng sang|24021|2402067972|10000|28/12/2016|Mr. Dương Văn Bắc|Số 313B - 315 Nam Kỳ Khởi Nghĩa - P. Xuân Hòa - Tp. Hồ Chí Minh|https://www.novaland.com.vn|- Kinh doanh BĐS, quyền sử dụng đất thuộc chủ sở hữu, chủ sử dụng hoặc đi thuê. - Sửa chữa thiết bị quang học. - Xây dựng nhà các loại. - Tư vấn, môi giới, đấu giá BĐS, đấu giá quyền sử dụng đất. - Tư vấn máy vi tính và quản trị hệ thống máy vi tính. - Dịch vụ lưu trú ngắn ngày.|12,20|29.305|Ricons, Hoa Binh, Coteccons, Central|Novaland Group, Nova Hospitality, Nova Service|CTCP Tập đoàn Đầu tư Địa ốc NoVa (Novaland)|TP. Hồ Chí Minh, Đồng Nai, Bình Thuận (NovaWorld Phan Thiết)
PDR|HOSE|Bất động sản|Khu đô thị / Căn hộ / Đất nền|Trung cấp - Cao cấp|9978|997809379|10000|30/07/2010|Mr. Nguyễn Văn Đạt|Số 39 Phạm Ngọc Thạch - P. Xuân Hòa - Tp. Hồ Chí Minh|https://www.phatdat.com.vn|- Đầu tư, phát triển các dự án BĐS dân dụng (nhà ở, căn hộ, biệt thự, khách sạn, resort…), công trình công nghiệp, cầu đường và cung cấp các dịch vụ về BĐS.|11,75|11.724|Coteccons, Central, Hòa Bình|Phát Đạt Corporation, Serenity, The EverRich|CTCP Phát triển Bất động sản Phát Đạt|TP. Hồ Chí Minh, Bình Định, Quảng Ngãi, Bà Rịa - Vũng Tàu
PTL|HOSE|Bất động sản|Căn hộ chung cư / Khu đô thị|Trung cấp|1000|100000000|10000|22/09/2010|Mr. Nguyễn Tuấn Anh|Số 12 Tân Trào - P. Tân Mỹ - Tp. Hồ Chí Minh|https://victorygroup.vn/|- Dịch vụ kinh doanh bất động sản. - Quản lý bất động sản. - Xây dựng, đầu tư xây dựng và kinh doanh hạ tầng kỹ thuật khu công nghiệp.|1,76|176|PVC, PetroCons|PVInvest (Dầu khí Phước Thái / PTL)|CTCP Đầu tư Dầu khí Phước Thái|TP. Hồ Chí Minh, Bà Rịa - Vũng Tàu
PV2|HNX|Bất động sản|Đầu tư BĐS / Tài chính|Trung cấp|374|36868800|10000|16/12/2010|Mr. Vũ Xuân Hân|Số 1 Phạm Văn Bạch - P. Cầu Giấy - Tp. Hà Nội|http://pv2.com.vn|- Đầu tư, kinh doanh bất động sản. - Đầu tư tài chính, khai thác các loại hình sau đầu tư….|1,70|63|PetroCons|PV2 (Đầu tư PV2)|CTCP Đầu tư PV2|Hà Nội & các tỉnh phía Bắc
QCG|HOSE|Bất động sản|Căn hộ chung cư / Đất nền / Khu đô thị|Trung cấp - Cao cấp|2751|275129141|10000|09/08/2010|Mr. Nguyễn Quốc Cường|Đường Nguyễn Chí Thanh - P. Hội Phú - T. Gia Lai|http://www.quoccuonggialai.com.vn|- Kinh doanh bất động sản. - Cao su. - Thủy điện.|10,25|2.820|QCG In-house, các nhà thầu TP.HCM|Quốc Cường Gia Lai|CTCP Quốc Cường Gia Lai|TP. Hồ Chí Minh, Gia Lai
RCL|HNX|Bất động sản|Nhà ở đô thị / Xây lắp|Trung cấp|144|14418317|10000|14/06/2007|Mr. Trần Văn Châu|Số 118 Hưng Phú - P. Chánh Hưng - Tp. Hồ Chí Minh|https://cholonres.com.vn|- Kinh doanh bất động sản, quyền sử dụng đất thuộc chủ sở hữu, chủ sử dụng hoặc đi thuê. - Môi giới bất động sản, dịch vụ sàn giao dịch bất động sản. Dịch vụ tư vấn, quản lý bất động sản. - Bán buôn vật liệu, thiết bị lắp đặt khác trong xây dựng.|10,90|157|Resco Cons|Địa ốc Chợ Lớn (Resco liên quan)|CTCP Địa ốc Chợ Lớn|TP. Hồ Chí Minh
SCR|HOSE|Bất động sản|Khu đô thị / Căn hộ chung cư (Sacomreal)|Trung cấp|3957|430595036|10000|18/11/2016|Mr. Võ Thanh Lâm|Số 512 Lý Thường Kiệt - P. Tân Sơn Nhất - Tp. Hồ Chí Minh|https://ttcland.vn/|- Kinh doanh bất động sản, quyền sử dụng đất thuộc chủ sở hữu, chủ sử dụng hoặc đi thuê. - Hoạt động hỗ trợ dịch vụ tài chính. - Giáo dục nghề nghiệp. - Bảo dưỡng, sữa chữa ô tô và xe có động cơ khác. - Bán phụ tùng và các bộ phận phụ trợ của ô tô và xe có động cơ khác...|4,23|1.821|TTC, Hòa Bình, Phước Thành|TTC Land (Tập đoàn TTC)|CTCP Địa ốc Sài Gòn Thương Tín (TTC Land)|TP. Hồ Chí Minh, Long An, Lâm Đồng, Bến Tre
SDU|HNX|Bất động sản|Căn hộ / Nhà ở đô thị|Bình dân - Trung cấp|200|20000000|10000|28/09/2009|Mr. Hoàng Văn Anh|Số 19 Phố Trúc Khê - P. Láng - Tp. Hà Nội|http://www.dothisongda.com.vn|- Kinh doanh các dịch vụ phục vụ đô thị. - Đầu tư tạo lập nhà, công trình xây dựng để bán, cho thuê, cho thuê mua. - Thuê nhà, công trình để cho thuê lại. - Đầu tư cải tạo đất và đầu tư cho các công trình hạ tầng trên đất thuê để cho thuê đất đã có hạ tầng. - Nhận chuyển nhượng quyền sử dụng đất, đầu tư công trình hạ tầng. - Giám sát, thi công, xây dựng và hoàn thiện các công trình xây dựng dân dụng, công nghiệp, hạ tầng, kỹ thuật.|7,30|146|Sông Đà Construction|Sông Đà Urban (SDU)|CTCP Đầu tư Xây dựng và Phát triển Đô thị Sông Đà|Hà Nội & các tỉnh phía Bắc
SGR|HOSE|Bất động sản|Khu đô thị / Căn hộ Sài Gòn|Trung cấp - Cao cấp|699|69874989|10000|15/01/2018|Mr. Phạm Thu|Cao ốc 63-65 Điện Biên Phủ - P. Gia Định - Tp. Hồ Chí Minh|https://www.saigonres.com.vn|- Xây dựng các công trình công nghiệp, dân dụng, giao thông thủy lợi, cấp thoát nước, điện và cầu cảng. - Đầu tư, xây dựng kinh doanh nhà ở. - Đầu tư, kinh doanh địa ốc. - Sàn giao dịch BĐS, quản lý BĐS. - Sản xuất, kinh doanh, XNK vật liệu xây dựng và trang trí nội thất. - Tư vấn và thiết kế xây dựng. - Cho thuê máy móc, thiết bị xây dựng...|9,99|698|Saigonres Cons, Hòa Bình|Địa ốc Sài Gòn (Saigonres)|CTCP Địa ốc Sài Gòn (Saigonres)|TP. Hồ Chí Minh & các tỉnh phía Nam
SJS|HOSE|Bất động sản|Khu đô thị lớn (Nam An Khánh, Lê Trọng Tấn)|Trung cấp - Cao cấp|2975|297474828|10000|06/07/2006|Mr. Bùi Quang Bách|Ô đất CT6 - Khu đô thị mới Nam An Khánh - X. An Khánh - Tp. Hà Nội|https://sjgroups.com.vn/|- Thương mại: Kinh doanh nhà ở, khu đô thị và công nghiệp, thi công xây lắp khu dân dụng và công nghiệp, sản xuất kinh doanh thiết bị vật liệu xây dựng, nhập khẩu máy móc thiết bị. - Dịch vụ: Tư vấn, đầu tư, lập và thực hiện các dự án đầu tư dân dụng, công nghiệp...|56,60|16.837|Sông Đà, Vinaconex|Sudico (Sông Đà Urban & Industrial Zone)|CTCP Đầu tư Phát triển Đô thị và Khu công nghiệp Sông Đà (Sudico)|Hà Nội, Hoà Bình (KĐT Nam An Khánh)
SZB|HNX|Bất động sản|Khu công nghiệp / Đất KCN Sonadezi|Không áp dụng (BĐS KCN)|300|30000000|10000|20/12/2019|Mr. Nguyễn Bá Chuyên|Số 1 Đường 3A - KCN Biên Hòa 2 - P. Long Hưng - Tp. Đồng Nai|https://szb.com.vn|- Kinh doanh hạ tầng khu công nghiệp. - Cho thuê văn phòng. - Kinh doanh nước sạch. - Hoạt động kinh doanh kho nội địa và ngoại quan hợp tác với ICD.|40,20|1.206|Sonadezi Construction|Sonadezi Biên Hòa (SZB)|CTCP Sonadezi Biên Hòa|Đồng Nai (Biên Hòa)
SZC|HOSE|Bất động sản|Khu công nghiệp & Đô thị Châu Đức|Không áp dụng (BĐS KCN)|1800|179985863|10000|15/01/2019|Mr. Nguyễn Văn Tuấn|Tầng 9 - Cao ốc Sonadezi - Số 1 - Đường 1 - Kp. An Hảo - P. Trấn Biên - Tp. Đồng Nai|https://sonadezichauduc.com.vn|- Đầu tư phát triển đô thị, khu công nghiệp, khu dân cư và sân golf. - Kinh doanh công trình kết cấu hạ tầng. - Giao dịch mua bán, chuyển nhượng, cho thuê, cho thuê mua bất động sản, môi giới bất động sản, định giá bất động sản, tư vấn bất động sản, quảng cáo bất động sản, quản lý bất động sản, sàn giao dịch bất động sản. - Kinh doanh thu phí đường bộ.|18,00|3.240|Sonadezi Cons, CC1|Sonadezi Châu Đức|CTCP Sonadezi Châu Đức|Bà Rịa - Vũng Tàu (Khu công nghiệp Châu Đức)
SZL|HOSE|Bất động sản|Khu công nghiệp / Hạ tầng Long Thành|Không áp dụng (BĐS KCN)|291|28352340|10000|09/09/2008|Mr. Phạm Anh Tuấn|KCN Long Thành - X. An Phước - T. Đồng Nai|http://www.szl.com.vn|- Khảo sát, thiết kế, đầu tư, xây dựng, quản lý, KD dịch vụ hạ tầng kỹ thuật KCN, nhà ở, nhà cho thuê. - Tư vấn cho các doanh nghiệp về lập, triển khai dự án kinh doanh. - Xây dựng công trình công nghiệp và dân dụng. - Cho thuê nhà xưởng, văn phòng, kho ngoại quan... - Giám sát công tác xây dựng và hoàn thiện công trình xây dựng và công nghiệp, công trình cầu - đường bộ.|50,60|1.435|Sonadezi Cons|Sonadezi Long Thành|CTCP Sonadezi Long Thành|Đồng Nai (Long Thành)
TAL|HOSE|Bất động sản|Khu công nghiệp / Đất KCN Tản Lĩnh|Không áp dụng (BĐS KCN)|5040|503999708|10000|01/08/2025|Mr. Nguyễn Trần Tùng|Tầng 1 - Tòa nhà N02-T1 - Khu Đoàn ngoại giao - Đường Xuân Tảo - P. Xuân Đỉnh - Tp. Hà Nội|https://tasecoland.vn/|- Bất động sản nhà ở thương mại, khu đô thị. - Bất động sản du lịch nghỉ dưỡng. - Bất động sản hạ tầng khu công nghiệp. - Hoạt động xây lắp. - Hoạt động quản lý vận hành sau đầu tư.|22,15|11.164|Nhà thầu địa phương|Đầu tư BĐS Tây Hồ|CTCP Đầu tư Bất động sản Tây Hồ|Hà Nội & các tỉnh phía Bắc
TCH|HOSE|Bất động sản|Căn hộ cao cấp / BĐS thương mại Hải Phòng|Trung cấp - Cao cấp|9121|912109224|10000|05/10/2016|Ms. Hoàng Thị Huyên|Số 116 Nguyễn Đức Cảnh - P. Lê Chân - Tp. Hải Phòng|https://www.hoanghuy.vn/|- Kinh doanh bất động sản, quyền sử dụng đất thuộc chủ sở hữu, chủ sử dụng hoặc đi thuê. - Sản xuất phụ tùng và bộ phận phụ trợ cho xe có động cơ và động cơ xe. - Xây dựng nhà để ở. - Xây dựng nhà không để ở. - Hoàn thiện công trình xây dựng. - Buôn bán ô tô và xe có động cơ khác. - Đại lý ô tô và xe có động cơ khác. - Bảo dưỡng, sửa chữa ô tô và xe có động cơ khác. - Bán phụ tùng và các bộ phận phụ trợ của ô tô và xe có động cơ khác.|11,70|10.672|Hoàng Huy Construction, Coteccons|Tập đoàn Hoàng Huy, CRV|CTCP Đầu tư Dịch vụ Tài chính Hoàng Huy|Hải Phòng & các tỉnh phía Bắc
TDC|HOSE|Bất động sản|Khu đô thị / Nhà ở Bình Dương|Trung cấp|1272|127228000|10000|04/05/2010|Mr. Đoàn Văn Thuận|26-27 Lô I - Đường Đồng Khởi - P. Bình Dương - Tp. Hồ Chí Minh|https://www.becamextdc.com.vn|- Kinh doanh bất động sản, quyền sử dụng đất thuộc chủ sở hữu, chủ sử dụng hoặc đi thuê; - Kinh doanh vật liệu xây dựng: Sắt thép, xi măng, nhựa đường, đá và các mặt hàng khác - Sản xuất vật liệu xây dựng: Bê tông tươi, sản phẩm cấu kiện và các mặt hàng khác - Xây dựng dân dụng và xây dựng công nghiệp|6,93|882|TDC In-house, Becamex|Becamex IDC, Kinh doanh và Phát triển Bình Dương (TDC)|CTCP Kinh doanh và Phát triển Bình Dương|Bình Dương & khu vực phía Nam
TDH|HOSE|Bất động sản|Căn hộ chung cư / Khu đô thị (Thủ Đức House)|Trung cấp|1127|112652767|10000|14/12/2006|Ms. Trần Thị Liên|Số 57 Song Hành - Kp.5 - P. Bình Trưng - Tp. Hồ Chí Minh|http://www.thuduchouse.com|- Kinh doanh địa ốc, đầu tư phát triển các dự án bất động sản, trung tâm thương mại, KCN. - Kinh doanh khai thác các dịch vụ địa ốc, khu đô thị và KCN. - Đầu tư kinh doanh tài chính, chứng khoán, ngân hàng...|3,10|349|TDH Cons, Hòa Bình|Thuduc House|CTCP Phát triển Nhà Thủ Đức (Thuduc House)|TP. Hồ Chí Minh & Bình Dương
TIG|HNX|Bất động sản|BĐS nghỉ dưỡng / Khu đô thị|Trung cấp - Cao cấp|1936|193606205|10000|08/10/2010|Mr. Nguyễn Phúc Long|Tầng 8 - Tháp B - Tòa nhà Sông Đà - Đường Phạm Hùng - P. Từ Liêm - Tp. Hà Nội|https://tig.vn/|- Bất động sản. - Đầu tư tài chính. - Chứng khoán và dịch vụ tài chính. - Du lịch, thương mại và dịch vụ. - Truyền thông tài chính và fintech.|5,80|1.123|TIG Cons, Vinaconex|Thăng Long Investment Group (TIG)|CTCP Tập đoàn Đầu tư Thăng Long|Hà Nội, Phú Thọ, Quảng Ninh
TIP|HOSE|Bất động sản|Khu công nghiệp / Hạ tầng Biên Hòa|Không áp dụng (BĐS KCN)|650|65007857|10000|06/06/2016|Mr. Phan Anh Dũng|KCN Tam Phước - Đường số 6 - P. Tam Phước - T. Đồng Nai|https://www.tinnghiaip.com.vn|- Đầu tư, kinh doanh hạ tầng KCN và khu dân cư. - Xây dựng các công trình dân dụng và công nghiệp. - Dịch vụ tư vấn, giám sát môi trường. - Sản xuất nước uống đóng chai. - Kinh doanh cây xanh, chăm sóc cây cảnh…|16,10|1.047|Tín Nghĩa Construction|Tín Nghĩa Corporation (TIP)|CTCP Phát triển Khu công nghiệp Tín Nghĩa|Đồng Nai (Biên Hòa, Long Thành)
TIX|HOSE|Bất động sản|Khu công nghiệp / Kho bãi Tân Tạo|Không áp dụng (BĐS KCN)|300|30000000|10000|25/11/2009|Mr. Trần Quang Trường|325 Lý Thường Kiệt - P. Tân Hòa - Tp. Hồ Chí Minh|https://www.tanimex.com.vn|- Đầu tư xây dựng và kinh doanh hạ tầng. - Đầu tư xây dựng và kinh doanh nhà xưởng, cao ốc văn phòng. - Kinh doanh bất động sản. - Đầu tư tài chính và các hoạt động khác...|39,00|1.170|Tân Tạo Cons|Tập đoàn Tân Tạo (ITA liên quan)|CTCP Sản xuất Kinh doanh Xuất nhập khẩu Tân Tạo|TP. Hồ Chí Minh (KCN Tân Tạo)
TN1|HOSE|Bất động sản|Bất động sản thương mại / Quản lý vận hành|Trung cấp - Cao cấp|601|60095480|10000|30/05/2019|Mr. Nguyễn Văn Hiệp|Tầng 25 - Tòa tháp A - Số 54A Nguyễn Chí Thanh - P. Láng - Tp. Hà Nội|https://roxkey.vn|- Dịch vụ quản lý tòa nhà, văn phòng, chung cư, khu công nghiệp và là đối tác tin cậy của nhiều Tập đoàn, doanh nghiệp trong lĩnh vực Bất động sản, bán lẻ, tài chính - ngân hàng. - Dịch vụ quản lý vận hành các khu công nghiệp lớn trên toàn quốc. - Các giải pháp và dịch vụ Công nghệ thông tin hàng đầu cho lĩnh vực tài chính - ngân hàng, khách sạn thông minh và bất động sản. - Dịch vụ bảo vệ chuyên nghiệp với đội ngũ nhân viên được đào tạo chuyên sâu. - Giải pháp Quản trị nhân sự khép kín, toàn|12,35|742|Cotana, Coteccons|TNS Holdings, TNR Holdings (Tập đoàn T&T)|CTCP Đầu tư TNR Holdings Việt Nam|Hà Nội, TP. Hồ Chí Minh & hệ thống toàn quốc
V21|HNX|Bất động sản|Xây dựng hạ tầng / Nhà ở|Trung cấp|120|11999789|10000|21/04/2010|Mr. Nguyễn Mạnh Hà|Tầng 3 - Tòa nhà Vinaconex 21 - Ngõ 804 đường Quang Trung - P. Dương Nội - Tp. Hà Nội|https://vinaconex21.vn/|- Thi công xây lắp các công trình dân dụng, CN, giao thông, thủy lợi, sân bay, bến cảng, đường hầm, cấp thoát nước, thủy điện, nhiệt điện, đường dây và trạm biến thế đến 500 KV, các công trình kỹ thuật hạ tầng, KDC, khu đô thị, khu CN - Tư vấn đầu tư và xây dựng các dự án - Khảo sát địa hình, địa chất thủy văn, đo đạc công trình, thí nghiệm - Đầu tư KD PT nhà, hạ tầng kỹ thuật khu đô thị, khu dân cư, khu kinh tế mới, khu CN...|5,90|71|Vinaconex 21|Vinaconex 21 (V21)|CTCP Vinaconex 21|Hà Nội & các tỉnh phía Bắc
VC3|HNX|Bất động sản|Khu đô thị / Xây lắp|Trung cấp|1364|138414258|10000|13/12/2007|Mr. Kiều Xuân Nam|Tầng 11 Tòa nhà Geleximco - Số 36 Hoàng Cầu - P. Ô Chợ Dừa - Tp. Hà Nội|https://nammekong.net|- Kinh doanh bất động sản. - Nhận thầu xây lắp các công trình dân dụng, công nghiệp, bưu điện, giao thông, xây lắp bến cảng, cầu cống, đường dây, trạm biến điện. - Thi công san lấp nền móng, xử lý nền đất yếu, các công trình xây dựng, cấp thoát nước. - Lắp đặt đường ống công nghệ, điện lạnh, trang trí nội thất, gia công, lắp đặt khung nhôm. - XD và KD nhà, cho thuê văn phòng. - Hoạt động mua bán nợ.|23,30|3.225|Constrexim, Vinaconex|VC3 (Constrexim số 3)|CTCP Tập đoàn Nam Cường / Constrexim 3|Hà Nội, Vĩnh Phúc, Quảng Bình, Bình Định
VC7|HNX|Bất động sản|Khu đô thị / Đầu tư BĐS|Trung cấp|961|96090556|10000|28/12/2007|Mr. Hoàng Trọng Đức|Tầng 3 -Tòa Vinaconex 7 - Số 61 - Đường Nguyễn Văn Giáp - P. Từ Liêm - Tp. Hà Nội|https://bgi.vn/|- Thi công xây lắp các công trình dân dụng, công nghiệp, bưu điện, các công trình thủy lợi, giao thông đường bộ các cấp, sân bay, bến cảng, cầu cống, các công trình kỹ thuật hạ tầng đô thị và khu công nghiệp, các công trình đường dây, trạm biến thế đến 110KV. - Thi công san lấp nền móng, xử lý nền đất yếu các công trình xây dựng cấp thoát nước. - Lắp đặt đường ống công nghệ và áp lực, điện lạnh. - Xây dựng và phát triển nhà ở. - Kinh doanh bất động sản. - Kinh doanh xuất nhập khẩu hàng hóa; Xuất|7,10|682|Vinaconex 7|VC7 (Bất động sản Vietuc / Vinaconex 7 cũ)|CTCP Tập đoàn Bất động sản Vietuc|Hà Nội, Quảng Ninh, Đà Nẵng
VHM|HOSE|Bất động sản|Khu đô thị quy mô lớn (Vinhomes Ocean Park, Smart City, Grand Park)|Trung cấp - Cao cấp - Hạng sang|82148|8214824008|10000|17/05/2018|Mr. Phạm Thiếu Hoa|Tòa nhà văn phòng Symphony - Đường Chu Huy Mân - Khu đô thị Vinhomes Riverside - P. Phúc Lợi - Tp. Hà Nội|https://vinhomes.vn/|- Phát triển và kinh doanh bất động sản, cho thuê văn phòng. - Cung cấp dịch vụ quản lý bất động sản và các dịch vụ liên quan.|72,00|591.467|Coteccons, Hòa Bình, Central, Vicons, Delta|Vingroup, Vincom Retail, Vinhomes Industrial Park|CTCP Vinhomes|Hà Nội, TP. Hồ Chí Minh, Hải Phòng, Hưng Yên, Quảng Ninh
VIC|HOSE|Bất động sản|Đa ngành (BĐS Vinhomes + Bán lẻ Vincom + VinFast)|Đa phân khúc (Bình dân tới Siêu sang)|77622|7762186429|10000|19/09/2007|Mr. Nguyễn Việt Quang|Số 07 đường Bằng Lăng 1 - Khu đô thị Vinhomes Riverside - P. Phúc Lợi - Tp. Hà Nội|https://www.vingroup.net/|- Kinh doanh BĐS. - Dịch vụ cho thuê văn phòng, nhà ở, máy móc thiết bị công trình. - Kinh doanh khách sạn, Dịch vụ vui chơi giải trí, làm đẹp, ăn uống. - Hoạt động sáng tác, nghệ thuật và giải trí. - Hoạt động tư vấn quản lý. - Hoạt động hỗ trợ dịch vụ tài chính chưa được phân vào đâu.|243,30|1.888.540|Coteccons, Delta, Vicons|Vingroup, Vinhomes, Vincom Retail, VinFast|Tập đoàn Vingroup - CTCP|Toàn quốc (Hà Nội, TP. Hồ Chí Minh, Hải Phòng, Nha Trang...)
VPH|HOSE|Bất động sản|Căn hộ chung cư / Đất nền (Vạn Phát Hưng)|Trung cấp|954|95357800|10000|09/09/2009|Mr. Võ Nguyễn Như Nguyện|Tầng 2 - Tòa nhà Tulip - Số 15 Hoàng Quốc Việt - P. Phú Thuận - Tp. Hồ Chí Minh|http://vanphathung.com|- Đầu tư phát triển dự án bất động sản và kinh doanh sản phẩm bất động sản... - Hoạt động vui chơi giải trí. - Hoạt động dịch vụ hỗ trợ kinh doanh. - Hoạt động của các câu lạc bộ thể thao.|2,98|284|VPH In-house, Phước Thành|Vạn Phát Hưng|CTCP Vạn Phát Hưng|TP. Hồ Chí Minh (Nhà Bè, Quận 7)
VPI|HOSE|Bất động sản|Khu đô thị / BĐS thương mại (Văn Phú - Invest)|Trung cấp - Cao cấp|3200|320049577|10000|29/06/2018|Mr. Tô Như Toàn|Số 104 Thái Thịnh - P. Đống Đa - Tp. Hà Nội|https://vanphu.vn|- Đầu tư, phát triển các dự án BĐS. - Quản lý và kinh doanh BĐS. - Tư vấn đầu tư. - Thiết kế, thi công, giám sát các công trình dân dụng, công nghiệp, giao thông và thủy lợi. - Sản xuất và kinh doanh vật liệu xây dựng. - Sản xuất và kinh doanh đồ gỗ, nội thất công trình.|59,60|19.075|Delta, Coteccons, Hòa Bình|Văn Phú - Invest|CTCP Đầu tư Văn Phú - Invest|Hà Nội, Phú Quốc, Nha Trang, Vĩnh Phúc
VRC|HOSE|Bất động sản|Đầu tư BĐS / Cảng biển|Trung cấp|500|50000000|10000|26/07/2010|Ms. Trần Thị Vân|Tầng 6 - Tòa nhà Smart View - 161A (1 phần)-163-165 - Đường Trần Hưng Đạo - P. Cầu Ông Lãnh - Tp. Hồ Chí Minh|http://vrc.com.vn|- Kinh doanh bất động sản. - Hoạt động đầu tư. - Hoạt động M&A.|11,45|573|Nhà thầu khu vực TP.HCM|Bất động sản và Đầu tư VRC|CTCP Bất động sản và Đầu tư VRC|TP. Hồ Chí Minh & Bà Rịa - Vũng Tàu
VRE|HOSE|Bất động sản|Trung tâm thương mại / Bán lẻ cho thuê (Vincom Center/Mega Mall)|Trung cấp - Cao cấp (Mặt bằng bán lẻ)|23288|2272318410|10000|06/11/2017|Ms. Phạm Thị Thu Hiền|Tòa nhà văn phòng Symphony - Đường Chu Huy Mân - KĐT sinh thái Vinhomes Riverside - P. Phúc Lợi - Tp. Hà Nội|https://vincom.com.vn|- Phát triển & vận hành bất động sản bán lẻ. - Bất động sản nhà phố thương mại để bán.|25,60|58.171|Delta, Hòa Bình, Coteccons|Vincom Retail (Hệ sinh thái Vingroup)|CTCP Vincom Retail|Toàn quốc (Hệ thống TTTM tại các thành phố lớn)
VTJ|HNX|Bất động sản|Khu công nghiệp / Xây lắp|Không áp dụng (BĐS KCN)|114|11400000|10000|26/04/2017|Mr. Lê Chí Long|Số 24 - Ngách 1 - Ngõ 46 - Đường Phạm Ngọc Thạch - P. Văn Miếu - Quốc Tử Giám - Tp. Hà Nội|https://vinainvest.com.vn|- Mua bán đồ dùng cá nhân và gia đình, thuốc lá điếu sản xuất trong nước và nguyên phụ liệu, máy móc, thiết bị ngành thuốc lá - Môi giới thương mại - Đại lý mua bán, ký gửi hàng - Tư vấn đầu tư, tư vấn quản lý kinh doanh - Lập dự án đầu tư và kinh doanh BĐS.|4,40|50|Vạn Tường Cons|Đầu tư và Xây dựng Vạn Tường|CTCP Đầu tư và Xây dựng Vạn Tường|Đà Nẵng & khu vực miền Trung
"""


# ============================================================
# TÊN NHẬN DIỆN
# Không sửa dữ liệu gốc trong COMPANY_DATA.
# Những dòng chưa chắc tên: chỉ khớp ticker, tránh gán sai.
# ============================================================

NAME_FIXES = {
    "EVG": "CTCP Tập đoàn Everland",
    "LGL": "CTCP Đầu tư và Phát triển Đô thị Long Giang",
    "NHA": "Tổng Công ty Đầu tư Phát triển Nhà và Đô thị Nam Hà Nội",
    "VC3": "CTCP Tập đoàn Nam Mê Kông",
    "TN1": "CTCP ROX Key Holdings",
}

TICKER_ONLY = {
    "DTD", "HAR", "NRC", "NTL", "PTL", "TAL", "VTJ",
}

EXTRA_ALIASES = {
    "AGG": ["An Gia"],
    "BCM": ["Becamex IDC"],
    "CDC": ["Chương Dương Corp"],
    "CEO": ["CEO Group", "Tập đoàn CEO", "Tập đoàn C.E.O"],
    "CIG": ["COMA 18"],
    "CKG": ["CIC Group"],
    "CRE": ["Cen Land", "CenLand"],
    "CSC": ["Cotana", "Cotana Group"],
    "DIG": ["DIC Corp", "DIC Corporation"],
    "DRH": ["DRH Holdings"],
    "DXG": ["Đất Xanh", "Đất Xanh Group"],
    "DXS": ["Đất Xanh Services", "Dat Xanh Services"],
    "EVG": ["Everland", "EverLand Group"],
    "FDC": ["Fideco"],
    "FIR": ["First Real"],
    "HDC": ["Hodeco"],
    "HDG": ["Tập đoàn Hà Đô", "Hà Đô Group"],
    "HLD": ["HUDland", "HUD Land"],
    "HPX": ["Hải Phát Invest"],
    "HQC": ["Địa ốc Hoàng Quân"],
    "IJC": ["Becamex IJC"],
    "ITC": ["Intresco"],
    "KBC": ["Đô thị Kinh Bắc", "Kinh Bắc City"],
    "KDH": ["Khang Điền"],
    "KHG": ["Khải Hoàn Land", "Khai Hoan Land"],
    "KOS": ["KOSY"],
    "KSF": ["Sunshine Homes"],
    "L14": ["Licogi 14"],
    "LDG": ["LDG Investment"],
    "LGL": ["Long Giang Land", "Longgiang Land"],
    "NBB": ["Đầu tư Năm Bảy Năm"],
    "NHA": ["Đầu tư Phát triển Nhà và Đô thị Nam Hà Nội"],
    "NLG": ["Nam Long", "Nam Long Group"],
    "NVL": ["Novaland", "Nova Land", "Đầu tư Địa ốc No Va"],
    "PDR": ["Phát Đạt"],
    "QCG": ["Quốc Cường Gia Lai"],
    "SCR": ["TTC Land"],
    "SGR": ["Saigonres"],
    "SJS": ["Sudico"],
    "SZB": ["Sonadezi Biên Hòa"],
    "SZC": ["Sonadezi Châu Đức"],
    "SZL": ["Sonadezi Long Thành"],
    "TDH": ["Thuduc House", "Thủ Đức House"],
    "TIG": ["Thăng Long Investment Group"],
    "TN1": ["ROX Key Holdings", "TNS Holdings"],
    "VC3": ["Nam Mê Kông", "Nam Mekong", "Vinaconex 3"],
    "VHM": ["Vinhomes"],
    "VIC": ["Vingroup"],
    "VPH": ["Vạn Phát Hưng"],
    "VPI": ["Văn Phú - Invest", "Văn Phú Invest"],
    "VRE": ["Vincom Retail"],
}

AMBIGUOUS_TICKERS = {"CEO", "API", "CDC"}

FIELDS = [
    "article_id", "source", "category", "url", "published_at",
    "title", "description", "content", "author",
    "ticker", "company", "match_keyword", "scraped_at",
    *PROFILE_FIELDS,
    "profile_source", "profile_as_of",
]


def norm(value):
    return " ".join(
        unicodedata.normalize("NFC", str(value or "")).split()
    )


def jd(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def digest(value):
    return hashlib.sha256(jd(value).encode("utf-8")).hexdigest()


def load_embedded_companies():
    profiles = {}

    for line in COMPANY_DATA.strip().splitlines():
        values = [norm(value) for value in line.split("|")]

        if len(values) != len(PROFILE_FIELDS):
            raise ValueError(
                f"Dòng {values[0]} có {len(values)} cột, phải có 19. "
                "Kiểm tra thao tác copy code."
            )

        profile = dict(zip(PROFILE_FIELDS, values))
        ticker = profile["Ma_CK"]

        if ticker in profiles:
            raise ValueError(f"Trùng ticker trong dữ liệu: {ticker}")

        profiles[ticker] = profile

    if len(profiles) != 83:
        raise ValueError(
            f"Chỉ nhận được {len(profiles)}/83 công ty. "
            "Bạn cần copy đủ khối COMPANY_DATA."
        )

    companies = {}

    for ticker, profile in profiles.items():
        name = NAME_FIXES.get(ticker, profile["Ten_Cong_Ty"])
        aliases = []

        if ticker not in TICKER_ONLY:
            expanded = re.sub(
                r"\bCTCP\b", "Công ty Cổ phần", name, flags=re.I
            )
            core = re.sub(r"\([^)]*\)", "", expanded)
            core = re.sub(
                r"\s*-\s*Công ty Cổ phần\s*$", "", core, flags=re.I
            )
            core = re.sub(
                r"^(?:Tổng Công ty(?: Cổ phần)?|Công ty Cổ phần)\s+",
                "", core, flags=re.I
            )
            core = norm(core)

            if "/" not in name:
                aliases.extend([name, expanded])
                if len(core) >= 8:
                    aliases.append(core)

            aliases.extend(EXTRA_ALIASES.get(ticker, []))

        companies[ticker] = {
            "name": name,
            "aliases": sorted(set(
                norm(alias).casefold()
                for alias in aliases
                if norm(alias)
            )),
        }

    owners = {}

    for ticker, info in companies.items():
        for alias in info["aliases"]:
            owners.setdefault(alias, set()).add(ticker)

    for info in companies.values():
        info["aliases"] = [
            alias for alias in info["aliases"]
            if len(owners[alias]) == 1
        ]

    return profiles, companies


def match_companies(article, companies):
    text = norm("\n".join(
        article.get(field, "")
        for field in ("title", "description", "content")
    ))
    folded = text.casefold()
    candidates = []

    for ticker, info in companies.items():
        for alias in info["aliases"]:
            for match in re.finditer(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
                folded,
            ):
                candidates.append((
                    match.start(), match.end(), ticker, alias
                ))

    # Tên dài ưu tiên, tránh "Đất Xanh" lấy nhầm
    # cụm "Dịch vụ Bất động sản Đất Xanh".
    occupied = bytearray(len(folded))
    found = {}

    for start, end, ticker, alias in sorted(
        candidates,
        key=lambda hit: (-(hit[1] - hit[0]), hit[0], hit[2]),
    ):
        if any(occupied[start:end]):
            continue

        occupied[start:end] = b"\x01" * (end - start)
        found.setdefault(ticker, set()).add(alias)

    for ticker in companies:
        pattern = r"(?<!\w)" + re.escape(ticker) + r"(?!\w)"

        # Mã phải viết hoa; không khớp NVL bên trong NVLX.
        if not re.search(pattern, text):
            continue

        if ticker in AMBIGUOUS_TICKERS and ticker not in found:
            hint = (
                r"(?:"
                r"(?:HOSE|HSX|HNX|UPCOM)\s*[:：-]?\s*"
                r"|(?:mã(?:\s+chứng khoán)?|cổ phiếu)\s*[:：-]?\s*"
                r")"
                + re.escape(ticker)
                + r"(?!\w)"
            )
            if not re.search(hint, text, flags=re.I):
                continue

        found.setdefault(ticker, set()).add(ticker)

    return [
        (ticker, companies[ticker]["name"], sorted(aliases))
        for ticker, aliases in sorted(found.items())
        if SELECTED_COMPANY is None or ticker == SELECTED_COMPANY
    ]


# ============================================================
# HTTP / PARSER
# ============================================================

class CrawlError(RuntimeError):
    pass


def parse_date(value):
    value = str(value or "").strip()

    if re.match(r"^\d{4}-\d{2}-\d{2}", value):
        date = isoparse(value)
    else:
        match = re.search(
            r"(\d{1,2})/(\d{1,2})/(\d{4})"
            r"(?:\s*[-–,]?\s*(\d{1,2}):(\d{2})(?::(\d{2}))?)?",
            value,
        )
        if not match:
            raise ValueError("Không đọc được ngày xuất bản")

        day, month, year, hour, minute, second = match.groups()
        date = datetime(
            int(year), int(month), int(day),
            int(hour or 0), int(minute or 0), int(second or 0),
        )

    return (
        date.replace(tzinfo=VN_TZ)
        if date.tzinfo is None
        else date.astimezone(VN_TZ)
    )


def walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def article_date(soup):
    candidates = []

    for selector in [
        'meta[property="article:published_time"]',
        'meta[name="pubdate"]',
        'meta[name="publishdate"]',
        'meta[name="datePublished"]',
        '[itemprop="datePublished"]',
    ]:
        for node in soup.select(selector):
            candidates.append(
                node.get("content")
                or node.get("datetime")
                or node.get_text(" ", strip=True)
            )

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            for obj in walk_json(json.loads(script.get_text())):
                kinds = obj.get("@type", [])
                kinds = [kinds] if isinstance(kinds, str) else kinds

                if (
                    isinstance(kinds, list)
                    and any(
                        isinstance(kind, str) and "Article" in kind
                        for kind in kinds
                    )
                    and obj.get("datePublished")
                ):
                    candidates.append(obj["datePublished"])
        except (ValueError, TypeError, RecursionError):
            continue

    for node in soup.select(
        "time[datetime], .pdate, .detail-time, .publish-time, .date"
    ):
        candidates.append(
            node.get("datetime") or node.get_text(" ", strip=True)
        )

    for value in candidates:
        try:
            return parse_date(value)
        except (ValueError, TypeError, OverflowError):
            continue

    raise CrawlError("Thiếu ngày xuất bản; không suy ngày từ URL")


def retry_wait(exc, attempt):
    wait = 2 ** attempt
    response = getattr(exc, "response", None)

    if response is not None:
        value = response.headers.get("Retry-After", "")
        if value.isdigit():
            wait = max(wait, int(value))

    return wait


def fetch_once(session, method, url, **kwargs):
    # Một lần gọi = một lần thử; không có adapter retry ngầm.
    time.sleep(REQUEST_DELAY)

    with session.request(
        method, url, timeout=(15, 60), stream=True, **kwargs
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        data = bytearray()

        for chunk in response.iter_content(65536):
            if len(data) + len(chunk) > MAX_HTML_BYTES:
                raise CrawlError("HTML vượt giới hạn 8 MB")
            data.extend(chunk)

    html = data.decode("utf-8-sig", errors="replace")

    if re.search(
        r"<title>[^<]*(just a moment|access denied|captcha)",
        html, re.I,
    ):
        raise CrawlError("Máy chủ trả trang chặn/challenge")

    if "json" in content_type.lower():
        try:
            html = json.loads(html)
        except ValueError as exc:
            raise CrawlError("Endpoint trả JSON hỏng") from exc

        if not isinstance(html, str):
            raise CrawlError("Endpoint trả JSON khác cấu trúc HTML")

    return html


def fetch_listing(session, method, url, **kwargs):
    for attempt in range(1, 4):
        try:
            return fetch_once(session, method, url, **kwargs)
        except (requests.RequestException, CrawlError) as exc:
            if attempt == 3:
                raise
            print(f"Trang danh sách lỗi {attempt}/3: {exc}", flush=True)
            time.sleep(retry_wait(exc, attempt))


LIST_LINK_SELECTOR = '.knswli-title a[href], .klw-top-news h2 a[href], .klw-top-news h3 a[href], li[role="article"] h3 a[href]'


def category_listing(session, category, page, state):
    init_url = BASE + '/' + category + ('.htm' if SITE == 'vietstock' else '.chn')
    if category not in state:
        initial = fetch_listing(session, 'GET', init_url)
        if SITE == 'kenh14':
            soup = BeautifulSoup(initial, 'html.parser')
            node = soup.select_one('#hdZoneId')
            zone = str(node.get('value', '')) if node else ''
            soup.decompose()
            if not zone.isdigit() or int(zone) <= 0:
                raise CrawlError('Không đọc được zone ID chuyên mục; chưa tăng trang.')
            state[category] = (zone, initial)
        else:
            state[category] = True
    if SITE == 'kenh14':
        zone, initial = state[category]
        html = initial if page == 1 else fetch_listing(
            session, 'GET',
            f'{BASE}/timeline/laytinmoitronglist-{zone}/page-{page}.chn',
            headers={'Referer': init_url})
        return html, None
    html = fetch_listing(session, 'POST', BASE + '/StartPage/ChannelContentPage',
        data={'channelID': CATEGORIES[category], 'page': page,
              'fromdate': DATE_START.strftime('%Y-%m-%d'),
              'todate': DATE_END_EXCLUSIVE.strftime('%Y-%m-%d')},
        headers={'Referer': init_url, 'X-Requested-With': 'XMLHttpRequest'})
    soup = BeautifulSoup(html, 'html.parser')
    node = soup.select_one('#totalChannelRow, input[name="totalChannelRow"]')
    value = str(node.get('value', '')) if node else ''
    soup.decompose()
    if not value.isdigit():
        match = re.search(r'totalChannelRow\s*[:=]\s*[\"\x27]?(\d+)', html)
        value = match.group(1) if match else ''
    total = int(value) if value.isdigit() else None
    if total == 0:
        raise CrawlError('Máy chủ báo 0 bài; chưa xác nhận hết khoảng ngày.')
    last = (total + 9) // 10 if total is not None else None
    if last is not None and page > last:
        raise CrawlError(f'Checkpoint trang {page} vượt trang cuối {last}; cần đối chiếu bộ lọc.')
    return html, last


def links_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []

    try:
        for node in soup.select(LIST_LINK_SELECTOR):
            try:
                parts = urlsplit(urljoin(BASE + "/", node["href"]))
            except ValueError:
                continue

            if (
                parts.scheme in ("http", "https")
                and parts.hostname in ("kenh14.vn", "www.kenh14.vn")
                and re.fullmatch(
                    r"/[^/]+-\d{8,}\.chn", parts.path, re.I
                )
            ):
                links.append(urlunsplit(
                    ("https", "kenh14.vn", parts.path, "", "")
                ))

        return list(dict.fromkeys(links))
    finally:
        soup.decompose()


def read_article(session, url):
    soup = BeautifulSoup(
        fetch_once(session, "GET", url), "html.parser"
    )

    try:
        date = article_date(soup)
        title = soup.select_one("h1")
        body = soup.select_one(
            '.detail-content.afcbc-body, .detail-content, '
            '[data-role="content"]'
        )

        if title is None or body is None:
            raise CrawlError("Không tìm thấy tiêu đề/vùng nội dung")

        def meta(selector):
            node = soup.select_one(selector)
            return str(node.get("content") or "") if node else ""

        sapo = soup.select_one(".knc-sapo, .detail-sapo, .sapo")
        description = (
            sapo.get_text(" ", strip=True) if sapo
            else meta(
                'meta[property="og:description"], '
                'meta[name="description"]'
            )
        )

        author_node = soup.select_one(".kbwcm-author, .detail-author, .author")
        author = (
            author_node.get_text(" ", strip=True) if author_node
            else meta('meta[name="author"]')
        )

        for node in body.select(
            "script, style, iframe, form, nav, noscript, "
            '.related-news, .readmore, [type="RelatedNews"], '
            '[type="RelatedOneNews"]'
        ):
            node.decompose()

        content = body.get_text("\n", strip=True)

        if len(content) < 100:
            raise CrawlError("Nội dung rỗng hoặc quá ngắn")

        return {
            "published_at": date.isoformat(),
            "title": title.get_text(" ", strip=True),
            "description": description,
            "content": content,
            "author": author,
        }
    finally:
        soup.decompose()


# ============================================================
# DRIVE / KHÓA / SQLITE
# ============================================================

def prepare_drive():
    from google.colab import drive

    if not (
        os.path.ismount("/content/drive")
        and Path("/content/drive/MyDrive").is_dir()
    ):
        try:
            drive.mount("/content/drive")
        except Exception as exc:
            raise RuntimeError(
                "Chưa cấp quyền Drive thành công. Nếu xuất hiện "
                "'credential propagation was unsuccessful', "
                "kiểm tra tài khoản, popup/cookie và cấp quyền lại. "
                "Crawler chưa bắt đầu chạy."
            ) from exc

    if not (
        os.path.ismount("/content/drive")
        and Path("/content/drive/MyDrive").is_dir()
    ):
        raise RuntimeError("Drive chưa mount; dừng để tránh lưu vào ổ tạm.")


@contextmanager
def writer_lock(path):
    # Chống chạy trùng trong cùng runtime.
    # Không chạy hai runtime cùng ghi vào một checkpoint.
    import fcntl

    key = hashlib.sha256(str(path).encode()).hexdigest()[:20]
    lockfile = Path(tempfile.gettempdir()) / f"kenh14_{key}.lock"

    with lockfile.open("a+b") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "Pipeline company đang chạy trong runtime này."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def open_db(path):
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA synchronous=FULL")
    checked = db.execute("PRAGMA quick_check").fetchall()
    if checked != [("ok",)]:
        db.close()
        raise RuntimeError(f"Checkpoint không toàn vẹn: {checked[:3]}; không tạo lại/xóa dữ liệu.")

    # Cùng schema cache bài với bản company trước.
    db.executescript("""
        CREATE TABLE IF NOT EXISTS article_cache(
            url TEXT PRIMARY KEY,
            category TEXT,
            page INTEGER,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            data_json TEXT,
            error TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS results(
            run TEXT,
            url TEXT,
            rows_json TEXT,
            PRIMARY KEY(run, url)
        );
        CREATE TABLE IF NOT EXISTS progress(
            run TEXT,
            category TEXT,
            next_page INTEGER,
            old_streak INTEGER,
            status TEXT,
            PRIMARY KEY(run, category)
        );
        CREATE TABLE IF NOT EXISTS page_signatures(
            run TEXT,
            category TEXT,
            page INTEGER,
            signature TEXT,
            PRIMARY KEY(run, category, page)
        );
    """)
    db.commit()
    return db


class Crawler:
    def __init__(self, profiles, companies):
        self.profiles = profiles
        self.companies = companies
        self.config = {
            "version": "kenh14_embedded_company_dates_v2",
            "profiles": profiles,
            "companies": companies,
            "selected": SELECTED_COMPANY,
            "date_start": DATE_START.isoformat(),
            "date_end_exclusive": DATE_END_EXCLUSIVE.isoformat(),
            "categories": CATEGORIES,
        }
        self.run_id = digest(self.config)[:12]
        self.db_path = OUTPUT_DIR / "company_checkpoint_dates_v2.sqlite3"
        self.csv_path = OUTPUT_DIR / f"kenh14_company_{self.run_id}.csv"
        self.added = 0

    def check_space(self):
        if shutil.disk_usage(OUTPUT_DIR).free < MIN_FREE_MB * 1024**2:
            raise OSError(f"Ổ lưu còn dưới {MIN_FREE_MB} MB.")

    def get_article(self, category, page, url):
        saved = self.db.execute(
            "SELECT status, attempts, data_json FROM article_cache WHERE url=?",
            (url,),
        ).fetchone()

        if saved and saved[0] == "ok":
            return json.loads(saved[2])

        attempts = saved[1] if saved else 0

        if attempts >= MAX_ARTICLE_ATTEMPTS:
            return None

        while attempts < MAX_ARTICLE_ATTEMPTS:
            attempts += 1

            # Ghi trước HTTP: mất kết nối runtime không reset số lần.
            with self.db:
                self.db.execute(
                    """
                    INSERT INTO article_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        category=excluded.category,
                        page=excluded.page,
                        status=excluded.status,
                        attempts=excluded.attempts,
                        error=excluded.error,
                        updated_at=excluded.updated_at
                    """,
                    (
                        url, category, page, "pending", attempts, None,
                        "Đã bắt đầu lần thử; có thể bị ngắt",
                        datetime.now(VN_TZ).isoformat(),
                    ),
                )

            try:
                article = read_article(self.session, url)
            except (requests.RequestException, CrawlError) as exc:
                status = (
                    "skipped"
                    if attempts >= MAX_ARTICLE_ATTEMPTS
                    else "pending"
                )

                with self.db:
                    self.db.execute(
                        """
                        UPDATE article_cache
                        SET status=?, error=?, updated_at=?
                        WHERE url=?
                        """,
                        (
                            status, str(exc),
                            datetime.now(VN_TZ).isoformat(), url,
                        ),
                    )

                print(
                    f"Lỗi {attempts}/{MAX_ARTICLE_ATTEMPTS}: {url}\n  {exc}",
                    flush=True,
                )

                if status == "skipped":
                    print("  → Bỏ qua, lần chạy sau không thử lại.", flush=True)
                    return None

                time.sleep(retry_wait(exc, attempts))
                continue

            with self.db:
                self.db.execute(
                    """
                    UPDATE article_cache
                    SET status='ok', data_json=?, error=NULL, updated_at=?
                    WHERE url=?
                    """,
                    (
                        jd(article), datetime.now(VN_TZ).isoformat(), url,
                    ),
                )

            return article

    def process(self, category, page, url):
        done = self.db.execute(
            "SELECT 1 FROM results WHERE run=? AND url=?",
            (self.run_id, url),
        ).fetchone()

        if done:
            return 0

        article = self.get_article(category, page, url)

        if article is None:
            return 0

        date = parse_date(article["published_at"])
        rows = []

        if DATE_START <= date < DATE_END_EXCLUSIVE:
            matched = match_companies(article, self.companies)
            article_id = re.search(r"-(\d+)\.chn$", url, re.I)

            base = {
                "article_id": (
                    article_id.group(1)
                    if article_id
                    else hashlib.sha256(url.encode()).hexdigest()[:20]
                ),
                "source": "kenh14",
                "category": category,
                "url": url,
                **article,
                "scraped_at": datetime.now(VN_TZ).isoformat(),
            }

            for ticker, company, aliases in matched:
                rows.append({
                    **base,
                    "ticker": ticker,
                    "company": company,
                    "match_keyword": "; ".join(aliases),
                    **self.profiles[ticker],
                    "profile_source": "user_table_embedded",
                    "profile_as_of": "",
                })

        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO results VALUES (?, ?, ?)",
                (self.run_id, url, jd(rows) if rows else None),
            )

        self.added += len(rows)
        return len(rows)

    def export(self):
        temporary = self.csv_path.with_suffix(".csv.tmp")
        total = 0

        with temporary.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()

            for (payload,) in self.db.execute(
                """
                SELECT rows_json FROM results
                WHERE run=? AND rows_json IS NOT NULL
                ORDER BY rowid
                """,
                (self.run_id,),
            ):
                for row in json.loads(payload):
                    # Cho phép dùng cache/schema của bản trước:
                    # chỉ xuất các trường thuộc bản hiện tại.
                    writer.writerow({
                        field: row.get(field, "") for field in FIELDS
                    })
                    total += 1

            file.flush()
            os.fsync(file.fileno())

        temporary.replace(self.csv_path)

        errors = [
            {
                "url": url,
                "category": category,
                "page": page,
                "attempts": attempts,
                "status": (
                    "skipped" if attempts >= MAX_ARTICLE_ATTEMPTS
                    else "pending"
                ),
                "error": error,
            }
            for url, category, page, attempts, error in self.db.execute(
                """
                SELECT url, category, page, attempts, error
                FROM article_cache WHERE status <> 'ok'
                """
            )
        ]

        (OUTPUT_DIR / "errors.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = {
            "run": self.run_id,
            "companies": len(self.companies),
            "csv_rows": total,
            "new_rows_this_execution": self.added,
            "skipped_errors": sum(
                item["status"] == "skipped" for item in errors
            ),
            "pending_errors": sum(
                item["status"] == "pending" for item in errors
            ),
            "categories": [
                {"category": category, "next_page": page, "status": status}
                for category, page, status in self.db.execute(
                    """
                    SELECT category, next_page, status
                    FROM progress WHERE run=?
                    """,
                    (self.run_id,),
                )
            ],
        }

        (OUTPUT_DIR / f"status_{self.run_id}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return total

    def crawl_category(self, category):
        saved = self.db.execute(
            'SELECT next_page, old_streak, status FROM progress WHERE run=? AND category=?',
            (self.run_id, category)).fetchone()
        if saved and saved[2].startswith('done:'):
            print(category, saved[2], flush=True)
            return
        page = start = saved[0] if saved else 1
        old_streak = saved[1] if saved else 0
        state = {}

        def progress(next_page, status):
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?, ?)',
                    (self.run_id, category, next_page, old_streak, status))

        print(f'[{SITE}] {category}: tiếp tục trang {start}', flush=True)
        try:
            for page in range(start, start + MAX_PAGES):
                self.check_space()
                progress(page, 'running')
                html, last = category_listing(self.session, category, page, state)
                urls = links_from_html(html)
                if not urls:
                    (OUTPUT_DIR / f'{category}_last_empty.html').write_text(html, encoding='utf-8')
                    raise CrawlError('Không có link; chưa xác nhận hoàn tất khoảng ngày.')
                signature = digest(sorted(urls))
                repeated = self.db.execute(
                    'SELECT page FROM page_signatures WHERE run=? AND category=? AND signature=? AND page<>?',
                    (self.run_id, category, signature, page)).fetchone()
                if repeated:
                    raise CrawlError(f'Trang {page} lặp trang {repeated[0]}; tạm dừng.')
                dates = []
                before = self.added
                for url in urls:
                    self.process(category, page, url)
                    saved_article = self.db.execute(
                        "SELECT data_json FROM article_cache WHERE url=? AND status='ok'", (url,)).fetchone()
                    if saved_article:
                        dates.append(parse_date(json.loads(saved_article[0])['published_at']))
                # Bài lỗi/không xác định ngày không được coi là bài cũ.
                all_known = len(dates) == len(urls)
                all_old = all_known and all(d < DATE_START for d in dates)
                old_streak = old_streak + 1 if all_old else 0
                reason = None
                if SITE == 'vietstock':
                    if any(d < DATE_START - timedelta(days=1) or d >= DATE_END_EXCLUSIVE + timedelta(days=1) for d in dates):
                        raise CrawlError('Ngày bài không khớp bộ lọc POST; cần kiểm tra, chưa tăng trang.')
                    if last is not None and page == last:
                        reason = 'done: hết danh sách của bộ lọc ngày; xem errors.json để biết bài lỗi'
                elif old_streak >= OLD_PAGE_STREAK:
                    reason = 'done: đủ trang liên tiếp trước start date (giả định danh sách theo ngày giảm dần)'
                with self.db:
                    self.db.execute('INSERT OR REPLACE INTO page_signatures VALUES (?, ?, ?, ?)',
                        (self.run_id, category, page, signature))
                    self.db.execute('INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?, ?)',
                        (self.run_id, category, page + 1, old_streak, reason or 'running'))
                span = f'{min(dates).date()} → {max(dates).date()}' if dates else 'chưa đọc được ngày'
                print(f'{category} | trang {page} | {span} | thêm {self.added-before} dòng', flush=True)
                # Xuất mỗi trang; CSV có thể dựng lại từ results nếu bị ngắt.
                print(f'CSV: {self.export():,} dòng', flush=True)
                if reason:
                    print(reason, flush=True)
                    return
            progress(page + 1, 'paused_page_budget')
            print(f'Hết {MAX_PAGES} trang/lượt; chạy lại tiếp trang {page+1}.', flush=True)
        except (requests.RequestException, CrawlError) as exc:
            progress(page, f'paused: {exc}')
            print(f'Tạm dừng {category} trang {page}: {exc}', flush=True)

    def run(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with writer_lock(self.db_path):
            self.db = open_db(self.db_path)

            try:
                self.check_space()
                (OUTPUT_DIR / f"config_{self.run_id}.json").write_text(
                    json.dumps(self.config, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                with requests.Session() as self.session:
                    self.session.headers.update({
                        "User-Agent": "Mozilla/5.0",
                        "Referer": BASE + "/",
                        "Accept-Language": "vi-VN,vi;q=0.9",
                        "Accept": "text/html,*/*;q=0.8",
                    })

                    print(
                        f"83 công ty | Run: {self.run_id}\n"
                        f"Khôi phục CSV: {self.export()} dòng\n"
                        f"Output: {OUTPUT_DIR}",
                        flush=True,
                    )

                    categories = (
                        list(CATEGORIES)
                        if SELECTED_CATEGORY == "all"
                        else [SELECTED_CATEGORY]
                    )

                    try:
                        for category in categories:
                            self.crawl_category(category)

                    except KeyboardInterrupt:
                        print(
                            "\nĐã ngắt; chạy lại để tiếp tục checkpoint.",
                            flush=True,
                        )

                    finally:
                        total = self.export()
                        print(
                            f"\nĐã lưu {total} dòng.\nCSV: {self.csv_path}",
                            flush=True,
                        )

            finally:
                self.db.close()


def main():
    if MAX_PAGES < 1:
        raise ValueError("MAX_PAGES phải >= 1")

    if SELECTED_CATEGORY not in {"all", *CATEGORIES}:
        raise ValueError("SELECTED_CATEGORY không hợp lệ")

    profiles, companies = load_embedded_companies()

    if SELECTED_COMPANY and SELECTED_COMPANY not in companies:
        raise ValueError(f"Không có công ty {SELECTED_COMPANY}")

    print("Đã nạp đủ 83 công ty × 19 cột ngay trong code.", flush=True)
    prepare_drive()
    Crawler(profiles, companies).run()


if __name__ == "__main__":
    main()