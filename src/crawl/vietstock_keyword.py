# -*- coding: utf-8 -*-
# VIETSTOCK KEYWORD — BỘ LỌC NHIỀU NHÓM TỪ KHÓA
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

BASE = "https://vietstock.vn"

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/crawl data/vietstock_keyword"
)

VN_TZ = timezone(timedelta(hours=7))
DATE_START = datetime(2024, 1, 1, tzinfo=VN_TZ)
DATE_END_EXCLUSIVE = datetime(2026, 9, 10, tzinfo=VN_TZ)

MAX_PAGES = 1000                  # Mỗi chuyên mục / mỗi lượt
MAX_ARTICLE_ATTEMPTS = 3          # Tổng cộng, kể cả chạy lại
REQUEST_DELAY = 1.5
MAX_HTML_BYTES = 8 * 1024 * 1024
MIN_FREE_MB = 100

MIN_CONCEPTS = 3
WINDOW_CHARS = 1000
SELECTED_CATEGORY = "all"         # all hoặc một tên trong CATEGORIES

# Không dừng dựa trên ngày của một bài hoặc một trang.
# Bài ngoài khoảng ngày bị loại, nhưng vẫn tiếp tục phân trang.

SITE = 'vietstock'
CATEGORIES = {'chung-khoan': 144, 'bat-dong-san': 763, 'doanh-nghiep': 733}
LIST_LINK_SELECTOR = '.channelContent a[href]'

ASSETS = {
    "bds_chung": [
        "bất động sản", "bđs", "nhà đất", "địa ốc",
    ],
    "nha_o": [
        "nhà ở", "thị trường nhà ở",
    ],
    "can_ho": [
        "căn hộ", "chung cư", "căn hộ chung cư",
    ],
    "dat_nen": [
        "đất nền", "lô đất",
    ],
    "nha_pho": [
        "nhà phố", "nhà mặt phố", "nhà mặt tiền",
    ],
    "lien_ke": [
        "nhà liền kề", "nhà ở liền kề",
    ],
    "biet_thu": [
        "biệt thự",
    ],
    "shophouse": [
        "shophouse", "nhà phố thương mại",
    ],
    "nghi_duong": [
        "bất động sản nghỉ dưỡng", "condotel", "căn hộ du lịch",
    ],
    "nha_xuong": [
        "nhà xưởng", "kho xưởng", "bất động sản công nghiệp",
    ],
}

CONTEXT = {
    "gia": [
        "giá nhà", "giá đất", "giá căn hộ", "giá chung cư",
        "giá bán", "giá thuê", "giá cho thuê",
    ],
    "mua_ban": [
        "mua nhà", "bán nhà", "mua đất", "bán đất",
        "mua căn hộ", "bán căn hộ",
    ],
    "cho_thue": [
        "thuê nhà", "thuê căn hộ", "thuê chung cư",
        "cho thuê nhà", "cho thuê căn hộ", "mặt bằng cho thuê",
    ],
    "nguon_cung": [
        "nguồn cung nhà ở", "nguồn cung căn hộ",
        "nguồn cung", "căn hộ mới",
    ],
    "nhu_cau": [
        "nhu cầu nhà ở", "nhu cầu mua nhà", "nhu cầu thuê nhà",
        "nhu cầu ở thực", "người mua nhà",
    ],
    "giao_dich": [
        "giao dịch nhà đất", "giao dịch bất động sản",
        "thanh khoản bất động sản", "tỷ lệ hấp thụ", "tỉ lệ hấp thụ",
    ],
    "khu_dan_cu": [
        "khu đô thị", "khu dân cư", "khu nhà ở",
        "dự án nhà ở", "dự án chung cư", "dự án căn hộ",
    ],
    "mo_ban": [
        "mở bán", "chào bán căn hộ", "mở bán nhà ở",
    ],
    "ban_giao": [
        "bàn giao nhà", "bàn giao căn hộ", "nhận bàn giao nhà",
    ],
}

CONCEPTS = {**ASSETS, **CONTEXT}

FIELDS = [
    "article_id",
    "source",
    "category",
    "url",
    "published_at",
    "title",
    "description",
    "content",
    "author",
    "keyword_groups",
    "matched_keywords",
    "scraped_at",
]


def norm(value):
    return " ".join(
        unicodedata.normalize("NFC", str(value or "")).split()
    )

def jd(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

def digest(value):
    return hashlib.sha256(jd(value).encode("utf-8")).hexdigest()

def phrase_present(text, alias):
    pattern = r"(?<!\w)" + re.escape(alias.casefold()) + r"(?!\w)"
    return re.search(pattern, text)

def has_asset(text):
    text = norm(text).casefold()

    return any(
        phrase_present(text, alias)
        for aliases in ASSETS.values()
        for alias in aliases
    )

def find_hits(text):
    text = norm(text).casefold()
    candidates = []

    for concept, aliases in CONCEPTS.items():
        for alias in aliases:
            pattern = (
                r"(?<!\w)" + re.escape(alias.casefold()) + r"(?!\w)"
            )

            for match in re.finditer(pattern, text):
                candidates.append(
                    (match.start(), match.end(), concept, alias)
                )

    occupied = []
    hits = {}

    # Ưu tiên cụm dài, không đếm lặp các cụm nằm lồng nhau.
    for start, end, concept, alias in sorted(
        candidates,
        key=lambda item: (
            -(item[1] - item[0]),
            item[0],
            item[2],
        ),
    ):
        if any(start < b and a < end for a, b in occupied):
            continue

        occupied.append((start, end))

        if alias not in hits.setdefault(concept, []):
            hits[concept].append(alias)

    return hits

def filter_article(article):
    lead = " ".join([
        article["title"],
        article["description"],
        article["content"][:700],
    ])

    if not has_asset(lead):
        return None

    text = norm(article["title"] + "\n" + article["content"])

    for start in range(0, len(text), WINDOW_CHARS // 2):
        window = text[start:start + WINDOW_CHARS]

        if (
            has_asset(window)
            and len(find_hits(window)) >= MIN_CONCEPTS
        ):
            return find_hits(text)

    return None

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
                and parts.hostname in ("vietstock.vn", "www.vietstock.vn")
                and re.fullmatch(
                    r"/\d{4}/\d{2}/[^/]+\.htm", parts.path, re.I
                )
            ):
                links.append(urlunsplit(
                    ("https", "vietstock.vn", parts.path, "", "")
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
            '#vst_detail, [itemprop="articleBody"], .article-body, .detail-content' 
        )

        if title is None or body is None:
            raise CrawlError("Không tìm thấy tiêu đề/vùng nội dung")

        def meta(selector):
            node = soup.select_one(selector)
            return str(node.get("content") or "") if node else ""

        sapo = soup.select_one(".pHead, .sapo")
        description = (
            sapo.get_text(" ", strip=True) if sapo
            else meta(
                'meta[property="og:description"], '
                'meta[name="description"]'
            )
        )

        author_node = soup.select_one(".pAuthor, .author")
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
    lockfile = Path(tempfile.gettempdir()) / f"vietstock_{key}.lock"

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
    def __init__(self):
        self.config = {
            "version": "vietstock_keyword_companybase_v1",
            "concepts": CONCEPTS,
            "assets": list(ASSETS),
            "min_concepts": MIN_CONCEPTS,
            "window_chars": WINDOW_CHARS,
            "date_start": DATE_START.isoformat(),
            "date_end_exclusive": DATE_END_EXCLUSIVE.isoformat(),
            "categories": CATEGORIES,
        }
        self.run_id = digest(self.config)[:12]
        self.db_path = OUTPUT_DIR / "keyword_checkpoint_companybase_v1.sqlite3"
        self.csv_path = OUTPUT_DIR / f"vietstock_keyword_{self.run_id}.csv"
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
            matches = filter_article(article)
            if matches:
                article_id = re.search(r"-(\d+)\.htm$", url, re.I)
                rows.append({
                    "article_id": article_id.group(1) if article_id else hashlib.sha256(url.encode()).hexdigest()[:20],
                    "source": "vietstock",
                    "category": category,
                    "url": url,
                    **article,
                    "keyword_groups": ",".join(matches),
                    "matched_keywords": jd(matches),
                    "scraped_at": datetime.now(VN_TZ).isoformat(),
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
            "min_concepts": MIN_CONCEPTS,
            "window_chars": WINDOW_CHARS,
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
            "SELECT next_page, status FROM progress WHERE run=? AND category=?",
            (self.run_id, category),
        ).fetchone()

        if saved and saved[1] == "done_filtered_listing":
            print(f"{category}: đã duyệt hết danh sách bộ lọc ngày; xem errors.json cho bài lỗi.", flush=True)
            return
        listing_state = {}
        start_page = saved[0] if saved else 1
        page = start_page

        try:
            pending = self.db.execute(
                """
                SELECT page, url FROM article_cache
                WHERE category=? AND status <> 'ok' AND attempts < ?
                """,
                (category, MAX_ARTICLE_ATTEMPTS),
            ).fetchall()

            for old_page, url in pending:
                self.check_space()
                self.process(category, old_page, url)

            for page in range(start_page, start_page + MAX_PAGES):
                self.check_space()

                # Giữ trang đang xử lý nếu runtime bị ngắt giữa trang.
                with self.db:
                    self.db.execute(
                        "INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?, ?)",
                        (self.run_id, category, page, 0, "running"),
                    )

                html, last_page = category_listing(
                    self.session, category, page, listing_state)

                urls = links_from_html(html)

                if not urls:
                    diagnostic = OUTPUT_DIR / f"{category}_{page}_empty.html"
                    diagnostic.write_text(html, encoding="utf-8")
                    raise CrawlError(
                        "Danh sách rỗng: có thể hết lịch sử hoặc HTML đổi. "
                        f"Xem {diagnostic.name}; chưa xác nhận crawl đủ."
                    )

                signature = digest(sorted(urls))
                repeated = self.db.execute(
                    """
                    SELECT page FROM page_signatures
                    WHERE run=? AND category=?
                      AND signature=? AND page<>?
                    """,
                    (self.run_id, category, signature, page),
                ).fetchone()

                if repeated:
                    raise CrawlError(
                        f"Trang {page} lặp trang {repeated[0]}; "
                        "cần kiểm tra endpoint phân trang."
                    )

                before = self.added

                for url in urls:
                    self.check_space()
                    self.process(category, page, url)

                with self.db:
                    self.db.execute(
                        "INSERT OR REPLACE INTO page_signatures VALUES (?, ?, ?, ?)",
                        (self.run_id, category, page, signature),
                    )
                    self.db.execute(
                        "INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?, ?)",
                        (self.run_id, category, page + 1, 0, "running"),
                    )

                print(
                    f"[KEYWORD] {category} | trang {page} | "
                    f"{len(urls)} URL | thêm {self.added - before} dòng",
                    flush=True,
                )

                if (page - start_page + 1) % 10 == 0:
                    print(f"CSV hiện có {self.export()} dòng.", flush=True)

                if last_page is not None and page == last_page:
                    with self.db:
                        self.db.execute(
                            "INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?, ?)",
                            (self.run_id, category, page + 1, 0, "done_filtered_listing"))
                    print(f"{category}: hết danh sách theo bộ lọc ngày; CSV {self.export()} dòng.", flush=True)
                    return

            with self.db:
                self.db.execute(
                    "INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?, ?)",
                    (
                        self.run_id, category, page + 1,
                        0, "paused_page_budget",
                    ),
                )

            print(
                f"{category}: hết ngân sách {MAX_PAGES} trang của lượt này; "
                f"chạy lại để tiếp tục trang {page + 1}.",
                flush=True,
            )

        except (requests.RequestException, CrawlError) as exc:
            with self.db:
                self.db.execute(
                    "INSERT OR REPLACE INTO progress VALUES (?, ?, ?, ?, ?)",
                    (self.run_id, category, page, 0, f"paused: {exc}"),
                )
            print(f"Tạm dừng chuyên mục {category}: {exc}", flush=True)

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
                        f"VIETSTOCK KEYWORD | Run: {self.run_id}\n"
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
    if MIN_CONCEPTS < 2 or WINDOW_CHARS < 100:
        raise ValueError("MIN_CONCEPTS >= 2 và WINDOW_CHARS >= 100")
    print(f"VIETSTOCK KEYWORD: ít nhất {MIN_CONCEPTS} nhóm trong cửa sổ {WINDOW_CHARS} ký tự.", flush=True)
    prepare_drive()
    Crawler().run()


if __name__ == "__main__":
    main()
