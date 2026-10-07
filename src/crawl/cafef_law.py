# -*- coding: utf-8 -*-
# CAFEF — CRAWL LUẬT / HÀNH CHÍNH BẤT ĐỘNG SẢN
# Độc lập. Không cần chạy pipeline khác trước.
# Dùng JSONL riêng; KHÔNG tự nhập checkpoint SQLite của bản CafeF cũ.
# Chỉ chạy một phiên ghi vào thư mục này.

import sys
import subprocess

try:
    import requests
    from bs4 import BeautifulSoup
    from dateutil import parser as dateparser
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "requests", "beautifulsoup4", "python-dateutil"
    ])
    import requests
    from bs4 import BeautifulSoup
    from dateutil import parser as dateparser

import csv
import json
import re
import os
import time
import random
import hashlib
import unicodedata

from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlsplit, urlunsplit
from contextlib import ExitStack


# ==================== CẤU HÌNH ====================

SOURCE = "cafef"
HOST = "https://cafef.vn"
OUT_DIR = Path("/content/drive/MyDrive/crawl data/cafef_luat")

CHECKPOINT = OUT_DIR / "cafef_law_checkpoint_like_kenh14_v1.jsonl"
OUTPUT = OUT_DIR / "cafef_law_raw_like_kenh14.csv"

DATE_START = "2024-01-01"
DATE_END = "2026-09-09"

MAX_PAGES = 1000              # Mỗi chuyên mục / mỗi lượt
MAX_ATTEMPTS = 3              # Tổng số lần tải mỗi URL qua mọi lượt
OLD_PAGES_TO_STOP = 3
WINDOW_CHARS = 2500
MAX_HTML_BYTES = 8 * 1024 * 1024
SELECTED_CATEGORY = "all"

CATEGORIES = {
    "bat-dong-san": 18835,
    "vi-mo": 18833,
    "doanh-nghiep": 18836,
}

VN_TZ = timezone(timedelta(hours=7))
START = datetime.fromisoformat(DATE_START)
END_EXCLUSIVE = datetime.fromisoformat(DATE_END) + timedelta(days=1)

FIELDS = [
    "article_id", "source", "category", "url",
    "published_at", "title", "description", "content", "author",
    "keyword_groups", "matched_keywords", "scraped_at",
]

csv.field_size_limit(100_000_000)


# ==================== BỘ LỌC ====================

PROPERTY = {
    "bds_dat": (
        "đất đai|đất ở|đất nền|nhà đất|quyền sử dụng đất|"
        "mục đích sử dụng đất|tiền sử dụng đất|tiền thuê đất|"
        "bảng giá đất|giá đất|thu hồi đất|giao đất|cho thuê đất"
    ).split("|"),
    "bds_nha": (
        "bất động sản|địa ốc|nhà ở|chung cư|căn hộ|nhà phố|"
        "biệt thự|nhà xưởng|khu đô thị|khu dân cư"
    ).split("|"),
    "bds_giay_to": (
        "sổ đỏ|sổ hồng|quyền sở hữu nhà"
    ).split("|"),
}

LEGAL = {
    "luat_van_ban": (
        "luật|nghị định|thông tư|pháp lệnh|văn bản quy phạm pháp luật"
    ).split("|"),
    "luat_thu_tuc": (
        "thủ tục hành chính|thủ tục cấp|thủ tục sang tên|"
        "thủ tục chuyển|thủ tục đăng ký|thủ tục cấp phép|"
        "hồ sơ cấp|hồ sơ xin cấp|giấy phép xây dựng|giấy chứng nhận"
    ).split("|"),
    "luat_van_ban_co_quan": (
        "nghị quyết|quyết định số|quyết định phê duyệt|"
        "công văn|chỉ thị|quy hoạch|quy định mới|quy định về"
    ).split("|"),
}

RULES = {
    "qd_ban_hanh": (
        "ban hành|sửa đổi|bổ sung|bãi bỏ|thay thế|có hiệu lực|"
        "hiệu lực thi hành|hướng dẫn|quy định|thông qua|"
        "dự thảo|lấy ý kiến"
    ).split("|"),
    "qd_thu_tuc": (
        "hồ sơ|điều kiện|trình tự|thời hạn|thẩm quyền|"
        "cấp phép|cấp giấy|sang tên|đăng ký biến động|"
        "tách thửa|hợp thửa|chuyển mục đích|cấp đổi|cấp lại|"
        "phân cấp|phân quyền|phê duyệt"
    ).split("|"),
    "qd_tai_chinh": (
        "mức thu|miễn giảm|miễn, giảm|lệ phí trước bạ|"
        "thuế sử dụng đất|bồi thường|tái định cư|xử phạt|mức phạt"
    ).split("|"),
}

AUTHORITIES = (
    "quốc hội|chính phủ|thủ tướng|ủy ban nhân dân|uỷ ban nhân dân|"
    "ubnd|hội đồng nhân dân|hđnd|bộ xây dựng|bộ tài chính|"
    "bộ tài nguyên và môi trường|bộ nông nghiệp và môi trường|"
    "bộ tư pháp|sở xây dựng|sở tài nguyên và môi trường|"
    "sở nông nghiệp và môi trường|văn phòng đăng ký đất đai"
).split("|")

STRONG_RULE_TERMS = set((
    "ban hành|sửa đổi|bãi bỏ|có hiệu lực|hiệu lực thi hành|"
    "dự thảo|lấy ý kiến|đăng ký biến động|tách thửa|hợp thửa|"
    "chuyển mục đích|cấp đổi|cấp lại|phân cấp|phân quyền|"
    "lệ phí trước bạ|thuế sử dụng đất|xử phạt|mức phạt"
).split("|"))

CONCEPTS = {**PROPERTY, **LEGAL, **RULES}


def norm(text):
    return " ".join(
        unicodedata.normalize("NFC", str(text or "")).split()
    )


def jdump(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(jdump(value).encode("utf-8")).hexdigest()


PATTERNS = [
    (
        group, term,
        re.compile(r"(?<!\w)" + re.escape(norm(term).casefold()) + r"(?!\w)")
    )
    for group, terms in CONCEPTS.items()
    for term in terms
]

EXCLUDED = re.compile(
    r"(?<!\w)(?:luật sư|kỷ luật|kỉ luật|quy luật|"
    r"luật chơi|luật bóng đá|luật rừng)(?!\w)"
)

AUTHORITY = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(x) for x in sorted(AUTHORITIES, key=len, reverse=True)
    )
    + r")(?!\w)"
)

STATE_CODE = re.compile(
    r"/(?:\d{4}/)?(?:nđ-cp|qđ-ttg|qđ-ubnd|nq-cp|nq-hđnd|qh\d+)(?!\w)"
)


def find_hits(text):
    text = EXCLUDED.sub(" ", norm(text).casefold())
    candidates = []

    for group, term, pattern in PATTERNS:
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), group, term))

    occupied = []
    hits = {}

    for start, end, group, term in sorted(
        candidates, key=lambda x: (-(x[1] - x[0]), x[0], x[2])
    ):
        if any(start < b and a < end for a, b in occupied):
            continue
        occupied.append((start, end))
        if term not in hits.setdefault(group, []):
            hits[group].append(term)

    return hits


def filter_article(article):
    text = norm(" ".join(
        article.get(key, "") for key in ("title", "description", "content")
    ))

    for start in range(0, len(text), WINDOW_CHARS // 2):
        end = min(start + WINDOW_CHARS, len(text))
        window = text[start:end]
        hits = find_hits(window)
        lower = window.casefold()
        authority = bool(AUTHORITY.search(lower) or STATE_CODE.search(lower))

        if not authority:
            hits.pop("luat_van_ban_co_quan", None)

        groups = set(hits)
        strong = {
            term
            for group, terms in hits.items()
            if group in RULES
            for term in terms
        } & STRONG_RULE_TERMS

        if groups & set(PROPERTY):
            if groups & set(LEGAL) or (authority and strong):
                return hits

        if end == len(text):
            break

    return None


# ==================== HTTP VÀ ĐỌC BÀI ====================

class CrawlError(Exception):
    pass


class Blocked(CrawlError):
    pass


def http(session, url):
    time.sleep(random.uniform(1.0, 1.8))
    try:
        with session.get(url, timeout=(15, 60), stream=True) as response:
            if response.status_code in (401, 403, 429):
                raise Blocked(
                    f"HTTP {response.status_code}: {url}. Tạm dừng chuyên mục."
                )
            response.raise_for_status()

            chunks = []
            size = 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > MAX_HTML_BYTES:
                    raise CrawlError("HTML vượt giới hạn kích thước.")
                chunks.append(chunk)

            html = b"".join(chunks).decode("utf-8-sig", errors="replace")
            if "<title>Just a moment" in html:
                raise Blocked("Website trả trang xác minh, chưa phải bài báo.")
            return html

    except requests.RequestException as exc:
        raise CrawlError(str(exc)) from exc


def listing_http(session, url):
    last = None
    for attempt in range(1, 4):
        try:
            return http(session, url)
        except Blocked:
            raise
        except CrawlError as exc:
            last = exc
            if attempt < 3:
                time.sleep(attempt * 3)
    raise CrawlError(f"Không tải được danh sách: {last}")


def canonical(value):
    if not value:
        return None

    try:
        parts = urlsplit(urljoin("https://cafef.vn/", value))
    except ValueError:
        return None

    if (
        parts.scheme not in ("http", "https")
        or parts.hostname not in ("cafef.vn", "www.cafef.vn")
        or not re.fullmatch(r"/[^/]+-\d{8,}\.chn", parts.path, re.I)
    ):
        return None

    return urlunsplit(("https", "cafef.vn", parts.path, "", ""))


def listing_links(html):
    soup = BeautifulSoup(html, "html.parser")
    try:
        return list(dict.fromkeys(
            url
            for node in soup.select("a[href]")
            if (url := canonical(node.get("href")))
        ))
    finally:
        soup.decompose()


def parse_date(value):
    value = norm(value)
    if not value:
        return None
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2}", value):
            result = dateparser.isoparse(value)
        else:
            match = re.search(
                r"\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2})?",
                value
            )
            if not match:
                return None
            result = dateparser.parse(match.group(), dayfirst=True)

        if result.tzinfo is not None:
            result = result.astimezone(VN_TZ).replace(tzinfo=None)
        return result
    except (ValueError, OverflowError):
        return None


def walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def article_date(soup):
    for selector in (
        'meta[property="article:published_time"]',
        'meta[itemprop="datePublished"]',
        'meta[name="pubdate"]',
        'time[itemprop="datePublished"]',
        'meta[name="publishdate"]',
        '[itemprop="datePublished"]',
        '.pdate', '.dateandcat', '.detail-time',
    ):
        node = soup.select_one(selector)
        if node:
            value = (
                node.get("content") or node.get("datetime")
                or node.get("value") or node.get_text(" ", strip=True)
            )
            parsed = parse_date(value)
            if parsed:
                return parsed

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            for obj in walk_json(json.loads(script.get_text())):
                parsed = parse_date(obj.get("datePublished"))
                if parsed:
                    return parsed
        except (ValueError, TypeError):
            pass
    return None


def read_article(html, category, url):
    soup = BeautifulSoup(html, "html.parser")
    published = article_date(soup)
    if published is None:
        raise CrawlError("Không xác định được ngày đăng.")

    date_text = published.isoformat(sep=" ")
    if not START <= published < END_EXCLUSIVE:
        return {"status": "outside", "date": date_text}

    title_node = soup.select_one("h1")
    body = soup.select_one(
        '.detail-content.afcbc-body, .detail-content, [data-role="content"], [itemprop="articleBody"]'
    )
    if title_node is None or body is None:
        raise CrawlError("Thiếu tiêu đề hoặc vùng nội dung bài.")

    title = norm(title_node.get_text(" ", strip=True))
    description_node = soup.select_one(".detail-sapo, .sapo, [data-role='sapo']")
    description = (
        norm(description_node.get_text(" ", strip=True))
        if description_node else ""
    )
    if not description:
        meta = soup.select_one('meta[name="description"]')
        description = norm(meta.get("content", "")) if meta else ""

    author_node = soup.select_one(".detail-author, .author, [itemprop='author']")
    author = norm(author_node.get_text(" ", strip=True)) if author_node else ""

    for node in body.select(
        "script, style, iframe, .VCSortableInPreviewMode[type='RelatedNews'], "
        ".link-source-wrapper, .sda_middle, .ads, .related-news"
    ):
        node.decompose()

    content = norm(body.get_text(" ", strip=True))
    if len(content) < 80:
        raise CrawlError("Nội dung quá ngắn hoặc trang chưa tải đúng.")

    article = {
        "article_id": hashlib.sha256(url.encode()).hexdigest()[:20],
        "source": SOURCE,
        "category": category,
        "url": url,
        "published_at": date_text,
        "title": title,
        "description": description,
        "content": content,
        "author": author,
    }

    hits = filter_article(article)
    if not hits:
        return {"status": "filtered", "date": date_text}

    article.update({
        "keyword_groups": jdump(sorted(hits)),
        "matched_keywords": jdump(hits),
        "scraped_at": datetime.now(VN_TZ).isoformat(),
    })
    return {"status": "saved", "date": date_text, "row": article}


# ==================== CHECKPOINT BỀN VỮNG ====================

def sync(handle):
    handle.flush()
    os.fsync(handle.fileno())


CONFIG = {
    "source": SOURCE,
    "version": "law_jsonl_v1",
    "dates": [DATE_START, DATE_END],
    "categories": CATEGORIES,
    "filter": digest([
        CONCEPTS, AUTHORITIES, sorted(STRONG_RULE_TERMS), WINDOW_CHARS
    ]),
}


def apply_event(state, event):
    kind = event["kind"]
    if kind == "attempt":
        state["attempts"][event["url"]] = event["number"]
    elif kind == "result":
        state["results"][event["url"]] = event["result"]
    elif kind == "page":
        category = event["category"]
        state["progress"][category] = event
        if event.get("signature"):
            state["signatures"].setdefault(category, set()).add(
                event["signature"]
            )


def load_state():
    state = {
        "attempts": {}, "results": {}, "progress": {}, "signatures": {}
    }
    if not CHECKPOINT.exists() or CHECKPOINT.stat().st_size == 0:
        return state

    before = (CHECKPOINT.stat().st_size, CHECKPOINT.stat().st_mtime_ns)
    header = False
    with CHECKPOINT.open("rb") as handle:
        for number, line in enumerate(handle, 1):
            if not line.endswith(b"\n"):
                raise RuntimeError(
                    f"Checkpoint dòng {number} chưa hoàn chỉnh. "
                    "Giữ nguyên file, chưa ghi thêm."
                )
            record = json.loads(line)
            event = record["event"]
            if record["checksum"] != digest(event):
                raise RuntimeError(f"Checkpoint sai checksum dòng {number}.")

            if not header:
                if event != {"kind": "config", "value": CONFIG}:
                    raise RuntimeError(
                        "Checkpoint thuộc cấu hình khác. Không tự reset tiến độ."
                    )
                header = True
            else:
                apply_event(state, event)

    after = (CHECKPOINT.stat().st_size, CHECKPOINT.stat().st_mtime_ns)
    if before != after:
        raise RuntimeError("Checkpoint đang bị phiên khác ghi.")
    return state


def emit(handle, state, event):
    record = {"event": event, "checksum": digest(event)}
    handle.write((jdump(record) + "\n").encode("utf-8"))
    sync(handle)
    apply_event(state, event)


def read_csv_seen():
    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        return set(), False

    with OUTPUT.open("rb") as handle:
        handle.seek(-1, 2)
        if handle.read(1) not in (b"\r", b"\n"):
            raise RuntimeError("CSV có dòng cuối chưa hoàn chỉnh.")

    seen = set()
    with OUTPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != FIELDS:
            raise RuntimeError("CSV có cấu trúc khác; không ghi đè.")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise RuntimeError("CSV có bản ghi thiếu/thừa cột.")
            if row["source"] != SOURCE:
                raise RuntimeError("CSV chứa dữ liệu nguồn khác.")
            seen.add(row["url"])
    return seen, True


def process(session, checkpoint, state, category, url):
    cached = state["results"].get(url)
    if cached is not None:
        return cached

    while state["attempts"].get(url, 0) < MAX_ATTEMPTS:
        number = state["attempts"].get(url, 0) + 1

        # Ghi số lần thử trước khi gửi HTTP.
        emit(checkpoint, state, {
            "kind": "attempt", "url": url, "number": number
        })

        try:
            result = read_article(http(session, url), category, url)
        except Blocked:
            raise
        except CrawlError as exc:
            print(f"  Lỗi {number}/{MAX_ATTEMPTS}: {exc}", flush=True)
            if number < MAX_ATTEMPTS:
                time.sleep(3)
                continue
            result = {
                "status": "failed", "date": "", "error": str(exc)
            }

        emit(checkpoint, state, {
            "kind": "result", "url": url, "result": result
        })
        return result

    # Lượt trước có thể ngắt sau khi đã ghi lần thử thứ 3.
    result = {
        "status": "failed", "date": "",
        "error": "Đã dùng đủ 3 lần thử; không tải lại."
    }
    emit(checkpoint, state, {
        "kind": "result", "url": url, "result": result
    })
    return result


# ==================== MAIN ====================

def main():
    import fcntl
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    if not os.path.ismount("/content/drive") or not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Drive chưa mount; chưa bắt đầu crawl.")

    if MAX_PAGES < 1 or MAX_ATTEMPTS < 1 or OLD_PAGES_TO_STOP < 1:
        raise ValueError("Giới hạn trang/lần thử/số trang cũ phải >= 1.")
    if START >= END_EXCLUSIVE:
        raise ValueError("Khoảng ngày không hợp lệ.")
    if SELECTED_CATEGORY != "all" and SELECTED_CATEGORY not in CATEGORIES:
        raise ValueError("SELECTED_CATEGORY không hợp lệ.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Khóa chỉ có tác dụng trong cùng runtime.
    with open("/content/cafef_law.lock", "a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Pipeline CafeF đang chạy trong runtime này.")

        state = load_state()
        seen, has_header = read_csv_seen()

        with ExitStack() as stack:
            checkpoint = stack.enter_context(CHECKPOINT.open("ab"))
            if CHECKPOINT.stat().st_size == 0:
                emit(checkpoint, state, {"kind": "config", "value": CONFIG})

            output = stack.enter_context(OUTPUT.open(
                "a", newline="", encoding="utf-8" if has_header else "utf-8-sig"
            ))
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            if not has_header:
                writer.writeheader()
                sync(output)

            def write_result(result):
                row = result.get("row")
                if row and row["url"] not in seen:
                    writer.writerow(row)
                    sync(output)
                    seen.add(row["url"])

            # Khôi phục dòng có checkpoint nhưng CSV còn thiếu.
            for result in state["results"].values():
                write_result(result)

            session = stack.enter_context(requests.Session())
            session.headers.update({
                "User-Agent": "Mozilla/5.0",
                "Referer": HOST + "/",
            })

            selected = (
                CATEGORIES if SELECTED_CATEGORY == "all"
                else {SELECTED_CATEGORY: CATEGORIES[SELECTED_CATEGORY]}
            )

            print("Checkpoint:", CHECKPOINT)
            print("CSV:", OUTPUT)
            print("Khoảng ngày:", DATE_START, "→", DATE_END)

            for category, zone_id in selected.items():
                progress = state["progress"].get(category, {})
                if progress.get("status") == "done_before_start":
                    print(f"\n{category}: đã dừng theo ngày ở lượt trước.")
                    continue

                page = progress.get("next_page", 1)
                streak = progress.get("old_streak", 0)

                try:
                    zone = str(zone_id)
                    previous_zone = progress.get("zone")
                    if previous_zone and previous_zone != zone:
                        raise CrawlError("ID chuyên mục đã thay đổi; không dùng lại số trang cũ.")
                    print(f"\n{category}: tiếp tục trang {page}, categoryID={zone}")

                    for _ in range(MAX_PAGES):
                        html = listing_http(
                            session, f"{HOST}/timelinelist/{zone}/{page}.chn")
                        urls = listing_links(html)
                        if not urls:
                            raise CrawlError(
                                "Danh sách rỗng hoặc HTML đổi; "
                                "chưa xác nhận đã đến DATE_START."
                            )

                        signature = digest(sorted(urls))
                        if signature in state["signatures"].get(category, set()):
                            raise CrawlError(
                                "Danh sách lặp trang đã xử lý; tạm dừng."
                            )

                        dates = []
                        unknown = 0
                        failures = 0

                        for url in urls:
                            result = process(
                                session, checkpoint, state, category, url
                            )
                            write_result(result)
                            published = parse_date(result.get("date"))
                            if published is None:
                                unknown += 1
                            else:
                                dates.append(published)
                            failures += result["status"] == "failed"

                        # Dùng ngày của mọi bài, không chỉ bài qua bộ lọc luật.
                        all_old = (
                            bool(dates) and unknown == 0 and max(dates) < START
                        )
                        streak = streak + 1 if all_old else 0
                        done = streak >= OLD_PAGES_TO_STOP

                        emit(checkpoint, state, {
                            "kind": "page", "category": category,
                            "next_page": page + 1,
                            "old_streak": streak,
                            "status": "done_before_start" if done else "running",
                            "signature": signature, "zone": zone,
                        })

                        date_range = (
                            f"{min(dates).date()} → {max(dates).date()}"
                            if dates else "không xác định"
                        )
                        print(
                            f"{category} | trang {page} | ngày {date_range} | "
                            f"lỗi {failures}, thiếu ngày {unknown} | "
                            f"CSV {len(seen):,} bài",
                            flush=True,
                        )

                        page += 1
                        if done:
                            print(
                                f"Đã gặp {OLD_PAGES_TO_STOP} trang liên tiếp "
                                f"có toàn bộ bài trước {DATE_START}; dừng chuyên mục."
                            )
                            break
                    else:
                        print(
                            f"Đạt giới hạn {MAX_PAGES} trang/lượt. "
                            f"Chạy lại tiếp tục trang {page}."
                        )

                except CrawlError as exc:
                    # Không tăng trang khi danh sách bị lỗi.
                    print(
                        f"TẠM DỪNG {category} tại trang {page}: {exc}",
                        flush=True,
                    )

            failed_total = sum(
                result["status"] == "failed"
                for result in state["results"].values()
            )
            print(f"\nĐã lưu {len(seen):,} bài:", OUTPUT)
            print(f"URL đã bỏ qua sau đủ lần thử: {failed_total:,}")
            print("Chạy lại cell để tiếp tục các chuyên mục chưa hoàn tất.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã ngắt. Chạy lại để tiếp tục checkpoint.")