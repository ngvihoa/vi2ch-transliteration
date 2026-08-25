#!/usr/bin/env python3
"""Crawl aligned Han-Viet/chữ Hán pairs for the Phú genre from Thi Viện."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from http.cookiejar import LWPCookieJar
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup, Tag


BASE_URL = "https://www.thivien.net"
SEARCH_URL = f"{BASE_URL}/search-poem.php?PoemType=2&ViewType=2"
GENRE_LABEL = "Phú"
REPORT_SCHEMA = "thivien-phu-crawl-v1"
POEM_PROGRESS_SCHEMA = "thivien-poem-crawl-progress-v1"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "raw-collections" / "poetry-collecions" / "phu"
DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "crawl_report.json"
DEFAULT_URL_CHECKPOINT = PIPELINE_ROOT / "outputs" / "poem_urls.json"
DEFAULT_COUNTRY_IDS: tuple[str, ...] | None = None
DEFAULT_AGE_PARTITION_COUNTS: dict[str, dict[str, int]] = {}
DEFAULT_SHALLOW_AGE_PARTITION_COUNTS: dict[str, dict[str, int]] = {}

POEM_LINK_RE = re.compile(r"/poem-([A-Za-z0-9_-]+)$")
AUTHOR_LINK_RE = re.compile(r"/author-([A-Za-z0-9_-]+)$")
NON_FILENAME_RE = re.compile(r"[^a-z0-9]+")
RESULT_COUNT_RE = re.compile(r"tổng số\s+\d+\s+trang\s+\(([\d.]+)\s+bài thơ\)")
TOO_MANY_RE = re.compile(r"Có quá nhiều\s+\(([\d.]+)\)\s+kết quả")
AUTHOR_COUNT_RE = re.compile(r"tổng số\s+\d+\s+trang\s+\(([\d.]+)\s+tác giả\)")
MAX_FILENAME_STEM_LENGTH = 180

# Các thời đại lá trên form Thi Viện. Chỉ dùng khi một quốc gia có hơn 100
# kết quả, nhằm tránh giới hạn chỉ xem được 10 trang đầu của website.
LEAF_AGES: dict[str, list[str]] = {
    "2": ["50", "52", "53", "54", "55", "56", "57", "2", "3"],
    "3": [
        "21", "22", "23", "24", "25", "26", "27", "5", "6", "7",
        "8", "9", "10", "12", "13", "28", "29", "14", "15",
        "16", "17", "18",
    ],
}


class CrawlError(RuntimeError):
    pass


class AccessChallengeError(CrawlError):
    pass


def is_access_challenge(html: str) -> bool:
    """Distinguish the blocking page from reCAPTCHA embedded in comments."""
    if "Xác nhận không phải máy truy cập tự động" in html:
        return True
    if 'class="g-recaptcha"' not in html:
        return False

    # Trang thơ bình thường có reCAPTCHA trong form bình luận. Chỉ xem
    # g-recaptcha là chặn truy cập khi response không còn nội dung trang thật.
    normal_page_markers = (
        'class="poem-content',
        "list-item-header",
        'name="PoemType"',
        'search-author.php',
    )
    return not any(marker in html for marker in normal_page_markers)


@dataclass(frozen=True)
class Poem:
    uid: str
    url: str
    title_vi: str
    lines_vi: list[str]
    lines_cn: list[str]


class HttpClient:
    def __init__(
        self,
        delay: float,
        jitter: float,
        pause_every: int,
        pause_min: float,
        pause_max: float,
        timeout: float,
        retries: int,
        user_agent: str = USER_AGENT,
        cookie_file: Path | None = None,
        captcha_pause_min: float = 0,
        captcha_pause_max: float = 0,
        captcha_retries: int = 0,
    ) -> None:
        self.delay = delay
        self.jitter = jitter
        self.pause_every = pause_every
        self.pause_min = pause_min
        self.pause_max = pause_max
        self.timeout = timeout
        self.retries = retries
        self.user_agent = user_agent
        self.cookie_file = cookie_file
        self.captcha_pause_min = captcha_pause_min
        self.captcha_pause_max = captcha_pause_max
        self.captcha_retries = captcha_retries
        self.cookies = LWPCookieJar(str(cookie_file) if cookie_file else None)
        if cookie_file is not None and cookie_file.exists():
            try:
                self.cookies.load(ignore_discard=True, ignore_expires=True)
            except (OSError, ValueError) as error:
                raise CrawlError(f"Cookie jar không hợp lệ ({cookie_file}): {error}") from error
        self._opener = build_opener(HTTPCookieProcessor(self.cookies))
        self._last_request_at: float | None = None
        self._request_count = 0

    def _save_cookies(self) -> None:
        if self.cookie_file is None:
            return
        self.cookie_file.parent.mkdir(parents=True, exist_ok=True)
        self.cookies.save(ignore_discard=True, ignore_expires=True)

    def _wait_before_request(self) -> None:
        if self.pause_every and self._request_count > 0:
            if self._request_count % self.pause_every == 0:
                pause = random.uniform(self.pause_min, self.pause_max)
                print(
                    f"[nghỉ định kỳ] Đã gửi {self._request_count} request; "
                    f"nghỉ {pause / 60:.1f} phút...",
                    flush=True,
                )
                time.sleep(pause)

        if self._last_request_at is not None:
            interval = self.delay + random.uniform(0, self.jitter)
            remaining = interval - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)

    def get_text(self, url: str) -> str:
        challenge_count = 0
        while True:
            try:
                return self._get_text_once(url)
            except AccessChallengeError:
                if challenge_count >= self.captcha_retries:
                    raise
                challenge_count += 1
                pause = random.uniform(
                    self.captcha_pause_min, self.captcha_pause_max
                )
                print(
                    f"[CAPTCHA] Ngừng request; nghỉ {pause / 60:.1f} phút "
                    f"rồi thử lại ({challenge_count}/{self.captcha_retries})...",
                    flush=True,
                )
                time.sleep(pause)

    def _get_text_once(self, url: str) -> str:
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "vi,en;q=0.7",
            },
        )
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                self._wait_before_request()
                self._last_request_at = time.monotonic()
                self._request_count += 1
                with self._opener.open(request, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    html = response.read().decode(charset, errors="strict")
                    self._save_cookies()
                    if is_access_challenge(html):
                        raise AccessChallengeError(
                            "Thi Viện đang yêu cầu xác minh CAPTCHA; "
                            "checkpoint vẫn được giữ nguyên"
                        )
                    return html
            except AccessChallengeError as error:
                # CAPTCHA là chặn chủ động, retry ngay chỉ làm tình hình xấu hơn.
                raise error
            except HTTPError as error:
                last_error = error
                if error.code not in {429, 500, 502, 503, 504}:
                    break
            except (URLError, TimeoutError, UnicodeDecodeError) as error:
                last_error = error
            if attempt < self.retries:
                time.sleep(min(2**attempt, 8))
        if isinstance(last_error, AccessChallengeError):
            raise last_error
        raise CrawlError(f"Không tải được {url}: {last_error}") from last_error


def normalized_lines(node: Tag) -> list[str]:
    for marker in node.find_all("sup"):
        marker.decompose()
    for line_break in node.find_all("br"):
        line_break.replace_with("\n")
    text = node.get_text("", strip=False)
    return [
        " ".join(unicodedata.normalize("NFC", line).split())
        for line in text.splitlines()
        if line.strip()
    ]


def parse_search_page(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()
    for anchor in soup.select("h4.list-item-header > a[href]"):
        url = urljoin(BASE_URL, str(anchor["href"]))
        if POEM_LINK_RE.search(urlparse(url).path) and url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def parse_total_pages(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    pages: list[int] = []

    # Danh sách nhỏ dùng <select name="Page">.
    page_select = soup.select_one('select[name="Page"]')
    if page_select is not None:
        pages.extend(
            int(str(option["value"]))
            for option in page_select.select("option[value]")
            if str(option["value"]).isdigit()
        )

    # Danh sách lớn dùng <input name="Page">; số trang cuối chỉ xuất hiện
    # trong các link phân trang, ví dụ ...&Page=129.
    for anchor in soup.select('a[href*="Page="]'):
        query = parse_qs(urlparse(str(anchor["href"])).query)
        pages.extend(int(value) for value in query.get("Page", []) if value.isdigit())

    return max(pages, default=1)


def parse_poem(html: str, url: str) -> Poem:
    soup = BeautifulSoup(html, "html.parser")
    section = soup.select_one("div.poem-content div.poem-view-separated")
    if section is None:
        raise CrawlError(f"Không tìm thấy phần nguyên tác/phiên âm: {url}")

    han_body = section.select_one("p.han-chinese")
    headings = section.find_all("h4", recursive=False)
    vi_heading = next((h for h in headings if not h.select_one(".han-chinese")), None)
    if han_body is None or vi_heading is None:
        raise CrawlError(f"Không tìm thấy cặp chữ Hán/Hán-Việt: {url}")
    vi_body = vi_heading.find_next_sibling("p")
    if vi_body is None or vi_body is han_body:
        raise CrawlError(f"Không tìm thấy phiên âm Hán-Việt: {url}")

    title_lines = normalized_lines(vi_heading)
    lines_cn = normalized_lines(han_body)
    lines_vi = normalized_lines(vi_body)
    if len(title_lines) != 1:
        raise CrawlError(f"Tiêu đề không hợp lệ: {url}")
    if not lines_cn or len(lines_cn) != len(lines_vi):
        raise CrawlError(
            f"Lệch dòng tại {url}: chữ Hán={len(lines_cn)}, Hán-Việt={len(lines_vi)}"
        )

    match = POEM_LINK_RE.search(urlparse(url).path)
    if match is None:
        raise CrawlError(f"URL không có UID: {url}")
    return Poem(match.group(1), url, title_lines[0], lines_vi, lines_cn)


def stem_with_uid(stem: str, uid: str) -> str:
    safe_uid = re.sub(r"[^A-Za-z0-9_-]+", "", uid)[:48]
    if not safe_uid:
        safe_uid = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:16]
    prefix_length = MAX_FILENAME_STEM_LENGTH - len(safe_uid) - 1
    return f"{stem[:max(1, prefix_length)]}-{safe_uid}"


def filename_stem(title: str, uid: str | None = None) -> str:
    ascii_title = (
        unicodedata.normalize("NFKD", title.replace("Đ", "D").replace("đ", "d"))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    stem = NON_FILENAME_RE.sub("", ascii_title)
    if not stem:
        raise CrawlError(f"Không thể tạo tên file từ {title!r}")
    if len(stem) > MAX_FILENAME_STEM_LENGTH:
        suffix = uid or hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]
        stem = stem_with_uid(stem, suffix)
    return stem


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_poem(poem: Poem, output_dir: Path, stem: str, overwrite: bool) -> Path:
    output = output_dir / f"{stem}.csv"
    if output.exists() and not overwrite:
        raise CrawlError(f"Output đã tồn tại ({output}); dùng --overwrite để thay thế")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["vi", "cn"])
    writer.writerows(zip(poem.lines_vi, poem.lines_cn, strict=True))
    atomic_write(output, buffer.getvalue())
    return output


def assert_allowed(client: HttpClient) -> None:
    parser = RobotFileParser()
    parser.set_url(ROBOTS_URL)
    parser.parse(client.get_text(ROBOTS_URL).splitlines())
    if not parser.can_fetch(USER_AGENT, SEARCH_URL):
        raise CrawlError(f"robots.txt không cho phép crawl {SEARCH_URL}")


def load_url_checkpoint(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CrawlError(f"Checkpoint URL không hợp lệ ({path}): {error}") from error
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("source") != SEARCH_URL
        or checkpoint.get("schema_version") not in {
            "thivien-partition-checkpoint-v2",
            "thivien-partition-checkpoint-v3",
            "thivien-partition-checkpoint-v4",
        }
    ):
        return None
    urls = checkpoint.get("urls")
    if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
        raise CrawlError(f"Checkpoint URL không có danh sách urls hợp lệ: {path}")
    return checkpoint


def save_url_checkpoint(path: Path, state: dict[str, object]) -> None:
    checkpoint = {
        "schema_version": "thivien-partition-checkpoint-v4",
        "source": SEARCH_URL,
        **state,
    }
    atomic_write(path, json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n")


def load_poem_progress(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CrawlError(f"Progress bài thơ không hợp lệ ({path}): {error}") from error
    if (
        not isinstance(progress, dict)
        or progress.get("schema_version") != POEM_PROGRESS_SCHEMA
        or progress.get("source") != SEARCH_URL
        or not isinstance(progress.get("items"), dict)
    ):
        raise CrawlError(f"Progress bài thơ không đúng schema/source: {path}")
    return {
        str(url): item
        for url, item in progress["items"].items()
        if isinstance(url, str) and isinstance(item, dict)
    }


def save_poem_progress(path: Path, items: dict[str, dict[str, object]]) -> None:
    progress = {
        "schema_version": POEM_PROGRESS_SCHEMA,
        "source": SEARCH_URL,
        "processed_urls": len(items),
        "items": items,
    }
    atomic_write(path, json.dumps(progress, ensure_ascii=False, indent=2) + "\n")


def progress_output_path(
    output_dir: Path, item: dict[str, object]
) -> Path | None:
    output = item.get("output")
    if not isinstance(output, str) or not output:
        return None
    return output_dir / output


def parse_result_count(html: str) -> int:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    match = RESULT_COUNT_RE.search(text) or TOO_MANY_RE.search(text)
    if match:
        return int(match.group(1).replace(".", ""))
    return len(parse_search_page(html))


def parse_country_ids(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    return [
        str(option["value"])
        for option in soup.select('select[name="Country"] option[value]')
        if str(option["value"]).isdigit()
    ]


def parse_author_count(html: str) -> int:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    match = AUTHOR_COUNT_RE.search(text)
    if match:
        return int(match.group(1).replace(".", ""))
    return len(parse_author_entries(html))


def parse_author_entries(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    authors: dict[str, str] = {}
    for anchor in soup.select("h4.list-item-header > a[href]"):
        url = urljoin(BASE_URL, str(anchor["href"]))
        if not AUTHOR_LINK_RE.search(urlparse(url).path):
            continue
        display_name = anchor.get_text(" ", strip=True)
        # Phần trước dấu " - " là tên Việt dùng được trong trường Author.
        authors[url] = display_name.split(" - ", 1)[0].strip()
    return authors


def partition_url(country: str, age: str | None = None, page: int | None = None) -> str:
    url = f"{SEARCH_URL}&Country={country}"
    if age is not None:
        url += f"&Age%5B%5D={age}"
    if page is not None:
        url += f"&Page={page}"
    return url


def author_uid(author_url: str) -> str:
    match = AUTHOR_LINK_RE.search(urlparse(author_url).path)
    if match is None:
        raise CrawlError(f"URL tác giả không có UID: {author_url}")
    return match.group(1)


def author_poem_url(country: str, age: str, author_url: str) -> str:
    # Trường Author chỉ là ô hiển thị của autocomplete. IAuthor mới là UID
    # được backend dùng để lọc đúng một tác giả.
    return f"{partition_url(country, age)}&{urlencode({'IAuthor': author_uid(author_url)})}"


def author_catalog_url(
    country: str,
    age: str,
    sort: str,
    sort_order: str,
    page: int | None = None,
) -> str:
    url = f"{BASE_URL}/search-author.php?Country={country}&Age%5B%5D={age}&Sort={sort}"
    # Thi Viện biểu diễn chiều tăng dần bằng cách không có SortOrder.
    if sort_order == "desc":
        url += "&SortOrder=desc"
    if page is not None:
        url += f"&Page={page}"
    return url


def discover_partition_authors(client: HttpClient, country: str, age: str) -> dict[str, str]:
    variants = [
        ("Author", "asc"),
        ("Author", "desc"),
        ("Date", "asc"),
        ("Date", "desc"),
        ("Views", "asc"),
        ("Views", "desc"),
        ("BirthYear", "asc"),
        ("BirthYear", "desc"),
        ("DeathYear", "asc"),
        ("DeathYear", "desc"),
        ("Poster", "asc"),
        ("Poster", "desc"),
    ]
    authors: dict[str, str] = {}
    expected: int | None = None
    label = f"country={country}&age={age}"

    for sort, order in variants:
        first_url = author_catalog_url(country, age, sort, order)
        print(f"[tác giả {label}] đang tải {sort}/{order}...", flush=True)
        first_html = client.get_text(first_url)
        if expected is None:
            expected = parse_author_count(first_html)
        authors.update(parse_author_entries(first_html))
        pages = min(parse_total_pages(first_html), 10)
        for page in range(2, pages + 1):
            html = client.get_text(author_catalog_url(country, age, sort, order, page))
            authors.update(parse_author_entries(html))
        print(
            f"[tác giả {label}] đã có {len(authors)}/{expected} tác giả",
            flush=True,
        )
        if len(authors) >= expected:
            return authors

    raise CrawlError(
        f"Không khám phá đủ tác giả cho {label}: {len(authors)}/{expected}"
    )


def collect_small_partition(
    client: HttpClient,
    first_html: str,
    base_url: str,
    label: str,
) -> list[str]:
    total_pages = parse_total_pages(first_html)
    urls = parse_search_page(first_html)
    print(
        f"[phân vùng {label}] trang 1/{total_pages}: {len(urls)} URL",
        flush=True,
    )
    for page in range(2, total_pages + 1):
        print(f"[phân vùng {label}] đang tải trang {page}/{total_pages}...", flush=True)
        page_urls = parse_search_page(client.get_text(f"{base_url}&Page={page}"))
        known = set(urls)
        urls.extend(url for url in page_urls if url not in known)
    return urls


def collect_capped_window(client: HttpClient, base_url: str, label: str) -> list[str]:
    """Collect the ten result pages Thi Viện exposes for an oversized query."""
    print(f"[cửa sổ {label}] đang tải trang 1...", flush=True)
    first_html = client.get_text(base_url)
    urls = parse_search_page(first_html)
    pages = min(parse_total_pages(first_html), 10)
    for page in range(2, pages + 1):
        html = client.get_text(f"{base_url}&Page={page}")
        known = set(urls)
        urls.extend(url for url in parse_search_page(html) if url not in known)
    print(f"[cửa sổ {label}] thu được {len(urls)} URL", flush=True)
    return urls


def collect_bidirectional_window(
    client: HttpClient,
    base_url: str,
    label: str,
    expected_count: int,
) -> list[str]:
    """Collect a 101..199-result partition from opposite 100-result windows."""
    if not 100 < expected_count < 200:
        raise CrawlError(
            f"Cửa sổ hai chiều chỉ hỗ trợ 101..199 kết quả: {expected_count}"
        )

    urls: list[str] = []
    for order in ("asc", "desc"):
        window_url = f"{base_url}&Sort=Date&SortOrder={order}"
        found = collect_capped_window(
            client, window_url, f"{label}-date-{order}"
        )
        known = set(urls)
        urls.extend(url for url in found if url not in known)

    if len(urls) != expected_count:
        raise CrawlError(
            f"Cửa sổ asc/desc không phủ đủ {label}: "
            f"{len(urls)}/{expected_count} URL"
        )
    print(
        f"[cửa sổ {label}] hợp nhất đủ {len(urls)}/{expected_count} URL",
        flush=True,
    )
    return urls


def collect_shallow_age_windows(
    client: HttpClient,
    checkpoint_path: Path,
    refresh_checkpoint: bool,
    configured_counts: dict[str, dict[str, int]],
    country_ids_override: list[str] | None = None,
) -> list[str]:
    """Collect selected country/age partitions without drilling into authors."""
    normalized_counts = {
        str(country): {str(age): int(count) for age, count in ages.items()}
        for country, ages in configured_counts.items()
    }
    requested_countries = (
        list(normalized_counts) if country_ids_override is None else country_ids_override
    )
    unknown = [country for country in requested_countries if country not in normalized_counts]
    if unknown:
        raise CrawlError(
            f"Shallow pipeline không cấu hình country ID: {unknown}; "
            f"chỉ cho phép {list(normalized_counts)}"
        )
    selected_counts = {
        country: normalized_counts[country] for country in requested_countries
    }
    partition_keys = {
        f"country={country}&age={age}"
        for country, ages in selected_counts.items()
        for age in ages
    }

    checkpoint = None if refresh_checkpoint else load_url_checkpoint(checkpoint_path)
    if checkpoint is not None and (
        checkpoint.get("strategy") != "shallow-country-age-date-windows-v1"
        or checkpoint.get("age_partition_counts") != selected_counts
    ):
        print("[checkpoint] Chiến lược shallow đã đổi; tạo lại danh sách URL.", flush=True)
        checkpoint = None
    if checkpoint is not None and checkpoint.get("complete") is True:
        urls = list(checkpoint["urls"])
        print(f"[danh sách] Dùng checkpoint shallow hoàn chỉnh: {len(urls)} URL", flush=True)
        return urls

    urls = list(checkpoint.get("urls", [])) if checkpoint else []
    completed = set(checkpoint.get("completed_partitions", [])) if checkpoint else set()
    partition_stats = dict(checkpoint.get("partition_stats", {})) if checkpoint else {}
    configured_results = sum(
        count for ages in selected_counts.values() for count in ages.values()
    )
    window_capacity = sum(
        min(count, 200) for ages in selected_counts.values() for count in ages.values()
    )

    def persist(complete: bool = False) -> None:
        save_url_checkpoint(checkpoint_path, {
            "strategy": "shallow-country-age-date-windows-v1",
            "expected_total": configured_results,
            "target_window_capacity": window_capacity,
            "country_ids": requested_countries,
            "age_partition_counts": selected_counts,
            "discovered_urls": len(urls),
            "completed_partitions": sorted(completed),
            "partition_stats": partition_stats,
            "unresolved_partitions": [],
            "complete": complete,
            "urls": urls,
        })

    for country, ages in selected_counts.items():
        for age, configured_count in ages.items():
            key = f"country={country}&age={age}"
            if key in completed:
                continue
            base_url = partition_url(country, age)
            if configured_count <= 100:
                print(f"[shallow {key}] lấy toàn bộ partition nhỏ...", flush=True)
                first_html = client.get_text(base_url)
                found = collect_small_partition(client, first_html, base_url, key)
                strategy = "all-pages"
            else:
                print(
                    f"[shallow {key}] {configured_count} bài; "
                    "lấy cửa sổ Date asc/desc...",
                    flush=True,
                )
                found = []
                for order in ("asc", "desc"):
                    window_url = f"{base_url}&Sort=Date"
                    if order == "desc":
                        window_url += "&SortOrder=desc"
                    window = collect_capped_window(
                        client, window_url, f"{key}&date-{order}"
                    )
                    known = set(found)
                    found.extend(url for url in window if url not in known)
                strategy = "date-asc-desc-windows"

            known_urls = set(urls)
            urls.extend(url for url in found if url not in known_urls)
            partition_stats[key] = {
                "configured_results": configured_count,
                "strategy": strategy,
                "collected_urls": len(found),
                "new_unique_urls": len(urls) - len(known_urls),
            }
            completed.add(key)
            persist()
            print(
                f"[shallow {key}] thu được {len(found)} URL; tổng unique={len(urls)}",
                flush=True,
            )

    complete = partition_keys.issubset(completed)
    persist(complete=complete)
    if not complete:
        missing = sorted(partition_keys - completed)
        raise CrawlError(f"Shallow discovery chưa hoàn tất: {missing}")
    return urls


def collect_urls(
    client: HttpClient,
    limit: int,
    checkpoint_path: Path,
    refresh_checkpoint: bool,
    country_ids_override: list[str] | None = None,
) -> list[str]:
    if DEFAULT_SHALLOW_AGE_PARTITION_COUNTS:
        urls = collect_shallow_age_windows(
            client,
            checkpoint_path,
            refresh_checkpoint,
            DEFAULT_SHALLOW_AGE_PARTITION_COUNTS,
            country_ids_override,
        )
        selected = list(urls)
        if limit:
            # A stable shuffled prefix samples across all configured age windows
            # and remains nested when the user later increases --limit.
            random.Random(42).shuffle(selected)
            selected = selected[:limit]
            print(
                f"[danh sách] Chọn mẫu cố định {len(selected)}/{len(urls)} "
                "URL shallow để crawl.",
                flush=True,
            )
        return selected

    # Các lần test có limit nhỏ chỉ cần trang đầu và không thay đổi checkpoint
    # của lần crawl toàn bộ.
    if limit:
        print("[danh sách] Đang tải trang 1...", flush=True)
        first_html = client.get_text(SEARCH_URL)
        total_pages = parse_total_pages(first_html)
        urls = parse_search_page(first_html)
        for page in range(2, total_pages + 1):
            if len(urls) >= limit:
                break
            print(f"[danh sách {page}/{total_pages}] Đang tải bản test...", flush=True)
            known_urls = set(urls)
            page_urls = parse_search_page(client.get_text(f"{SEARCH_URL}&Page={page}"))
            urls.extend(url for url in page_urls if url not in known_urls)
        urls = urls[:limit]
        print(f"[danh sách] Đã lấy {len(urls)} URL thử nghiệm.", flush=True)
        return urls

    checkpoint = None if refresh_checkpoint else load_url_checkpoint(checkpoint_path)
    if checkpoint is not None and checkpoint.get("complete") is True:
        urls = list(checkpoint["urls"])
        print(f"[danh sách] Dùng checkpoint hoàn chỉnh: {len(urls)} URL", flush=True)
        return urls

    urls = list(checkpoint["urls"]) if checkpoint else []
    completed = set(checkpoint.get("completed_partitions", [])) if checkpoint else set()
    previous_unresolved = list(checkpoint.get("unresolved_partitions", [])) if checkpoint else []
    partition_stats = dict(checkpoint.get("partition_stats", {})) if checkpoint else {}
    unresolved: list[dict[str, object]] = []

    configured_age_counts = {
        str(country): {str(age): int(count) for age, count in ages.items()}
        for country, ages in DEFAULT_AGE_PARTITION_COUNTS.items()
    }
    stored_age_counts = checkpoint.get("age_partition_counts", {}) if checkpoint else {}
    if checkpoint and configured_age_counts and stored_age_counts != configured_age_counts:
        configured_countries = set(configured_age_counts)
        for country in configured_countries:
            country_prefix = f"country={country}"
            completed = {
                key for key in completed
                if key != country_prefix and not key.startswith(f"{country_prefix}&age=")
            }
            partition_stats = {
                key: value for key, value in partition_stats.items()
                if key != country_prefix and not key.startswith(f"{country_prefix}&age=")
            }
        previous_unresolved = [
            item for item in previous_unresolved
            if not any(
                str(item.get("partition", "")).startswith(f"country={country}")
                for country in configured_countries
            )
        ]
        completed.discard("recovery-country-asc")
        completed.discard("recovery-country-desc")
        print(
            "[checkpoint] Cấu hình age đã thay đổi; mở lại các phân vùng "
            f"country={','.join(sorted(configured_countries))}.",
            flush=True,
        )

    if checkpoint is None:
        print("[danh sách] Đang đọc tổng kết quả và danh sách quốc gia...", flush=True)
        root_html = client.get_text(SEARCH_URL)
        expected_total = parse_result_count(root_html)
        country_ids = parse_country_ids(root_html)
        if not country_ids:
            raise CrawlError("Không đọc được danh sách quốc gia từ form tìm kiếm")
        if country_ids_override is not None:
            available_country_ids = set(country_ids)
            unknown = [
                country for country in country_ids_override
                if country not in available_country_ids
            ]
            if unknown:
                raise CrawlError(f"Country ID không có trong form tìm kiếm: {unknown}")
            country_ids = list(country_ids_override)
    else:
        expected_total = int(checkpoint.get("expected_total", 0))
        stored_country_ids = checkpoint.get("country_ids", [])
        country_ids = [str(value) for value in stored_country_ids if str(value).isdigit()]
        if country_ids_override is not None:
            country_ids = list(country_ids_override)
        if not country_ids:
            country_ids = sorted(
                set(LEAF_AGES).union(
                    match.group(1)
                    for key in completed
                    if (match := re.fullmatch(r"country=(\d+)", key))
                ),
                key=int,
            )
        if expected_total <= 0 or not country_ids:
            raise CrawlError("Checkpoint thiếu expected_total/country_ids để tiếp tục")
        print(
            f"[danh sách] Tiếp tục từ checkpoint: {len(urls)}/{expected_total} URL; "
            "không tải lại form gốc.",
            flush=True,
        )

    # v2/v3 dùng Author=<tên>, có thể bỏ sót do tên thay thế/trùng tên. Giữ
    # toàn bộ URL đã tìm được, nhưng mở lại hai cây country/age để chuyển sang
    # IAuthor=<UID>. Các author checkpoint mới sẽ được ghi theo schema v4.
    if checkpoint and checkpoint.get("schema_version") != "thivien-partition-checkpoint-v4":
        for country in LEAF_AGES:
            prefix = f"country={country}"
            completed = {
                key for key in completed
                if key != prefix and not key.startswith(f"{prefix}&age=")
            }
        completed = {
            key for key in completed
            if "&author=" not in key and not key.startswith("recovery-")
        }

    # Di trú checkpoint v2: giữ URL đã có nhưng mở lại phân vùng từng bị >100.
    for item in previous_unresolved:
        if not isinstance(item, dict) or not isinstance(item.get("partition"), str):
            continue
        partition = str(item["partition"])
        completed.discard(partition)
        if "&age=" in partition:
            completed.discard(partition.split("&age=", 1)[0])

    def persist(complete: bool = False) -> None:
        save_url_checkpoint(checkpoint_path, {
            "strategy": "country-age-v1",
            "expected_total": expected_total,
            "country_ids": country_ids,
            "age_partition_counts": configured_age_counts,
            "discovered_urls": len(urls),
            "completed_partitions": sorted(completed),
            "partition_stats": partition_stats,
            "unresolved_partitions": unresolved,
            "complete": complete,
            "urls": urls,
        })

    def extend_discovered(found: list[str]) -> int:
        known = set(urls)
        urls.extend(url for url in found if url not in known)
        return len(urls) - len(known)

    def collect_by_authors(
        country: str,
        age: str,
        age_key: str,
        age_count: int,
    ) -> bool:
        print(
            f"[phân vùng {age_key}] {age_count} kết quả; chia tiếp theo tác giả",
            flush=True,
        )
        try:
            authors = discover_partition_authors(client, country, age)
        except CrawlError as error:
            unresolved.append({
                "partition": age_key,
                "results": age_count,
                "error": str(error),
            })
            persist()
            return False

        resolved = True
        for author_url, author_name in authors.items():
            uid = author_uid(author_url)
            author_key = f"{age_key}&author_uid={uid}"
            if author_key in completed:
                continue
            query_url = author_poem_url(country, age, author_url)
            print(
                f"[phân vùng tác giả] {author_name} ({age_key})...",
                flush=True,
            )
            author_html = client.get_text(query_url)
            author_count = parse_result_count(author_html)
            if author_count > 100:
                unresolved.append({
                    "partition": author_key,
                    "results": author_count,
                    "error": "Một tác giả vẫn vượt giới hạn 100 bài",
                })
                resolved = False
            else:
                found = collect_small_partition(
                    client,
                    author_html,
                    query_url,
                    f"{age_key}&author={author_name}",
                )
                extend_discovered(found)
            completed.add(author_key)
            persist()
        return resolved

    def collect_country_ages(country: str) -> bool:
        country_resolved = True
        configured_counts = configured_age_counts.get(country)
        ages = list(configured_counts) if configured_counts is not None else LEAF_AGES[country]
        for age in ages:
            age_key = f"country={country}&age={age}"
            if age_key in completed:
                continue
            age_url = partition_url(country, age)
            expected_count = configured_counts.get(age) if configured_counts else None
            age_resolved = True
            found: list[str] = []

            if expected_count == 0:
                partition_stats[age_key] = {
                    "expected_results": 0,
                    "collected_urls": 0,
                    "new_unique_urls": 0,
                }
                completed.add(age_key)
                persist()
                print(f"[phân vùng {age_key}] 0 kết quả; bỏ qua request", flush=True)
                continue

            print(f"[phân vùng {age_key}] đang kiểm tra...", flush=True)
            if expected_count is not None and expected_count > 100:
                age_count = expected_count
                age_html = None
            else:
                age_html = client.get_text(age_url)
                actual_count = parse_result_count(age_html)
                age_count = expected_count if expected_count is not None else actual_count
                if expected_count is not None and actual_count != expected_count:
                    unresolved.append({
                        "partition": age_key,
                        "results": actual_count,
                        "error": (
                            f"Số kết quả thay đổi: cấu hình={expected_count}, "
                            f"website={actual_count}"
                        ),
                    })
                    country_resolved = False
                    persist()
                    continue

            if age_count <= 100:
                assert age_html is not None
                found = collect_small_partition(
                    client, age_html, age_url, age_key
                )
                if len(found) != age_count:
                    unresolved.append({
                        "partition": age_key,
                        "results": age_count,
                        "error": f"Thu được {len(found)}/{age_count} URL",
                    })
                    age_resolved = False
            elif age_count < 200:
                try:
                    found = collect_bidirectional_window(
                        client, age_url, age_key, age_count
                    )
                except CrawlError as error:
                    unresolved.append({
                        "partition": age_key,
                        "results": age_count,
                        "error": str(error),
                    })
                    age_resolved = False
            else:
                age_resolved = collect_by_authors(
                    country, age, age_key, age_count
                )

            if age_resolved:
                added = extend_discovered(found) if found else 0
                partition_stats[age_key] = {
                    "expected_results": age_count,
                    "collected_urls": len(found),
                    "new_unique_urls": added,
                }
                completed.add(age_key)
            else:
                country_resolved = False
            persist()
        return country_resolved

    for country in country_ids:
        country_key = f"country={country}"
        if country_key in completed:
            continue
        country_url = partition_url(country)
        print(f"[phân vùng {country_key}] đang kiểm tra...", flush=True)
        country_html = client.get_text(country_url)
        country_count = parse_result_count(country_html)
        country_resolved = True

        if country_count <= 100:
            found = collect_small_partition(client, country_html, country_url, country_key)
            extend_discovered(found)
        elif country_count < 200:
            try:
                found = collect_bidirectional_window(
                    client, country_url, country_key, country_count
                )
                extend_discovered(found)
            except CrawlError as error:
                if country in LEAF_AGES:
                    print(
                        f"[phân vùng {country_key}] cửa sổ hai chiều thất bại; "
                        "chia tiếp theo thời kỳ",
                        flush=True,
                    )
                    country_resolved = collect_country_ages(country)
                else:
                    unresolved.append({
                        "partition": country_key,
                        "results": country_count,
                        "error": str(error),
                    })
                    country_resolved = False
        elif country in LEAF_AGES:
            country_resolved = collect_country_ages(country)
        else:
            unresolved.append({
                "partition": country_key,
                "results": country_count,
                "error": "Phân vùng >=200 và chưa có cấu hình chia thời kỳ",
            })
            country_resolved = False

        if country_resolved:
            completed.add(country_key)
        persist()
        status = "hoàn tất" if country_key in completed else "chưa hoàn tất"
        print(f"[phân vùng {country_key}] {status}; tổng {len(urls)}/{expected_total} URL", flush=True)

    # Một số bài thiếu metadata country/age nên không xuất hiện trong các phân
    # vùng trên. Sort theo Country ở hai chiều đưa nhóm rỗng ra một trong hai
    # đầu danh sách, cho phép phục hồi chúng trong giới hạn 10 trang.
    if len(urls) < expected_total:
        for order in ("asc", "desc"):
            recovery_key = f"recovery-country-{order}"
            if recovery_key in completed:
                continue
            recovery_url = f"{SEARCH_URL}&Sort=Country"
            if order == "desc":
                recovery_url += "&SortOrder=desc"
            found = collect_capped_window(client, recovery_url, recovery_key)
            known = set(urls)
            urls.extend(url for url in found if url not in known)
            completed.add(recovery_key)
            persist()
            print(
                f"[cửa sổ {recovery_key}] tổng {len(urls)}/{expected_total} URL",
                flush=True,
            )

    complete = not unresolved and len(urls) == expected_total
    persist(complete=complete)
    if not complete:
        raise CrawlError(
            f"Khám phá URL chưa đầy đủ: {len(urls)}/{expected_total}; "
            f"phân vùng chưa giải quyết: {unresolved}"
        )
    return urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Crawl thể loại {GENRE_LABEL} chữ Hán từ Thi Viện."
    )
    parser.add_argument("--limit", type=int, default=5, help="0 để crawl toàn bộ.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--url-checkpoint", type=Path, default=DEFAULT_URL_CHECKPOINT)
    parser.add_argument(
        "--country-ids",
        default=None,
        help=(
            "Chỉ duyệt các country ID phân cách bằng dấu phẩy, ví dụ 2,3. "
            "Mặc định do từng pipeline cấu hình hoặc dùng toàn bộ form."
        ),
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Checkpoint từng bài; mặc định nằm cạnh crawl_report.json.",
    )
    parser.add_argument(
        "--refresh-url-checkpoint",
        action="store_true",
        help="Bỏ qua checkpoint URL cũ và đọc lại danh sách từ trang 1.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Khoảng cách tối thiểu giữa hai request (mặc định: 5 giây).",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=3.0,
        help="Thêm ngẫu nhiên 0..N giây giữa các request (mặc định: 3).",
    )
    parser.add_argument(
        "--pause-every",
        type=int,
        default=40,
        help="Nghỉ dài sau mỗi N request; 0 để tắt (mặc định: 20).",
    )
    parser.add_argument(
        "--pause-min",
        type=float,
        default=300.0,
        help="Thời gian nghỉ dài tối thiểu, tính bằng giây (mặc định: 300).",
    )
    parser.add_argument(
        "--pause-max",
        type=float,
        default=600.0,
        help="Thời gian nghỉ dài tối đa, tính bằng giây (mặc định: 600).",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--captcha-pause-min",
        type=float,
        default=600.0,
        help="Nghỉ tối thiểu khi gặp CAPTCHA, tính bằng giây (mặc định: 600).",
    )
    parser.add_argument(
        "--captcha-pause-max",
        type=float,
        default=900.0,
        help="Nghỉ tối đa khi gặp CAPTCHA, tính bằng giây (mặc định: 900).",
    )
    parser.add_argument(
        "--captcha-retries",
        type=int,
        default=3,
        help="Số lần chờ và thử lại CAPTCHA trước khi dừng hẳn (mặc định: 3).",
    )
    parser.add_argument(
        "--trust-progress",
        action="store_true",
        help=(
            "Tin trạng thái written/existing trong progress JSON và bỏ qua URL "
            "mà không kiểm tra file CSV còn tồn tại."
        ),
    )
    parser.add_argument(
        "--retry-skipped",
        action="store_true",
        help="Thử lại URL từng bị đánh dấu skipped trong progress JSON.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.limit < 0
        or args.delay < 0
        or args.jitter < 0
        or args.pause_every < 0
        or args.pause_min < 0
        or args.pause_max < args.pause_min
        or args.timeout <= 0
        or args.retries < 0
        or args.captcha_pause_min < 0
        or args.captcha_pause_max < args.captcha_pause_min
        or args.captcha_retries < 0
    ):
        parser.error(
            "limit/delay/jitter/pause/retries phải hợp lệ, timeout phải > 0 "
            "và các giá trị pause-max phải >= pause-min"
        )
    if args.progress is None:
        args.progress = args.report.parent / "poem_crawl_progress.json"
    if args.country_ids is None:
        args.country_ids = (
            list(DEFAULT_COUNTRY_IDS) if DEFAULT_COUNTRY_IDS is not None else None
        )
    else:
        country_ids = [value.strip() for value in args.country_ids.split(",")]
        if not country_ids or any(not value.isdigit() for value in country_ids):
            parser.error("--country-ids cần là các số phân cách bằng dấu phẩy")
        args.country_ids = list(dict.fromkeys(country_ids))
    return args


def main() -> int:
    args = parse_args()
    client = HttpClient(
        args.delay,
        args.jitter,
        args.pause_every,
        args.pause_min,
        args.pause_max,
        args.timeout,
        args.retries,
        captcha_pause_min=args.captcha_pause_min,
        captcha_pause_max=args.captcha_pause_max,
        captcha_retries=args.captcha_retries,
    )
    assert_allowed(client)
    urls = collect_urls(
        client,
        args.limit,
        args.url_checkpoint,
        args.refresh_url_checkpoint,
        args.country_ids,
    )
    if not urls:
        raise CrawlError(f"Không tìm thấy bài {GENRE_LABEL} chữ Hán nào")

    progress_items = load_poem_progress(args.progress)
    used_stems = {
        Path(str(item["output"])).stem
        for item in progress_items.values()
        if isinstance(item.get("output"), str)
    }
    for index, url in enumerate(urls, start=1):
        previous = progress_items.get(url)
        if previous is not None and not args.overwrite:
            status = previous.get("status")
            output = progress_output_path(args.output_dir, previous)
            progress_says_complete = status in {"written", "existing"}
            output_exists = output is not None and output.exists()
            if progress_says_complete and (args.trust_progress or output_exists):
                label = output.name if output is not None else "đã ghi trong JSON"
                print(f"[{index}/{len(urls)}] ĐÃ CÓ: {label}", flush=True)
                continue
            if status == "skipped" and not args.retry_skipped:
                print(f"[{index}/{len(urls)}] ĐÃ BỎ QUA: {url}", flush=True)
                continue

        # Lỗi mạng/CAPTCHA phải dừng để progress các bài trước được giữ nguyên,
        # không được ghi nhầm thành một bài có nội dung không hợp lệ.
        html = client.get_text(url)
        try:
            poem = parse_poem(html, url)
            previous_output = (
                progress_output_path(args.output_dir, previous)
                if previous is not None
                else None
            )
            if (
                previous is not None
                and previous.get("status") in {"written", "existing"}
                and previous_output is not None
            ):
                stem = previous_output.stem
            else:
                stem = filename_stem(poem.title_vi, poem.uid)
                if stem in used_stems:
                    stem = stem_with_uid(stem, poem.uid)
            used_stems.add(stem)
            output = args.output_dir / f"{stem}.csv"
            if output.exists() and not args.overwrite:
                status = "existing"
            else:
                write_poem(poem, args.output_dir, stem, args.overwrite)
                status = "written"
            progress_items[url] = {
                "status": status,
                "output": output.name,
                "title_vi": poem.title_vi,
                "sentence_pairs": len(poem.lines_vi),
            }
            save_poem_progress(args.progress, progress_items)
            label = "ĐÃ CÓ" if status == "existing" else "ĐÃ GHI"
            print(
                f"[{index}/{len(urls)}] {label}: {poem.title_vi}: "
                f"{len(poem.lines_vi)} cặp câu",
                flush=True,
            )
        except CrawlError as error:
            progress_items[url] = {
                "status": "skipped",
                "url": url,
                "error": str(error),
            }
            save_poem_progress(args.progress, progress_items)
            print(f"[{index}/{len(urls)}] BỎ QUA: {error}", flush=True)

    selected_items = [progress_items[url] for url in urls if url in progress_items]
    successful = [
        item
        for item in selected_items
        if item.get("status") in {"written", "existing"}
    ]
    skipped = [item for item in selected_items if item.get("status") == "skipped"]
    sentence_pairs = sum(int(item.get("sentence_pairs", 0)) for item in successful)

    report = {
        "schema_version": REPORT_SCHEMA,
        "source": SEARCH_URL,
        "discovered_poems": len(urls),
        "written_csv_files": len(successful),
        "sentence_pairs": sentence_pairs,
        "skipped_poems": len(skipped),
        "skipped": skipped,
    }
    atomic_write(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Hoàn tất {len(successful)}/{len(urls)} CSV; "
        f"bỏ qua {len(skipped)} bài. Report: {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
