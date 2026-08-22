#!/usr/bin/env python3
"""Crawl aligned Han-Viet/chữ Hán pairs for the Phú genre from Thi Viện."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup, Tag


BASE_URL = "https://www.thivien.net"
SEARCH_URL = f"{BASE_URL}/search-poem.php?PoemType=2&ViewType=2"
GENRE_LABEL = "Phú"
REPORT_SCHEMA = "thivien-phu-crawl-v1"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
USER_AGENT = "vi2ch-dataset-research/1.0 (polite crawler; contact repository owner)"

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "raw-collections" / "poetry-collecions" / "phu"
DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "crawl_report.json"
DEFAULT_URL_CHECKPOINT = PIPELINE_ROOT / "outputs" / "poem_urls.json"

POEM_LINK_RE = re.compile(r"/poem-([A-Za-z0-9_-]+)$")
AUTHOR_LINK_RE = re.compile(r"/author-([A-Za-z0-9_-]+)$")
NON_FILENAME_RE = re.compile(r"[^a-z0-9]+")
RESULT_COUNT_RE = re.compile(r"tổng số\s+\d+\s+trang\s+\(([\d.]+)\s+bài thơ\)")
TOO_MANY_RE = re.compile(r"Có quá nhiều\s+\(([\d.]+)\)\s+kết quả")
AUTHOR_COUNT_RE = re.compile(r"tổng số\s+\d+\s+trang\s+\(([\d.]+)\s+tác giả\)")

# Các thời đại lá trên form Thi Viện. Chỉ dùng khi một quốc gia có hơn 100
# kết quả, nhằm tránh giới hạn chỉ xem được 10 trang đầu của website.
LEAF_AGES: dict[str, list[str]] = {
    "2": ["50", "52", "53", "54", "55", "56", "57", "2", "3"],
    "3": [
        "21", "22", "23", "24", "25", "26", "27", "7", "8", "9",
        "10", "12", "28", "29", "14", "15", "16", "17", "18",
    ],
}


class CrawlError(RuntimeError):
    pass


class AccessChallengeError(CrawlError):
    pass


@dataclass(frozen=True)
class Poem:
    uid: str
    url: str
    title_vi: str
    lines_vi: list[str]
    lines_ch: list[str]


class HttpClient:
    def __init__(self, delay: float, timeout: float, retries: int) -> None:
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self._last_request_at: float | None = None

    def get_text(self, url: str) -> str:
        if self._last_request_at is not None:
            remaining = self.delay - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)

        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "vi,en;q=0.7",
            },
        )
        last_error: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                self._last_request_at = time.monotonic()
                with urlopen(request, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    html = response.read().decode(charset, errors="strict")
                    if (
                        "Xác nhận không phải máy truy cập tự động" in html
                        or 'class="g-recaptcha"' in html
                    ):
                        raise AccessChallengeError(
                            "Thi Viện đang yêu cầu xác minh CAPTCHA; "
                            "checkpoint vẫn được giữ nguyên"
                        )
                    return html
            except AccessChallengeError as error:
                last_error = error
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
    lines_ch = normalized_lines(han_body)
    lines_vi = normalized_lines(vi_body)
    if len(title_lines) != 1:
        raise CrawlError(f"Tiêu đề không hợp lệ: {url}")
    if not lines_ch or len(lines_ch) != len(lines_vi):
        raise CrawlError(
            f"Lệch dòng tại {url}: chữ Hán={len(lines_ch)}, Hán-Việt={len(lines_vi)}"
        )

    match = POEM_LINK_RE.search(urlparse(url).path)
    if match is None:
        raise CrawlError(f"URL không có UID: {url}")
    return Poem(match.group(1), url, title_lines[0], lines_vi, lines_ch)


def filename_stem(title: str) -> str:
    ascii_title = (
        unicodedata.normalize("NFKD", title.replace("Đ", "D").replace("đ", "d"))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    stem = NON_FILENAME_RE.sub("", ascii_title)
    if not stem:
        raise CrawlError(f"Không thể tạo tên file từ {title!r}")
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
    writer.writerow(["vi", "ch"])
    writer.writerows(zip(poem.lines_vi, poem.lines_ch, strict=True))
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


def collect_urls(
    client: HttpClient,
    limit: int,
    checkpoint_path: Path,
    refresh_checkpoint: bool,
) -> list[str]:
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
    unresolved: list[dict[str, object]] = []

    if checkpoint is None:
        print("[danh sách] Đang đọc tổng kết quả và danh sách quốc gia...", flush=True)
        root_html = client.get_text(SEARCH_URL)
        expected_total = parse_result_count(root_html)
        country_ids = parse_country_ids(root_html)
        if not country_ids:
            raise CrawlError("Không đọc được danh sách quốc gia từ form tìm kiếm")
    else:
        expected_total = int(checkpoint.get("expected_total", 0))
        stored_country_ids = checkpoint.get("country_ids", [])
        country_ids = [str(value) for value in stored_country_ids if str(value).isdigit()]
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
            "discovered_urls": len(urls),
            "completed_partitions": sorted(completed),
            "unresolved_partitions": unresolved,
            "complete": complete,
            "urls": urls,
        })

    for country in country_ids:
        country_key = f"country={country}"
        if country_key in completed:
            continue
        country_url = partition_url(country)
        print(f"[phân vùng {country_key}] đang kiểm tra...", flush=True)
        country_html = client.get_text(country_url)
        country_count = parse_result_count(country_html)

        if country_count <= 100:
            found = collect_small_partition(client, country_html, country_url, country_key)
            known = set(urls)
            urls.extend(url for url in found if url not in known)
        elif country in LEAF_AGES:
            country_resolved = True
            for age in LEAF_AGES[country]:
                age_key = f"country={country}&age={age}"
                if age_key in completed:
                    continue
                age_url = partition_url(country, age)
                print(f"[phân vùng {age_key}] đang kiểm tra...", flush=True)
                age_html = client.get_text(age_url)
                age_count = parse_result_count(age_html)
                if age_count > 100:
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
                        country_resolved = False
                        persist()
                        continue

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
                            country_resolved = False
                        else:
                            found = collect_small_partition(
                                client,
                                author_html,
                                query_url,
                                f"{age_key}&author={author_name}",
                            )
                            known = set(urls)
                            urls.extend(url for url in found if url not in known)
                        completed.add(author_key)
                        persist()
                    if any(item.get("partition", "").startswith(age_key) for item in unresolved):
                        country_resolved = False
                        continue
                else:
                    found = collect_small_partition(client, age_html, age_url, age_key)
                    known = set(urls)
                    urls.extend(url for url in found if url not in known)
                completed.add(age_key)
                persist()
        else:
            unresolved.append({"partition": country_key, "results": country_count})

        if country_count <= 100 or country not in LEAF_AGES or country_resolved:
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
        "--refresh-url-checkpoint",
        action="store_true",
        help="Bỏ qua checkpoint URL cũ và đọc lại danh sách từ trang 1.",
    )
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.delay < 0 or args.timeout <= 0 or args.retries < 0:
        parser.error("Tham số limit/delay/retries phải >= 0 và timeout phải > 0")
    return args


def main() -> int:
    args = parse_args()
    client = HttpClient(args.delay, args.timeout, args.retries)
    assert_allowed(client)
    urls = collect_urls(
        client,
        args.limit,
        args.url_checkpoint,
        args.refresh_url_checkpoint,
    )
    if not urls:
        raise CrawlError(f"Không tìm thấy bài {GENRE_LABEL} chữ Hán nào")

    used_stems: set[str] = set()
    written = 0
    sentence_pairs = 0
    skipped: list[dict[str, str]] = []
    for index, url in enumerate(urls, start=1):
        try:
            poem = parse_poem(client.get_text(url), url)
            stem = filename_stem(poem.title_vi)
            if stem in used_stems:
                stem = f"{stem}-{poem.uid}"
            used_stems.add(stem)
            write_poem(poem, args.output_dir, stem, args.overwrite)
            written += 1
            sentence_pairs += len(poem.lines_vi)
            print(f"[{index}/{len(urls)}] {poem.title_vi}: {len(poem.lines_vi)} cặp câu")
        except CrawlError as error:
            skipped.append({"url": url, "error": str(error)})
            print(f"[{index}/{len(urls)}] BỎ QUA: {error}")

    report = {
        "schema_version": REPORT_SCHEMA,
        "source": SEARCH_URL,
        "discovered_poems": len(urls),
        "written_csv_files": written,
        "sentence_pairs": sentence_pairs,
        "skipped_poems": len(skipped),
        "skipped": skipped,
    }
    atomic_write(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Đã ghi {written} CSV; bỏ qua {len(skipped)} bài. Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
