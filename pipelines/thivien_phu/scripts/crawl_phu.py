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
from urllib.parse import parse_qs, urljoin, urlparse
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
NON_FILENAME_RE = re.compile(r"[^a-z0-9]+")


class CrawlError(RuntimeError):
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
                    return response.read().decode(charset, errors="strict")
            except HTTPError as error:
                last_error = error
                if error.code not in {429, 500, 502, 503, 504}:
                    break
            except (URLError, TimeoutError, UnicodeDecodeError) as error:
                last_error = error
            if attempt < self.retries:
                time.sleep(min(2**attempt, 8))
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
    if not isinstance(checkpoint, dict) or checkpoint.get("source") != SEARCH_URL:
        return None
    urls = checkpoint.get("urls")
    if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
        raise CrawlError(f"Checkpoint URL không có danh sách urls hợp lệ: {path}")
    return checkpoint


def save_url_checkpoint(
    path: Path,
    urls: list[str],
    total_pages: int,
    last_completed_page: int,
    complete: bool,
) -> None:
    checkpoint = {
        "schema_version": "thivien-url-checkpoint-v1",
        "source": SEARCH_URL,
        "total_pages": total_pages,
        "last_completed_page": last_completed_page,
        "discovered_urls": len(urls),
        "complete": complete,
        "urls": urls,
    }
    atomic_write(path, json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n")


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
        print(
            f"[danh sách] Dùng checkpoint hoàn chỉnh: {len(urls)} URL "
            f"từ {checkpoint_path}",
            flush=True,
        )
        return urls

    if checkpoint is not None:
        urls = list(checkpoint["urls"])
        total_pages = int(checkpoint.get("total_pages", 1))
        last_completed_page = int(checkpoint.get("last_completed_page", 0))
        start_page = last_completed_page + 1
        print(
            f"[danh sách] Tiếp tục checkpoint tại trang {start_page}/{total_pages}; "
            f"đã có {len(urls)} URL.",
            flush=True,
        )
    else:
        print("[danh sách] Đang tải trang 1...", flush=True)
        first_html = client.get_text(SEARCH_URL)
        total_pages = parse_total_pages(first_html)
        urls = parse_search_page(first_html)
        start_page = 2
        save_url_checkpoint(checkpoint_path, urls, total_pages, 1, total_pages == 1)
        print(
            f"[danh sách 1/{total_pages}] {len(urls)} URL; "
            f"đã lưu checkpoint.",
            flush=True,
        )

    for page in range(start_page, total_pages + 1):
        print(f"[danh sách {page}/{total_pages}] Đang tải...", flush=True)
        page_urls = parse_search_page(client.get_text(f"{SEARCH_URL}&Page={page}"))
        known_urls = set(urls)
        urls.extend(url for url in page_urls if url not in known_urls)
        save_url_checkpoint(
            checkpoint_path,
            urls,
            total_pages,
            page,
            page == total_pages,
        )
        print(
            f"[danh sách {page}/{total_pages}] +{len(page_urls)} kết quả, "
            f"tổng {len(urls)} URL; đã lưu checkpoint.",
            flush=True,
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
