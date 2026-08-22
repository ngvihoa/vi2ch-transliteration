#!/usr/bin/env python3
"""Crawl aligned Han-Viet/Kinh Thi text pairs from thivien.net."""

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
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup, Tag


BASE_URL = "https://www.thivien.net"
GROUP_URL = (
    f"{BASE_URL}/Kh%E1%BB%95ng-T%E1%BB%AD/"
    "Thi-kinh-Kinh-thi/group-ZDB2Tl5514uy8PI478SU_g"
)
ROBOTS_URL = f"{BASE_URL}/robots.txt"
USER_AGENT = "vi2ch-dataset-research/1.0 (polite crawler; contact repository owner)"

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "raw-collections" / "poetry-collecions"
DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "crawl_report.json"

POEM_LINK_RE = re.compile(r"/poem-([A-Za-z0-9_-]+)$")
GROUP_LINK_RE = re.compile(r"/group-[A-Za-z0-9_-]+$")
NON_FILENAME_RE = re.compile(r"[^a-z0-9]+")


class CrawlError(RuntimeError):
    """Raised when source data cannot be fetched or aligned safely."""


@dataclass(frozen=True)
class Poem:
    uid: str
    url: str
    title_vi: str
    title_ch: str
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
    # Thi Viện bọc một số từ trong các span chú giải. Chỉ thẻ <br> biểu thị
    # xuống dòng thơ; dùng separator của get_text() sẽ tách nhầm các span đó.
    # Các số chú thích nằm trong <sup> cũng không thuộc nội dung phiên âm.
    for footnote_marker in node.find_all("sup"):
        footnote_marker.decompose()
    for line_break in node.find_all("br"):
        line_break.replace_with("\n")
    text = node.get_text("", strip=False)
    return [
        " ".join(unicodedata.normalize("NFC", line).split())
        for line in text.splitlines()
        if line.strip()
    ]


def unique_matching_urls(anchors: list[Tag], pattern: re.Pattern[str]) -> list[str]:
    """Resolve matching links while retaining their first-seen order."""
    urls: list[str] = []
    seen: set[str] = set()
    for anchor in anchors:
        absolute_url = urljoin(BASE_URL, str(anchor["href"]))
        if pattern.search(urlparse(absolute_url).path) and absolute_url not in seen:
            urls.append(absolute_url)
            seen.add(absolute_url)
    return urls


def parse_section_urls(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    return unique_matching_urls(
        list(soup.select("h4.poem-group-title > a[href]")), GROUP_LINK_RE
    )


def parse_group_poem_urls(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    return unique_matching_urls(
        list(soup.select("div.poem-group-list a[href]")), POEM_LINK_RE
    )


def parse_poem(html: str, url: str) -> Poem:
    soup = BeautifulSoup(html, "html.parser")
    section = soup.select_one("div.poem-content div.poem-view-separated")
    if section is None:
        raise CrawlError(f"Không tìm thấy phần nguyên tác/phiên âm: {url}")

    han_body = section.select_one("p.han-chinese")
    if han_body is None:
        raise CrawlError(f"Không tìm thấy nội dung chữ Hán: {url}")

    headings = section.find_all("h4", recursive=False)
    han_heading = next((heading for heading in headings if heading.select_one(".han-chinese")), None)
    vi_heading = next((heading for heading in headings if not heading.select_one(".han-chinese")), None)
    if han_heading is None or vi_heading is None:
        raise CrawlError(f"Không tìm thấy cặp tiêu đề: {url}")

    vi_body = vi_heading.find_next_sibling("p")
    if vi_body is None or vi_body is han_body:
        raise CrawlError(f"Không tìm thấy phiên âm Hán-Việt: {url}")

    title_ch_lines = normalized_lines(han_heading)
    title_vi_lines = normalized_lines(vi_heading)
    lines_ch = normalized_lines(han_body)
    lines_vi = normalized_lines(vi_body)
    if len(title_ch_lines) != 1 or len(title_vi_lines) != 1:
        raise CrawlError(f"Tiêu đề không hợp lệ: {url}")
    if not lines_ch or len(lines_ch) != len(lines_vi):
        raise CrawlError(
            f"Lệch dòng tại {url}: chữ Hán={len(lines_ch)}, Hán-Việt={len(lines_vi)}"
        )

    match = POEM_LINK_RE.search(urlparse(url).path)
    if match is None:
        raise CrawlError(f"URL bài thơ không có UID: {url}")

    return Poem(
        uid=match.group(1),
        url=url,
        title_vi=title_vi_lines[0],
        title_ch=title_ch_lines[0],
        lines_vi=lines_vi,
        lines_ch=lines_ch,
    )


def filename_stem(title: str) -> str:
    ascii_title = (
        unicodedata.normalize("NFKD", title.replace("Đ", "D").replace("đ", "d"))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    stem = NON_FILENAME_RE.sub("", ascii_title)
    if not stem:
        raise CrawlError(f"Không thể tạo tên file từ tiêu đề {title!r}")
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
    output_path = output_dir / f"{stem}.csv"
    if output_path.exists() and not overwrite:
        raise CrawlError(f"Output đã tồn tại ({output_path}); dùng --overwrite để thay thế")

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["vi", "ch"])
    writer.writerows(zip(poem.lines_vi, poem.lines_ch, strict=True))
    atomic_write(output_path, buffer.getvalue())
    return output_path


def assert_allowed(client: HttpClient) -> None:
    robots_text = client.get_text(ROBOTS_URL)
    parser = RobotFileParser()
    parser.set_url(ROBOTS_URL)
    parser.parse(robots_text.splitlines())
    if not parser.can_fetch(USER_AGENT, GROUP_URL):
        raise CrawlError(f"robots.txt không cho phép crawl {GROUP_URL}")


def collect_urls(client: HttpClient, limit: int) -> list[str]:
    root_html = client.get_text(GROUP_URL)
    section_urls = parse_section_urls(root_html)
    if not section_urls:
        raise CrawlError("Không tìm thấy các phần Quốc phong, Nhã và Tụng")

    urls: list[str] = []
    for section_url in section_urls:
        section_html = client.get_text(section_url)
        for url in parse_group_poem_urls(section_html):
            if url not in urls:
                urls.append(url)
        if limit and len(urls) >= limit:
            break
    return urls if limit == 0 else urls[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl Kinh Thi thành một file CSV song song cho mỗi bài thơ."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Số bài tối đa; 0 để crawl toàn bộ (mặc định: 5).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--delay", type=float, default=1.0, help="Giây nghỉ giữa các request.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit phải >= 0")
    if args.delay < 0:
        parser.error("--delay phải >= 0")
    if args.timeout <= 0:
        parser.error("--timeout phải > 0")
    if args.retries < 0:
        parser.error("--retries phải >= 0")
    return args


def main() -> int:
    args = parse_args()
    client = HttpClient(delay=args.delay, timeout=args.timeout, retries=args.retries)
    assert_allowed(client)
    urls = collect_urls(client, args.limit)
    if not urls:
        raise CrawlError("Danh sách tìm kiếm không trả về bài thơ nào")

    used_stems: set[str] = set()
    written: list[Path] = []
    skipped: list[dict[str, str]] = []
    sentence_pairs = 0
    for index, url in enumerate(urls, start=1):
        try:
            poem = parse_poem(client.get_text(url), url)
        except CrawlError as error:
            skipped.append({"url": url, "error": str(error)})
            print(f"[{index}/{len(urls)}] BỎ QUA: {error}", flush=True)
            continue
        stem = filename_stem(poem.title_vi)
        if stem in used_stems:
            stem = f"{stem}-{poem.uid}"
        used_stems.add(stem)
        path = write_poem(poem, args.output_dir, stem, args.overwrite)
        written.append(path)
        sentence_pairs += len(poem.lines_vi)
        print(
            f"[{index}/{len(urls)}] {poem.title_vi}: "
            f"{len(poem.lines_vi)} cặp câu -> {stem}.csv",
            flush=True,
        )

    report = {
        "schema_version": "thivien-kinh-thi-crawl-v1",
        "source": GROUP_URL,
        "discovered_poems": len(urls),
        "written_csv_files": len(written),
        "sentence_pairs": sentence_pairs,
        "skipped_poems": len(skipped),
        "skipped": skipped,
    }
    atomic_write(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Đã ghi {len(written)} file CSV / {sentence_pairs} cặp câu vào "
        f"{args.output_dir}; bỏ qua {len(skipped)} bài. Báo cáo: {args.report}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
