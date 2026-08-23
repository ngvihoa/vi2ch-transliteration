#!/usr/bin/env python3
"""Remove Thi Vien sitemap URLs belonging to selected authors."""

from __future__ import annotations

import argparse
import html
import os
import re
import stat
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_XML_DIR = PROJECT_ROOT / "raw-collections" / "thivien_xml"

LOC_RE = re.compile(r"<loc>([^<]+)</loc>")
POEM_SEGMENT_RE = re.compile(r"poem-[A-Za-z0-9_-]+")


def normalize_author(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Tên tác giả không được để trống")

    if "://" in value:
        path_parts = [part for part in urlparse(value).path.split("/") if part]
        if not path_parts:
            raise ValueError(f"URL tác giả không có path: {value}")
        value = path_parts[0]
    else:
        value = value.strip("/").split("/", 1)[0]

    return unquote(value)


def resolve_authors_file(path: Path) -> Path:
    """Prefer the supplied path, then look beside this script."""
    if path.is_absolute() or path.exists():
        return path
    script_relative_path = SCRIPT_DIR / path
    if script_relative_path.exists():
        return script_relative_path
    return path


def load_authors(values: list[str], files: list[Path]) -> set[str]:
    authors: set[str] = set()
    for value in values:
        authors.add(normalize_author(value))
    for supplied_path in files:
        path = resolve_authors_file(supplied_path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            try:
                authors.add(normalize_author(value))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not authors:
        raise ValueError("Cần truyền ít nhất một tác giả hoặc --authors-file")
    return authors


def author_from_url(url: str) -> str | None:
    parsed = urlparse(html.unescape(url))
    if parsed.netloc not in {"thivien.net", "www.thivien.net"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    return unquote(parts[0])


def is_poem_url(url: str) -> bool:
    parsed = urlparse(html.unescape(url))
    return any(POEM_SEGMENT_RE.fullmatch(part) for part in parsed.path.split("/"))


def should_remove(line: str, authors: set[str], scope: str) -> bool:
    match = LOC_RE.search(line)
    if match is None:
        return False
    url = match.group(1)
    if author_from_url(url) not in authors:
        return False
    return scope == "all" or is_poem_url(url)


def process_xml(path: Path, authors: set[str], scope: str, dry_run: bool) -> int:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    removed = 0
    try:
        with (
            path.open("r", encoding="utf-8", newline="") as source,
            os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output,
        ):
            for line in source:
                if should_remove(line, authors, scope):
                    removed += 1
                else:
                    output.write(line)

        if dry_run or removed == 0:
            temporary.unlink()
        else:
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Xóa URL Thi Viện của tác giả được chỉ định khỏi các sitemap XML."
    )
    parser.add_argument(
        "authors",
        nargs="*",
        help="Tên/slug/URL tác giả, ví dụ Jan-Neruda hoặc Jaroslav-Vrchlický.",
    )
    parser.add_argument(
        "--authors-file",
        type=Path,
        action="append",
        default=[],
        help="File UTF-8, mỗi dòng chứa một tên/slug/URL tác giả; có thể dùng nhiều lần.",
    )
    parser.add_argument("--xml-dir", type=Path, default=DEFAULT_XML_DIR)
    parser.add_argument(
        "--scope",
        choices=("poem", "all"),
        default="poem",
        help="poem chỉ xóa bài thơ; all xóa mọi URL dưới segment tác giả.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ đếm, không thay đổi XML.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        authors = load_authors(args.authors, args.authors_file)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"Lỗi danh sách tác giả: {error}") from error

    paths = sorted(args.xml_dir.glob("*.xml"))
    if not paths:
        raise SystemExit(f"Không tìm thấy file XML trong {args.xml_dir}")

    total = 0
    mode = "DRY-RUN" if args.dry_run else "ĐÃ XÓA"
    for path in paths:
        removed = process_xml(path, authors, args.scope, args.dry_run)
        total += removed
        print(f"[{mode}] {path.name}: {removed} URL")
    print(f"[{mode}] Tổng: {total} URL / {len(authors)} tác giả")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
