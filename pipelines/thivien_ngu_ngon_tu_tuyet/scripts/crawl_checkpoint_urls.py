#!/usr/bin/env python3
"""Crawl only the Ngũ ngôn tứ tuyệt URLs already stored in a checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
SHARED_CRAWLER = (
    PROJECT_ROOT / "pipelines" / "thivien_phu" / "scripts" / "crawl_phu.py"
)

SEARCH_URL = "https://www.thivien.net/search-poem.php?PoemType=3&ViewType=2"
DEFAULT_URL_CHECKPOINT = PIPELINE_ROOT / "outputs" / "poem_urls.json"
DEFAULT_PROGRESS = PIPELINE_ROOT / "outputs" / "csv_crawl_progress.json"
DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "crawl_report.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "raw-collections"
    / "poetry-collecions"
    / "ngu-ngon-tu-tuyet"
)
PROGRESS_SCHEMA = "thivien-checkpoint-url-crawl-v1"
REPORT_SCHEMA = "thivien-ngu-ngon-tu-tuyet-crawl-v1"


def load_shared_crawler():
    spec = importlib.util.spec_from_file_location(
        "_thivien_checkpoint_shared_crawler", SHARED_CRAWLER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp crawler dùng chung: {SHARED_CRAWLER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SEARCH_URL = SEARCH_URL
    module.GENRE_LABEL = "Đường luật - ngũ ngôn tứ tuyệt"
    return module


def load_checkpoint_urls(path: Path) -> list[str]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Không đọc được checkpoint URL ({path}): {error}") from error

    if not isinstance(checkpoint, dict) or checkpoint.get("source") != SEARCH_URL:
        raise SystemExit(f"Checkpoint không thuộc thể ngũ ngôn tứ tuyệt: {path}")
    raw_urls = checkpoint.get("urls")
    if not isinstance(raw_urls, list) or not all(
        isinstance(url, str) for url in raw_urls
    ):
        raise SystemExit(f"Checkpoint không có danh sách urls hợp lệ: {path}")

    # Giữ nguyên thứ tự đã khám phá và loại URL trùng.
    return list(dict.fromkeys(raw_urls))


def load_progress(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Progress không hợp lệ ({path}): {error}") from error
    if (
        not isinstance(progress, dict)
        or progress.get("schema_version") != PROGRESS_SCHEMA
        or progress.get("source") != SEARCH_URL
        or not isinstance(progress.get("items"), dict)
    ):
        raise SystemExit(f"Progress không đúng schema/source: {path}")
    return {
        str(url): item
        for url, item in progress["items"].items()
        if isinstance(url, str) and isinstance(item, dict)
    }


def save_progress(crawler, path: Path, items: dict[str, dict[str, object]]) -> None:
    payload = {
        "schema_version": PROGRESS_SCHEMA,
        "source": SEARCH_URL,
        "processed_urls": len(items),
        "items": items,
    }
    crawler.atomic_write(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def output_from_item(output_dir: Path, item: dict[str, object]) -> Path | None:
    output = item.get("output")
    if not isinstance(output, str) or not output:
        return None
    return output_dir / output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tạo CSV từ các URL ngũ ngôn tứ tuyệt đã có trong poem_urls.json; "
            "không khám phá lại URL."
        )
    )
    parser.add_argument("--url-checkpoint", type=Path, default=DEFAULT_URL_CHECKPOINT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Giới hạn số URL để thử; 0 dùng toàn bộ URL trong checkpoint.",
    )
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--jitter", type=float, default=3.0)
    parser.add_argument("--pause-every", type=int, default=40)
    parser.add_argument("--pause-min", type=float, default=300.0)
    parser.add_argument("--pause-max", type=float, default=600.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--retry-skipped",
        action="store_true",
        help="Thử lại các URL từng bị bỏ qua vì nội dung không hợp lệ.",
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
    ):
        parser.error(
            "limit/delay/jitter/pause/retries phải hợp lệ, timeout phải > 0 "
            "và pause-max phải >= pause-min"
        )
    return args


def main() -> int:
    args = parse_args()
    crawler = load_shared_crawler()
    urls = load_checkpoint_urls(args.url_checkpoint)
    if args.limit:
        urls = urls[: args.limit]
    if not urls:
        raise SystemExit("Checkpoint không có URL nào để crawl")

    items = load_progress(args.progress)
    used_stems = {
        Path(str(item["output"])).stem
        for item in items.values()
        if isinstance(item.get("output"), str)
    }
    client = crawler.HttpClient(
        args.delay,
        args.jitter,
        args.pause_every,
        args.pause_min,
        args.pause_max,
        args.timeout,
        args.retries,
    )
    crawler.assert_allowed(client)

    for index, url in enumerate(urls, start=1):
        previous = items.get(url)
        if previous is not None:
            status = previous.get("status")
            output = output_from_item(args.output_dir, previous)
            if status in {"written", "existing"} and output is not None and output.exists():
                print(f"[{index}/{len(urls)}] ĐÃ CÓ: {output.name}", flush=True)
                continue
            if status == "skipped" and not args.retry_skipped:
                print(f"[{index}/{len(urls)}] ĐÃ BỎ QUA: {url}", flush=True)
                continue

        # Lỗi mạng/CAPTCHA phải dừng hẳn; progress của các bài trước đã được lưu.
        html = client.get_text(url)
        try:
            poem = crawler.parse_poem(html, url)
            stem = crawler.filename_stem(poem.title_vi)
            if stem in used_stems:
                stem = f"{stem}-{poem.uid}"
            used_stems.add(stem)
            output = args.output_dir / f"{stem}.csv"

            if output.exists() and not args.overwrite:
                status = "existing"
            else:
                crawler.write_poem(poem, args.output_dir, stem, args.overwrite)
                status = "written"

            items[url] = {
                "status": status,
                "output": output.name,
                "title_vi": poem.title_vi,
                "sentence_pairs": len(poem.lines_vi),
            }
            save_progress(crawler, args.progress, items)
            label = "ĐÃ CÓ" if status == "existing" else "ĐÃ GHI"
            print(
                f"[{index}/{len(urls)}] {label}: {poem.title_vi} "
                f"({len(poem.lines_vi)} cặp câu)",
                flush=True,
            )
        except crawler.CrawlError as error:
            items[url] = {"status": "skipped", "url": url, "error": str(error)}
            save_progress(crawler, args.progress, items)
            print(f"[{index}/{len(urls)}] BỎ QUA: {error}", flush=True)

    selected_items = [items[url] for url in urls if url in items]
    successful = [
        item for item in selected_items if item.get("status") in {"written", "existing"}
    ]
    skipped = [item for item in selected_items if item.get("status") == "skipped"]
    report = {
        "schema_version": REPORT_SCHEMA,
        "source": SEARCH_URL,
        "url_checkpoint": str(args.url_checkpoint),
        "discovered_poems": len(urls),
        "written_csv_files": len(successful),
        "sentence_pairs": sum(
            int(item.get("sentence_pairs", 0)) for item in successful
        ),
        "skipped_poems": len(skipped),
        "skipped": skipped,
    }
    crawler.atomic_write(
        args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"Hoàn tất {len(successful)}/{len(urls)} CSV; "
        f"bỏ qua {len(skipped)} bài. Report: {args.report}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
