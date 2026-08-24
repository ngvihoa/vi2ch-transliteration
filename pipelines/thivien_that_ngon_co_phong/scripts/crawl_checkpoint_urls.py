#!/usr/bin/env python3
# Chạy từ root vi2ch-model:
# python3 pipelines/thivien_that_ngon_co_phong/scripts/crawl_checkpoint_urls.py
"""Create Thất ngôn cổ phong CSV files from the existing URL checkpoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
BASE_SCRIPT = (
    PROJECT_ROOT
    / "pipelines"
    / "thivien_ngu_ngon_tu_tuyet"
    / "scripts"
    / "crawl_checkpoint_urls.py"
)


def load_checkpoint_crawler():
    spec = importlib.util.spec_from_file_location(
        "_thivien_shared_checkpoint_crawler", BASE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp checkpoint crawler: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    crawler = load_checkpoint_crawler()
    crawler.SEARCH_URL = (
        "https://www.thivien.net/search-poem.php?PoemType=8&ViewType=2"
    )
    crawler.GENRE_LABEL = "Thất ngôn cổ phong"
    crawler.REPORT_SCHEMA = "thivien-that-ngon-co-phong-crawl-v1"
    crawler.DEFAULT_URL_CHECKPOINT = PIPELINE_ROOT / "outputs" / "poem_urls.json"
    crawler.DEFAULT_PROGRESS = (
        PIPELINE_ROOT / "outputs" / "csv_crawl_progress.json"
    )
    crawler.DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "crawl_report.json"
    crawler.DEFAULT_COOKIE_FILE = PIPELINE_ROOT / "outputs" / "http_cookies.txt"
    crawler.DEFAULT_OUTPUT_DIR = (
        PROJECT_ROOT
        / "raw-collections"
        / "poetry-collecions"
        / "that-ngon-co-phong"
    )
    return crawler.main()


if __name__ == "__main__":
    raise SystemExit(main())
