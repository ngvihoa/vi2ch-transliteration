#!/usr/bin/env python3
# Chạy thử nhẹ từ root vi2ch-model:
# python3 pipelines/thivien_that_ngon_bat_cu/scripts/crawl_that_ngon_bat_cu.py --limit 100
#
# Crawl toàn bộ URL tìm được từ các cửa sổ shallow (không exhaustive):
# python3 pipelines/thivien_that_ngon_bat_cu/scripts/crawl_that_ngon_bat_cu.py --limit 0
"""Crawl a shallow test corpus of Thất ngôn bát cú poems."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
BASE_SCRIPT = PROJECT_ROOT / "pipelines" / "thivien_phu" / "scripts" / "crawl_phu.py"

AGE_COUNTS = {
    "52": 3,
    "53": 347,
    "54": 54,
    "55": 1426,
    "56": 266,
    "57": 1695,
    "2": 784,
}


def load_crawler():
    spec = importlib.util.spec_from_file_location("_thivien_shared_crawler", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp crawler dùng chung: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_crawler(crawler):
    crawler.SEARCH_URL = "https://www.thivien.net/search-poem.php?PoemType=7&ViewType=2"
    crawler.GENRE_LABEL = "Thất ngôn bát cú"
    crawler.REPORT_SCHEMA = "thivien-that-ngon-bat-cu-shallow-crawl-v1"
    crawler.DEFAULT_COUNTRY_IDS = ("2",)
    crawler.DEFAULT_SHALLOW_AGE_PARTITION_COUNTS = {"2": AGE_COUNTS}
    crawler.DEFAULT_OUTPUT_DIR = (
        PROJECT_ROOT / "raw-collections" / "poetry-collecions" / "that-ngon-bat-cu"
    )
    crawler.DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "crawl_report.json"
    crawler.DEFAULT_URL_CHECKPOINT = PIPELINE_ROOT / "outputs" / "poem_urls.json"
    return crawler


def main() -> int:
    return configure_crawler(load_crawler()).main()


if __name__ == "__main__":
    raise SystemExit(main())
