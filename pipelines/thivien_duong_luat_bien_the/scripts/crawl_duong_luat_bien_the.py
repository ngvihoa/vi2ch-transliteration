#!/usr/bin/env python3
# Chạy từ root vi2ch-model:
# python3 pipelines/thivien_duong_luat_bien_the/scripts/crawl_duong_luat_bien_the.py --limit 0
"""Crawl Đường luật biến thể by configuring the shared Thi Viện crawler."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
BASE_SCRIPT = PROJECT_ROOT / "pipelines" / "thivien_phu" / "scripts" / "crawl_phu.py"


def load_crawler():
    spec = importlib.util.spec_from_file_location("_thivien_shared_crawler", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp crawler dùng chung: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    crawler = load_crawler()
    crawler.SEARCH_URL = "https://www.thivien.net/search-poem.php?PoemType=23&ViewType=2"
    crawler.GENRE_LABEL = "Đường luật biến thể"
    crawler.REPORT_SCHEMA = "thivien-duong-luat-bien-the-crawl-v1"
    crawler.DEFAULT_COUNTRY_IDS = ("2", "3")
    crawler.DEFAULT_OUTPUT_DIR = (
        PROJECT_ROOT / "raw-collections" / "poetry-collecions" / "duong-luat-bien-the"
    )
    crawler.DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "crawl_report.json"
    crawler.DEFAULT_URL_CHECKPOINT = PIPELINE_ROOT / "outputs" / "poem_urls.json"
    return crawler.main()


if __name__ == "__main__":
    raise SystemExit(main())
