#!/usr/bin/env python3
# Chạy từ root vi2ch-model:
# python3 pipelines/thivien_that_ngon_bat_cu/scripts/merge_that_ngon_bat_cu_csvs.py
"""Merge per-poem Thất ngôn bát cú CSV files into one atomic output."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
BASE_SCRIPT = (
    PROJECT_ROOT / "pipelines" / "thivien_phu" / "scripts" / "merge_phu_csvs.py"
)


def load_merger():
    spec = importlib.util.spec_from_file_location("_thivien_shared_merger", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp merger dùng chung: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    merger = load_merger()
    merger.MERGE_LABEL = "Thất ngôn bát cú"
    merger.DEFAULT_INPUT_DIR = (
        PROJECT_ROOT / "raw-collections" / "poetry-collecions" / "that-ngon-bat-cu"
    )
    merger.DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs" / "that-ngon-bat-cu.csv"
    return merger.main()


if __name__ == "__main__":
    raise SystemExit(main())
