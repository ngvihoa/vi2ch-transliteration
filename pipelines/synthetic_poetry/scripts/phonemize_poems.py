#!/usr/bin/env python3
"""Phonemize normalized Vietnamese poetry with the external vPhon tool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_INPUT = PIPELINE_ROOT / "outputs" / "poem_lines.jsonl"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs" / "poem_ipa.jsonl"
DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "poem_ipa_report.json"
DEFAULT_VPHON = PROJECT_ROOT / "tools" / "vPhon" / "vPhon.py"
DEFAULT_OVERRIDES = SCRIPT_DIR / "ipa_overrides.json"
DEFAULT_LOCK = SCRIPT_DIR / "vphon.lock.json"
CHAO_PATTERN = re.compile(r"^(?P<segments>.*?)(?P<tone>[1-5g]+)$")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_no}")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_overrides(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        overrides = json.load(handle)
    if not isinstance(overrides, dict) or not all(
        isinstance(source, str) and isinstance(target, str) and source and target
        for source, target in overrides.items()
    ):
        raise ValueError(f"IPA overrides must be a non-empty string-to-string object: {path}")
    return overrides


def load_and_verify_lock(vphon_script: Path, lock_path: Path) -> dict[str, object]:
    with lock_path.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    expected_files = lock.get("files") if isinstance(lock, dict) else None
    if not isinstance(expected_files, dict):
        raise ValueError(f"Invalid vPhon lock file: {lock_path}")
    paths = {"vPhon.py": vphon_script, "rules.py": vphon_script.with_name("rules.py")}
    for name, path in paths.items():
        expected = expected_files.get(name)
        if not isinstance(expected, str):
            raise ValueError(f"Missing hash for {name} in {lock_path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"vPhon dependency mismatch for {path}: expected {expected}, got {actual}"
            )
    return lock


def backend_metadata(
    vphon_script: Path,
    dialect: str,
    overrides_path: Path,
    overrides: dict[str, str],
    lock: dict[str, object],
) -> dict[str, object]:
    rules_path = vphon_script.with_name("rules.py")
    return {
        "name": lock["name"],
        "version": lock["version"],
        "source": lock["repository"],
        "commit": lock["commit"],
        "license": lock["license"],
        "dialect": dialect,
        "representation": "phonemic",
        "tone_system": "chao",
        "predictable_glottal_onsets": "suppressed",
        "vphon_sha256": _sha256(vphon_script),
        "rules_sha256": _sha256(rules_path),
        "override_count": len(overrides),
        "overrides_sha256": _sha256(overrides_path),
        "arguments": ["--dialect", dialect, "--chao", "--nosuper", "--phonemic", "--glottal"],
    }


def run_vphon(
    syllables: Iterable[str], vphon_script: Path, dialect: str
) -> dict[str, str]:
    unique = sorted(set(syllables))
    if not unique:
        return {}
    if not vphon_script.is_file():
        raise FileNotFoundError(
            f"vPhon not found at {vphon_script}. Clone https://github.com/kirbyj/vPhon "
            "into tools/vPhon or pass --vphon-script."
        )
    if not vphon_script.with_name("rules.py").is_file():
        raise FileNotFoundError(f"vPhon rules.py not found next to {vphon_script}")

    command = [
        sys.executable,
        str(vphon_script),
        "--dialect",
        dialect,
        "--chao",
        "--nosuper",
        "--phonemic",
        "--glottal",
    ]
    completed = subprocess.run(
        command,
        input="\n".join(unique) + "\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"vPhon failed with exit code {completed.returncode}: {detail}")
    outputs = completed.stdout.splitlines()
    if len(outputs) != len(unique):
        raise RuntimeError(
            f"vPhon returned {len(outputs)} lines for {len(unique)} input syllables"
        )
    return dict(zip(unique, outputs))


def parse_transcription(text: str) -> tuple[str | None, str | None, str]:
    """Split vPhon's Chao output into segmental IPA, tone, and status."""
    if text.startswith("[") and text.endswith("]"):
        return None, None, "unrecognized"
    match = CHAO_PATTERN.fullmatch(text)
    if not match or not match.group("segments"):
        return None, None, "malformed_output"
    return match.group("segments"), match.group("tone"), "ok"


def build_ipa_rows(
    source_rows: Iterable[dict[str, object]],
    transcriptions: dict[str, str],
    backend: dict[str, object],
    pronunciation_overrides: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    pronunciation_overrides = pronunciation_overrides or {}
    output: list[dict[str, object]] = []
    for source in source_rows:
        if not source.get("include_in_benchmark", False):
            continue
        source_syllables = source.get("syllables")
        if not isinstance(source_syllables, list) or not all(
            isinstance(syllable, str) for syllable in source_syllables
        ):
            raise ValueError(f"Invalid syllables for line {source.get('line_id', '<unknown>')}")

        ipa_syllables: list[dict[str, object]] = []
        statuses: list[str] = []
        for syllable in source_syllables:
            phonemized_as = pronunciation_overrides.get(syllable, syllable)
            raw_ipa = transcriptions.get(phonemized_as)
            if raw_ipa is None:
                segments, tone, status = None, None, "missing_output"
            else:
                segments, tone, status = parse_transcription(raw_ipa)
            statuses.append(status)
            ipa_syllables.append(
                {
                    "text": syllable,
                    "phonemized_as": phonemized_as,
                    "normalization": "override" if phonemized_as != syllable else "identity",
                    "ipa": raw_ipa if status == "ok" else None,
                    "ipa_segments": segments,
                    "tone_chao": tone,
                    "status": status,
                    "backend_output": raw_ipa,
                }
            )

        line_status = "ok" if all(status == "ok" for status in statuses) else "partial"
        row = dict(source)
        row.update(
            {
                "schema_version": "poem-ipa-v1",
                "phonemization": backend,
                "ipa_syllables": ipa_syllables,
                "ipa": " ".join(
                    item["ipa"] if isinstance(item["ipa"], str) else "<?>"
                    for item in ipa_syllables
                ),
                "phonemization_status": line_status,
            }
        )
        output.append(row)
    return output


def build_report(
    source_rows: list[dict[str, object]],
    ipa_rows: list[dict[str, object]],
    backend: dict[str, object],
) -> dict[str, object]:
    syllable_statuses: Counter[str] = Counter()
    failed_tokens: Counter[str] = Counter()
    overridden_tokens: Counter[tuple[str, str]] = Counter()
    unique_tokens: set[str] = set()
    for row in ipa_rows:
        for item in row["ipa_syllables"]:
            syllable_statuses[item["status"]] += 1
            unique_tokens.add(item["text"])
            if item["normalization"] == "override":
                overridden_tokens[(item["text"], item["phonemized_as"])] += 1
            if item["status"] != "ok":
                failed_tokens[item["text"]] += 1

    return {
        "schema_version": "poem-ipa-report-v1",
        "backend": backend,
        "source_lines": len(source_rows),
        "eligible_lines": sum(bool(row.get("include_in_benchmark")) for row in source_rows),
        "output_lines": len(ipa_rows),
        "complete_lines": sum(row["phonemization_status"] == "ok" for row in ipa_rows),
        "partial_lines": sum(row["phonemization_status"] == "partial" for row in ipa_rows),
        "unique_syllables": len(unique_tokens),
        "syllable_status_counts": dict(sorted(syllable_statuses.items())),
        "failed_syllables": [
            {"text": text, "occurrences": count}
            for text, count in sorted(failed_tokens.items(), key=lambda item: (-item[1], item[0]))
        ],
        "pronunciation_overrides": [
            {"text": source, "phonemized_as": target, "occurrences": count}
            for (source, target), count in sorted(
                overridden_tokens.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Vietnamese IPA poetry ground truth with vPhon.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--vphon-script", type=Path, default=DEFAULT_VPHON)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--vphon-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--dialect",
        choices=("n", "c", "s", "o"),
        default="n",
        help="vPhon dialect: northern, central, southern, or spelling pronunciation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = read_jsonl(args.input)
    eligible = [row for row in source_rows if row.get("include_in_benchmark", False)]
    overrides = load_overrides(args.overrides)
    lock = load_and_verify_lock(args.vphon_script, args.vphon_lock)
    syllables = [
        overrides.get(syllable, syllable)
        for row in eligible
        for syllable in row.get("syllables", [])
    ]
    transcriptions = run_vphon(syllables, args.vphon_script, args.dialect)
    backend = backend_metadata(args.vphon_script, args.dialect, args.overrides, overrides, lock)
    ipa_rows = build_ipa_rows(source_rows, transcriptions, backend, overrides)
    report = build_report(source_rows, ipa_rows, backend)

    _atomic_write_text(
        args.output,
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ipa_rows) + "\n",
    )
    _atomic_write_text(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Phonemized {report['output_lines']} lines; complete {report['complete_lines']}, "
        f"partial {report['partial_lines']}; unique syllables {report['unique_syllables']}."
    )
    print(args.output)
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
