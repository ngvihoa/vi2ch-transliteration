#!/usr/bin/env python3
"""Decode Hanzi candidate lattices with Vietnamese poetic rhyme constraints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from map_ipa_to_pinyin import (
    DEFAULT_DEPS,
    DEFAULT_LOCK,
    activate_and_verify_dependencies,
    clean_ipa,
    sequence_distance,
    tokenize_ipa,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PIPELINE_ROOT.parent.parent
DEFAULT_CANDIDATES = PIPELINE_ROOT / "outputs" / "hanzi_candidates.jsonl"
DEFAULT_LINES = PIPELINE_ROOT / "outputs" / "poem_pinyin.jsonl"
DEFAULT_OUTPUT = PIPELINE_ROOT / "outputs" / "poem_hanzi_decoded.jsonl"
DEFAULT_REPORT = PIPELINE_ROOT / "outputs" / "decoder_report.json"
DEFAULT_RHYME_WEIGHT = 0.12
Node = tuple[int, int]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _node(line_index: int, syllable_index: int, lines: list[dict[str, object]]) -> Node | None:
    syllables = lines[line_index]["pinyin_syllables"]
    if not syllables:
        return None
    resolved = syllable_index if syllable_index >= 0 else len(syllables) + syllable_index
    if not 0 <= resolved < len(syllables):
        return None
    return line_index, resolved


def build_rhyme_edges(lines: list[dict[str, object]]) -> list[tuple[Node, Node]]:
    """Build forward rhyme links from adjacent lines in the same work."""
    edges: set[tuple[Node, Node]] = set()
    for index in range(len(lines) - 1):
        current, following = lines[index], lines[index + 1]
        if current["work"] != following["work"]:
            continue
        left_role, right_role = current["line_role"], following["line_role"]
        form = current["form"]
        positions: tuple[int, int] | None = None

        if form in {"luc_bat", "luc_bat_mixed"}:
            if left_role == "luc" and right_role == "bat":
                positions = (-1, 5)
            elif left_role == "bat" and right_role == "luc":
                positions = (-1, -1)

        elif form == "song_that_luc_bat":
            if left_role == "that" and right_role == "that":
                positions = (-1, 4)
            elif left_role == "that" and right_role == "luc":
                positions = (-1, -1)
            elif left_role == "luc" and right_role == "bat":
                positions = (-1, 5)
            elif left_role == "bat" and right_role == "that":
                positions = (-1, 4)

        if positions is None:
            continue
        left = _node(index, positions[0], lines)
        right = _node(index + 1, positions[1], lines)
        if left is not None and right is not None and left != right:
            edges.add((left, right))
    return sorted(edges)


def graph_components(edges: list[tuple[Node, Node]]) -> list[list[Node]]:
    adjacency: dict[Node, set[Node]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(adjacency)
    components = []
    while unseen:
        start = min(unseen)
        queue = deque([start])
        component = []
        unseen.remove(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def order_path(component: list[Node], edges: list[tuple[Node, Node]]) -> list[Node]:
    members = set(component)
    adjacency: dict[Node, list[Node]] = defaultdict(list)
    for left, right in edges:
        if left in members and right in members:
            adjacency[left].append(right)
            adjacency[right].append(left)
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        raise ValueError("Rhyme graph contains a branch; expected path components")
    endpoints = sorted(node for node in component if len(adjacency[node]) == 1)
    if not endpoints:
        raise ValueError("Rhyme graph contains a cycle; expected path components")
    ordered = []
    previous = None
    current = endpoints[0]
    while True:
        ordered.append(current)
        following = [node for node in adjacency[current] if node != previous]
        if not following:
            break
        previous, current = current, following[0]
    return ordered


def optimize_path(
    path: list[Node],
    options: dict[Node, list[dict[str, object]]],
    rhyme_weight: float,
    rhyme_distance: Callable[[dict[str, object], dict[str, object]], float],
) -> dict[Node, int]:
    """Viterbi decode one rhyme-chain path."""
    costs = [float(option["selection_score"]) for option in options[path[0]]]
    backpointers: list[list[int]] = []
    for previous_node, node in zip(path, path[1:]):
        next_costs = []
        pointers = []
        for current_option in options[node]:
            choices = [
                previous_cost
                + float(current_option["selection_score"])
                + rhyme_weight * rhyme_distance(previous_option, current_option)
                for previous_cost, previous_option in zip(costs, options[previous_node])
            ]
            best_previous = min(range(len(choices)), key=lambda index: choices[index])
            next_costs.append(choices[best_previous])
            pointers.append(best_previous)
        costs = next_costs
        backpointers.append(pointers)

    selected = [min(range(len(costs)), key=lambda index: costs[index])]
    for pointers in reversed(backpointers):
        selected.append(pointers[selected[-1]])
    selected.reverse()
    return dict(zip(path, selected))


def build_option_lookup(
    lines: list[dict[str, object]], candidate_rows: list[dict[str, object]]
) -> dict[Node, list[dict[str, object]]]:
    candidates = {row["candidate_set_id"]: row["candidates"] for row in candidate_rows}
    options = {}
    for line_index, line in enumerate(lines):
        for syllable_index, syllable in enumerate(line["pinyin_syllables"]):
            set_id = syllable["candidate_set_id"]
            if set_id not in candidates or not candidates[set_id]:
                raise ValueError(f"Missing Hanzi candidates for {set_id}")
            options[(line_index, syllable_index)] = candidates[set_id]
    return options


@lru_cache(maxsize=None)
def pinyin_rhyme_tokens(pinyin: str, matched_ipa: str) -> tuple[str, ...]:
    from pinyin_to_ipa import pinyin_to_ipa
    from pypinyin.contrib.tone_convert import to_initials

    base = pinyin[:-1]
    has_initial = bool(to_initials(base, strict=True))
    variants = list(pinyin_to_ipa(pinyin))
    chosen = variants[0]
    for variant in variants:
        if "".join(clean_ipa(part) for part in variant) == matched_ipa:
            chosen = variant
            break
    parts = [clean_ipa(part) for part in chosen]
    if has_initial:
        parts = parts[1:]
    return tuple(phone for part in parts for phone in tokenize_ipa(part))


def candidate_rhyme_distance(
    left: dict[str, object], right: dict[str, object]
) -> float:
    return sequence_distance(
        pinyin_rhyme_tokens(left["pinyin"], left["ipa"]),
        pinyin_rhyme_tokens(right["pinyin"], right["ipa"]),
    )


def decode(
    lines: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    rhyme_weight: float,
    distance_function: Callable[[dict[str, object], dict[str, object]], float] = candidate_rhyme_distance,
) -> tuple[dict[Node, int], list[tuple[Node, Node]]]:
    options = build_option_lookup(lines, candidate_rows)
    selections = {node: 0 for node in options}
    edges = build_rhyme_edges(lines)
    for component in graph_components(edges):
        path = order_path(component, edges)
        selections.update(optimize_path(path, options, rhyme_weight, distance_function))
    return selections, edges


def build_decoded_lines(
    lines: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    selections: dict[Node, int],
) -> list[dict[str, object]]:
    options = build_option_lookup(lines, candidate_rows)
    output = []
    for line_index, line in enumerate(lines):
        decoded = []
        for syllable_index, source in enumerate(line["pinyin_syllables"]):
            node = (line_index, syllable_index)
            choice_index = selections[node]
            choice = options[node][choice_index]
            decoded.append(
                {
                    "text": source["text"],
                    "source_ipa": source["source_ipa"],
                    "candidate_set_id": source["candidate_set_id"],
                    "char": choice["char"],
                    "pinyin": choice["pinyin"],
                    "selection_score": choice["selection_score"],
                    "provenance": choice["provenance"],
                    "requires_review": choice["requires_review"],
                    "candidate_rank": choice_index + 1,
                    "changed_from_greedy": choice_index != 0,
                }
            )
        output.append(
            {
                "schema_version": "poem-hanzi-decoded-v1",
                "label_quality": "synthetic_silver",
                "decoder": "rhyme-graph-viterbi-v1",
                "line_id": line["line_id"],
                "work": line["work"],
                "form": line["form"],
                "line_role": line["line_role"],
                "vi": line["text"],
                "hanzi": "".join(item["char"] for item in decoded),
                "pinyin": " ".join(item["pinyin"] for item in decoded),
                "requires_review": any(item["requires_review"] for item in decoded),
                "changed_from_greedy": any(item["changed_from_greedy"] for item in decoded),
                "syllables": decoded,
            }
        )
    return output


def edge_average(
    edges: list[tuple[Node, Node]],
    options: dict[Node, list[dict[str, object]]],
    selections: dict[Node, int],
    distance_function: Callable[[dict[str, object], dict[str, object]], float],
) -> float:
    if not edges:
        return 0.0
    return sum(
        distance_function(options[left][selections[left]], options[right][selections[right]])
        for left, right in edges
    ) / len(edges)


def build_report(
    lines: list[dict[str, object]],
    decoded_lines: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    selections: dict[Node, int],
    edges: list[tuple[Node, Node]],
    rhyme_weight: float,
) -> dict[str, object]:
    options = build_option_lookup(lines, candidate_rows)
    greedy = {node: 0 for node in options}
    total_syllables = len(options)
    changed = sum(index != 0 for index in selections.values())
    selected_options = [options[node][index] for node, index in selections.items()]
    greedy_local = sum(float(node_options[0]["selection_score"]) for node_options in options.values())
    decoded_local = sum(
        float(options[node][index]["selection_score"]) for node, index in selections.items()
    )
    greedy_rhyme = edge_average(edges, options, greedy, candidate_rhyme_distance)
    decoded_rhyme = edge_average(edges, options, selections, candidate_rhyme_distance)
    return {
        "schema_version": "decoder-report-v1",
        "label_quality": "synthetic_silver",
        "decoder": "rhyme-graph-viterbi-v1",
        "line_count": len(lines),
        "syllable_count": total_syllables,
        "rhyme_edge_count": len(edges),
        "rhyme_component_count": len(graph_components(edges)),
        "rhyme_weight": rhyme_weight,
        "greedy_average_local_selection_score": round(greedy_local / total_syllables, 6),
        "decoded_average_local_selection_score": round(decoded_local / total_syllables, 6),
        "greedy_average_rhyme_distance": round(greedy_rhyme, 6),
        "decoded_average_rhyme_distance": round(decoded_rhyme, 6),
        "greedy_total_objective": round(greedy_local + rhyme_weight * greedy_rhyme * len(edges), 6),
        "decoded_total_objective": round(decoded_local + rhyme_weight * decoded_rhyme * len(edges), 6),
        "changed_syllables": changed,
        "changed_syllable_rate": round(changed / total_syllables, 6),
        "changed_lines": sum(line["changed_from_greedy"] for line in decoded_lines),
        "lines_requiring_review": sum(line["requires_review"] for line in decoded_lines),
        "reference_selected_syllables": sum(
            option["provenance"] == "xinhua_english_reference" for option in selected_options
        ),
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


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    _atomic_write_text(path, "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode Hanzi with Vietnamese poetic rhyme constraints.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--lines", type=Path, default=DEFAULT_LINES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--deps", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--rhyme-weight", type=float, default=DEFAULT_RHYME_WEIGHT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rhyme_weight < 0:
        raise ValueError("--rhyme-weight must be non-negative")
    activate_and_verify_dependencies(args.deps, args.lock)
    candidate_rows = read_jsonl(args.candidates)
    lines = read_jsonl(args.lines)
    selections, edges = decode(lines, candidate_rows, args.rhyme_weight)
    decoded_lines = build_decoded_lines(lines, candidate_rows, selections)
    report = build_report(
        lines, decoded_lines, candidate_rows, selections, edges, args.rhyme_weight
    )
    _write_jsonl(args.output, decoded_lines)
    _atomic_write_text(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Decoded {report['line_count']} lines across {report['rhyme_edge_count']} rhyme links; "
        f"changed {report['changed_syllables']} syllables; average rhyme distance "
        f"{report['greedy_average_rhyme_distance']:.3f} -> "
        f"{report['decoded_average_rhyme_distance']:.3f}."
    )
    print(args.output)
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
