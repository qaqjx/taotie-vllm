#!/usr/bin/env python3
"""Aggregate per-method throughput CSVs into combined tables."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-method throughput CSVs into combined CSV/Markdown tables."
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Result mapping in METHOD=path/to/csv format. Repeat for each method.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write aggregated outputs into.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="throughput_summary_combined",
        help="Prefix for generated aggregated output files.",
    )
    return parser.parse_args()


def parse_result_mappings(items: List[str]) -> List[Tuple[str, Path]]:
    mappings: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --result value: {item!r}. Expected METHOD=path.")
        method, path_str = item.split("=", 1)
        method = method.strip()
        path = Path(path_str).expanduser()
        if not method:
            raise ValueError(f"Invalid --result value: {item!r}. Empty method.")
        if not path.exists():
            raise FileNotFoundError(f"Result CSV not found: {path}")
        mappings.append((method, path))
    if not mappings:
        raise ValueError("At least one --result METHOD=path pair is required.")
    return mappings


def _sort_qps_key(value: str) -> float:
    return float("inf") if value == "inf" else float(value)


def load_rows(method: str, csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "method": method,
                    "target_qps": row["target_qps"],
                    "attempted_requests": int(row["attempted_requests"]),
                    "successful_requests": int(row["successful_requests"]),
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                    "achieved_throughput_rps": float(row["achieved_throughput_rps"])
                    if row["achieved_throughput_rps"]
                    else None,
                    "success_rate": float(row["success_rate"]) if row["success_rate"] else None,
                    "latency_avg_seconds": float(row["latency_avg_seconds"])
                    if row["latency_avg_seconds"]
                    else None,
                    "latency_p50_seconds": float(row["latency_p50_seconds"])
                    if row["latency_p50_seconds"]
                    else None,
                    "latency_p90_seconds": float(row["latency_p90_seconds"])
                    if row["latency_p90_seconds"]
                    else None,
                }
            )
    return rows


def write_long_csv(rows: List[dict], output_path: Path) -> None:
    fieldnames = [
        "method",
        "target_qps",
        "attempted_requests",
        "successful_requests",
        "elapsed_seconds",
        "achieved_throughput_rps",
        "success_rate",
        "latency_avg_seconds",
        "latency_p50_seconds",
        "latency_p90_seconds",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_wide_csv(rows: List[dict], methods: List[str], output_path: Path) -> None:
    by_qps: Dict[str, Dict[str, dict]] = {}
    for row in rows:
        by_qps.setdefault(row["target_qps"], {})[row["method"]] = row

    qps_values = sorted(by_qps.keys(), key=_sort_qps_key)
    fieldnames = ["target_qps"]
    for method in methods:
        fieldnames.extend(
            [
                f"{method}_throughput_rps",
                f"{method}_success_rate",
                f"{method}_latency_avg_ms",
                f"{method}_latency_p50_ms",
                f"{method}_latency_p90_ms",
            ]
        )

    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for qps in qps_values:
            output_row = {"target_qps": qps}
            for method in methods:
                row = by_qps[qps].get(method)
                if row is None:
                    continue
                output_row[f"{method}_throughput_rps"] = row["achieved_throughput_rps"]
                output_row[f"{method}_success_rate"] = row["success_rate"]
                output_row[f"{method}_latency_avg_ms"] = (
                    row["latency_avg_seconds"] * 1000 if row["latency_avg_seconds"] is not None else ""
                )
                output_row[f"{method}_latency_p50_ms"] = (
                    row["latency_p50_seconds"] * 1000 if row["latency_p50_seconds"] is not None else ""
                )
                output_row[f"{method}_latency_p90_ms"] = (
                    row["latency_p90_seconds"] * 1000 if row["latency_p90_seconds"] is not None else ""
                )
            writer.writerow(output_row)


def _format_throughput(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "N/A"


def _format_percent(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def _format_ms(seconds: float | None) -> str:
    return f"{seconds * 1000:.2f} ms" if seconds is not None else "N/A"


def build_markdown(rows: List[dict], methods: List[str]) -> str:
    by_qps: Dict[str, Dict[str, dict]] = {}
    for row in rows:
        by_qps.setdefault(row["target_qps"], {})[row["method"]] = row

    qps_values = sorted(by_qps.keys(), key=_sort_qps_key)
    lines: List[str] = []
    lines.append("# Throughput Comparison")
    lines.append("")
    lines.append(
        f"Generated at {datetime.now(timezone.utc).isoformat()} from {len(methods)} method result files."
    )
    lines.append("")

    def _build_table(title: str, formatter) -> None:
        lines.append(f"## {title}")
        lines.append("")
        header = ["Target QPS", *methods]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for qps in qps_values:
            row_values = [qps]
            for method in methods:
                row = by_qps[qps].get(method)
                row_values.append(formatter(row) if row else "N/A")
            lines.append("| " + " | ".join(row_values) + " |")
        lines.append("")

    _build_table(
        "Achieved Throughput (req/s)",
        lambda row: _format_throughput(row["achieved_throughput_rps"]),
    )
    _build_table(
        "Success Rate",
        lambda row: _format_percent(row["success_rate"]),
    )
    _build_table(
        "Average Latency",
        lambda row: _format_ms(row["latency_avg_seconds"]),
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    mappings = parse_result_mappings(args.result)
    methods = [method for method, _ in mappings]

    rows: List[dict] = []
    for method, csv_path in mappings:
        rows.extend(load_rows(method, csv_path))

    rows.sort(key=lambda row: (_sort_qps_key(row["target_qps"]), methods.index(row["method"])))

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else mappings[0][1].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{args.output_prefix}_{timestamp}.csv"
    wide_csv_path = output_dir / f"{args.output_prefix}_{timestamp}_wide.csv"
    md_path = output_dir / f"{args.output_prefix}_{timestamp}.md"

    write_long_csv(rows, csv_path)
    write_wide_csv(rows, methods, wide_csv_path)
    markdown = build_markdown(rows, methods)
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown + "\n")

    print(f"Saved combined CSV to: {csv_path}")
    print(f"Saved combined wide CSV to: {wide_csv_path}")
    print(f"Saved combined Markdown table to: {md_path}")


if __name__ == "__main__":
    main()
