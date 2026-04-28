#!/usr/bin/env python3
"""Export queue memory observed at each request send time."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-results", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client_results = json.loads(
        Path(args.client_results).read_text(encoding="utf-8")
    )
    rows = sorted(
        client_results.get("results", []),
        key=lambda row: float(row.get("sent_at_wall", 0.0)),
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "id",
                "sent_at_wall",
                "arrival_queue_ts",
                "arrival_queue_bytes",
                "arrival_queue_mib",
                "arrival_pending_count",
                "arrival_rss_bytes",
                "ttft_s",
                "latency_s",
                "success",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.get("id"),
                    "sent_at_wall": row.get("sent_at_wall"),
                    "arrival_queue_ts": row.get("arrival_queue_ts"),
                    "arrival_queue_bytes": row.get("arrival_queue_bytes"),
                    "arrival_queue_mib": row.get("arrival_queue_mib"),
                    "arrival_pending_count": row.get("arrival_pending_count"),
                    "arrival_rss_bytes": row.get("arrival_rss_bytes"),
                    "ttft_s": row.get("ttft_s"),
                    "latency_s": row.get("latency_s"),
                    "success": row.get("success"),
                    "status": row.get("status"),
                }
            )

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
