#!/usr/bin/env python3
"""Plot xj_project CPU compression queue backlog from LMCACHE_XJ_QUEUE_LOG."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-log", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        raise RuntimeError(f"No queue records found in {path}")
    return records


def write_csv(records: list[dict], path: Path) -> None:
    start_ts = records[0]["ts"]
    fields = [
        "time_s",
        "event",
        "remote_queue_mib",
        "remote_pending_count",
        "rss_gib",
        "local_store_entries",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "time_s": record["ts"] - start_ts,
                    "event": record.get("event"),
                    "remote_queue_mib": (
                        record.get("remote_queue_bytes", 0) / 1024 / 1024
                    ),
                    "remote_pending_count": record.get("remote_pending_count", 0),
                    "rss_gib": (
                        record.get("rss_bytes", 0) / 1024 / 1024 / 1024
                    ),
                    "local_store_entries": record.get("local_store_entries", 0),
                    "path": record.get("path", ""),
                }
            )


def plot(records: list[dict], output_png: Path) -> None:
    if plt is None:
        plot_with_pillow(records, output_png)
        return

    start_ts = records[0]["ts"]
    times = [record["ts"] - start_ts for record in records]
    queue_mib = [record.get("remote_queue_bytes", 0) / 1024 / 1024 for record in records]
    rss_gib = [record.get("rss_bytes", 0) / 1024 / 1024 / 1024 for record in records]
    pending = [record.get("remote_pending_count", 0) for record in records]

    fig, ax_queue = plt.subplots(figsize=(11, 5))
    ax_queue.plot(times, queue_mib, color="#d62728", linewidth=2, label="CPU compression queue")
    ax_queue.set_xlabel("Time since xj_project adapter init (s)")
    ax_queue.set_ylabel("Queued raw KV bytes (MiB)", color="#d62728")
    ax_queue.tick_params(axis="y", labelcolor="#d62728")
    ax_queue.grid(True, linestyle="--", alpha=0.35)

    ax_rss = ax_queue.twinx()
    ax_rss.plot(times, rss_gib, color="#1f77b4", linewidth=1.8, label="Server RSS")
    ax_rss.scatter(times, pending, color="#2ca02c", s=10, alpha=0.35, label="Pending count")
    ax_rss.set_ylabel("RSS (GiB) / pending count", color="#1f77b4")
    ax_rss.tick_params(axis="y", labelcolor="#1f77b4")

    lines, labels = ax_queue.get_legend_handles_labels()
    lines2, labels2 = ax_rss.get_legend_handles_labels()
    ax_queue.legend(lines + lines2, labels + labels2, loc="upper left")
    ax_queue.set_title("xj_project online serving CPU memory backlog trend")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)


def _scale(values: list[float], low: int, high: int) -> list[int]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [(low + high) // 2 for _ in values]
    return [
        int(high - (value - min_value) * (high - low) / (max_value - min_value))
        for value in values
    ]


def plot_with_pillow(records: list[dict], output_png: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    start_ts = records[0]["ts"]
    times = [record["ts"] - start_ts for record in records]
    queue_mib = [record.get("remote_queue_bytes", 0) / 1024 / 1024 for record in records]
    rss_gib = [record.get("rss_bytes", 0) / 1024 / 1024 / 1024 for record in records]

    width, height = 1400, 700
    left, right, top, bottom = 95, 75, 70, 95
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    max_time = max(times) if times else 1.0
    x_points = [
        int(left + (time_value / max_time) * plot_width) if max_time else left
        for time_value in times
    ]
    queue_y = _scale(queue_mib, top, top + plot_height)
    rss_y = _scale(rss_gib, top, top + plot_height)

    axis_color = (80, 80, 80)
    grid_color = (220, 220, 220)
    queue_color = (214, 39, 40)
    rss_color = (31, 119, 180)

    draw.rectangle((left, top, left + plot_width, top + plot_height), outline=axis_color)
    for tick in range(6):
        y = top + int(plot_height * tick / 5)
        draw.line((left, y, left + plot_width, y), fill=grid_color)
        queue_value = max(queue_mib) * (5 - tick) / 5
        rss_value = min(rss_gib) + (max(rss_gib) - min(rss_gib)) * (5 - tick) / 5
        draw.text((8, y - 6), f"{queue_value:.0f} MiB", fill=queue_color, font=font)
        draw.text((left + plot_width + 8, y - 6), f"{rss_value:.1f} GiB", fill=rss_color, font=font)

    for tick in range(6):
        x = left + int(plot_width * tick / 5)
        draw.line((x, top, x, top + plot_height), fill=grid_color)
        draw.text((x - 18, top + plot_height + 12), f"{max_time * tick / 5:.0f}s", fill=axis_color, font=font)

    if len(x_points) > 1:
        draw.line(list(zip(x_points, queue_y)), fill=queue_color, width=4)
        draw.line(list(zip(x_points, rss_y)), fill=rss_color, width=3)

    title = "xj_project online serving CPU memory backlog trend"
    draw.text((left, 25), title, fill=(20, 20, 20), font=font)
    draw.line((left, height - 35, left + 40, height - 35), fill=queue_color, width=4)
    draw.text((left + 50, height - 42), "Queued raw KV bytes", fill=queue_color, font=font)
    draw.line((left + 260, height - 35, left + 300, height - 35), fill=rss_color, width=3)
    draw.text((left + 310, height - 42), "Server RSS", fill=rss_color, font=font)
    draw.text((left + 500, height - 42), f"max queue={max(queue_mib):.1f} MiB, max RSS={max(rss_gib):.2f} GiB", fill=(20, 20, 20), font=font)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.queue_log))
    write_csv(records, Path(args.output_csv))
    plot(records, Path(args.output_png))
    max_queue = max(record.get("remote_queue_bytes", 0) for record in records)
    max_pending = max(record.get("remote_pending_count", 0) for record in records)
    max_rss = max(record.get("rss_bytes", 0) for record in records)
    print(f"records={len(records)}")
    print(f"max_queue_mib={max_queue / 1024 / 1024:.2f}")
    print(f"max_pending_count={max_pending}")
    print(f"max_rss_gib={max_rss / 1024 / 1024 / 1024:.2f}")
    print(f"wrote {args.output_png}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
