#!/usr/bin/env python3
"""Run a cpu-compress queue-size experiment with xj CPU compression enabled."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_DATASET = str(ROOT / "exp" / "data" / "wikimqa_s.jsonl")
DEFAULT_QUEUE_LOG = "xj_queue.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--num-requests", type=int, default=120)
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--source-limit", type=int, default=20)
    parser.add_argument("--source-offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--xj-num-workers", type=int, default=32)
    parser.add_argument("--xj-max-rss-gib", type=float, default=200.0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Defaults to timestamped results dir.",
    )
    return parser.parse_args()


def build_output_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "base_dir": base_dir,
        "queue_log": base_dir / DEFAULT_QUEUE_LOG,
        "server_log": base_dir / "server.log",
        "client_log": base_dir / "client.log",
        "client_results": base_dir / "client_results.json",
        "plot_png": base_dir / "xj_cpu_memory_backlog.png",
        "plot_csv": base_dir / "xj_cpu_memory_backlog.csv",
        "summary": base_dir / "summary.json",
        "run_env": base_dir / "run_env.txt",
    }


def make_output_dir(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        base_dir = RESULTS_DIR / f"xj_cpucompress_dedup_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return build_output_paths(base_dir)


def build_lmcache_extra_config(
    args: argparse.Namespace, paths: dict[str, Path]
) -> str:
    return json.dumps(
        {
            "xj_project": {
                "enabled": True,
                "store_enabled": True,
                "prefetch_enabled": True,
                "compress_type": "OURS",
                "s3_config": "/home/xujie/xj_project/config/s3.ini",
                "num_workers": args.xj_num_workers,
                "max_rss_gib": args.xj_max_rss_gib,
                "queue_log_path": str(paths["queue_log"]),
                "queue_log_interval": 0.2,
                "run_namespace": paths["base_dir"].name,
            }
        }
    )


def build_server_env(
    args: argparse.Namespace, paths: dict[str, Path]
) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [
        "/home/xujie/xj_project",
        env.get("PYTHONPATH", ""),
    ]
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "LMCACHE_ENABLE_PROFILING": os.environ.get(
                "LMCACHE_ENABLE_PROFILING", "1"
            ),
            "CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices),
            "CATKV_OPS_QUEUE_TRACE_STDOUT": "1",
            "LMCACHE_CHUNK_SIZE": "256",
            "LMCACHE_ENABLE_BLENDING": "True",
            "LMCACHE_BLEND_SPECIAL_STR": " # # ",
            "LMCACHE_USE_LAYERWISE": "True",
            "LMCACHE_BLEND_CHECK_LAYERS": "1",
            "LMCACHE_BLEND_RECOMPUTE_RATIOS": "0.15",
            "LMCACHE_BLEND_MIN_TOKENS": "64",
            "LMCACHE_EXTRA_CONFIG": build_lmcache_extra_config(args, paths),
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "5",
            "LMCACHE_COMPRESS_TYPE": "OURS",
            "LMCACHE_USE_XJ_PROJECT": "1",
            "LMCACHE_XJ_STORE": "1",
            "LMCACHE_XJ_PREFETCH": "1",
            "LMCACHE_XJ_S3_CONFIG": "/home/xujie/xj_project/config/s3.ini",
            "LMCACHE_XJ_NUM_WORKERS": str(args.xj_num_workers),
            "LMCACHE_XJ_MAX_RSS_GIB": str(args.xj_max_rss_gib),
            "LMCACHE_XJ_QUEUE_LOG": str(paths["queue_log"]),
            "LMCACHE_XJ_QUEUE_LOG_INTERVAL": "0.2",
            "PYTHONPATH": ":".join(part for part in pythonpath_parts if part),
        }
    )
    env["LD_LIBRARY_PATH"] = ":".join(
        part
        for part in [
            str(ROOT / ".venv" / "lib" / "python3.13" / "site-packages" / "torch" / "lib"),
            "/usr/local/cuda/lib64",
            env.get("LD_LIBRARY_PATH", ""),
        ]
        if part
    )
    return env


def wait_for_health(port: int, timeout_s: int = 300) -> None:
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}")


def start_server(args: argparse.Namespace, paths: dict[str, Path]) -> subprocess.Popen:
    env = build_server_env(args, paths)
    command = [
        str(ROOT / ".venv" / "bin" / "vllm"),
        "serve",
        args.model,
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--enforce-eager",
        "--disable-log-requests",
        "-tp",
        "1",
    ]
    server_log = paths["server_log"].open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    proc._server_log_handle = server_log  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
            proc.wait(timeout=5)
    handle = getattr(proc, "_server_log_handle", None)
    if handle is not None:
        handle.close()


def run_client(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "exp" / "request_rate" / "xj_memory_pressure.py"),
        "--dataset",
        args.dataset,
        "--model",
        args.model,
        "--api-base",
        f"http://127.0.0.1:{args.port}",
        "--num-requests",
        str(args.num_requests),
        "--request-rate",
        str(args.request_rate),
        "--source-limit",
        str(args.source_limit),
        "--source-offset",
        str(args.source_offset),
        "--repeat-dataset",
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--queue-log",
        str(paths["queue_log"]),
        "--server-log",
        str(paths["server_log"]),
        "--output",
        str(paths["client_results"]),
    ]
    with paths["client_log"].open("w", encoding="utf-8") as client_log:
        subprocess.run(
            command,
            cwd=str(ROOT),
            stdout=client_log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def plot_queue(paths: dict[str, Path]) -> None:
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "exp" / "request_rate" / "plot_xj_queue_log.py"),
        "--queue-log",
        str(paths["queue_log"]),
        "--output-png",
        str(paths["plot_png"]),
        "--output-csv",
        str(paths["plot_csv"]),
    ]
    subprocess.run(command, cwd=str(ROOT), check=True)


def write_summary(paths: dict[str, Path]) -> None:
    client = json.loads(paths["client_results"].read_text(encoding="utf-8"))
    queue_records = [
        json.loads(line)
        for line in paths["queue_log"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    peak_queue = max(queue_records, key=lambda record: record.get("remote_queue_bytes", 0))
    peak_rss = max(queue_records, key=lambda record: record.get("rss_bytes", 0))
    summary = {
        "output_dir": str(paths["base_dir"]),
        "success_count": client["success_count"],
        "num_requests": client["num_requests"],
        "max_queue_mib": peak_queue.get("remote_queue_bytes", 0) / 1024 / 1024,
        "max_pending_count": peak_queue.get("remote_pending_count", 0),
        "max_rss_gib": peak_rss.get("rss_bytes", 0) / 1024 / 1024 / 1024,
        "queue_log": str(paths["queue_log"]),
        "client_results": str(paths["client_results"]),
        "plot_png": str(paths["plot_png"]),
        "plot_csv": str(paths["plot_csv"]),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = make_output_dir(args)
    paths["run_env"].write_text(
        "\n".join(
            [
                f"model={args.model}",
                f"dataset={args.dataset}",
                f"cuda_visible_devices={args.cuda_visible_devices}",
                f"num_requests={args.num_requests}",
                f"request_rate={args.request_rate}",
                f"source_limit={args.source_limit}",
                f"source_offset={args.source_offset}",
                f"xj_num_workers={args.xj_num_workers}",
                f"xj_max_rss_gib={args.xj_max_rss_gib}",
            ]
        ),
        encoding="utf-8",
    )

    server_proc = None
    try:
        server_proc = start_server(args, paths)
        wait_for_health(args.port)
        run_client(args, paths)
        plot_queue(paths)
        write_summary(paths)
    finally:
        stop_server(server_proc)

    print(paths["base_dir"])


if __name__ == "__main__":
    main()
