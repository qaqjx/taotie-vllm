from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "exp"
    / "request_rate"
    / "run_cpu_compress_dedup_experiment.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_cpu_compress_dedup_experiment", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_output_paths_uses_named_artifacts(tmp_path):
    module = load_module()
    paths = module.build_output_paths(tmp_path)

    assert paths["queue_log"].name == "xj_queue.jsonl"
    assert paths["client_results"].name == "client_results.json"
    assert paths["plot_png"].name == "xj_cpu_memory_backlog.png"
    assert paths["summary"].name == "summary.json"


def test_build_lmcache_extra_config_carries_xj_queue_settings(tmp_path):
    module = load_module()
    paths = module.build_output_paths(tmp_path)
    args = SimpleNamespace(xj_num_workers=32, xj_max_rss_gib=200.0)

    config = json.loads(module.build_lmcache_extra_config(args, paths))

    xj_config = config["xj_project"]
    assert xj_config["enabled"] is True
    assert xj_config["store_enabled"] is True
    assert xj_config["prefetch_enabled"] is True
    assert xj_config["compress_type"] == "OURS"
    assert xj_config["s3_config"] == "/home/xujie/xj_project/config/s3.ini"
    assert xj_config["num_workers"] == 32
    assert xj_config["max_rss_gib"] == 200.0
    assert xj_config["queue_log_path"] == str(paths["queue_log"])
    assert xj_config["queue_log_interval"] == 0.2
    assert xj_config["run_namespace"] == tmp_path.name


def test_build_server_env_exposes_xj_project_on_pythonpath(tmp_path):
    module = load_module()
    paths = module.build_output_paths(tmp_path)
    args = SimpleNamespace(
        cuda_visible_devices="1",
        xj_num_workers=32,
        xj_max_rss_gib=200.0,
    )

    env = module.build_server_env(args, paths)

    assert "/home/xujie/xj_project" in env["PYTHONPATH"].split(":")
    assert env["LMCACHE_ENABLE_PROFILING"] == "1"
