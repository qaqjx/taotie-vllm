from __future__ import annotations

import importlib
import json

import torch

from lmcache.v1.compute.blend.compress.abstract import CompressType
from lmcache.v1.compute.blend.xj_project_adapter import (
    XJProjectAdapterConfig,
    XJProjectBlendAdapter,
)


def test_adapter_reports_disabled_when_feature_flag_is_false():
    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(enabled=False, config_path=None)
    )

    assert adapter.available is False
    assert adapter.unavailable_reason == "disabled"
    assert adapter.supports(CompressType.OURS) is False


def test_adapter_reports_import_error_when_catkv_ops_is_missing(monkeypatch):
    def _fake_import_module(name: str):
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(enabled=True, config_path="/tmp/s3.ini")
    )

    assert adapter.available is False
    assert adapter.unavailable_reason.startswith("import-error:")
    assert adapter.supports(CompressType.OURS) is False


def test_adapter_supports_only_ours(monkeypatch):
    class _FakeOps:
        class CPUMemoryStore:
            def __init__(self, pin_memory=True):
                self.pin_memory = pin_memory

            def enable_remote_upload(self, *args, **kwargs):
                return None

        class S3Schedule:
            def __init__(self, config_path, workers):
                self.config_path = config_path
                self.workers = workers

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)

    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(
            enabled=True,
            config_path="/tmp/s3.ini",
            dtype=torch.bfloat16,
        )
    )

    assert adapter.available is True
    assert adapter.supports(CompressType.OURS) is True
    assert adapter.supports(CompressType.NONE) is False


def test_adapter_env_overrides_worker_counts(monkeypatch):
    class _FakeStore:
        def __init__(self, pin_memory=True):
            self.pin_memory = pin_memory
            self.enable_calls = []

        def enable_remote_upload(self, *args, **kwargs):
            self.enable_calls.append((args, kwargs))

    class _FakeSchedule:
        def __init__(self, config_path, workers):
            self.config_path = config_path
            self.workers = workers

    class _FakeOps:
        CPUMemoryStore = _FakeStore
        S3Schedule = _FakeSchedule

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)
    monkeypatch.setenv("LMCACHE_XJ_NUM_WORKERS", "32")
    monkeypatch.setenv("LMCACHE_XJ_PREFETCH_WORKERS", "24")

    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(enabled=True, config_path="/tmp/s3.ini")
    )

    assert adapter._store.enable_calls[0][1]["num_workers"] == 32
    assert adapter._scheduler.workers == 24


def test_adapter_can_enable_prefetch_without_store(monkeypatch):
    class _FakeStore:
        constructed = False

        def __init__(self, pin_memory=True):
            _FakeStore.constructed = True

    class _FakeSchedule:
        def __init__(self, config_path, workers):
            self.config_path = config_path
            self.workers = workers

    class _FakeOps:
        CPUMemoryStore = _FakeStore
        S3Schedule = _FakeSchedule

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)

    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(
            enabled=True,
            config_path="/tmp/s3.ini",
            store_enabled=False,
            prefetch_enabled=True,
        )
    )

    assert adapter.available is True
    assert _FakeStore.constructed is False
    assert adapter._store is None
    assert adapter._scheduler is not None
    assert adapter.supports_store(CompressType.OURS) is False
    assert adapter.supports_prefetch(CompressType.OURS) is True


def test_adapter_can_emit_queue_snapshots_to_stdout(monkeypatch, capsys):
    class _FakeStore:
        def __init__(self, pin_memory=True):
            self.pin_memory = pin_memory

        def enable_remote_upload(self, *args, **kwargs):
            return None

        def remote_pending_count(self):
            return 3

        def remote_current_queue_bytes(self):
            return 1024

        def remote_queue_stats(self):
            return {"rss_drop_count": 0}

        def __len__(self):
            return 7

    class _FakeSchedule:
        def __init__(self, config_path, workers):
            self.config_path = config_path
            self.workers = workers

    class _FakeOps:
        CPUMemoryStore = _FakeStore
        S3Schedule = _FakeSchedule

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)
    monkeypatch.setenv("LMCACHE_XJ_QUEUE_LOG_STDOUT", "1")

    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(enabled=True, config_path="/tmp/s3.ini")
    )
    adapter._write_queue_log("tick", "chunk-a")

    captured = capsys.readouterr()
    assert "XJ_QUEUE" in captured.out
    assert "\"event\": \"tick\"" in captured.out


def test_adapter_can_emit_queue_snapshots_to_path_from_config(
    monkeypatch, tmp_path
):
    class _FakeStore:
        def __init__(self, pin_memory=True):
            self.pin_memory = pin_memory

        def enable_remote_upload(self, *args, **kwargs):
            return None

        def remote_pending_count(self):
            return 3

        def remote_current_queue_bytes(self):
            return 1024

        def remote_queue_stats(self):
            return {"rss_drop_count": 0}

        def __len__(self):
            return 7

    class _FakeSchedule:
        def __init__(self, config_path, workers):
            self.config_path = config_path
            self.workers = workers

    class _FakeOps:
        CPUMemoryStore = _FakeStore
        S3Schedule = _FakeSchedule

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)
    monkeypatch.delenv("LMCACHE_XJ_QUEUE_LOG", raising=False)
    monkeypatch.delenv("LMCACHE_XJ_QUEUE_LOG_STDOUT", raising=False)
    monkeypatch.delenv("LMCACHE_XJ_QUEUE_LOG_INTERVAL", raising=False)

    queue_log = tmp_path / "xj_queue.jsonl"
    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(
            enabled=True,
            config_path="/tmp/s3.ini",
            queue_log_path=str(queue_log),
            queue_log_stdout=False,
            queue_log_interval=0.1,
        )
    )
    adapter._write_queue_log("tick", "chunk-a")

    record = json.loads(queue_log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["event"] == "tick"
    assert record["path"] == "chunk-a"
    assert record["remote_queue_bytes"] == 1024
    assert record["remote_pending_count"] == 3


def test_adapter_forwards_offload(monkeypatch):
    class _FakeStore:
        def __init__(self, pin_memory=True):
            self.pin_memory = pin_memory
            self.offload_calls = []

        def enable_remote_upload(self, *args, **kwargs):
            return None

        def offload(self, path, data, uuid):
            self.offload_calls.append((path, data, uuid))
            return "submitted"

    class _FakeSchedule:
        def __init__(self, config_path, workers):
            self.config_path = config_path
            self.workers = workers

    class _FakeOps:
        CPUMemoryStore = _FakeStore
        S3Schedule = _FakeSchedule

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)

    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(enabled=True, config_path="/tmp/s3.ini")
    )

    payload = {"key": torch.ones(1), "value": torch.zeros(1)}
    result = adapter.offload("chunk-a", payload, group_uuid="req-1")

    assert result == "submitted"
    assert adapter._store.offload_calls == [("chunk-a", payload, "req-1")]


def test_adapter_prefetch_and_result_roundtrip(monkeypatch):
    class _FakeStore:
        def __init__(self, pin_memory=True):
            self.pin_memory = pin_memory
            self.load_batch_calls = []

        def enable_remote_upload(self, *args, **kwargs):
            return None

        def load_batch(self, paths, device="cpu"):
            self.load_batch_calls.append((list(paths), device))
            return [
                {
                    "key_sv_quantized": torch.ones(1),
                    "key_sv_meta": torch.ones(1) * 2,
                    "key_residual_sv": torch.ones(1) * 3,
                    "uuid": torch.tensor([7]),
                    "u_quantized": torch.ones(1) * 4,
                    "u_meta": torch.ones(1) * 5,
                    "value_sv_quantized": torch.ones(1) * 6,
                    "value_sv_meta": torch.ones(1) * 7,
                    "value_residual_sv": torch.ones(1) * 8,
                },
                {
                    "key_sv_quantized": torch.ones(1) * 9,
                    "key_sv_meta": torch.ones(1) * 10,
                    "key_residual_sv": torch.ones(1) * 11,
                    "uuid": torch.tensor([8]),
                    "u_quantized": torch.ones(1) * 12,
                    "u_meta": torch.ones(1) * 13,
                    "value_sv_quantized": torch.ones(1) * 14,
                    "value_sv_meta": torch.ones(1) * 15,
                    "value_residual_sv": torch.ones(1) * 16,
                },
            ]

    class _FakeSchedule:
        def __init__(self, config_path, workers):
            self.config_path = config_path
            self.workers = workers
            self.prefetch_calls = []

        def submit_batch_load_to_gpu(self, paths, device_id):
            self.prefetch_calls.append((tuple(paths), device_id))
            return "task-7"

        def get_batch_load_to_gpu_result(self, task_id):
            assert task_id == "task-7"
            return [
                {
                    "key_sv_quantized": torch.ones(1),
                    "key_sv_meta": torch.ones(1) * 2,
                    "key_residual_sv": torch.ones(1) * 3,
                    "uuid": torch.tensor([7]),
                },
                {
                    "u_quantized": torch.ones(1) * 4,
                    "u_meta": torch.ones(1) * 5,
                    "value_sv_quantized": torch.ones(1) * 6,
                    "value_sv_meta": torch.ones(1) * 7,
                    "value_residual_sv": torch.ones(1) * 8,
                },
                {
                    "key_sv_quantized": torch.ones(1) * 9,
                    "key_sv_meta": torch.ones(1) * 10,
                    "key_residual_sv": torch.ones(1) * 11,
                    "uuid": torch.tensor([8]),
                },
                {
                    "u_quantized": torch.ones(1) * 12,
                    "u_meta": torch.ones(1) * 13,
                    "value_sv_quantized": torch.ones(1) * 14,
                    "value_sv_meta": torch.ones(1) * 15,
                    "value_residual_sv": torch.ones(1) * 16,
                },
            ]

    class _FakeOps:
        CPUMemoryStore = _FakeStore
        S3Schedule = _FakeSchedule

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)

    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(enabled=True, config_path="/tmp/s3.ini")
    )

    task_id = adapter.prefetch_remote(["chunk-a", "chunk-b"], device="cuda:3")
    payloads = adapter.get_prefetch_result(task_id)

    assert adapter._store.load_batch_calls == []
    assert adapter._scheduler.prefetch_calls == [
        (
            (
                "chunk-a_key_sv",
                "chunk-a_other",
                "chunk-b_key_sv",
                "chunk-b_other",
            ),
            3,
        )
    ]
    assert payloads[0]["key_sv_quantized"].item() == 1
    assert payloads[0]["value_sv_quantized"].item() == 6
    assert payloads[0]["u_quantized"].item() == 4
    assert payloads[0]["uuid"].item() == 7
    assert payloads[1]["key_sv_quantized"].item() == 9
    assert payloads[1]["value_sv_quantized"].item() == 14


def test_adapter_expands_split_paths_and_merges_tuple_gpu_results(monkeypatch):
    class _FakeStore:
        def __init__(self, pin_memory=True):
            self.pin_memory = pin_memory
            self.load_batch_calls = []

        def enable_remote_upload(self, *args, **kwargs):
            return None

        def load_batch(self, paths, device="cpu"):
            self.load_batch_calls.append((list(paths), device))
            return [
                {
                    "key_sv_quantized": torch.ones(
                        1, 1, 1, dtype=torch.uint8
                    ),
                    "key_sv_meta": torch.ones(1, 1, 2),
                    "key_residual_sv": torch.ones(1, 1, 2),
                    "uuid": torch.ones(1, dtype=torch.uint8),
                    "u_quantized": torch.zeros(
                        2, 1, 1, dtype=torch.uint8
                    ),
                    "u_meta": torch.zeros(2, 1, 2),
                    "value_sv_quantized": torch.zeros(
                        1, 1, 1, dtype=torch.uint8
                    ),
                    "value_sv_meta": torch.zeros(1, 1, 2),
                    "value_residual_sv": torch.zeros(1, 1, 2),
                }
            ]

    class _FakeSchedule:
        def __init__(self, config_path, workers):
            self.config_path = config_path
            self.workers = workers

    class _FakeOps:
        CPUMemoryStore = _FakeStore
        S3Schedule = _FakeSchedule

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)

    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(enabled=True, config_path="/tmp/s3.ini")
    )
    task_id = adapter.prefetch_remote(["chunk-a"], device="cuda:2")
    payloads = adapter.get_prefetch_result(task_id)

    assert adapter._store.load_batch_calls == [(["chunk-a"], "cuda:2")]
    assert len(payloads) == 1
    assert payloads[0]["key_sv_quantized"].shape[0] == 1
    assert payloads[0]["value_sv_quantized"].shape[0] == 1
    assert payloads[0]["u_quantized"].shape[0] == 2
    assert payloads[0]["uuid"].shape[0] == 1
    assert torch.equal(payloads[0]["key_residual_sv"], torch.ones(1, 1, 2))
    assert torch.equal(
        payloads[0]["value_residual_sv"], torch.zeros(1, 1, 2)
    )


def test_adapter_returns_empty_payload_when_split_half_is_missing(monkeypatch):
    class _FakeStore:
        def __init__(self, pin_memory=True):
            self.pin_memory = pin_memory
            self.load_batch_calls = []

        def enable_remote_upload(self, *args, **kwargs):
            return None

        def load_batch(self, paths, device="cpu"):
            self.load_batch_calls.append((list(paths), device))
            return [{}]

    class _FakeSchedule:
        def __init__(self, config_path, workers):
            self.config_path = config_path
            self.workers = workers

    class _FakeOps:
        CPUMemoryStore = _FakeStore
        S3Schedule = _FakeSchedule

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)

    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(enabled=True, config_path="/tmp/s3.ini")
    )
    task_id = adapter.prefetch_remote(["chunk-a"], device="cuda")

    assert adapter.get_prefetch_result(task_id) == [{}]
    assert adapter._store.load_batch_calls == [(["chunk-a"], "cuda:0")]
