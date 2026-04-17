from __future__ import annotations

import importlib

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

        def enable_remote_upload(self, *args, **kwargs):
            return None

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
                {"key": torch.ones(1), "value": torch.zeros(1)},
                {"key": torch.zeros(1), "value": torch.ones(1)},
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

    assert task_id == "task-7"
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
    assert payloads[0]["key"].item() == 1
    assert payloads[1]["value"].item() == 1


def test_adapter_expands_split_paths_and_merges_tuple_gpu_results(monkeypatch):
    class _FakeStore:
        def __init__(self, pin_memory=True):
            self.pin_memory = pin_memory

        def enable_remote_upload(self, *args, **kwargs):
            return None

    class _FakeSchedule:
        def __init__(self, config_path, workers):
            self.prefetch_calls = []

        def submit_batch_load_to_gpu(self, paths, device_id):
            self.prefetch_calls.append((tuple(paths), device_id))
            return "task-9"

        def get_batch_load_to_gpu_result(self, task_id):
            assert task_id == "task-9"
            return (
                [{}, {}],
                [
                    {
                        "key_sv_quantized": torch.ones(
                            1, 1, 1, dtype=torch.uint8
                        ),
                        "key_sv_meta": torch.ones(1, 1, 2),
                        "key_residual_sv": torch.ones(1, 1, 2),
                        "uuid": torch.ones(1, dtype=torch.uint8),
                    },
                    {
                        "u_quantized": torch.zeros(
                            2, 1, 1, dtype=torch.uint8
                        ),
                        "u_meta": torch.zeros(2, 1, 2),
                        "value_sv_quantized": torch.zeros(
                            1, 1, 1, dtype=torch.uint8
                        ),
                        "value_sv_meta": torch.zeros(1, 1, 2),
                        "value_residual_sv": torch.zeros(1, 1, 2),
                    },
                ],
            )

    class _FakeOps:
        CPUMemoryStore = _FakeStore
        S3Schedule = _FakeSchedule

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)

    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(enabled=True, config_path="/tmp/s3.ini")
    )
    task_id = adapter.prefetch_remote(["chunk-a"], device="cuda:2")
    payloads = adapter.get_prefetch_result(task_id)

    assert adapter._scheduler.prefetch_calls == [
        (("chunk-a_key_sv", "chunk-a_other"), 2)
    ]
    assert len(payloads) == 1
    assert payloads[0]["v_quantized"].shape[0] == 2
    assert payloads[0]["v_meta"].shape[0] == 2
    assert torch.equal(payloads[0]["key_residual_sv"], torch.ones(1, 1, 2))
    assert torch.equal(
        payloads[0]["value_residual_sv"], torch.zeros(1, 1, 2)
    )


def test_adapter_returns_empty_payload_when_split_half_is_missing(monkeypatch):
    class _FakeStore:
        def __init__(self, pin_memory=True):
            self.pin_memory = pin_memory

        def enable_remote_upload(self, *args, **kwargs):
            return None

    class _FakeSchedule:
        def __init__(self, config_path, workers):
            pass

        def submit_batch_load_to_gpu(self, paths, device_id):
            return "task-missing"

        def get_batch_load_to_gpu_result(self, task_id):
            assert task_id == "task-missing"
            return ([{}, {}], [{"key_sv_quantized": torch.ones(1)}, {}])

    class _FakeOps:
        CPUMemoryStore = _FakeStore
        S3Schedule = _FakeSchedule

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)

    adapter = XJProjectBlendAdapter(
        XJProjectAdapterConfig(enabled=True, config_path="/tmp/s3.ini")
    )
    task_id = adapter.prefetch_remote(["chunk-a"], device="cuda")

    assert adapter.get_prefetch_result(task_id) == [{}]
