from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Optional

import torch

from lmcache.v1.compute.blend.compress.abstract import CompressType


@dataclass(frozen=True)
class XJProjectAdapterConfig:
    enabled: bool
    config_path: Optional[str]
    ratio: float = 0.15
    num_workers: int = 32
    max_queue_bytes: int = 0
    dtype: torch.dtype = torch.bfloat16
    prefetch_workers: int = 16


class XJProjectBlendAdapter:
    def __init__(self, config: XJProjectAdapterConfig):
        self.config = config
        self.available = False
        self.unavailable_reason: Optional[str] = None
        self._ops = None
        self._store = None
        self._scheduler = None
        self._split_prefetch_tasks: dict[Any, int] = {}

        if not config.enabled:
            self.unavailable_reason = "disabled"
            return

        try:
            self._ops = importlib.import_module("catKV_ops")
        except ImportError as exc:
            self.unavailable_reason = f"import-error:{exc}"
            return

        self._store = self._ops.CPUMemoryStore(pin_memory=True)
        if config.config_path:
            self._store.enable_remote_upload(
                config.config_path,
                ratio=config.ratio,
                dtype=config.dtype,
                num_workers=config.num_workers,
                max_queue_bytes=config.max_queue_bytes,
            )
            self._scheduler = self._ops.S3Schedule(
                config.config_path, config.prefetch_workers
            )

        self.available = True

    def supports(self, compress_type: CompressType) -> bool:
        return self.available and compress_type == CompressType.OURS

    def _device_to_index(self, device: str | torch.device) -> int:
        if isinstance(device, torch.device):
            if device.type != "cuda":
                return -1
            return 0 if device.index is None else device.index

        if device == "cpu":
            return -1
        if device.startswith("cuda:"):
            return int(device.split(":", maxsplit=1)[1])
        if device == "cuda":
            return 0
        return -1

    def offload(self, path: str, tensors: dict[str, torch.Tensor], group_uuid: str):
        if self._store is None:
            raise RuntimeError("xj_project CPUMemoryStore is unavailable")
        return self._store.offload(path, tensors, group_uuid)

    def _split_remote_paths(self, paths: list[str]) -> list[str]:
        return [
            split_path
            for path in paths
            for split_path in (f"{path}_key_sv", f"{path}_other")
        ]

    def _prefer_gpu_results(self, result):
        if not isinstance(result, tuple) or len(result) != 2:
            return result

        cpu_results, gpu_results = result
        normalized = []
        for cpu_result, gpu_result in zip(cpu_results, gpu_results, strict=False):
            normalized.append(gpu_result if gpu_result else cpu_result)
        if len(gpu_results) > len(cpu_results):
            normalized.extend(gpu_results[len(cpu_results) :])
        elif len(cpu_results) > len(gpu_results):
            normalized.extend(cpu_results[len(gpu_results) :])
        return normalized

    def _merge_split_payloads(self, result, logical_path_count: int):
        if not isinstance(result, list):
            return result
        if len(result) == logical_path_count:
            return result

        merged_payloads = []
        for idx in range(logical_path_count):
            key_sv_payload_idx = idx * 2
            other_payload_idx = key_sv_payload_idx + 1
            if other_payload_idx >= len(result):
                merged_payloads.append({})
                continue

            key_sv_payload = result[key_sv_payload_idx]
            other_payload = result[other_payload_idx]
            merged_payloads.append(
                self._merge_split_payload(key_sv_payload, other_payload)
            )
        return merged_payloads

    def _merge_split_payload(self, key_sv_payload, other_payload):
        if not key_sv_payload or not other_payload:
            return {}

        required_key_sv_keys = {
            "key_sv_quantized",
            "key_sv_meta",
            "key_residual_sv",
        }
        required_other_keys = {
            "u_quantized",
            "u_meta",
            "value_sv_quantized",
            "value_sv_meta",
            "value_residual_sv",
        }
        if not required_key_sv_keys.issubset(key_sv_payload) or not (
            required_other_keys.issubset(other_payload)
        ):
            return {}

        return {
            "u_quantized": other_payload["u_quantized"],
            "u_meta": other_payload["u_meta"],
            "v_quantized": torch.cat(
                [
                    key_sv_payload["key_sv_quantized"],
                    other_payload["value_sv_quantized"],
                ],
                dim=0,
            ),
            "v_meta": torch.cat(
                [key_sv_payload["key_sv_meta"], other_payload["value_sv_meta"]],
                dim=0,
            ),
            "key_residual_sv": key_sv_payload["key_residual_sv"],
            "value_residual_sv": other_payload["value_residual_sv"],
        }

    def prefetch_remote(self, paths: list[str], device: str | torch.device):
        if self._scheduler is None:
            raise RuntimeError("xj_project S3Schedule is unavailable")
        split_paths = self._split_remote_paths(paths)
        task_id = self._scheduler.submit_batch_load_to_gpu(
            split_paths, self._device_to_index(device)
        )
        self._split_prefetch_tasks[task_id] = len(paths)
        return task_id

    def get_prefetch_result(self, task_id):
        if self._scheduler is None:
            raise RuntimeError("xj_project S3Schedule is unavailable")
        result = self._prefer_gpu_results(
            self._scheduler.get_batch_load_to_gpu_result(task_id)
        )
        logical_path_count = self._split_prefetch_tasks.pop(task_id, None)
        if logical_path_count is None:
            return result
        return self._merge_split_payloads(result, logical_path_count)

    def shutdown(self) -> None:
        if self._store is not None and hasattr(self._store, "wait_remote_all"):
            self._store.wait_remote_all()
