from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
import threading
import time
from typing import Any, Optional

import torch
import catKV_ops
from lmcache.v1.compute.blend.compress.abstract import CompressType
ENABLE_PROFILING = os.environ.get("LMCACHE_ENABLE_PROFILING", "0") == "1"

def profile_log(msg: str, *args, **kwargs):
    if ENABLE_PROFILING:
        print(f"[PROFILE] {msg}", flush=True)


@dataclass(frozen=True)
class XJProjectAdapterConfig:
    enabled: bool
    config_path: Optional[str]
    store_enabled: bool = True
    prefetch_enabled: bool = True
    ratio: float = 0.15
    num_workers: int = 32
    max_queue_bytes: int = 0
    dtype: torch.dtype = torch.bfloat16
    prefetch_workers: int = 16
    queue_log_path: Optional[str] = None
    queue_log_stdout: Optional[bool] = None
    queue_log_interval: Optional[float] = None


class XJProjectBlendAdapter:
    def __init__(self, config: XJProjectAdapterConfig):
        self.config = config
        self.available = False
        self.unavailable_reason: Optional[str] = None
        self._ops = None
        self._store = None
        self._scheduler = None
        self._split_prefetch_tasks: dict[Any, int] = {}
        self._queue_log_path = (
            config.queue_log_path
            if config.queue_log_path is not None
            else os.environ.get("LMCACHE_XJ_QUEUE_LOG")
        )
        self._queue_log_stdout = (
            config.queue_log_stdout
            if config.queue_log_stdout is not None
            else os.environ.get("LMCACHE_XJ_QUEUE_LOG_STDOUT", "0") == "1"
        )
        self._queue_log_interval = (
            config.queue_log_interval
            if config.queue_log_interval is not None
            else float(os.environ.get("LMCACHE_XJ_QUEUE_LOG_INTERVAL", "0.5"))
        )
        self._queue_monitor_stop = threading.Event()
        self._queue_monitor_thread: Optional[threading.Thread] = None

        if not config.enabled or not (config.store_enabled or config.prefetch_enabled):
            self.unavailable_reason = "disabled"
            return
        
        self._ops = catKV_ops
    
        num_workers = self._int_env(
            "LMCACHE_XJ_NUM_WORKERS", config.num_workers
        )
        prefetch_workers = self._int_env(
            "LMCACHE_XJ_PREFETCH_WORKERS", config.prefetch_workers
        )
        if config.store_enabled:
            self._store = self._ops.CPUMemoryStore(pin_memory=True)

        if config.store_enabled and config.config_path:
            self._store.enable_remote_upload(
                config.config_path,
                ratio=config.ratio,
                dtype=config.dtype,
                num_workers=num_workers,
                max_queue_bytes=config.max_queue_bytes,
            )
        if config.prefetch_enabled and config.config_path:
            self._scheduler = self._ops.S3Schedule(
                config.config_path, prefetch_workers
            )

        self.available = True
        self._start_queue_monitor()
        self.key = {}
        self.cpu_transfer_gpu_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _int_env(self, name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            parsed = int(value)
        except ValueError:
            return default
        return parsed if parsed > 0 else default

    def _current_rss_bytes(self) -> Optional[int]:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as statm:
                pages = int(statm.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            return None

    def _queue_snapshot(self, event: str, path: Optional[str] = None) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "ts": time.time(),
            "event": event,
            "rss_bytes": self._current_rss_bytes(),
        }
        if path is not None:
            snapshot["path"] = path
        if self._store is None:
            return snapshot

        if hasattr(self._store, "remote_pending_count"):
            snapshot["remote_pending_count"] = self._store.remote_pending_count()
        if hasattr(self._store, "remote_current_queue_bytes"):
            snapshot["remote_queue_bytes"] = self._store.remote_current_queue_bytes()
        if hasattr(self._store, "remote_queue_stats"):
            snapshot["remote_queue_stats"] = self._store.remote_queue_stats()
        try:
            snapshot["local_store_entries"] = len(self._store)
        except Exception:
            pass
        return snapshot

    def _write_queue_log(self, event: str, path: Optional[str] = None) -> None:
        if not self._queue_log_path and not self._queue_log_stdout:
            return
        snapshot = self._queue_snapshot(event, path)
        serialized = json.dumps(snapshot, sort_keys=True)
        if self._queue_log_path:
            with open(self._queue_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(serialized + "\n")
        if self._queue_log_stdout:
            print(f"[XJ_QUEUE] {serialized}", flush=True)

    def _queue_monitor_loop(self) -> None:
        while not self._queue_monitor_stop.wait(self._queue_log_interval):
            try:
                self._write_queue_log("tick")
            except Exception:
                pass

    def _start_queue_monitor(self) -> None:
        if (
            (not self._queue_log_path and not self._queue_log_stdout)
            or self._queue_monitor_thread is not None
        ):
            return
        self._write_queue_log("init")
        self._queue_monitor_thread = threading.Thread(
            target=self._queue_monitor_loop,
            name="xj-project-queue-monitor",
            daemon=True,
        )
        self._queue_monitor_thread.start()

    def supports(self, compress_type: CompressType) -> bool:
        return self.supports_store(compress_type) or self.supports_prefetch(
            compress_type
        )

    def supports_store(self, compress_type: CompressType) -> bool:
        return (
            self.available
            and self.config.store_enabled
            and self._store is not None
            and compress_type == CompressType.OURS
        )

    def supports_prefetch(self, compress_type: CompressType) -> bool:
        return (
            self.available
            and self.config.prefetch_enabled
            and self._scheduler is not None
            and compress_type == CompressType.OURS
        )

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
        
        profile_log(f"offload: offloading path {path} with tensor keys {tensors.keys()} , shapes {[tensor.shape for tensor in tensors.values()]}")
        result = self._store.offload(path, tensors, group_uuid)
        self._write_queue_log("offload", path)
        return result

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

        merged = dict(other_payload)
        merged.update(key_sv_payload)
        return merged

    def prefetch_remote(self, paths: list[str], device: str | torch.device):
        # load from cpu memory
        cpu_data = self._store.load_batch(paths, f"cpu")
        miss_paths = []
        for path, data in zip(paths, cpu_data, strict=False):
            if data:
                profile_log(f"prefetch_remote: loaded from CPU for path {path} with data keys {data.keys() if isinstance(data, dict) else 'N/A'}")
                for key in data.keys():
                    value = data[key]
                    profile_log(f"prefetch_remote: preparing to transfer key {key} , shape {value.shape}")
                    with torch.cuda.stream(self.cpu_transfer_gpu_stream):
                        data[key] = value.to(f"cuda:{self._device_to_index(device)}")
            else:
                miss_paths.append(path)
        task_id = paths[0]
        profile_log(f"prefetch_remote: loaded from CPU for task_id {task_id}, miss_paths: {miss_paths}")
        if miss_paths != []:
            split_paths = self._split_remote_paths(miss_paths)
            task_id = self._scheduler.submit_batch_load_to_gpu(
                split_paths, self._device_to_index(device)
            )
            self._split_prefetch_tasks[task_id] = len(miss_paths)
            self.key[task_id] = cpu_data , task_id
        else:
            self.key[task_id] = cpu_data, None
        return task_id

    def get_prefetch_result(self, task_id):
        result_from_cpu, task_flag = self.key.pop(task_id)
        profile_log(f"get_prefetch_result: retrieved from CPU for task_id {task_id} with flag {task_flag}")
        if task_flag is not None:
            result = self._prefer_gpu_results(
                self._scheduler.get_batch_load_to_gpu_result(task_id)
            )
            logical_path_count = self._split_prefetch_tasks.pop(task_id, None)
            result_from_s3 = self._merge_split_payloads(result, logical_path_count)
            count = 0
            for idx,data in enumerate(result_from_cpu):
                if not data:
                    result_from_cpu[idx] = result_from_s3[count]
                    profile_log(f"get_prefetch_result: merged data for task_id {task_id} at index {idx}")
                    profile_log(f"get_prefetch_result: merged data keys {result_from_cpu[idx].keys() if isinstance(result_from_cpu[idx], dict) else 'N/A'} for task_id {task_id} at index {idx} , remote keys{result_from_s3[count].keys() if isinstance(result_from_s3[count], dict) else 'N/A'}")
                    count += 1
            self.key.pop(task_id,None)
        
        return result_from_cpu

    def shutdown(self) -> None:
        self._queue_monitor_stop.set()
        if self._queue_monitor_thread is not None:
            self._queue_monitor_thread.join(timeout=1.0)
            self._queue_monitor_thread = None
        self._write_queue_log("shutdown")
        if self._store is not None and hasattr(self._store, "wait_remote_all"):
            self._store.wait_remote_all()
