from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import threading
import time

import torch

from lmcache.v1.compute.blend.compress.abstract import CompressType
from lmcache.v1.compute.blend.context_manager import ContextManager
from lmcache.v1.compute.blend.db import DataCenter
from lmcache.v1.compute.blend.taotie_blender import build_hash_text
from lmcache.v1.compute.blend.kvmanager import KVCacheManager
from lmcache.v1.compute.models.t_llama import TaoTieLMCLlamaModel


class _FakeKVManager:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.compress_type = CompressType.NONE
        self.seen_keys = []

    def retrieve_keys(self, keys):
        self.seen_keys = list(keys)
        return "task-id"

    def retrieve_by_task_id(self, task_id):
        assert task_id == "task-id"
        return list(self.payloads)


def test_build_hash_text_adds_run_namespace():
    tokens = torch.tensor([1, 2, 3, 4])

    value = build_hash_text(tokens, world_size=1, run_namespace="run-a")

    assert value.startswith("run-a-wordl_size1")


class _FakeSplitKVManager:
    def __init__(self, task_payloads):
        self.compress_type = CompressType.OURS
        self.task_payloads = task_payloads
        self.seen_keys = []
        self.retrieve_calls = []
        self.compressor = None

    def retrieve_keys(self, keys):
        keys = list(keys)
        self.seen_keys.append(keys)
        task_id = f"task-{len(self.seen_keys)}"
        self.retrieve_calls.append((task_id, keys))
        return task_id

    def retrieve_by_task_id(self, task_id):
        return list(self.task_payloads[task_id])


class _RetrySplitKVManager:
    def __init__(self):
        self.compress_type = CompressType.OURS
        self.retrieve_keys_calls = []
        self.retrieve_by_task_calls = []
        self.materialize_calls = 0

    def _supports_xj_project_prefetch(self):
        return True

    def retrieve_keys(self, keys):
        self.retrieve_keys_calls.append(list(keys))
        return f"retry-task-{len(self.retrieve_keys_calls)}"

    def retrieve_by_task_id(self, task_id):
        self.retrieve_by_task_calls.append(task_id)
        self.materialize_calls += 1
        if self.materialize_calls == 1:
            return [
                {
                    "u_quantized": torch.ones(1),
                    "u_meta": torch.ones(1),
                    "value_sv_quantized": torch.ones(1),
                    "value_sv_meta": torch.ones(1),
                    "value_residual_sv": torch.ones(1),
                }
            ]
        return [
            {
                "key_sv_quantized": torch.ones(1) * 2,
                "key_sv_meta": torch.ones(1) * 3,
                "key_residual_sv": torch.ones(1) * 4,
                "u_quantized": torch.ones(1) * 5,
                "u_meta": torch.ones(1) * 6,
                "value_sv_quantized": torch.ones(1) * 7,
                "value_sv_meta": torch.ones(1) * 8,
                "value_residual_sv": torch.ones(1) * 9,
            }
        ]


class _AlwaysEmptyXJKVManager:
    def __init__(self):
        self.compress_type = CompressType.OURS
        self.retrieve_keys_calls = []
        self.retrieve_by_task_calls = []

    def _supports_xj_project_prefetch(self):
        return True

    def retrieve_keys(self, keys):
        self.retrieve_keys_calls.append(list(keys))
        return "empty-task"

    def retrieve_by_task_id(self, task_id):
        self.retrieve_by_task_calls.append(task_id)
        return [{}]


def _build_context_manager(payloads):
    manager = ContextManager.__new__(ContextManager)
    manager.layer_num = 3
    manager.batch_size = 1
    manager.num_heads_kv = 1
    manager.head_dim = 2
    manager.dtype = torch.float32
    manager.device = "cpu"
    manager.compress_type = "NONE"
    manager.kv_manager = _FakeKVManager(payloads)
    manager.lengths = [0, 0, 0]
    manager.kv = [None, None, None]
    manager.reuse_kv_finished = [False, False, False]
    manager.tasks = [None, None, None]
    manager.ping_pong_cache = None
    manager.compress_data = [None, None, None]
    manager.active_hash_text = []
    manager.active_indices = []
    manager.indices = []
    return manager


def test_prefetch_chunk_kv_prunes_empty_s3_results():
    kept_payload = {
        "key": torch.ones(1, 2),
        "value": torch.ones(1, 2),
    }
    manager = _build_context_manager([kept_payload, []])

    manager.prefetch_chunk_kv(
        ["chunk-a", "chunk-b"],
        [[0, 2], [2, 4]],
        kv_len=4,
        layer_idx=1,
    )

    assert manager.has_valid_reuse() is True
    assert manager.get_effective_hash_text() == ["chunk-a"]
    assert manager.get_effective_indices() == [[0, 2]]
    assert manager.compress_data[1] == [kept_payload]
    assert manager.ping_pong_cache is not None
    assert len(manager.kv_manager.seen_keys) == 2


def test_prefetch_chunk_kv_disables_reuse_when_all_results_are_empty():
    manager = _build_context_manager([[], []])

    manager.prefetch_chunk_kv(
        ["chunk-a", "chunk-b"],
        [[0, 2], [2, 4]],
        kv_len=4,
        layer_idx=1,
    )

    assert manager.has_valid_reuse() is False
    assert manager.get_effective_hash_text() == []
    assert manager.get_effective_indices() == []
    assert manager.ping_pong_cache is None


def test_prefetch_chunk_kv_loads_split_ours_payloads_from_key_sv_and_other():
    manager = _build_context_manager([])
    manager.compress_type = "OURS"
    manager.kv_manager = _FakeSplitKVManager(
        {
            "task-1": [
                {
                    "key_sv_quantized": torch.ones(1),
                    "key_sv_meta": torch.ones(1) * 2,
                    "key_residual_sv": torch.ones(1) * 3,
                    "uuid": torch.tensor([1]),
                },
                {
                    "key_sv_quantized": torch.ones(1) * 4,
                    "key_sv_meta": torch.ones(1) * 5,
                    "key_residual_sv": torch.ones(1) * 6,
                    "uuid": torch.tensor([2]),
                },
            ],
            "task-2": [
                {
                    "u_quantized": torch.ones(1) * 7,
                    "u_meta": torch.ones(1) * 8,
                    "value_sv_quantized": torch.ones(1) * 9,
                    "value_sv_meta": torch.ones(1) * 10,
                    "value_residual_sv": torch.ones(1) * 11,
                },
                {
                    "u_quantized": torch.ones(1) * 12,
                    "u_meta": torch.ones(1) * 13,
                    "value_sv_quantized": torch.ones(1) * 14,
                    "value_sv_meta": torch.ones(1) * 15,
                    "value_residual_sv": torch.ones(1) * 16,
                },
            ],
        }
    )

    manager.prefetch_chunk_kv(
        ["chunk-a", "chunk-b"],
        [[0, 2], [2, 4]],
        kv_len=4,
        layer_idx=1,
    )

    assert manager.kv_manager.seen_keys == [
        [
            "vllm/kvcache/OURS/chunk-a-layer_1-device_cpu.bin_key_sv",
            "vllm/kvcache/OURS/chunk-b-layer_1-device_cpu.bin_key_sv",
        ],
        [
            "vllm/kvcache/OURS/chunk-a-layer_1-device_cpu.bin_other",
            "vllm/kvcache/OURS/chunk-b-layer_1-device_cpu.bin_other",
        ],
    ]
    assert isinstance(manager.compress_data[1], list)
    assert manager.compress_data[1][0]["key_sv_quantized"].item() == 1
    assert manager.compress_data[1][0]["value_sv_quantized"].item() == 9
    assert manager.compress_data[1][1]["key_sv_quantized"].item() == 4
    assert manager.compress_data[1][1]["value_sv_quantized"].item() == 14


def test_materialize_ours_payload_retries_until_split_halves_are_available(monkeypatch):
    monkeypatch.setenv("LMCACHE_XJ_RETRIEVE_RETRY_TIMEOUT_S", "1")
    monkeypatch.setenv("LMCACHE_XJ_RETRIEVE_RETRY_INTERVAL_S", "0")
    manager = _build_context_manager([])
    manager.compress_type = "OURS"
    manager.kv_manager = _RetrySplitKVManager()
    manager.active_hash_text = ["chunk-a"]
    manager.active_indices = [[0, 2]]
    manager.indices = [[0, 2]]
    manager.compress_data[2] = "initial-task"

    payloads = manager._materialize_compress_data(2)

    assert payloads[0]["key_sv_quantized"].item() == 2
    assert payloads[0]["value_sv_quantized"].item() == 7
    assert manager.kv_manager.retrieve_by_task_calls == [
        "initial-task",
        "retry-task-1",
    ]
    assert manager.kv_manager.retrieve_keys_calls == [
        ["vllm/kvcache/OURS/chunk-a-layer_2-device_cpu.bin"]
    ]


def test_prefetch_chunk_kv_marks_xj_empty_payloads_as_misses_without_retry(
    monkeypatch,
):
    monkeypatch.setenv("LMCACHE_XJ_RETRIEVE_RETRY_TIMEOUT_S", "0.01")
    monkeypatch.setenv("LMCACHE_XJ_RETRIEVE_RETRY_INTERVAL_S", "0")
    manager = _build_context_manager([])
    manager.compress_type = "OURS"
    manager.kv_manager = _AlwaysEmptyXJKVManager()

    manager.prefetch_chunk_kv(
        ["chunk-a"],
        [[0, 2]],
        kv_len=2,
        layer_idx=1,
    )

    assert manager.has_valid_reuse() is False
    assert manager.missing_hash_text == ["chunk-a"]
    assert manager.missing_indices == [[0, 2]]
    assert manager.kv_manager.retrieve_by_task_calls == ["empty-task"]
    assert manager.kv_manager.retrieve_keys_calls == [
        ["vllm/kvcache/OURS/chunk-a-layer_1-device_cpu.bin"]
    ]


def test_get_reuse_kv_retries_when_xj_prefetch_returns_incomplete_list(monkeypatch):
    monkeypatch.setenv("LMCACHE_XJ_RETRIEVE_RETRY_TIMEOUT_S", "1")
    monkeypatch.setenv("LMCACHE_XJ_RETRIEVE_RETRY_INTERVAL_S", "0")
    manager = _build_context_manager([])
    manager.compress_type = "OURS"
    manager.kv_manager = _RetrySplitKVManager()
    manager.kv_manager.compressor = type(
        "Compressor",
        (),
        {
            "decompress": lambda self, payload, kv_len: (
                torch.zeros(1, 2, 1, 2),
                torch.zeros(1, 2, 1, 2),
            )
        },
    )()
    manager.active_hash_text = ["chunk-a"]
    manager.active_indices = [[0, 2]]
    manager.indices = [[0, 2]]
    manager.ping_pong_cache = [
        {"key": torch.zeros(1, 2, 2), "value": torch.zeros(1, 2, 2)}
        for _ in range(manager.layer_num)
    ]
    manager.compress_data[2] = [
        {
            "u_quantized": torch.ones(1),
            "u_meta": torch.ones(1),
            "value_sv_quantized": torch.ones(1),
            "value_sv_meta": torch.ones(1),
            "value_residual_sv": torch.ones(1),
        }
    ]

    manager.get_reuse_kv(2)

    assert manager.reuse_kv_finished[2] is True
    assert manager.kv_manager.retrieve_keys_calls == [
        ["vllm/kvcache/OURS/chunk-a-layer_2-device_cpu.bin"]
    ]


class _FakeLayerNorm:
    def __call__(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _FakeQKVProj:
    def __call__(self, hidden_states):
        seq_len = hidden_states.size(0)
        return torch.zeros(seq_len, 384), None


class _FakeOProj:
    def __call__(self, attn_output):
        return attn_output, None


class _FakeMLP:
    def __call__(self, hidden_states):
        return hidden_states


class _FakeLayer:
    def __init__(self):
        self.input_layernorm = _FakeLayerNorm()
        self.post_attention_layernorm = _FakeLayerNorm()
        self.self_attn = SimpleNamespace(
            qkv_proj=_FakeQKVProj(),
            o_proj=_FakeOProj(),
            q_size=128,
            kv_size=128,
            attn=SimpleNamespace(num_heads=1, num_kv_heads=1, head_size=128),
        )
        self.mlp = _FakeMLP()


class _FakeVllmModel:
    def __init__(self, num_layers):
        layers = [_FakeLayer() for _ in range(num_layers)]
        self.model = SimpleNamespace(
            layers=layers,
            start_layer=0,
            end_layer=num_layers,
        )

    def get_input_embeddings(self, input_ids):
        return torch.zeros(len(input_ids), 128)


def _build_taotie_model(num_layers):
    model = TaoTieLMCLlamaModel.__new__(TaoTieLMCLlamaModel)
    torch.nn.Module.__init__(model)
    model.vllm_model = _FakeVllmModel(num_layers)
    model.num_layers = num_layers
    model.vllm_attn_layers = [
        layer.self_attn.attn for layer in model.vllm_model.model.layers
    ]
    model.lmc_attn_layers = []
    model.blender = None
    return model


class _FakeRuntimeContextManager:
    def __init__(self, keep_reuse: bool):
        self.keep_reuse = keep_reuse
        self.reset_called = False
        self.prefill_calls = []
        self.prefill_select_token_calls = []
        self.prefill_blend_calls = []
        self.active_hash_text = []
        self.active_indices = []

    def reset(self):
        self.reset_called = True

    def prefetch_chunk_kv(self, hash_text, indices, input_len, layer_idx):
        if layer_idx == 1:
            if self.keep_reuse:
                self.active_hash_text = [hash_text[0]]
                self.active_indices = [indices[0]]
            else:
                self.active_hash_text = []
                self.active_indices = []

    def has_valid_reuse(self) -> bool:
        return len(self.active_indices) > 0

    def get_effective_hash_text(self):
        return list(self.active_hash_text)

    def get_effective_indices(self):
        return [list(index) for index in self.active_indices]

    def prefill(self, q, k, v, layer_idx, blend_meta):
        self.prefill_calls.append((layer_idx, blend_meta["indices"]))
        return torch.zeros(1, q.size(1), 1, 128)

    def prefill_select_token(self, q, k, v, layer_idx, blend_meta):
        self.prefill_select_token_calls.append((layer_idx, blend_meta["indices"]))
        return torch.zeros(1, 2, 1, 128), torch.tensor([0, 1])

    def prefill_blend(self, q, k, v, layer_idx, positions, blend_meta):
        self.prefill_blend_calls.append((layer_idx, blend_meta["indices"]))
        return torch.zeros(1, positions.numel(), 1, 128)


def test_compute_layer_uses_reuse_after_partial_prune(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    model = _build_taotie_model(num_layers=2)
    context_manager = _FakeRuntimeContextManager(keep_reuse=True)
    blend_meta = {
        "context_manager": context_manager,
        "gpu_connector": None,
        "hash_text": ["chunk-a", "chunk-b"],
        "flag": 1,
        "indices": [[0, 2], [2, 4]],
        "state": "retrieve",
    }

    model.compute_layer(torch.tensor([1, 2, 3, 4]), blend_meta)

    assert context_manager.reset_called is True
    assert context_manager.prefill_select_token_calls == [(1, [[0, 2]])]
    assert context_manager.prefill_blend_calls == []
    assert context_manager.prefill_calls == [(0, [[0, 2]])]


def test_compute_layer_falls_back_to_prefill_when_all_hits_are_pruned(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    model = _build_taotie_model(num_layers=2)
    context_manager = _FakeRuntimeContextManager(keep_reuse=False)
    blend_meta = {
        "context_manager": context_manager,
        "gpu_connector": None,
        "hash_text": ["chunk-a", "chunk-b"],
        "flag": 1,
        "indices": [[0, 2], [2, 4]],
        "state": "retrieve",
    }

    model.compute_layer(torch.tensor([1, 2, 3, 4]), blend_meta)

    assert context_manager.reset_called is True
    assert context_manager.prefill_select_token_calls == []
    assert context_manager.prefill_blend_calls == []
    assert context_manager.prefill_calls == [(0, []), (1, [])]


def test_compute_layer_resets_context_manager_for_store_requests(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    model = _build_taotie_model(num_layers=2)
    context_manager = _FakeRuntimeContextManager(keep_reuse=False)
    blend_meta = {
        "context_manager": context_manager,
        "gpu_connector": None,
        "hash_text": "chunk-a",
        "flag": 0,
        "indices": [[0, 4]],
        "state": "store",
    }

    model.compute_layer(torch.tensor([1, 2, 3, 4]), blend_meta)

    assert context_manager.reset_called is True
    assert context_manager.prefill_calls == [(0, [[0, 4]]), (1, [[0, 4]])]


class _FakePrefillAttention:
    def prefill(self, query, key, value, mask=None):
        return query.squeeze(0)


class _FakeStoreKVManager:
    def __init__(self):
        self.offload_calls = []
        self.store_calls = []

    def offload_layer_data(self, key, data, layer_idx=None, group_uuid=None):
        self.offload_calls.append((key, data, layer_idx, group_uuid))
        return True

    def store_data(self, key, data, score=None, layer_idx=None):
        self.store_calls.append((key, data, layer_idx))
        return True


def _build_store_context_manager(kv_manager):
    manager = ContextManager.__new__(ContextManager)
    manager.position_embedding = (
        lambda query, key, positions: (query.contiguous(), key.contiguous())
    )
    manager.attention = _FakePrefillAttention()
    manager.initialized = True
    manager.device = "cpu"
    manager.layer_num = 1
    manager.batch_size = 1
    manager.dtype = torch.float32
    manager.num_heads = 1
    manager.num_heads_kv = 1
    manager.head_dim = 2
    manager.compress_type = "NONE"
    manager.kv_manager = kv_manager
    manager.lengths = [0]
    manager.kv = [None]
    manager.reuse_kv_finished = [False]
    manager.tasks = [None]
    manager.ping_pong_cache = None
    manager.compress_data = [None]
    manager.active_hash_text = []
    manager.active_indices = []
    manager.indices = []
    manager.requested_hash_text = []
    manager.requested_indices = []
    manager.missing_hash_text = []
    manager.missing_indices = []
    manager._transfer_page = lambda layer_idx, blend_meta: None
    return manager


def test_prefill_store_path_uses_layer_offload():
    kv_manager = _FakeStoreKVManager()
    manager = _build_store_context_manager(kv_manager)
    q = torch.zeros(1, 4, 1, 2)
    k = torch.zeros(1, 4, 1, 2)
    v = torch.zeros(1, 4, 1, 2)

    manager.prefill(
        q,
        k,
        v,
        0,
        {
            "state": "store",
            "hash_text": "chunk-a",
            "kvcaches": None,
            "slot_mapping": None,
        },
    )

    assert len(kv_manager.offload_calls) == 1
    assert kv_manager.store_calls == []
    assert kv_manager.offload_calls[0][3] is None


def test_prefill_store_path_passes_group_uuid():
    kv_manager = _FakeStoreKVManager()
    manager = _build_store_context_manager(kv_manager)
    q = torch.zeros(1, 4, 1, 2)
    k = torch.zeros(1, 4, 1, 2)
    v = torch.zeros(1, 4, 1, 2)

    manager.prefill(
        q,
        k,
        v,
        0,
        {
            "state": "store",
            "hash_text": "chunk-a",
            "group_uuid": "req-42",
            "kvcaches": None,
            "slot_mapping": None,
        },
    )

    assert kv_manager.offload_calls[0][3] == "req-42"


def test_prefill_retrieve_path_offloads_missing_chunks():
    kv_manager = _FakeStoreKVManager()
    manager = _build_store_context_manager(kv_manager)
    manager.missing_hash_text = ["chunk-miss"]
    manager.missing_indices = [[1, 3]]
    q = torch.zeros(1, 4, 1, 2)
    k = torch.arange(8, dtype=torch.float32).view(1, 4, 1, 2)
    v = torch.arange(8, dtype=torch.float32).view(1, 4, 1, 2)

    manager.prefill(
        q,
        k,
        v,
        0,
        {
            "state": "retrieve",
            "hash_text": ["chunk-hit"],
            "indices": [[0, 1]],
            "kvcaches": None,
            "slot_mapping": None,
        },
    )

    assert len(kv_manager.offload_calls) == 1
    offload_key, offload_data, layer_idx, group_uuid = kv_manager.offload_calls[0]
    assert "chunk-miss" in offload_key
    assert layer_idx == 0
    assert group_uuid is None
    assert offload_data[0].shape[1] == 2
    assert offload_data[1].shape[1] == 2


def test_prefill_retrieve_missing_chunks_reuse_group_uuid():
    kv_manager = _FakeStoreKVManager()
    manager = _build_store_context_manager(kv_manager)
    manager.missing_hash_text = ["chunk-miss"]
    manager.missing_indices = [[1, 3]]
    q = torch.zeros(1, 4, 1, 2)
    k = torch.arange(8, dtype=torch.float32).view(1, 4, 1, 2)
    v = torch.arange(8, dtype=torch.float32).view(1, 4, 1, 2)

    manager.prefill(
        q,
        k,
        v,
        0,
        {
            "state": "retrieve",
            "hash_text": ["chunk-hit"],
            "indices": [[0, 1]],
            "group_uuid": "req-43",
            "kvcaches": None,
            "slot_mapping": None,
        },
    )

    assert kv_manager.offload_calls[0][3] == "req-43"


def test_prefill_blend_offloads_missing_chunks():
    kv_manager = _FakeStoreKVManager()
    manager = _build_store_context_manager(kv_manager)
    manager.missing_hash_text = ["chunk-miss"]
    manager.missing_indices = [[1, 3]]
    manager.get_reuse_kv = lambda layer_idx: None
    manager.ping_pong_cache = [
        {
            "key": torch.zeros(1, 4, 1, 2),
            "value": torch.zeros(1, 4, 1, 2),
        }
    ]
    query = torch.zeros(1, 2, 1, 2)
    key = torch.ones(1, 2, 1, 2)
    value = torch.ones(1, 2, 1, 2)

    manager.prefill_blend(
        query,
        key,
        value,
        0,
        torch.tensor([1, 2]),
        {
            "state": "retrieve",
            "input_len": 4,
            "select_config": {"mask_type": "True"},
            "kvcaches": None,
            "slot_mapping": None,
        },
    )

    assert len(kv_manager.offload_calls) == 1
    offload_key, offload_data, layer_idx, group_uuid = kv_manager.offload_calls[0]
    assert "chunk-miss" in offload_key
    assert layer_idx == 0
    assert group_uuid is None
    assert offload_data[0].shape[1] == 2
    assert offload_data[1].shape[1] == 2


def test_prefill_blend_missing_chunks_reuse_group_uuid():
    kv_manager = _FakeStoreKVManager()
    manager = _build_store_context_manager(kv_manager)
    manager.missing_hash_text = ["chunk-miss"]
    manager.missing_indices = [[1, 3]]
    manager.get_reuse_kv = lambda layer_idx: None
    manager.ping_pong_cache = [
        {
            "key": torch.zeros(1, 4, 1, 2),
            "value": torch.zeros(1, 4, 1, 2),
        }
    ]
    query = torch.zeros(1, 2, 1, 2)
    key = torch.ones(1, 2, 1, 2)
    value = torch.ones(1, 2, 1, 2)

    manager.prefill_blend(
        query,
        key,
        value,
        0,
        torch.tensor([1, 2]),
        {
            "state": "retrieve",
            "group_uuid": "req-44",
            "input_len": 4,
            "select_config": {"mask_type": "True"},
            "kvcaches": None,
            "slot_mapping": None,
        },
    )

    assert kv_manager.offload_calls[0][3] == "req-44"


def test_offload_layer_data_falls_back_to_store_data_for_cpu_tensors():
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.compress_type = CompressType.NONE
    fallback_calls = []

    def _fallback_store(key, data, score=None, layer_idx=None):
        fallback_calls.append((key, data, layer_idx))
        return True

    manager.store_data = _fallback_store

    result = manager.offload_layer_data(
        "chunk-a-layer_0-device_cpu.bin",
        (torch.zeros(1, 2, 1, 2), torch.zeros(1, 2, 1, 2)),
        layer_idx=0,
    )

    assert result is True
    assert len(fallback_calls) == 1


def test_offload_layer_data_falls_back_for_unsupported_compression(monkeypatch):
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.compress_type = CompressType.OURS
    fallback_calls = []

    class _FakeCudaTensor:
        device = "cuda:0"

    def _fallback_store(key, data, score=None, layer_idx=None):
        fallback_calls.append((key, data, layer_idx))
        return True

    manager.store_data = _fallback_store
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    result = manager.offload_layer_data(
        "chunk-a-layer_0-device_cuda:0.bin",
        (_FakeCudaTensor(), _FakeCudaTensor()),
        layer_idx=0,
    )

    assert result is True
    assert len(fallback_calls) == 1


class _FakeAsyncCudaTensor:
    def __init__(self, gate: threading.Event):
        self.device = "cuda:0"
        self._gate = gate

    def size(self, dim=None):
        shape = (1, 2, 1, 2)
        if dim is None:
            return shape
        return shape[dim]


class _FakePinnedTensor:
    def __init__(self, gate: threading.Event):
        self._gate = gate
        self.copy_calls = 0

    def copy_(self, src, non_blocking=True):
        self.copy_calls += 1
        self._gate.wait(timeout=1.0)
        return self


class _FakePinnedAllocator:
    def __init__(self, gate: threading.Event):
        self._gate = gate

    def allocate(self, size):
        return _FakePinnedTensor(self._gate)


class _FakeCudaEvent:
    pass


class _FakeCudaStream:
    def __init__(self):
        self.waited_events = []
        self.synchronize_calls = 0
        self.recorded_events = []

    def wait_event(self, event):
        self.waited_events.append(event)

    def synchronize(self):
        self.synchronize_calls += 1

    def record_event(self, event):
        self.recorded_events.append(event)


def test_offload_layer_data_returns_before_async_store_is_published(monkeypatch):
    gate = threading.Event()
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.compress_type = CompressType.NONE
    manager.device = "cuda:0"
    manager.keys_set = set()
    manager._offload_lock = threading.Lock()
    manager._offload_queue = None
    manager._offload_worker = None
    manager._pending_offload_futures = []
    manager.transfer_stream = _FakeCudaStream()
    manager.pinned_buffer_allocator = _FakePinnedAllocator(gate)

    store_calls = []

    class _FakeDB:
        def store_data(
            self, key, data, compress_flag=True, async_disk_save=False
        ):
            store_calls.append((key, data, compress_flag, async_disk_save))
            return True

    manager.db = _FakeDB()

    producer_stream = _FakeCudaStream()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: producer_stream,
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    future = manager.offload_layer_data(
        "chunk-a-layer_0-device_cuda:0.bin",
        (_FakeAsyncCudaTensor(gate), _FakeAsyncCudaTensor(gate)),
        layer_idx=0,
    )

    assert future.done() is False
    assert store_calls == []

    gate.set()
    assert future.result(timeout=1.0) is True
    assert len(store_calls) == 1
    assert store_calls[0][3] is True

    manager.shutdown_offload_worker(wait=True)


class _FakeXJAdapter:
    def __init__(self):
        self.offload_calls = []
        self.prefetch_calls = []
        self.results = {
            "task-1": [{"key": torch.ones(1), "value": torch.ones(1)}]
        }

    def supports(self, compress_type):
        return compress_type == CompressType.OURS

    def supports_store(self, compress_type):
        return self.supports(compress_type)

    def supports_prefetch(self, compress_type):
        return self.supports(compress_type)

    def offload(self, path, tensors, group_uuid):
        self.offload_calls.append((path, tensors, group_uuid))
        return "offload-submitted"

    def prefetch_remote(self, paths, device):
        self.prefetch_calls.append((tuple(paths), device))
        return "task-1"

    def get_prefetch_result(self, task_id):
        return self.results[task_id]


def test_offload_layer_data_uses_xj_adapter_for_ours():
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.compress_type = CompressType.OURS
    manager.device = "cuda:0"
    manager._xj_adapter = _FakeXJAdapter()
    manager.store_data = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("legacy path should not run")
    )

    result = manager.offload_layer_data(
        "vllm/kvcache/OURS/chunk-a-layer_0-device_cuda:0.bin",
        (torch.zeros(1, 2, 1, 2), torch.zeros(1, 2, 1, 2)),
        layer_idx=0,
        group_uuid="req-1",
    )

    assert result == "offload-submitted"
    assert manager._xj_adapter.offload_calls[0][2] == "req-1"


def test_offload_layer_data_falls_back_when_xj_store_is_disabled():
    class _PrefetchOnlyAdapter(_FakeXJAdapter):
        def supports_store(self, compress_type):
            return False

        def supports_prefetch(self, compress_type):
            return compress_type == CompressType.OURS

    manager = KVCacheManager.__new__(KVCacheManager)
    manager.compress_type = CompressType.OURS
    manager.device = "cuda:0"
    manager._xj_adapter = _PrefetchOnlyAdapter()
    fallback_calls = []

    def _fallback_store(key, data, score=None, layer_idx=None):
        fallback_calls.append((key, data, layer_idx))
        return True

    manager.store_data = _fallback_store

    result = manager.offload_layer_data(
        "vllm/kvcache/OURS/chunk-a-layer_0-device_cuda:0.bin",
        (torch.zeros(1, 2, 1, 2), torch.zeros(1, 2, 1, 2)),
        layer_idx=0,
        group_uuid="req-1",
    )

    assert result is True
    assert fallback_calls[0][2] == 0
    assert manager._xj_adapter.offload_calls == []


def test_retrieve_keys_and_result_use_xj_adapter_for_ours():
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.compress_type = CompressType.OURS
    manager.device = "cuda:0"
    manager._xj_adapter = _FakeXJAdapter()

    task_id = manager.retrieve_keys(["chunk-a", "chunk-b"])
    payloads = manager.retrieve_by_task_id(task_id)

    assert task_id == "task-1"
    assert manager._xj_adapter.prefetch_calls == [(("chunk-a", "chunk-b"), "cuda:0")]
    assert payloads[0]["key"].item() == 1


def test_retrieve_keys_still_use_xj_adapter_when_store_is_disabled():
    class _PrefetchOnlyAdapter(_FakeXJAdapter):
        def supports_store(self, compress_type):
            return False

        def supports_prefetch(self, compress_type):
            return compress_type == CompressType.OURS

    manager = KVCacheManager.__new__(KVCacheManager)
    manager.compress_type = CompressType.OURS
    manager.device = "cuda:0"
    manager._xj_adapter = _PrefetchOnlyAdapter()
    manager.db = type(
        "DB",
        (),
        {"retrieve_keys": lambda self, keys: (_ for _ in ()).throw(AssertionError())},
    )()

    task_id = manager.retrieve_keys(["chunk-a"])

    assert task_id == "task-1"
    assert manager._xj_adapter.prefetch_calls == [(("chunk-a",), "cuda:0")]


def test_offload_compress_data_uses_xj_adapter_for_ours():
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.compress_type = CompressType.OURS
    manager.device = "cuda:0"
    manager._xj_adapter = _FakeXJAdapter()
    manager.store_data = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("legacy Python compress path should not run")
    )

    result = manager.offload_compress_data(
        "vllm/kvcache/OURS/chunk-a-layer_0-device_cuda:0.bin",
        (torch.zeros(1, 2, 1, 2), torch.zeros(1, 2, 1, 2)),
        layer_idx=0,
    )

    assert result == "offload-submitted"
    assert manager._xj_adapter.offload_calls[0][0].endswith("chunk-a-layer_0-device_cuda:0.bin")


def test_datacenter_async_disk_save_tracks_pending_task_and_waits_on_read():
    data_center = DataCenter.__new__(DataCenter)
    data_center.offload_mode = DataCenter.__init__.__globals__["OffloadMode"].DISK
    data_center.cpu_buffer_pool = type(
        "Pool",
        (),
        {
            "__init__": lambda self: setattr(self, "items", {}),
            "add_data": lambda self, key, data: self.items.__setitem__(key, data)
            or True,
            "get_data": lambda self, key: self.items.get(key),
            "clean": lambda self: self.items.clear(),
        },
    )()
    waits = []
    async_saves = []

    class _FakeDisk:
        def save_data_async(self, key, data, compress_flag=False):
            async_saves.append((key, data, compress_flag))
            return 7

        def wait(self, task_id):
            waits.append(task_id)

        def load_data(self, key):
            return {"key": torch.ones(1), "value": torch.ones(1)}

    data_center.disk_io_manager = _FakeDisk()
    data_center.pending_save_tasks = {}

    payload = {"key": torch.zeros(1), "value": torch.zeros(1)}
    assert data_center.store_data("chunk-a", payload, async_disk_save=True) is True
    assert data_center.pending_save_tasks == {"chunk-a": 7}

    data_center.cpu_buffer_pool.items.pop("chunk-a")
    loaded = data_center.retrieve_data("chunk-a")

    assert waits == [7]
    assert data_center.pending_save_tasks == {}
    assert loaded["key"].item() == 1


def test_datacenter_retrieve_keys_keeps_missing_key_placeholders():
    data_center = DataCenter.__new__(DataCenter)
    data_center.offload_mode = DataCenter.__init__.__globals__["OffloadMode"].DISK
    data_center.device = "cuda:0"
    data_center.cpu_buffer_pool = type(
        "Pool",
        (),
        {
            "__init__": lambda self: setattr(self, "items", {}),
            "add_data": lambda self, key, data: self.items.__setitem__(key, data)
            or True,
            "get_data": lambda self, key: self.items.get(key),
            "clean": lambda self: self.items.clear(),
        },
    )()
    data_center.pending_save_tasks = {}
    submit_calls = []

    class _FakeDisk:
        def exists(self, key):
            return key == "chunk-a"

        def load_datas(self, keys, device):
            submit_calls.append((list(keys), device))
            return 11

        def load_task(self, task_id):
            assert task_id == 11
            return [], [{"key": torch.ones(1), "value": torch.ones(1)}]

    data_center.disk_io_manager = _FakeDisk()

    task = data_center.retrieve_keys(["chunk-a", "chunk-b"])
    cpu_results, gpu_results = data_center.retrive_by_task(task)

    assert submit_calls == [(["chunk-a"], "cuda:0")]
    assert cpu_results == [[], []]
    assert gpu_results[1] == []
    assert gpu_results[0]["key"].item() == 1


def test_retrieve_by_task_id_skips_transfer_for_missing_placeholders():
    class _NoPrefetchAdapter(_FakeXJAdapter):
        def supports_prefetch(self, compress_type):
            return False

    class _FakeCompressor:
        def __init__(self):
            self.transfer_calls = []

        def transfer(self, data):
            self.transfer_calls.append(data)
            return {"transferred": data["key"].item()}

    class _FakeDB:
        def retrive_by_task(self, task_id):
            assert task_id == "task-1"
            return [], [{"key": torch.ones(1), "value": torch.ones(1)}, []]

    manager = KVCacheManager.__new__(KVCacheManager)
    manager.compress_type = CompressType.OURS
    manager.device = "cuda:0"
    manager._xj_adapter = _NoPrefetchAdapter()
    manager.db = _FakeDB()
    manager.compressor = _FakeCompressor()

    payloads = manager.retrieve_by_task_id("task-1")

    assert payloads == [{"transferred": 1.0}, []]
    assert len(manager.compressor.transfer_calls) == 1
