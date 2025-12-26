from asyncio.log import logger
import os
import threading
from sympy import sequence
import torch
import torch.distributed as dist
from typing import Any, Dict, List, Optional
import flashinfer
import lmcache.c_ops as lmc_ops

from lmcache.v1.compute.blend.Cacheblend import CacheBlend
from lmcache.v1.compute.blend.kvmanager import KVCacheManager

store_kvcache_dir = "kvcache/"

def get_kvcache_filename(hash_str : str, layer_idx = -1, device = "cuda:0"):

    filename = os.path.join(store_kvcache_dir, str(hash_str) + "-layer_" + str(layer_idx) + "-device_" + str(device) + ".bin")
    return filename

class AdaptiveKVCacheAttention:
    def __init__(self, phase):
        self.phase = phase

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):

        # q, k, v: (batch_size, num_heads, len_q, dim_head)
        q = query.contiguous().squeeze(0)
        k = key.contiguous().squeeze(0)
        v = value.contiguous().squeeze(0)

        if mask is None:
            o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
        else:
            o = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)

        return o

    def decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        # q, k, v: (batch_size, num_heads, len_q, dim_head)
        q = query.contiguous().squeeze(0)
        k = key.contiguous().squeeze(0)
        v = value.contiguous().squeeze(0)
       
        q = q.squeeze(0)
        o = flashinfer.single_decode_with_kv_cache(
            q, k, v, use_tensor_cores=True
        )

        return o.unsqueeze(0)


class ContextManager:
    def __init__(
        self,
        position_embedding,
        layer_num: int,
        num_heads_kv: int = 8,
        head_dim: int = 128,
        batch_size: int = 1,
        dtype = torch.bfloat16,
        gpu_connector = None,
        device: str = "cuda",
    ):
        self.gpu_connector = gpu_connector
        self.dtype = dtype
        self.batch_size = batch_size
        self.head_dim = head_dim
        self.num_heads_kv = num_heads_kv
        self.position_embedding = position_embedding
        self.attention = AdaptiveKVCacheAttention(phase="prefill")
        self.initialized = False
        self.device = device
        self.layer_num = layer_num
        self.phase = "prefill"
        self.shape = (self.batch_size , 1 , self.num_heads_kv , self.head_dim) 
        self.kv_manager = KVCacheManager.get_instance(shape = self.shape ,layer_num = self.layer_num , device=self.device)

        self.lengths = [0 for _ in range(layer_num)]
        self.kv = [None for _ in range(layer_num)]

        self.reuse_kv_finished = [False for _ in range(layer_num)]
        self.tasks = [None for _ in range(layer_num)] # prefetch tasks

        # self.transfer_stream = [torch.cuda.Stream(device=self.device) for _ in range(2)]

    def reset(self):
        self.lengths = [0 for _ in range(self.layer_num)]
        self.kv = [None for _ in range(self.layer_num)]
        self.reuse_kv_finished = [False for _ in range(self.layer_num)]
        self.tasks = [None for _ in range(self.layer_num)] # prefetch tasks

    def _transfer_page(self, layer_idx: int , blend_meta: Dict[str, Any] ):
        kvcaches = blend_meta["kvcaches"]
        slot_mapping = blend_meta["slot_mapping"]

        if self.kv[layer_idx] is None:
            return

        kv = torch.cat(self.kv[layer_idx], dim=0).contiguous().reshape(2, -1, self.num_heads_kv * self.head_dim)
        lmc_ops.single_layer_kv_transfer(
            kv,
            kvcaches[layer_idx],
            slot_mapping,
            True,
            False,
            (kvcaches[0].shape[0] == 2)
        )
        self.kv[layer_idx] = None

    def init(self, query , key):
        logger.info(f"ContextManager: Initializing with query and key shapes. Query shape: {query.shape}, Key shape: {key.shape}")
        assert query.dim() == 4
        batch_size, _, num_heads, head_dim = query.shape

        self.num_heads = num_heads
        self.dtype = query.dtype

        self.initialized = True

    def update_kv(self, key, value, layer_idx: int):
        """
        Update key and value tensors to the context manager.
        key: (batch_size, num_heads_kv, len_k, head_dim)
        value: (batch_size, num_heads_kv, len_k, head_dim)
        """
        self.kv[layer_idx] = (key, value)

    def save_all_kv_tensors(self, text_hash:str,  scores = None):
        """
        Save the key and value tensors to the disk.
        hash_str: the hash string of the text
        """ 

        for idx , kv in enumerate(self.kv):
            self.kv_manager.store_data(get_kvcache_filename(text_hash, layer_idx=idx ,device=self.device), kv, scores , layer_idx=idx)
 
    def store_chunks_kv(self, store_text_hashs: List , store_indices: List, layer_idx: int):

        for text_hash , indices in zip(store_text_hashs , store_indices):
            key , value = self.kv[layer_idx]
            key = key[:, indices[0]:indices[1], :, :]
            value = value[:, indices[0]:indices[1], :, :]

            # slice the chunk kv cache
            self.kv_manager.offload_compress_data(
                get_kvcache_filename(text_hash , layer_idx=layer_idx , device=self.device),
                (key, value)
            )


    def prefetch_chunk_kv(self, text_hash: list , indices: list , kv_len: int , layer_idx: int):
        """
        Prefetch the kv cache from the SSD to the CPU.
        text_hash: list of hash strings
        """
        if layer_idx >= self.layer_num:
            return

        if layer_idx == 1:
            self.all_reuse_cache = [
                {
                    "key": torch.zeros((self.batch_size, kv_len, self.num_heads_kv * self.head_dim), dtype=self.dtype, device=self.device),
                    "value": torch.zeros((self.batch_size, kv_len, self.num_heads_kv * self.head_dim), dtype=self.dtype, device=self.device)
                }
                for _ in range(self.layer_num)
            ]
            self.compress_data = [None for _ in range(self.layer_num) ]
            self.indices = indices

        self.compress_data[layer_idx] = self.kv_manager.retrieve_keys(
            [
                get_kvcache_filename(text , layer_idx=layer_idx , device=self.device)
                    for text in text_hash
            ]
        )

    def get_reuse_kv(self, layer_idx: int):
        if self.reuse_kv_finished[layer_idx]:
            return

        if not isinstance(self.compress_data[layer_idx], list):
            self.compress_data[layer_idx] = self.kv_manager.retrieve_by_task_id(task_id=self.compress_data[layer_idx])

        for idx , compressed_data in enumerate(self.compress_data[layer_idx]):
            key , value = self.kv_manager.compressor.decompress(compressed_data, kv_len=self.indices[idx][1] - self.indices[idx][0])

            self.all_reuse_cache[layer_idx]["key"][: , self.indices[idx][0] : self.indices[idx][1], :].copy_(key.reshape(self.all_reuse_cache[layer_idx]["key"][: , self.indices[idx][0] : self.indices[idx][1], :].shape) , non_blocking=True)
            self.all_reuse_cache[layer_idx]["value"][: , self.indices[idx][0] : self.indices[idx][1], :].copy_(value.reshape(self.all_reuse_cache[layer_idx]["value"][: , self.indices[idx][0] : self.indices[idx][1], :].shape) , non_blocking=True)

        self.all_reuse_cache[layer_idx]["key"] = self.all_reuse_cache[layer_idx]["key"].reshape(self.batch_size, -1, self.num_heads_kv, self.head_dim)
        self.all_reuse_cache[layer_idx]["value"] = self.all_reuse_cache[layer_idx]["value"].reshape(self.batch_size, -1, self.num_heads_kv, self.head_dim)

        self.compress_data[layer_idx] = None

        self.reuse_kv_finished[layer_idx] = True

    def prefill(
        self,
        pre_rope_query: torch.Tensor,
        pre_rope_key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        blend_meta: Optional[Dict[str, Any]] = None,
    ):
        """
        pre_rope_query: (batch_size,len_q ,num_heads, head_dim)
        pre_rope_key: (batch_size, len_k, num_heads_kv, head_dim)
        value: (batch_size, len_k, num_heads_kv, head_dim)
        """

        len_k = pre_rope_key.size(1)
        if not self.initialized:
            self.init(pre_rope_query, pre_rope_key)

        positions = torch.arange(0, len_k, device=self.device) + self.lengths[layer_idx]

        post_rope_query, post_rope_key = self.position_embedding(
            pre_rope_query.contiguous(), 
            pre_rope_key.contiguous(), 
            positions.unsqueeze(0).expand(self.batch_size, -1)
        )

        # step 1 : compute the attention
        o = self.attention.prefill(post_rope_query, post_rope_key, value)
        o = o.unsqueeze(0)
        
        if blend_meta["state"] == "store":
            self.kv_manager.store_data(
                get_kvcache_filename(blend_meta["hash_text"], layer_idx=layer_idx , device=self.device),
                (pre_rope_key , value),
                layer_idx=layer_idx
            )
        # step 2 : offload the kv cache to manager
        self.update_kv(post_rope_key, value, layer_idx)

        self.lengths[layer_idx] += len_k

        self._transfer_page(layer_idx , blend_meta)
        assert o.size(1) == post_rope_query.size(1)

        return o
    
    def prefill_select_token(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        blend_meta: Dict[str, Any],
    ):
        """
        query: (batch_size,len_q ,num_heads, head_dim)
        key: (batch_size, len_k, num_heads_kv, head_dim)
        value: (batch_size, len_k, num_heads_kv, head_dim)

        result: (o , recomputed_token_idx)
        o: (batch_size, num_heads, len_q, head_dim)
        recomputed_token_idx: (len_q * recompute_ratio)

        1. get the previous kv from the CPU cache
        2. select the recomputed token index
        3. compute the attention
        """
        assert query.size(1) == blend_meta["input_len"]

        self.get_reuse_kv(layer_idx)
        blender = CacheBlend(blend_meta , device=query.device)
  
        recomputed_idx = blender.blend_forward(
            query, key, value, (
                self.all_reuse_cache[layer_idx]["key"], 
                self.all_reuse_cache[layer_idx]["value"]
            )
        )

        # print(f"Layer {layer_idx} recompute {recomputed_idx.size(0)}/{query.size(1)} tokens.")
        # step 3: compute the attention
        o = self.prefill(query, key, value, layer_idx ,blend_meta)

        o = o[:, recomputed_idx, :, :]
        return o, recomputed_idx

    
    def prefill_blend(
        self,
        pre_rope_query: torch.Tensor,
        pre_rope_key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
        blend_meta: Dict[str, Any],
    ):
        """
        query: (batch_size,len_q ,num_heads, head_dim)
        key: (batch_size, len_k, num_heads_kv, head_dim)
        value: (batch_size, len_k, num_heads_kv, head_dim)
        positions: (len_q)

        result: (out , recomputed_token_idx)
        out: (batch_size, num_heads, len_q, head_dim)
        recomputed_token_idx: (len_q * recompute_ratio)
        """

        # step1: concatenate the key and value
        kv_len = blend_meta["input_len"]

        self.get_reuse_kv(layer_idx)
        # Shard along the head dimension
        assert self.all_reuse_cache is not None, "!!! should retrieve all layers' kv in prefill_select_token"
        retrieve_layer_key = self.all_reuse_cache[layer_idx]["key"]
        retrieve_layer_value = self.all_reuse_cache[layer_idx]["value"]

        assert pre_rope_key.size(2) == self.num_heads_kv
        retrieve_layer_key[:, positions, ...] = pre_rope_key
        retrieve_layer_value[:, positions, ...] = value
    
        # rotary the query and key
        post_rope_query, _ = self.position_embedding(
            pre_rope_query.contiguous(),
            torch.zeros_like(pre_rope_query),
            positions.unsqueeze(0).expand(self.batch_size, -1)
        )
        _, post_rope_key = self.position_embedding(
            torch.zeros_like(retrieve_layer_key),
            retrieve_layer_key.contiguous(),
            torch.arange(0, kv_len, device=self.device)
            .unsqueeze(0)
            .expand(self.batch_size, -1),
        )
        
        recomputed_len = positions.size(0)
        # generate the mask
        if "mask_type" not in blend_meta["select_config"] or blend_meta["select_config"]["mask_type"] == "True":
            mask = positions.unsqueeze(1) >= torch.arange(kv_len, device=self.device)
            # True Mask: 
            #   [[1, 0, 0, 0, 0],
            #    [1, 1, 1, 0, 0],
            #    [1, 1, 1, 1, 0]]
        elif blend_meta["select_config"]["mask_type"] == "Top-Right":
            mask = torch.tril(torch.ones(positions.size(0), kv_len, device=self.device))
            # Top-Right Mask:
            #   [[1, 0, 0, 0, 0],
            #    [1, 1, 0, 0, 0],
            #    [1, 1, 1, 0, 0]]
        elif blend_meta["select_config"]["mask_type"] == "Bottom-Right":
            mask = (
                    torch.arange(kv_len, device=self.device).unsqueeze(0) 
                    <= 
                    torch.arange(kv_len - recomputed_len, kv_len, device=self.device).unsqueeze(1)
                ).float()            # Bottom-Right mask
            #   [[1, 1, 1, 0, 0],
            #    [1, 1, 1, 1, 0],
            #    [1, 1, 1, 1, 1]]

        out = self.attention.prefill(post_rope_query, post_rope_key, retrieve_layer_value , mask).unsqueeze(0)

        # For tensor parallel, we need to update KV with the full tensors for proper storage
        self.update_kv(post_rope_key, retrieve_layer_value, layer_idx)
        self.lengths[layer_idx] += kv_len

        self.all_reuse_cache[layer_idx] = None
        assert out.size(1) == positions.size(0)
        self._transfer_page(layer_idx , blend_meta)

        return out
