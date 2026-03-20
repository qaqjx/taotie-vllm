from asyncio.log import logger
from multiprocessing.pool import ThreadPool
import os
import threading
import time
from typing import Any, Dict, List, Optional

import torch

from lmcache.v1.compute.blend.compress.abstract import CompressType

# Profiling toggle
ENABLE_PROFILING = os.environ.get("LMCACHE_ENABLE_PROFILING", "0") == "1"

def profile_log(msg: str):
    if ENABLE_PROFILING:
        print(f"[PROFILE] {msg}", flush=True)


def get_compress_type_from_env() -> CompressType:
    """
    Get compression type from environment variable LMCACHE_COMPRESS_TYPE.
    Supported values: NONE, KIVI_2BIT, OURS, SVDQ
    Default: KIVI_2BIT
    """
    env_value = os.environ.get("LMCACHE_COMPRESS_TYPE", "KIVI_2BIT").upper()
    compress_map = {
        "NONE": CompressType.NONE,
        "KIVI_2BIT": CompressType.KIVI_2BIT,
        "OURS": CompressType.OURS,
        "SVDQ": CompressType.SVDQ,
    }
    if env_value in compress_map:
        logger.info(f"[KVCacheManager] Using compress type from env: {env_value}")
        return compress_map[env_value]
    else:
        logger.warning(f"[KVCacheManager] Unknown compress type '{env_value}', using KIVI_2BIT")
        return CompressType.KIVI_2BIT
from lmcache.v1.compute.blend.compress.kivi import Kivi2Bit
from lmcache.v1.compute.blend.compress.normal import Normal
from lmcache.v1.compute.blend.compress.our import Ours
from lmcache.v1.compute.blend.compress.svdq import SVDQ
from lmcache.v1.compute.blend.db import DataCenter



class CompressFactory:
    """
    Factory class for creating compression instances.
    """
    @staticmethod
    def create_compressor(compress_type: CompressType, compress_config: dict = None, layer_num : int = 0, device="cuda"):
        compress_config = compress_config or {}
        
        if compress_type == CompressType.NONE:
            return Normal(device=device, config=compress_config)
        elif compress_type == CompressType.KIVI_2BIT:
            return Kivi2Bit(device=device, config=compress_config)
        elif compress_type == CompressType.OURS:
            return Ours(device=device)
        elif compress_type == CompressType.SVDQ:
            return SVDQ(device=device, config=compress_config)
  
class MemoryPinnedBuffer:
    """
    A class to manage pinned memory buffers.
    """
    MAX_PINNED_MEMORY = 16 * 1024 * 1024 * 1024  # 16 GB

    def __init__(self, shape):
        assert shape != None
        self.max_len = self.MAX_PINNED_MEMORY // (shape[2] * shape[3] * 2 * 2) 
        shape = (shape[0] , self.max_len , shape[2] , shape[3])

        self.shape = shape
        self.buffer = torch.empty(self.shape, dtype=torch.bfloat16).pin_memory()
        self.offset = 0

    def allocate(self, size: int):
        """
            Allocate a buffer of the given size from the pinned memory.
            if not enough memory, reset the buffer.
        """

        if self.offset + size > self.max_len:
            self.reset()
        start = self.offset
        self.offset += size
        return self.buffer[: , start:self.offset]

    def reset(self):
        self.offset = 0

class KVCacheManager:
    """
    A singleton class to manage key-value cache.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, shape = None, layer_num = None, device = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = KVCacheManager(shape = shape, layer_num = layer_num , device = device)
        return cls._instance
    
    @classmethod
    def clean_instance(cls):
        if cls._instance is not None:
            cls._instance.clean()

    def __init__(
        self,
        layer_num: int,
        device="cuda",
        shape=None,
        max_buffer_size=0 * 1024 * 1024 * 1024,
        compress_type=None,  # If None, read from env LMCACHE_COMPRESS_TYPE
        compress_config=None,
        io_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the KVCacheManager with specified parameters.

        Args:
            layer_num: Number of layers in the model.
            device: Device to use for computation.
            shape: Shape of the KV cache tensors.
            max_buffer_size: Maximum buffer size for CPU buffer pool.
            offload_mode: Offload mode (CPU or DISK).
            io_mode: IO mode (SAFETENSOR, BIN, CPP, S3, etc.).
            compress_type: Compression type.
            compress_config: Compression configuration.
            io_config: IO configuration (e.g., {"config_path": "path/to/s3.ini"} for S3).
        """
        self.device = device
        self.max_buffer_size = max_buffer_size

        self.io_config = io_config or {}
        self.db = DataCenter(
            max_buffer_size=max_buffer_size,
            device=device,
            io_config=self.io_config,
        )
        # If compress_type is None, read from environment variable
        if compress_type is None:
            compress_type = get_compress_type_from_env()
        self.compress_type = compress_type
        self.layer_num = layer_num
        self.compress_config = compress_config or {}
        self.compressor = CompressFactory.create_compressor(compress_type, self.compress_config, layer_num=layer_num , device=self.device)

        self.pinned_buffer_allocator = MemoryPinnedBuffer(shape)
        
        self.transfer_stream = torch.cuda.Stream(device=self.device)

        self.keys_set = set()    

    def check_keys_exist(self, keys: list) -> list[bool]:
        """
        Check if the given keys exist in the KV cache.
        """
        # keys: list of str ["xx/xx/filename"]
        # get the filenames
        
        return [key.split("/")[-1] in self.keys_set for key in keys]

    def set_compress_type(self, compress_type: CompressType, compress_config=None):
        """
        Set the compression type for the KV cache.
        """
        self.compress_type = compress_type
        self.compress_config = compress_config or {}
        self.compressor = CompressFactory.create_compressor(compress_type, self.compress_config, layer_num=self.layer_num, device=self.device)


    def retrieve_data(self, key: str , kv, layer_idx: int , stream=None):
        """
        Load data from a file.
        """
        data = self.db.retrieve_data(key)
      
        data = self.compressor.decompress(data , kv , layer_idx )
        
        return  data
    
    def store_data(self, key: str, data , score = None , layer_idx: int = None):
        """
        Save data to a file or cpu.
        """
        data = self.compressor.compress(data , score , layer_idx=layer_idx)
        # if layer_idx == 0:
        #     print(f"[KVManager.store_data] layer={layer_idx}, key={key}, data_keys={list(data.keys())}", flush=True)
        flag = self.db.store_data(key, data)
        
        if flag and "layer_0" in key:
            # only need to store the filename
            self.keys_set.add(key.split("/")[-1])

        return  flag

    def offload_compress_data(self , key: str, data: List[torch.Tensor]):
        sequence_len = data[0].size(1)

        # allocate pinned memory
        key_pinned = self.pinned_buffer_allocator.allocate(sequence_len)
        value_pinned = self.pinned_buffer_allocator.allocate(sequence_len)

        # copy to pinned memory
        key_pinned.copy_(data[0] , non_blocking=True)
        value_pinned.copy_(data[1] , non_blocking=True)
    
        torch.cuda.current_stream(device=self.device).synchronize()
        data = {"key": key_pinned , "value": value_pinned}
        flag = self.db.store_data(key , data , compress_flag=True)

        return flag
        
    def retrieve_by_task_id(self, task_id):
        t0 = time.perf_counter()
        compress_data_cpu , compress_data_gpu = self.db.retrive_by_task(task_id)
        t1 = time.perf_counter()
        profile_log(f"retrieve_by_task_id: db.retrive_by_task took {(t1-t0)*1000:.2f}ms")

        if isinstance(compress_data_gpu, list):
            result = []
            t_transfer_start = time.perf_counter()
            for i, data in enumerate(compress_data_gpu):
                # if i == 0:
                #     print(f"[KVManager.retrieve_by_task_id] data[0] keys before transfer={list(data.keys()) if isinstance(data, dict) else type(data)}", flush=True)
                transferred = self.compressor.transfer(data)
                # if i == 0:
                #     print(f"[KVManager.retrieve_by_task_id] data[0] keys after transfer={list(transferred.keys()) if isinstance(transferred, dict) else type(transferred)}", flush=True)
                result.append(transferred)
            t_transfer_end = time.perf_counter()
            profile_log(f"retrieve_by_task_id: transfer {len(compress_data_gpu)} items took {(t_transfer_end-t_transfer_start)*1000:.2f}ms")
            return result
        if compress_data_gpu is not None:
            # print(f"[KVManager.retrieve_by_task_id] single data keys before transfer={list(compress_data_gpu.keys()) if isinstance(compress_data_gpu, dict) else type(compress_data_gpu)}", flush=True)
            return self.compressor.transfer(compress_data_gpu)
        return compress_data_gpu
  
    def retrieve_keys(self , keys: List[str]):
        # if keys and len(keys) > 0:
        #     print(f"[KVManager.retrieve_keys] keys={keys}", flush=True)
        t_start = time.perf_counter()
        task_id = self.db.retrieve_keys(keys)
        t_db_end = time.perf_counter()
        profile_log(f"retrieve_keys: db.retrieve_keys for {len(keys)} keys took {(t_db_end - t_start) * 1000:.2f}ms")
        if isinstance(task_id, list):
            result = []
            t_transfer_start = time.perf_counter()
            for i, data in enumerate(task_id):
                # if i == 0:
                #     print(f"[KVManager.retrieve_keys] loaded data[0] keys={list(data.keys()) if isinstance(data, dict) else type(data)}", flush=True)
                result.append(self.compressor.transfer(data))
            t_transfer_end = time.perf_counter()
            profile_log(f"retrieve_keys: compressor.transfer for {len(task_id)} entries took {(t_transfer_end - t_transfer_start) * 1000:.2f}ms")
            return result
        return task_id

    def decompress(self, compressed_data , kv_len):
        t_decompress_start = time.perf_counter()
        decompressed = self.compressor.decompress(compressed_data , kv_len)
        t_decompress_end = time.perf_counter()
        profile_log(f"decompress: compressor.decompress for kv_len={kv_len} took {(t_decompress_end - t_decompress_start) * 1000:.2f}ms")
        return decompressed

    def clean(self):
        self.db.clean()
