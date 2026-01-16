from asyncio.log import logger
from multiprocessing.pool import ThreadPool
import os
import threading
import time
from typing import Any, Dict, List, Optional

import torch

from lmcache.v1.compute.blend.compress.abstract import CompressType
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
        max_buffer_size=16 * 1024 * 1024 * 1024,
        compress_type=CompressType.NONE,
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
        compress_data_cpu , compress_data_gpu = self.db.retrive_by_task(task_id)
        return compress_data_gpu
  
    
    def retrieve_keys(self , keys: List[str]):

        task_id = self.db.retrieve_keys(keys)
        if isinstance(task_id, list):
            result = []
            for data in task_id:
                result.append(self.compressor.transfer(data))
            return result
        return task_id

    def decompress(self, compressed_data , kv_len):
        return self.compressor.decompress(compressed_data , kv_len)

    def clean(self):
        self.db.clean()