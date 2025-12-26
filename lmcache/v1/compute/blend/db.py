
from enum import Enum
from pickletools import pydict
from typing import Dict, List, Optional, Any

import torch
from lmcache.v1.compute.blend.cpu_buffer import MAX_BUFFER_SIZE, CPUBufferPool
from lmcache.v1.compute.blend.s3_manager import S3DiskManager

class OffloadMode(Enum):
    CPU = "cpu"
    DISK = "disk"

class DataCenter:
    """
    Base class for data center operations.
    """
    def __init__(
        self,
        max_buffer_size=MAX_BUFFER_SIZE,
        offload_mode=OffloadMode.DISK,
        device="cuda",
        io_config: Optional[Dict[str, Any]] = None,
    ):
        self.max_buffer_size = max_buffer_size
        self.offload_mode = offload_mode
        self.io_config = io_config or {}
        self.disk_io_manager = S3DiskManager()
        self.cpu_buffer_pool = CPUBufferPool(max_size=max_buffer_size)
        self.device = device

    def retrieve_data(self, key: str):
        """
        Load data from a file.
        """
        data = self.cpu_buffer_pool.get_data(key)
        if self.offload_mode == OffloadMode.CPU:
            return data
        
        if data is None:
            data = self.disk_io_manager.load_data(key)
            if data is not None:
                self.cpu_buffer_pool.add_data(key, data)
                return data
    
        return data
    

    def retrieve_keys(self, keys: List[str]):
        # get by cpu buffer
        # result = [self.cpu_buffer_pool.get_data(key) for key in keys]
        
        # disk_keys = [keys[i] for i in range(len(keys)) if result[i] is None]
        # if disk_keys == []:
        #     return result

        task_id = self.disk_io_manager.load_datas(keys, self.device) 
        # for i, key in enumerate(disk_keys):
        #     if disk_result_cpu[i] is not None:
        #         self.cpu_buffer_pool.add_data(key, disk_result_cpu[i].clone())
        
        return task_id
        # merge result

        for i, key in enumerate(disk_keys):
            if disk_result[i] is not None:
                self.cpu_buffer_pool.add_data(key, disk_result[i])
        result = [data if data is not None else self.cpu_buffer_pool.get_data(key) for data, key in zip(result, keys)]
        return result

    def retrive_by_task(self, task_id):
        result = self.disk_io_manager.load_task(task_id)
        return result

    def store_data(self, key: str, data , compress_flag=False):
        """
        Save data to a file.
        """
        self.cpu_buffer_pool.add_data(key, data)

        if self.offload_mode == OffloadMode.DISK:
            self.disk_io_manager.save_data(key, data, compress_flag=compress_flag)

        return True
    
    def clean(self):
        self.cpu_buffer_pool.clean()