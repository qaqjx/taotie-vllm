

from lmcache.v1.compute.blend.compress.abstract import AbstractCompress

class Normal(AbstractCompress):
    """
    SVD compression algorithm.
    """
    def __init__(self, device="cuda", config: dict = None):
        super().__init__(device=device)
        config = config or {}

    def compress(self, data, score=None , layer_idx = None) -> dict:
        """
        Not compress the given data .
        """

        key , value = data
        data_dict = {
            "key": key.contiguous().cpu().pin_memory(),
            "value": value.contiguous().cpu().pin_memory(),
        }

        return data_dict
    
    def transfer(self, compressed_data):
        key = compressed_data["key"].to(self.device, non_blocking=True)
        value = compressed_data["value"].to(self.device, non_blocking=True)
        return key, value

    def decompress(self, compressed_data , kv_len):
        """
        Decompress the given compressed data using SVD.
        """
        if isinstance(compressed_data, tuple):
            return compressed_data[0] , compressed_data[1]
        return compressed_data["key"] , compressed_data["value"]