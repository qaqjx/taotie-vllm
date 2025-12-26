
from enum import Enum

class CompressType(Enum):
    """
    Enum for different compression types.
    """
    NONE = "None"
    SVD = "svd"
    SQ8 = "sq8"
    SQ4 = "sq4"
    SVD_SQ4 = "svd_sq4"
    KIVI_2BIT = "kivi_2bit"
    CACHEGEN = "cachegen"
    SVDSQ4_DYNAMICRESIDUAL = "SVDSQ4_DynamicResidual"
    OURS = "ours"
    SVDQ = "svdq"


class AbstractCompress:

    def __init__(self, device="cuda"):
        """
        Initialize the compression algorithm with a device.
        """
        self.device = device
        
    """
    Abstract base class for compression algorithms.
    """
    def compress(self, data):
        """
        Compress the given data.
        """
        pass
    
    def decompress(self, compressed_data , kv_len):
        """
        Decompress the given compressed data.
        """
        pass

    def transfer(self, compressed_data):
        pass