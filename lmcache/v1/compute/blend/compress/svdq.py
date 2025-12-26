

import torch
from lmcache.v1.compute.blend.compress.abstract import AbstractCompress
class SVDQ(AbstractCompress):
    """
    SVD compression algorithm.
    """
    def __init__(self, device="cuda", config: dict = None):
        super().__init__(device=device)
        self.sechdual_bit =  [8,4,4]

    def second_compress(self, data, prefix = "key") -> dict:
        us  , v = data
        us = us.transpose_(-1 , -2) # [bs , rank , sequence]

        chunk_us = us.chunk(len(self.sechdual_bit), dim=-2)

        packed_chunks = []
        for idx , (chunk , bit) in enumerate(zip(chunk_us , self.sechdual_bit)):
            # chunk quantization to bit

            chunk_min = torch.min(chunk, dim=-1, keepdim=True).values
            chunk_max = torch.max(chunk, dim=-1, keepdim=True).values
            chunk_range = chunk_max - chunk_min + 1e-6
            chunk_normalized = (chunk - chunk_min) / chunk_range
            chunk_quantized = torch.clamp((chunk_normalized * (2**bit - 1)).round(), 0, (2**bit - 1)).to(torch.uint8)

            group = 8 // bit

            if chunk_quantized.size(-1) % group != 0:
                pad_size = group - (chunk_quantized.size(-1) % group)
                chunk_quantized = torch.nn.functional.pad(chunk_quantized, (0, pad_size), mode='constant', value=0)

            chunk_quantized_group = chunk_quantized.chunk(dim = -1 , chunks=group)

            chunk_quantized_packed = None
            for i in range(group):
                if i == 0:
                    chunk_quantized_packed = chunk_quantized_group[0]
                else:
                    chunk_quantized_packed |= (chunk_quantized_group[i] << (i * bit))

            chunk_quantized_packed = chunk_quantized_packed.to(torch.uint8).cpu().contiguous()
            chunk_meta = torch.cat([chunk_min, chunk_max], dim=-1).cpu()
            packed_chunks.append({f"chunk_{idx}_quantized": chunk_quantized_packed, f"chunk_{idx}_meta": chunk_meta})
        
        # combine all chunks to a dict
        combined_dict = {}
        for chunk_dict in packed_chunks:
            combined_dict.update(chunk_dict)

        combined_dict[f"v"] = v.cpu()
        return combined_dict

    def second_decompress(self, compressed_data , kv_len) -> tuple:
        v = compressed_data["v"]
        type = v.dtype
        us = None
        for idx , bit in enumerate(self.sechdual_bit):
            chunk_quantized = compressed_data[f"chunk_{idx}_quantized"].to(self.device)
            chunk_meta = compressed_data[f"chunk_{idx}_meta"].to(self.device)

            split_dim = chunk_meta.size(-1) // 2
            chunk_min = chunk_meta[:, :, :split_dim]
            chunk_max = chunk_meta[:, :, split_dim:]
            chunk_range = chunk_max - chunk_min + 1e-6

            group = 8 // bit

            chunk_quantized_group = []
            for i in range(group):
                shifted = (chunk_quantized >> (i * bit)) & (2**bit - 1)
                chunk_quantized_group.append(shifted)

            chunk_quantized_reconstructed = torch.cat(chunk_quantized_group, dim=-1).to(torch.uint8)
            # print(f"Decompressing chunk {idx} with bit {bit}, shape: {chunk_quantized_reconstructed.shape}")
            # print(f"Chunk min shape: {chunk_min.shape}, Chunk max shape: {chunk_max.shape}")
            chunk_dequantized = (chunk_quantized_reconstructed.to(type) / (2**bit - 1)) * chunk_range + chunk_min
            chunk_dequantized = chunk_dequantized[:, : , :kv_len]
            if idx == 0:
                us = chunk_dequantized
            else:
                us = torch.cat([us , chunk_dequantized] , dim=-2)

        us = us.transpose_(-1 , -2) # [bs , sequence , rank]

        return us , v

    def svd(self , tensor , rank) -> tuple:
        type = tensor.dtype
        bs, sequence_length, _, _ = tensor.shape

        all_us = []
        all_v = []
        for i in range(bs) :
            layer = tensor[i:i+1].reshape(1, sequence_length, -1)
            u, s, v = torch.linalg.svd(layer.float(), full_matrices=False)
            us = (u[:, :, :rank] @ torch.diag_embed(s[:, :rank])).to(type)
            v = v[:, :rank, :].to(type)
            all_us.append(u)
            all_v.append(v)
        u = torch.cat(all_us, dim=0)
        v = torch.cat(all_v, dim=0)
        return us , v

    def compress(self, data,score:None, layer_idx: int=0) -> dict:
        """
        Compress the given data using SVD.
        """
        key, value = data
        assert key.device != "cpu", "SVD compression only supports CUDA tensors"

        e_rank = min(key.size(-1) * key.size(-2), key.size(-3)) // 8 * len(self.sechdual_bit)
        key_us,  key_v = self.svd(key, e_rank)
        value_us, value_v = self.svd(value, e_rank)

        us = torch.cat([key_us, value_us], dim=0)
        v = torch.cat([key_v, value_v], dim=0)

        data = (us , v)
        data_dict = self.second_compress(data, prefix="both")

        return data_dict

    def transfer(self, compressed_data):
        # transfer to device
        for key in compressed_data:
            compressed_data[key] = compressed_data[key].to(self.device)
        return compressed_data

    def decompress(self, compressed_data: dict , kv_len) -> tuple:
        """
        Decompress the given compressed data using SVD.
        """
        us, v = self.second_decompress(compressed_data, kv_len=kv_len)
        # print(f"u shape: {u.shape}, sv shape: {sv.shape}")
        kv = torch.matmul(us, v) 

        splits = kv.size(0) // 2

        return kv[:splits] , kv[splits:]