

import time
import torch

from lmcache.v1.compute.blend.compress.abstract import AbstractCompress

def calculate_elbow(singular_value: torch.Tensor) -> float:
    """
    Find the elbow point for each sample in a batch of singular values,
    then return the mean elbow dimension (soft-argmax not used; hard argmax + float mean).

    Args:
        singular_value (Tensor): Shape [B, N], each row is a sorted singular value vector (descending)

    Returns:
        float: Mean elbow dimension across the batch (scalar, detached float)
    """
    device = singular_value.device
    B, N = singular_value.shape
    
    log_s = torch.log(singular_value + 1e-16)  # [B, N]
    # print(f"singular value {singular_value[: , -1]} ,log s {log_s[: , -1]}")
    # x coordinates: same for all batches -> [0, 1, ..., N-1]
    x_coords = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N]
    
    # Coordinates: [B, N, 2] where last dim is (x, log_s)
    all_coords = torch.stack([x_coords.expand(B, N), log_s], dim=-1)  # [B, N, 2]

    # First and last points: shape [B, 2]
    first_point = all_coords[:, 0, :]  # [B, 2]
    last_point = all_coords[:, -1, :]  # [B, 2]

    # Vector from first to last: [B, 2]
    line_vec = last_point - first_point  # [B, 2]

    # Vectors from first point to each point: [B, N, 2]
    point_vec = all_coords - first_point.unsqueeze(1)  # [B, N, 2]

    # Cross product in 2D: |line_vec_x * point_vec_y - line_vec_y * point_vec_x|
    cross_product = torch.abs(
        line_vec[:, 0:1] * point_vec[:, :, 1] - line_vec[:, 1:2] * point_vec[:, :, 0]
    )  # [B, N]

    # Norm of line vector: [B]
    line_norm = torch.linalg.norm(line_vec, dim=1)  # [B]

    # Handle near-zero norm (constant lines)
    small_norm_mask = line_norm < 1e-10
    line_norm = torch.where(small_norm_mask, torch.ones_like(line_norm), line_norm)

    # Distance from each point to the line: [B, N]
    distances = cross_product / line_norm.unsqueeze(1)  # [B, N]

    # Find elbow index for each batch
    _, indices = torch.max(distances, dim=1)  # indices: [B], long tensor
    mean_elbow_dim = indices.float().mean().item()

    return mean_elbow_dim

class UniformAllocate():
    def __init__(self):
        pass
    def _calculate_ranks(self, total_budget_bytes: int, num_layers: int, dim: int, token: int) -> int:
        budget_per_rank = 4 * 16 + (token + dim) * 4  # 4 bytes per float
        return total_budget_bytes * 8 // budget_per_rank
        
    def allocate(self, total_budget_bytes: int, num_layers: int, dim: int, token_num: int,
                 score: list = None, max_rank: int = 0) -> list:

        all_rank = self._calculate_ranks(total_budget_bytes, num_layers, dim, token_num)
        
        rank_per_layer = int(all_rank // num_layers)
        ranks = [rank_per_layer] * num_layers
        return ranks


class Ours(AbstractCompress):
    def __init__(self, device="cuda", config: dict = None):
        super().__init__(device=device)
        self.ratio = 0.05
        self.last_final_ranks = None

    def sq_compress(self, data) -> dict:
        u , v = data
        assert u.device != "cpu", "SVD compression only supports CUDA tensors"

        type = u.dtype
        u_min = torch.min(u, dim=-1, keepdim=True).values
        u_max = torch.max(u, dim=-1, keepdim=True).values
        u_range = u_max - u_min + 1e-6
        u_normalized = (u - u_min) / u_range
        u_quantized = torch.clamp((u_normalized * 15).round(), 0, 15).to(torch.uint8)
        
        u_chunk = u_quantized.chunk(dim = -1 , chunks=2)
        # padding chunk 1 to make shapes match
        if u_chunk[0].shape[-1] != u_chunk[1].shape[-1]:
            pad_size = u_chunk[0].shape[-1] - u_chunk[1].shape[-1]
            u_chunk_1_padded = torch.nn.functional.pad(u_chunk[1], (0, pad_size), mode='constant', value=0)        
        else:
            u_chunk_1_padded = u_chunk[1]

        u_quantized = u_chunk[0] * 16 + u_chunk_1_padded
        u_meta = torch.cat([u_min, u_max], dim=-1).cpu()
        
        v_min = torch.min(v, dim=-1, keepdim=True).values
        v_max = torch.max(v, dim=-1, keepdim=True).values
        v_range = v_max - v_min + 1e-6
        v_normalized = (v - v_min) / v_range
        v_quantized = torch.clamp((v_normalized * 15).round(), 0, 15).to(torch.uint8)
        v_chunk = v_quantized.chunk(dim = -1 , chunks=2)
        if v_chunk[0].shape[-1] != v_chunk[1].shape[-1]:
            pad_size = v_chunk[0].shape[-1] - v_chunk[1].shape[-1]
            v_chunk_1_padded = torch.nn.functional.pad(v_chunk[1], (0, pad_size), mode='constant', value=0)
        else:
            v_chunk_1_padded = v_chunk[1]

        v_quantized = v_chunk[0] * 16 + v_chunk_1_padded
        v_meta = torch.cat([v_min, v_max], dim=-1).cpu()

        split_dim = v.size(0) // 2
        residual_key_sv = v[:split_dim,:self.key_residual_dim,:]
        residual_value_sv = v[split_dim:,:self.value_residual_dim,:]

        residual_key_u = u[:split_dim,:self.key_residual_dim,:]
        residual_value_u = u[split_dim:,:self.value_residual_dim,:]

        return {
            f"u_quantized": u_quantized.cpu().contiguous(),
            f"u_meta": u_meta.cpu(),

            f"v_quantized": v_quantized.cpu(),
            f"v_meta": v_meta.cpu(),

            f"key_residual_sv": residual_key_sv.cpu(),
            f"value_residual_sv": residual_value_sv.cpu(),

            # f"key_residual_u": residual_key_u.cpu().contiguous(),
            # f"value_residual_u": residual_value_u.cpu().contiguous(),
        }

    def sq_decompress(self, compressed_data ,kv_len) -> tuple:
        u_quantized = compressed_data[f"u_quantized"]
        u_meta = compressed_data[f"u_meta"]
        v_quantized = compressed_data[f"v_quantized"]
        v_meta = compressed_data[f"v_meta"]
        residual_key_v = compressed_data[f"key_residual_sv"]
        residual_value_v = compressed_data[f"value_residual_sv"]
        # residual_key_u = compressed_data[f"key_residual_u"]
        # residual_value_u = compressed_data[f"value_residual_u"]

        u_0 = u_quantized // 16
        u_1 = u_quantized % 16
        u_quantized = torch.cat([u_0, u_1], dim=-1).to(torch.uint8)

        split_dim = u_meta.size(-1) // 2
        type = u_meta.dtype
        u_dequantized = u_quantized.to(type) / 15 * (u_meta[: , : , split_dim:] - u_meta[: , : , :split_dim]) + u_meta[: , : , :split_dim]
        u_dequantized = u_dequantized[:, : , :kv_len]
      
        v_0 = v_quantized // 16
        v_1 = v_quantized % 16
        v_quantized = torch.cat([v_0, v_1], dim=-1).to(torch.uint8)
        v_dequantized = v_quantized.to(type) / 15 * (v_meta[: , : , split_dim:] - v_meta[: , : , :split_dim]) + v_meta[: , : , :split_dim]

        split_dim = v_dequantized.size(0) // 2

        residual_key_v_dim = min(v_dequantized.size(1), residual_key_v.size(1))
        residual_value_v_dim = min(v_dequantized.size(1), residual_value_v.size(1))
        # residual_key_u_dim = min(u_dequantized.size(1), residual_key_u.size(1))
        # residual_value_u_dim = min(u_dequantized.size(1), residual_value_u.size(1))

        v_dequantized[:split_dim, :residual_key_v_dim, :] = residual_key_v[: ,:residual_key_v_dim, :]
        v_dequantized[split_dim:, :residual_value_v_dim, :] = residual_value_v[: ,:residual_value_v_dim, :]

        # u_dequantized[:split_dim, :residual_key_u_dim, :] = residual_key_u[: ,:residual_key_u_dim, :]
        # u_dequantized[split_dim:, :residual_value_u_dim, :] = residual_value_u[: ,:residual_value_u_dim, :]

        return u_dequantized, v_dequantized

    def svd(self, x):
        layer_num = x.shape[0]
        all_u = []
        all_s = []
        all_v = []
        for i in range(0, layer_num):
            u , s , v = torch.linalg.svd(x[i : i + 1].reshape(-1, x.shape[1], x.shape[2] * x.shape[3]).float(), full_matrices=False)
            all_u.append(u.to(x.dtype))
            all_s.append(s)
            all_v.append(v)
        all_s = torch.cat(all_s, dim=0)
        return all_u , all_s , all_v

    def compress(self, all_layer_kv: list, scores: list = None, layer_idx: int = 0 ) -> list:
        """
        Compresses KV caches for all layers with adaptive rank allocation.

        Args:
            all_layer_kv (list): A list of (key, value) tuples.
            all_scale_factors (list): A list of floats, one importance score for each layer.
            global_compression_ratio (float): The target compression ratio for the entire KV cache.

        Returns:
            list: A list of compressed data dictionaries, one for each layer.
        """
        # 1. Calculate original total memory and target budget
        key , value = all_layer_kv

        num_layers , seq_len , num_heads , head_dim = key.shape
        total_original_mem_bytes = num_layers * seq_len * num_heads * head_dim * (key.element_size()) * self.ratio
        # 2. Gather metadata for rank calculation
        # 3. SVD
        key_u , key_s , key_v = self.svd(key)
        value_u , value_s , value_v = self.svd(value)
        key_residual_dim = int(calculate_elbow(key_s))
        value_residual_dim = int(calculate_elbow(value_s))

        final_ranks = 0
        if layer_idx < 1:
            final_ranks = UniformAllocate().allocate(
                total_budget_bytes=total_original_mem_bytes,
                num_layers=1,
                dim=head_dim * num_heads,
                token_num=seq_len
            )[0]
        else:  
            final_ranks = UniformAllocate().allocate(
                total_budget_bytes=total_original_mem_bytes,
                num_layers=1,
                dim=head_dim * num_heads,
                token_num=seq_len
            )[0]
            # final_ranks = DynamicAllocate().allocate(
            #     total_budget_bytes=total_original_mem_bytes * (len(scores) - 1) ,
            #     num_layers=(len(scores) - 1),
            #     dim=head_dim * num_heads,
            #     token_num=seq_len,
            #     score=scores,
            #     max_rank=min(head_dim * num_heads, seq_len)
            # )
            # final_ranks  = final_ranks[layer_idx - 1]

        max_residual_rank = max(final_ranks - 1, 0)
        self.key_residual_dim = min(key_residual_dim, max_residual_rank)
        self.value_residual_dim = min(value_residual_dim, max_residual_rank)

        # only one layer data
        u = torch.cat([key_u[0][:, : , : final_ranks], value_u[0 ][:, : , : final_ranks]], dim=0).transpose(-1, -2)
        s = torch.cat([key_s[0 : 1, : final_ranks], value_s[0 : 1, : final_ranks]], dim=0)
        v = torch.cat([key_v[0][: , : final_ranks, :], value_v[0][:, : final_ranks, :]], dim=0)    
        sv = torch.diag_embed(s) @ v

        # Store for debugging purposes.
        self.last_final_ranks = final_ranks
        data = self.sq_compress((u, sv.to(u.dtype)))

        return data
    
    def transfer(self, compressed_data):
        compressed_data[f"u_quantized"] = compressed_data[f"u_quantized"].to(self.device, non_blocking=True)
        compressed_data[f"u_meta"] = compressed_data[f"u_meta"].to(self.device, non_blocking=True)

        compressed_data[f"v_quantized"] = compressed_data[f"v_quantized"].to(self.device, non_blocking=True)
        compressed_data[f"v_meta"] = compressed_data[f"v_meta"].to(self.device, non_blocking=True)

        compressed_data[f"key_residual_sv"] = compressed_data[f"key_residual_sv"].to(self.device, non_blocking=True)
        compressed_data[f"value_residual_sv"] = compressed_data[f"value_residual_sv"].to(self.device, non_blocking=True)

        # compressed_data[f"key_residual_u"] = compressed_data[f"key_residual_u"].to(self.device, non_blocking=True)
        # compressed_data[f"value_residual_u"] = compressed_data[f"value_residual_u"].to(self.device, non_blocking=True)
        return compressed_data

    def decompress(self, compressed_data , kv_len):
        u , sv = self.sq_decompress(compressed_data, kv_len)

        u = u.transpose(-1, -2).contiguous()
        kv = torch.matmul(u , sv)

        return kv[0] , kv[1]
