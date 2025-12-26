import torch


import torch

from lmcache.v1.compute.blend.compress.abstract import AbstractCompress

def quantized_last_dim(tensor, num_bits):
    """
    Quantizes the last dimension of the tensor to the specified number of bits.
    """

    data_min = tensor.min(dim=-1, keepdim=True).values
    data_range = tensor.max(dim=-1, keepdim=True).values - data_min + 1e-6  # Prevent division by zero
    data_normalized = (tensor - data_min) / data_range

    tensor_quantized = torch.clamp((data_normalized * (2**num_bits - 1)).round(), 0, 2**num_bits - 1).to(torch.uint8)
    chunk_num = 8 // num_bits

    if chunk_num > 1:
        tensor_quantized = tensor_quantized.view(*tensor_quantized.shape[:-1], -1, chunk_num)
        shifts = torch.arange(chunk_num, device=tensor.device) * num_bits
        tensor_quantized = (tensor_quantized << shifts).sum(dim=-1).to(torch.uint8)

    return tensor_quantized, data_min, data_range

def dequantized_last_dim(tensor_quantized, data_min, data_range, num_bits):
    """
    Dequantizes the last dimension of the tensor from the specified number of bits.
    """
    chunk_num = 8 // num_bits
    
    if chunk_num > 1:
        tensor_quantized = tensor_quantized.unsqueeze(-1)
        shifts = torch.arange(chunk_num, device=tensor_quantized.device) * num_bits
        tensor_quantized = (tensor_quantized >> shifts) & (2**num_bits - 1)
        tensor_quantized = tensor_quantized.reshape(*tensor_quantized.shape[:-2], -1)
    # print(f"dequantized tensor shape: {tensor_quantized.shape} , data_min shape: {data_min.shape} , data_range shape: {data_range.shape}")

    tensor_dequantized = tensor_quantized.to(data_min.dtype) / (2**num_bits - 1) * data_range + data_min
    return tensor_dequantized

def quantized_last_dim_group(tensor, num_bits, group_size=32):
    """
    Group-wise quantization of the last dimension of the tensor.
    """
    tensor = tensor.view(*tensor.shape[:-1], -1, group_size)
    return quantized_last_dim(tensor, num_bits)

def dequantized_last_dim_group(tensor_quantized, data_min, data_range, num_bits):
    """
    Group-wise dequantization of the last dimension of the tensor.
    """
    tensor_dequantized = dequantized_last_dim(tensor_quantized, data_min, data_range, num_bits)
    return tensor_dequantized.view(*tensor_dequantized.shape[:-2], -1)


import triton
import triton.language as tl
import random
import numpy as np
import torch

def quant_and_pack_last_dim(tensor: torch.FloatTensor, bits: int):
	assert len(tensor.shape) >= 2
	shape = tensor.shape
	# ================== Get Scale & Zeros ===============

	# Quantize
	max_int = 2 ** bits - 1
	mn = torch.min(tensor, dim=-1, keepdim=True).values
	mx = torch.max(tensor, dim=-1, keepdim=True).values
	scale = (mx - mn) / max_int
	tensor = tensor	 - mn
	tensor.div_(scale)
	tensor = tensor.clamp_(0, max_int).round().to(torch.int32)
	tensor = tensor.view(shape)
	# Pack
	code = pack_tensor(tensor, bits, pack_dim=len(shape)-1)
	return code, scale, mn

def quant_and_pack_kcache(k: torch.FloatTensor, group_size: int, bits: int):
	assert len(k.shape) == 4
	shape = k.shape
	B, nh, T, D = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B, nh, num_groups, group_size, D)
	# Quantize
	max_int = 2 ** bits - 1
	data = k.view(new_shape)
	mn = torch.min(data, dim=-2, keepdim=True)[0]
	mx = torch.max(data, dim=-2, keepdim=True)[0]
	scale =  (mx - mn) / max_int
	data = data - mn
	data.div_(scale)
	data = data.clamp_(0, max_int).round_().to(torch.int32)
	data = data.view(shape)
	code = pack_tensor(data, bits, pack_dim=2)
	return code, scale, mn

def quant_and_pack_vcache(v: torch.FloatTensor, group_size: int, bits: int):
	shape = v.shape
	assert len(shape) == 4
	assert v.shape[-1] % group_size == 0
	num_groups = shape[-1] // group_size
	new_shape = (shape[:-1] + (num_groups, group_size))
	# Quantize
	max_int = 2 ** bits - 1
	data = v.view(new_shape)
	mn = torch.min(data, dim=-1, keepdim=True)[0]
	mx = torch.max(data, dim=-1, keepdim=True)[0]
	scale = (mx - mn) / max_int
	data = data - mn
	data.div_(scale)
	data = data.clamp_(0, max_int).round_().to(torch.int32)
	data = data.view(shape)
	# Pack
	code = pack_tensor(data, bits, pack_dim=3)
	return code, scale, mn


def unpack_and_dequant_kcache(k_code: torch.FloatTensor, 
							  scale: torch.FloatTensor, 
							  mn: torch.FloatTensor,
							  group_size: int, 
							  bits: int,
							  ):
	pack_dim = 2
	assert bits in [2, 4, 8]
	assert len(k_code.shape) == 4
	data = unpack_tensor(k_code, bits, pack_dim=pack_dim)
	shape = data.shape
	num_groups = shape[pack_dim] // group_size
	data = data.view(shape[:pack_dim] + (num_groups, group_size,) + shape[pack_dim+1:])
	data = data.to(torch.float16)
	data = data * scale + mn 
	return data.view(shape)

	
def unpack_and_dequant_vcache(v_code: torch.FloatTensor, 
							  scale: torch.FloatTensor, 
							  mn: torch.FloatTensor,
							  group_size: int, 
							  bits: int,
							  ):
	assert bits in [2, 4, 8]
	assert len(v_code.shape) == 4
	data = unpack_tensor(v_code, bits, pack_dim=3)
	shape = data.shape
	num_groups = shape[-1] // group_size
	data = data.view(shape[:-1] + (num_groups, group_size,))
	data = data.to(torch.float16)
	data = data * scale + mn 
	return data.view(shape)


def pack_tensor(data, bits, pack_dim):
	# Pack
	shape = data.shape
	feat_per_int = 32 // bits
	assert bits in [2,4,8], "Only 2, 4, 8 bits are supported"
	assert shape[pack_dim] % feat_per_int == 0, "Dimension length must be divisible by number of features per int"
	# BS, nh, T, nd // 16 # 16 is for 2bit
	code = torch.zeros(shape[:pack_dim] + (shape[pack_dim] // feat_per_int,)+shape[pack_dim+1:], 
					dtype=torch.int32, 
					device=data.device)
	i = 0
	row = 0
	unpacked_indices = [slice(None)] * len(data.shape)
	packed_indices = [slice(None)] * len(data.shape)
	while row < code.shape[pack_dim]:
		packed_indices[pack_dim] = row
		for j in range(i, i + (32 // bits)):
			unpacked_indices[pack_dim] = j
			code[packed_indices] |= data[unpacked_indices] << (bits * (j - i))
		i += 32 // bits
		row += 1
	return code


def unpack_tensor(v_code: torch.FloatTensor, 
				  bits: int, 
				  pack_dim: int):
	assert bits in [2,4,8]
	shape = v_code.shape
	feat_per_int = 32 // bits
	new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim+1:]
	unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
	i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
	j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
	num = 0xFF >> (8 - bits)
	packed_indices = [slice(None)] * len(new_shape)
	packed_indices[pack_dim] = i
	if pack_dim == 2:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)) & num
	elif pack_dim == 3:
		unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
	else:
		raise NotImplementedError
	return unpacked_v_code


@triton.jit
def _pack_along_last_dim(
	bits: tl.constexpr,
	intensor_ptr,
	code_ptr,
	N,
	num_feats: tl.constexpr,
	feat_per_int: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	num_int_per_y_dim = num_feats // feat_per_int
	bid = tl.program_id(axis=0)
	yid = tl.program_id(axis=1)
	offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int # offset of the first element at current tile
	packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
	for i in range(feat_per_int):
		ptr = block_start + i
		element = tl.load(ptr, mask=offs_N<N, other=0.)
		element = element << (i * bits)
		# Combine the value using bitwise OR
		packed = packed | element
	tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)



@triton.jit
def _minmax_along_last_dim(
	x_ptr,
	mn_ptr, mx_ptr,
	total_elements: tl.constexpr, 
	N: tl.constexpr,
	num_groups: tl.constexpr, 
	group_size: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	bid = tl.program_id(axis=0)
	offsets_b = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	offsets = offsets_b[:, None] * group_size + tl.arange(0, group_size)[None, :]
	mask = offsets < total_elements
	x = tl.load(x_ptr + offsets, mask=mask)
	mx_val = tl.max(x, axis=1)
	mn_val = tl.min(x, axis=1)
	# tl.device_print('shape', mn_val[:, None].shape)
	tl.store(mn_ptr+offsets_b, mn_val, mask=offsets_b<N*num_groups)
	tl.store(mx_ptr+offsets_b, mx_val, mask=offsets_b<N*num_groups)


# def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
# 	assert len(data.shape) == 4
# 	shape = data.shape
# 	B, nh, D, T = shape
# 	# ================== Get Scale & Zeros ===============
# 	assert T % group_size == 0
# 	num_groups = T // group_size
# 	new_shape = (B * nh * D, num_groups, group_size)
# 	scale_mn_shape = B, nh, D, num_groups
# 	# Quantize
# 	max_int = 2 ** bit - 1
# 	data = data.view(new_shape)
# 	mn = torch.min(data, dim=-1, keepdim=True)[0]
# 	mx = torch.max(data, dim=-1, keepdim=True)[0]
# 	# B, nh, D, T // group_size, 1
# 	scale = (mx - mn) / max_int
# 	data = data - mn
# 	data.div_(scale)
# 	data = data.clamp_(0, max_int).round_().to(torch.int32)
# 	scale, mn = scale.squeeze(-1), mn.squeeze(-1)
# 	data = data.view(-1, T)
# 	feat_per_int = 32 // bit
# 	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
# 	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
# 	if B <= 4:
# 		BLOCK_SIZE_N = 32
# 	else:
# 		BLOCK_SIZE_N = 128
# 	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
# 	_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
# 								data.shape[1], feat_per_int, 
# 								BLOCK_SIZE_N=BLOCK_SIZE_N, 
# 								num_warps=8)
# 	return code.view(B, nh, D, -1), scale.view(scale_mn_shape), mn.view(scale_mn_shape)
	
	

def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
	assert len(data.shape) == 4
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	scale_mn_shape = B, nh, D, num_groups
	# Quantize
	data = data.reshape(new_shape)
	mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
	with torch.cuda.device(data.device):
		_minmax_along_last_dim[grid](data, mn, mx,
							 data.numel(), data.shape[0], num_groups, group_size,
							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
	scale = (mx - mn) / (2 ** bit - 1)
	data = data - mn.unsqueeze(-1)
	data.div_(scale.unsqueeze(-1))
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
	data = data.view(-1, T)
	feat_per_int = 32 // bit
	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	with torch.cuda.device(data.device):
		_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)
	


class Kivi2Bit(AbstractCompress):
    """
    2-bit compression algorithm.
    """
    def __init__(self, device="cuda", config: dict = None):
        # print("Using SQ4 compression")
        super().__init__(device=device)
        config = config or {}
        self.residual_length = config.get("residual_length", 128)
        self.group_size = config.get("group_size", 32)
        self.num_bits = 2
        # print("Using KIVI 2-bit compression with residual length:", self.residual_length)

    def compress(self, data , score = None,layer_idx=None, ) -> dict:
        """
        Compress the given data using 2-bit.
        """
        key, value = data
        assert key.device != "cpu", "2-bit compression only supports CUDA tensors"
        if key.size(1) < self.residual_length + 1:
            return {
                "key": key.cpu(),
                "value": value.cpu(),
            }

        type = key.dtype

        residual_length = key.size(1) % self.residual_length
        if residual_length != 0:
            key_residual = key[:, :residual_length, ::]
            value_residual = value[:, :residual_length , ::]

        # transpose key to [B ,  H , D , S]
        key = key[:, residual_length:, ::].permute(0, 2, 3, 1).contiguous()
        value = value[:, residual_length: , ::]

        key_quantized , key_min , key_range = triton_quantize_and_pack_along_last_dim(
            key , self.group_size , 2
        )

        value_quantized , value_min , value_range = triton_quantize_and_pack_along_last_dim(
            value , self.group_size , 2
        )
       
        dict = {
            "key_quantized": key_quantized.cpu(),
            "key_min": key_min.to(type).cpu(),
            "key_range": key_range.to(type).cpu(),
            "value_quantized": value_quantized.cpu(),
            "value_min": value_min.to(type).cpu(),
            "value_range": value_range.to(type).cpu(),
        }

        if residual_length != 0:
            dict["key_residual"] = key_residual.cpu()
            dict["value_residual"] = value_residual.cpu()

        return dict

    def transfer(self, compressed_data):
        if "key" in compressed_data and "value" in compressed_data:
            compressed_data["key"] = compressed_data["key"].to(self.device)
            compressed_data["value"] = compressed_data["value"].to(self.device)
            return compressed_data

        compressed_data["key_quantized"] = compressed_data["key_quantized"].to(self.device)
        compressed_data["key_min"] = compressed_data["key_min"].to(self.device)
        compressed_data["key_range"] = compressed_data["key_range"].to(self.device)

        compressed_data["value_quantized"] = compressed_data["value_quantized"].to(self.device)
        compressed_data["value_min"] = compressed_data["value_min"].to(self.device)
        compressed_data["value_range"] = compressed_data["value_range"].to(self.device)

        if "key_residual" in compressed_data and "value_residual" in compressed_data:
            compressed_data["key_residual"] = compressed_data["key_residual"].to(self.device)
            compressed_data["value_residual"] = compressed_data["value_residual"].to(self.device)
            return compressed_data
        return compressed_data
    
    def decompress(self, compressed_data, kv_len) -> tuple:
        """
        Decompress the given compressed data using 2-bit.
        """
        compressed_data = self.transfer(compressed_data)        

        if "key" in compressed_data and "value" in compressed_data:
            return compressed_data["key"] , compressed_data["value"]

        key_quantized = compressed_data["key_quantized"]
        key_min = compressed_data["key_min"]
        key_range = compressed_data["key_range"]
        value_quantized = compressed_data["value_quantized"]
        value_min = compressed_data["value_min"]
        value_range = compressed_data["value_range"]

        key = unpack_and_dequant_vcache(key_quantized, key_min.unsqueeze(-1), key_range.unsqueeze(-1), self.group_size, 2)
        value = unpack_and_dequant_vcache(value_quantized, value_min.unsqueeze(-1), value_range.unsqueeze(-1), self.group_size, 2)

        key = key.permute(0, 3, 1, 2).contiguous()


        if "key_residual" in compressed_data and "value_residual" in compressed_data:
            key_residual = compressed_data["key_residual"]
            value_residual = compressed_data["value_residual"]

            key = torch.cat([key_residual, key], dim=1)
            value = torch.cat([value_residual, value], dim=1)
  

        # print(f"key sum {torch.sum(key)} , value sum {torch.sum(value)}")
        assert key.shape == value.shape, "Key and value shapes do not match after decompression"

        return key, value
    