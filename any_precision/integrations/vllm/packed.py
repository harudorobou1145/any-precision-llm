import numpy as np
import torch


def pack_indices(indices, parent_bits=8):
    if indices.ndim != 2 or indices.shape[1] % 32:
        raise ValueError('Indices must be a matrix with input width divisible by 32.')
    if not 3 <= parent_bits <= 8:
        raise ValueError('Parent precision must be between 3 and 8.')
    if indices.dtype != torch.uint8 or indices.max().item() >= 2 ** parent_bits:
        raise ValueError('Indices must be uint8 and fit the parent precision.')
    values = indices.detach().cpu().numpy()
    planes = np.stack([np.packbits((values >> shift) & 1, axis=1)
                       for shift in range(parent_bits - 1, -1, -1)])
    byte_count = planes.shape[-1]
    positions = np.arange(byte_count)
    mapped = np.empty_like(positions)
    for start in range(0, byte_count, 128):
        count = min(128, byte_count - start)
        offsets = np.arange(count)
        threads = count // 4
        mapped[start:start + count] = start + offsets % threads * 4 + ((offsets // threads) ^ 3)
    packed = planes[:, :, np.argsort(mapped)].copy().view(np.int32)
    return torch.from_numpy(packed).to(indices.device)


def compressed_linear(inputs, qweight, codebook, bits, bias=None):
    if torch.is_grad_enabled() and (inputs.requires_grad or codebook.requires_grad):
        raise RuntimeError('Compressed rollout is inference-only; use the trainable actor for backward.')
    if bits < 3 or bits > qweight.shape[0]:
        raise ValueError('Unsupported precision.')
    if qweight.dtype != torch.int32 or qweight.ndim != 3:
        raise ValueError('Expected int32 bitplanes.')
    if qweight.shape[1] % 16 or inputs.shape[-1] != qweight.shape[2] * 32:
        raise ValueError('Kernel requires output width divisible by 16 and matching input width.')
    if codebook.shape != (qweight.shape[1], 2 ** bits):
        raise ValueError('Codebook shape does not match the selected precision.')
    if not inputs.is_cuda or inputs.device != qweight.device or inputs.device != codebook.device:
        raise ValueError('Inputs, bitplanes and codebook must be on the same CUDA device.')
    from any_precision_ext import matmul_kbit
    flat = inputs.reshape(-1, inputs.shape[-1]).to(torch.float16).contiguous()
    result = torch.empty((flat.shape[0], qweight.shape[1]), device=inputs.device, dtype=torch.float16)
    lookup = codebook.to(torch.float16).contiguous()
    for start in range(0, flat.shape[0], 8):
        result[start:start + 8] = matmul_kbit(flat[start:start + 8], qweight, lookup, bits)
    result = result.reshape(*inputs.shape[:-1], qweight.shape[1]).to(inputs.dtype)
    return result if bias is None else result + bias