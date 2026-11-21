import re

import torch

from .quantization import AnyPrecisionLinearMethod
from .packed import pack_indices


def projection_sources(name):
    match = re.search(r'\.layers\.(\d+)\.(mlp|linear_attn|self_attn)\.([^.]*)$', name)
    if match is None:
        raise ValueError(f'Unsupported compressed layer: {name}')
    index, branch, projection = match.groups()
    fused = {'mlp.gate_up_proj': ('gate_proj', 'up_proj'),
             'linear_attn.in_proj_qkvz': ('in_proj_qkv', 'in_proj_z'),
             'self_attn.qkv_proj': ('q_proj', 'k_proj', 'v_proj')}
    return index, tuple(f'{branch}.{part}' for part in fused.get(f'{branch}.{projection}', (projection,)))


def compressed_layers(model):
    layers = [(name, layer) for name, layer in model.named_modules()
              if isinstance(getattr(layer, 'quant_method', None), AnyPrecisionLinearMethod)]
    if not layers:
        raise ValueError('No compressed AnyPrecision layers found.')
    return layers


def set_precision(model, bits):
    layers = compressed_layers(model)
    if any(bits not in layer.quant_method.precisions for _, layer in layers):
        raise ValueError(f'Unsupported student precision: {bits}')
    for _, layer in layers:
        layer.quant_method.set_precision(layer, bits)


@torch.no_grad()
def update_from_actor(model, weights):
    state = dict(weights)
    plan = []
    for name, layer in compressed_layers(model):
        index, projections = projection_sources(name)
        books = {}
        for bits in layer.quant_method.precisions:
            pieces = []
            for projection in projections:
                suffix = f'.layers.{index}.{projection}.'
                sources = {}
                for key in ('delta', f'base_lut{bits}'):
                    candidates = [value for source, value in state.items() if source.endswith(suffix + key)]
                    if len(candidates) != 1:
                        raise ValueError(f'Expected exactly one actor tensor for {suffix}{key}')
                    sources[key] = candidates[0]
                base = sources[f'base_lut{bits}']
                delta = sources['delta']
                if delta.ndim != 2 or base.shape != (delta.shape[0], 2 ** bits):
                    raise ValueError(f'Invalid shared codebook shape for {suffix}')
                pieces.append(base.float() + delta.float().reshape(delta.shape[0], 2 ** bits, -1).mean(-1))
            book = torch.cat(pieces, dim=0)
            if book.shape != getattr(layer, f'lut{bits}').shape or not torch.isfinite(book).all():
                raise ValueError(f'Invalid rollout codebook for {name} at {bits} bits')
            books[bits] = book
        index_pieces = []
        for projection in projections:
            suffix = f'.layers.{index}.{projection}.indices'
            candidates = [value for source, value in state.items() if source.endswith(suffix)]
            if len(candidates) != 1:
                raise ValueError(f'Expected exactly one actor tensor for {suffix}')
            indices = candidates[0]
            if indices.ndim != 2 or indices.dtype != torch.uint8:
                raise ValueError(f'Invalid actor indices for {suffix}')
            index_pieces.append(pack_indices(indices, max(layer.quant_method.precisions)))
        packed = torch.cat(index_pieces, dim=1)
        if packed.shape != layer.qweight.shape:
            raise ValueError(f'Invalid rollout index shape for {name}')
        plan.append((layer, packed, books))
    for layer, packed, books in plan:
        layer.quant_method.load_compressed(layer, packed, books)
    return len(plan)