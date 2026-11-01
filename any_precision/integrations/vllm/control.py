import re

import torch

from .quantization import AnyPrecisionLinearMethod


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
        match = re.search(r'\.layers\.(\d+)\.mlp\.(gate_up_proj|gate_proj|up_proj|down_proj)$', name)
        if match is None:
            raise ValueError(f'Unsupported compressed layer: {name}')
        projections = ('gate_proj', 'up_proj') if match.group(2) == 'gate_up_proj' else (match.group(2),)
        books = {}
        for bits in layer.quant_method.precisions:
            pieces = []
            for projection in projections:
                suffix = f'.layers.{match.group(1)}.mlp.{projection}.'
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
        plan.append((layer, books))
    for layer, books in plan:
        layer.quant_method.update_codebooks(layer, books)
    return len(plan)