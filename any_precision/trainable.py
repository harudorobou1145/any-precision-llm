from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional


class TrainableAnyPrecisionLinear(nn.Module):
    """Fixed prefix indices with a shared, differentiable codebook update."""

    def __init__(self, indices, codebooks, bias=None):
        super().__init__()
        self.precisions = sorted(codebooks)
        self.parent_bits = max(self.precisions)
        self.precision = self.parent_bits
        if self.parent_bits > 8 or min(self.precisions) < 1:
            raise ValueError('Only 1 through 8 bit indices are supported.')
        if indices.ndim != 2 or indices.numel() == 0:
            raise ValueError('Indices must be a nonempty matrix.')
        if indices.min().item() < 0 or indices.max().item() >= 2 ** self.parent_bits:
            raise ValueError('Indices are outside the parent codebook.')
        self.out_features, self.in_features = indices.shape
        self.register_buffer('indices', indices.to(torch.uint8))
        for bits, values in codebooks.items():
            if values.shape != (self.out_features, 2 ** bits):
                raise ValueError('Codebook shape does not match indices and precision.')
            self.register_buffer(f'base_lut{bits}', values.detach().clone())
        self.delta = nn.Parameter(torch.zeros_like(codebooks[self.parent_bits], dtype=torch.float32))
        self.register_buffer('bias', None if bias is None else bias.detach().clone())

    def codebook(self, bits):
        if bits not in self.precisions:
            raise ValueError(f'Unsupported precision: {bits}')
        delta = self.delta.reshape(self.out_features, 2 ** bits, -1).mean(dim=-1)
        return getattr(self, f'base_lut{bits}').float() + delta

    def forward(self, inputs):
        indices = self.indices.long() >> (self.parent_bits - self.precision)
        weight = self.codebook(self.precision).gather(1, indices).to(inputs.dtype)
        bias = None if self.bias is None else self.bias.to(inputs.dtype)
        return functional.linear(inputs, weight, bias)


def install_trainable_qwen(model, cache_path, precisions=(4, 8)):
    """Install MLP codebooks from an existing group-count-one PTQ cache."""
    cache_path = Path(cache_path)
    precisions = list(precisions)
    available = sorted(int(path.name[4:]) for path in cache_path.glob('lut_*'))
    if not available or not precisions or not set(precisions).issubset(available):
        raise ValueError('Requested precisions are missing from the quantization cache.')
    parent_bits = max(available)
    if parent_bits not in precisions:
        raise ValueError('Include the parent precision to define the shared codebook.')
    if model.config.model_type != 'qwen3_5':
        raise ValueError('This installer supports the Qwen3.5 MLP adaptation only.')
    layers = model.model.language_model.layers
    replacements = []
    for layer_index, layer in enumerate(layers):
        indices = torch.load(cache_path / 'weights' / f'l{layer_index}.pt', weights_only=False)
        luts = {bits: torch.load(cache_path / f'lut_{bits}' / f'l{layer_index}.pt',
                                 weights_only=False) for bits in precisions}
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            key = f'mlp.{name}'
            original = getattr(layer.mlp, name)
            index = torch.as_tensor(indices[key])
            if index.shape != (original.out_features, 1, original.in_features):
                raise ValueError(f'Unsupported index shape for layer {layer_index} {key}.')
            codebooks = {bits: torch.as_tensor(luts[bits][key])[:, 0, :] for bits in precisions}
            replacement = TrainableAnyPrecisionLinear(index[:, 0, :], codebooks, original.bias)
            replacements.append((layer.mlp, name, replacement.to(original.weight.device)))
    model.requires_grad_(False)
    for parent, name, replacement in replacements:
        setattr(parent, name, replacement)
    return model


@contextmanager
def student_precision(model, bits):
    """Keep this context active through backward when using checkpointing."""
    modules = [module for module in model.modules() if isinstance(module, TrainableAnyPrecisionLinear)]
    if not modules or any(bits not in module.precisions for module in modules):
        raise ValueError(f'No compatible trainable student at precision {bits}.')
    previous = [module.precision for module in modules]
    try:
        for module in modules:
            module.precision = bits
        yield model
    finally:
        for module, precision in zip(modules, previous):
            module.precision = precision