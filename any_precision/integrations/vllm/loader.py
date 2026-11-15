from pathlib import Path
from copy import deepcopy
import re

import torch
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from .packed import pack_indices
from .quantization import AnyPrecisionLinearMethod


@register_model_loader('anyprecision')
class AnyPrecisionModelLoader(DefaultModelLoader):
    def __init__(self, load_config):
        self.anyprecision_options = dict(load_config.model_loader_extra_config)
        base_config = deepcopy(load_config)
        base_config.model_loader_extra_config = {}
        base_config.load_format = 'auto'
        super().__init__(base_config)

    def load_weights(self, model, model_config):
        options = self.anyprecision_options
        cache = Path(options['cache_path'])
        precision = int(options.get('precision', 8))
        weights = self.get_all_weights(model_config, model)
        loaded = model.load_weights((name, value) for name, value in weights
                                    if not re.search(r'\.mlp\.(gate_proj|up_proj|down_proj)\.weight$', name))
        compressed_names = set()
        layer_cache = {}
        for name, layer in model.named_modules():
            method = getattr(layer, 'quant_method', None)
            if not isinstance(method, AnyPrecisionLinearMethod):
                continue
            match = re.search(r'\.layers\.(\d+)\.mlp\.(gate_up_proj|gate_proj|up_proj|down_proj)$', name)
            if match is None:
                raise ValueError(f'Unsupported compressed layer: {name}')
            index = int(match.group(1))
            if index not in layer_cache:
                indices = torch.load(cache / 'weights' / f'l{index}.pt', map_location='cpu', weights_only=False)
                books = {bits: torch.load(cache / f'lut_{bits}' / f'l{index}.pt',
                                          map_location='cpu', weights_only=False)
                         for bits in method.precisions}
                layer_cache[index] = indices, books
            indices, books = layer_cache[index]
            projections = ('gate_proj', 'up_proj') if match.group(2) == 'gate_up_proj' else (match.group(2),)
            packed = torch.cat([pack_indices(torch.as_tensor(indices[f'mlp.{projection}'][:, 0, :]),
                                             max(method.precisions))
                                for projection in projections], dim=1)
            codebooks = {bits: torch.cat([torch.as_tensor(books[bits][f'mlp.{projection}'][:, 0, :])
                                         for projection in projections], dim=0)
                         for bits in method.precisions}
            method.load_compressed(layer, packed, codebooks)
            method.set_precision(layer, precision)
            compressed_names.update(f'{name}.{suffix}' for suffix in ('qweight', *(f'lut{bits}' for bits in method.precisions)))
        if not compressed_names:
            raise RuntimeError('No AnyPrecision layers were installed.')
        missing = set(dict(model.named_parameters())) - set(loaded) - compressed_names
        if missing:
            raise RuntimeError(f'Unloaded model parameters: {sorted(missing)}')