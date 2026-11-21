from pathlib import Path
from copy import deepcopy

import torch
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from .packed import pack_indices
from .quantization import AnyPrecisionLinearMethod
from .control import projection_sources


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
        source_names = set()
        for name, layer in model.named_modules():
            if isinstance(getattr(layer, 'quant_method', None), AnyPrecisionLinearMethod):
                index, projections = projection_sources(name)
                source_names.update(f'.layers.{index}.{projection}.weight' for projection in projections)
        for path in (cache / 'weights').glob('l*.pt'):
            index = int(path.stem[1:])
            cached = torch.load(path, map_location='cpu', weights_only=False)
            expected = {f'.layers.{index}.{key}.weight' for key in cached}
            actual = {suffix for suffix in source_names if suffix.startswith(f'.layers.{index}.')}
            if expected != actual:
                raise ValueError(f'Cache and vLLM projection coverage differ at layer {index}; '
                                 'set full_projection to match the cache.')
        weights = self.get_all_weights(model_config, model)
        loaded = model.load_weights((name, value) for name, value in weights
                                    if not any(name.endswith(suffix) for suffix in source_names))
        compressed_names = set()
        layer_cache = {}
        for name, layer in model.named_modules():
            method = getattr(layer, 'quant_method', None)
            if not isinstance(method, AnyPrecisionLinearMethod):
                continue
            index, projections = projection_sources(name)
            index = int(index)
            if index not in layer_cache:
                indices = torch.load(cache / 'weights' / f'l{index}.pt', map_location='cpu', weights_only=False)
                books = {bits: torch.load(cache / f'lut_{bits}' / f'l{index}.pt',
                                          map_location='cpu', weights_only=False)
                         for bits in method.precisions}
                layer_cache[index] = indices, books
            indices, books = layer_cache[index]
            packed = torch.cat([pack_indices(torch.as_tensor(indices[projection][:, 0, :]),
                                             max(method.precisions))
                                for projection in projections], dim=1)
            codebooks = {bits: torch.cat([torch.as_tensor(books[bits][projection][:, 0, :])
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