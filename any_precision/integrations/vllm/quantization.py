import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase

from .packed import compressed_linear, decode_weight


@register_quantization_config('anyprecision')
class AnyPrecisionConfig(QuantizationConfig):
    def __init__(self, precisions=(4, 8), full_projection=False):
        super().__init__()
        self.full_projection = full_projection
        self.precisions = sorted(set(precisions))
        if not self.precisions or any(bits < 2 or bits > 8 for bits in self.precisions):
            raise ValueError('Precisions must be between 2 and 8.')

    def get_name(self):
        return 'anyprecision'

    def get_supported_act_dtypes(self):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 70

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls(config.get('precisions', (4, 8)), config.get('full_projection', False))

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            targets = ('.mlp.gate_up_proj', '.mlp.gate_proj', '.mlp.up_proj', '.mlp.down_proj')
            if self.full_projection:
                targets += ('.linear_attn.in_proj_qkvz', '.linear_attn.out_proj',
                            '.self_attn.qkv_proj', '.self_attn.o_proj')
            if prefix.endswith(targets):
                return AnyPrecisionLinearMethod(self.precisions)
            return UnquantizedLinearMethod()
        return None


class AnyPrecisionLinearMethod(QuantizeMethodBase):
    def __init__(self, precisions):
        self.precisions = tuple(precisions)

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        output_width = sum(output_partition_sizes)
        if input_size_per_partition != input_size or output_width != output_size:
            raise ValueError('AnyPrecision currently requires tensor_parallel_size=1.')
        if input_size % 32 or output_width % 16:
            raise ValueError('Unsupported AnyPrecision kernel layer dimensions.')
        layer.register_parameter('qweight', torch.nn.Parameter(torch.empty(
            max(self.precisions), output_width, input_size // 32, dtype=torch.int32), requires_grad=False))
        for bits in self.precisions:
            layer.register_parameter(f'lut{bits}', torch.nn.Parameter(torch.empty(
                output_width, 2 ** bits, dtype=params_dtype), requires_grad=False))
        layer.anyprecision_bits = max(self.precisions)
        layer.anyprecision_ready = False
        layer.anyprecision_dense_cache = {}

    @torch.no_grad()
    def load_compressed(self, layer, qweight, codebooks):
        if qweight.shape != layer.qweight.shape or qweight.dtype != torch.int32:
            raise ValueError('Packed weight shape or dtype mismatch.')
        for bits in self.precisions:
            if bits not in codebooks or codebooks[bits].shape != getattr(layer, f'lut{bits}').shape:
                raise ValueError('Missing or incompatible codebook.')
        layer.qweight.copy_(qweight)
        self.update_codebooks(layer, codebooks)
        layer.anyprecision_ready = True

    @torch.no_grad()
    def update_codebooks(self, layer, codebooks):
        for bits in self.precisions:
            if bits not in codebooks or codebooks[bits].shape != getattr(layer, f'lut{bits}').shape:
                raise ValueError('Missing or incompatible codebook.')
        for bits in self.precisions:
            getattr(layer, f'lut{bits}').copy_(codebooks[bits])
        layer.anyprecision_dense_cache.clear()

    def set_precision(self, layer, bits):
        if bits not in self.precisions:
            raise ValueError('Unsupported student precision.')
        layer.anyprecision_bits = bits

    def apply(self, layer, inputs, bias=None):
        if not layer.anyprecision_ready:
            raise RuntimeError('Load AnyPrecision bitplanes and codebooks before inference.')
        bits = layer.anyprecision_bits
        if bits == 2 or inputs.dtype == torch.bfloat16:
            key = (bits, inputs.dtype)
            if key not in layer.anyprecision_dense_cache:
                layer.anyprecision_dense_cache[key] = decode_weight(
                    layer.qweight, getattr(layer, f'lut{bits}'), bits, inputs.dtype)
            return torch.nn.functional.linear(inputs, layer.anyprecision_dense_cache[key], bias)
        return compressed_linear(inputs, layer.qweight, getattr(layer, f'lut{bits}'), bits, bias)