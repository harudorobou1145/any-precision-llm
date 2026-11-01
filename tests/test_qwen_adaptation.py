import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import yaml
from transformers import LlamaConfig, AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def load_source(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QwenAdaptationTests(unittest.TestCase):
    def test_model_class_selection(self):
        utils = load_source('loader_utils', 'any_precision/analyzer/utils.py')
        self.assertIs(utils.get_model_class(LlamaConfig()), AutoModelForCausalLM)
        from transformers import Qwen3_5Config
        self.assertEqual(utils.get_model_class(Qwen3_5Config()).__name__,
                         'Qwen3_5ForConditionalGeneration')
        with torch.device('meta'):
            model = utils.build_model(Qwen3_5Config(), torch.bfloat16)
        self.assertEqual(model.__class__.__name__, 'Qwen3_5ForConditionalGeneration')
        self.assertEqual(model.model.language_model.embed_tokens.weight.dtype,
                         torch.bfloat16)

    def test_explicit_mlp_config(self):
        path = ROOT / 'any_precision/analyzer/architectures/Qwen3_5ForConditionalGeneration.yaml'
        config = yaml.safe_load(path.read_text())
        self.assertEqual(config['arch_config']['model_name'], 'model.language_model')
        self.assertEqual(config['arch_config']['module_names'],
                         ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'])

    def test_linear_dtype_boundary(self):
        module = load_source('ap_linear', 'any_precision/modules/AnyPrecisionLinear.py')
        for token_count in (1, 9):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(token_count=token_count, dtype=dtype):
                    def matmul(inputs, weights, lut, bits):
                        self.assertEqual(inputs.dtype, torch.float16)
                        self.assertEqual(lut.dtype, torch.float16)
                        return inputs * 2

                    def dequant(weights, lut, bits):
                        return torch.eye(32, dtype=torch.float16) * 2

                    with patch.object(module, 'matmul_kbit', matmul), \
                            patch.object(module, 'dequant_kbit', dequant):
                        linear = module.AnyPrecisionLinear(32, 32, [4, 8],
                                                          bias=False, dtype=torch.float16)
                        inputs = torch.ones(token_count, 32, dtype=dtype)
                        outputs = linear(inputs)
                        self.assertEqual(outputs.dtype, dtype)
                        torch.testing.assert_close(outputs, inputs * 2)


if __name__ == '__main__':
    unittest.main()