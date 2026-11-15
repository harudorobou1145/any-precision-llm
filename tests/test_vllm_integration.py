import unittest
import os
import asyncio
import time
from types import SimpleNamespace

import numpy as np

import torch

from any_precision.integrations.vllm.control import compressed_layers, set_precision, update_from_actor
from any_precision.integrations.vllm.packed import compressed_linear, decode_weight, pack_indices
from any_precision.integrations.vllm.quantization import AnyPrecisionLinearMethod


class VllmIntegrationTests(unittest.TestCase):
    def test_w2_bf16_decode_and_cache_refresh(self):
        for parent in (2, 8):
            indices = torch.randint(0, 2 ** parent, (32, 160), dtype=torch.uint8)
            packed = pack_indices(indices, parent)
            for bits in sorted({2, parent}):
                book = torch.randn(32, 2 ** bits).bfloat16()
                expected = book.gather(1, indices.long() >> (parent - bits))
                torch.testing.assert_close(decode_weight(packed, book, bits, torch.bfloat16), expected)
        layer = torch.nn.Module()
        method = AnyPrecisionLinearMethod((2, 4, 8))
        method.create_weights(layer, 160, [32], 160, 32, torch.bfloat16)
        indices = torch.randint(0, 256, (32, 160), dtype=torch.uint8)
        books = {bits: torch.randn(32, 2 ** bits).bfloat16() for bits in (2, 4, 8)}
        method.load_compressed(layer, pack_indices(indices), books)
        inputs = torch.randn(3, 160).bfloat16()
        for bits in (2, 4, 8):
            method.set_precision(layer, bits)
            expected = torch.nn.functional.linear(inputs, books[bits].gather(1, indices.long() >> (8 - bits)))
            torch.testing.assert_close(method.apply(layer, inputs), expected)
        self.assertEqual(len(layer.anyprecision_dense_cache), 3)
        method.update_codebooks(layer, {bits: book + 1 for bits, book in books.items()})
        self.assertFalse(layer.anyprecision_dense_cache)
        expected = torch.nn.functional.linear(inputs, (books[8] + 1).gather(1, indices.long()))
        torch.testing.assert_close(method.apply(layer, inputs), expected)

    @unittest.skipUnless(os.environ.get('ANYPRECISION_MODEL_TEST') == '1', 'explicit full-model check')
    def test_real_qwen_next_token_matches_hf(self):
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from vllm import LLM, SamplingParams
        from any_precision.trainable import install_trainable_qwen, student_precision

        model_path = os.environ['ANYPRECISION_MODEL']
        cache = os.environ['ANYPRECISION_CACHE']
        precisions = [int(value) for value in os.environ.get('ANYPRECISION_TEST_BITS', '4,8').split(',')]
        dtype_name = os.environ.get('ANYPRECISION_TEST_DTYPE', 'float16')
        dtype = getattr(torch, dtype_name)
        engine = LLM(model=model_path, quantization='anyprecision', load_format='anyprecision',
                 model_loader_extra_config={'cache_path': cache, 'precision': max(precisions)}, dtype=dtype_name,
                 hf_overrides={'quantization_config': {'quant_method': 'anyprecision', 'precisions': precisions}},
                     tensor_parallel_size=1, enforce_eager=True, max_model_len=256,
                     max_num_seqs=8, max_num_batched_tokens=256, gpu_memory_utilization=0.15,
                     enable_prefix_caching=False, limit_mm_per_prompt={'image': 0, 'video': 0})
        reference = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation='sdpa').cuda().eval()
        install_trainable_qwen(reference, cache, precisions=precisions)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        token_ids = tokenizer('Two plus two equals', add_special_tokens=False)['input_ids']
        for bits in precisions:
            engine.reset_prefix_cache()
            engine.apply_model(lambda net: set_precision(net, bits))
            result = engine.generate([{'prompt_token_ids': token_ids}],
                                     SamplingParams(temperature=0, max_tokens=1, logprobs=5), use_tqdm=False)
            with torch.no_grad(), student_precision(reference, bits):
                logits = reference(input_ids=torch.tensor([token_ids], device='cuda')).logits[0, -1].float()
            expected_token = logits.argmax().item()
            output = result[0].outputs[0]
            self.assertEqual(output.token_ids[0], expected_token, f'{bits}-bit greedy token mismatch')
            expected_logprob = logits.log_softmax(-1)[expected_token].item()
            self.assertAlmostEqual(output.logprobs[0][expected_token].logprob, expected_logprob, delta=0.1)
        if os.environ.get('ANYPRECISION_BENCHMARK') == '1':
            for bits in (2, 4):
                def clear_cache(net):
                    for layer in net.modules():
                        getattr(layer, 'anyprecision_dense_cache', {}).clear()
                engine.apply_model(clear_cache)
                engine.apply_model(lambda net: set_precision(net, bits))
                torch.cuda.synchronize()
                started = time.monotonic()
                engine.generate([{'prompt_token_ids': token_ids}] * 8,
                                SamplingParams(temperature=0, max_tokens=128, min_tokens=128,
                                               ignore_eos=True), use_tqdm=False)
                vllm_seconds = time.monotonic() - started
                inputs = torch.tensor([token_ids] * 8, device='cuda')
                torch.cuda.synchronize()
                started = time.monotonic()
                with torch.no_grad(), student_precision(reference, bits):
                    reference.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                       do_sample=False, max_new_tokens=128, min_new_tokens=128,
                                       eos_token_id=None, pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                hf_seconds = time.monotonic() - started
                print(f'BENCHMARK W{bits} batch8 tokens128: vllm={vllm_seconds:.3f}s '
                      f'hf={hf_seconds:.3f}s speedup={hf_seconds / vllm_seconds:.2f}x', flush=True)
        if os.environ.get('ANYPRECISION_MOPD_TEST') == '1':
            from tensordict import TensorDict
            from verl import DataProto
            from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

            rollout = object.__new__(vLLMRollout)
            rollout.anyprecision = True
            rollout.inference_engine = engine
            rollout.config = SimpleNamespace(response_length=2, calculate_log_probs=False)
            rollout.sampling_params = SamplingParams(temperature=0, max_tokens=2, detokenize=False)
            rollout.pad_token_id = tokenizer.pad_token_id
            rollout.lora_kwargs = {}
            prompts = DataProto(batch=TensorDict({
                'input_ids': torch.tensor([token_ids, token_ids], device='cuda'),
                'attention_mask': torch.ones(2, len(token_ids), dtype=torch.long, device='cuda'),
                'position_ids': torch.arange(len(token_ids), device='cuda')[None].repeat(2, 1),
            }, batch_size=2), non_tensor_batch={'student_bits': np.array([8, 4])},
                meta_info={'eos_token_id': tokenizer.eos_token_id, 'do_sample': True})
            generated = rollout.generate_sequences(prompts)
            self.assertEqual(generated.batch['responses'].shape, (2, 2))
            self.assertEqual(generated.non_tensor_batch['student_bits'].tolist(), [8, 4])
            asyncio.run(rollout.update_weights(reference.state_dict().items()))
            for _, layer in compressed_layers(engine.llm_engine.model_executor.driver_worker.worker.model_runner.model):
                self.assertTrue(layer.anyprecision_ready)

    def make_model(self, device='cpu'):
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        model.model.layers[0].mlp = torch.nn.Module()
        layer = torch.nn.Module()
        method = AnyPrecisionLinearMethod((4, 8))
        method.create_weights(layer, 32, [16, 16], 32, 32, torch.float16)
        layer.quant_method = method
        model.model.layers[0].mlp.gate_up_proj = layer
        return model.to(device), layer

    def test_actor_update_and_atomic_validation(self):
        model, layer = self.make_model()
        state = {}
        for projection in ('gate_proj', 'up_proj'):
            prefix = f'model.language_model.layers.0.mlp.{projection}.'
            state[prefix + 'delta'] = torch.ones(16, 256)
            for bits in (4, 8):
                state[prefix + f'base_lut{bits}'] = torch.full((16, 2 ** bits), 2.0)
        pointer = layer.qweight.data_ptr()
        self.assertEqual(update_from_actor(model, state.items()), 1)
        for bits in (4, 8):
            set_precision(model, bits)
            self.assertEqual(layer.anyprecision_bits, bits)
            torch.testing.assert_close(getattr(layer, f'lut{bits}'), torch.full((32, 2 ** bits), 3.0).half())
        self.assertEqual(pointer, layer.qweight.data_ptr())
        state.pop('model.language_model.layers.0.mlp.up_proj.base_lut8')
        snapshot = layer.lut4.clone()
        with self.assertRaises(ValueError):
            update_from_actor(model, state.items())
        torch.testing.assert_close(layer.lut4, snapshot)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_packed_matmul_on_nondefault_stream(self):
        torch.manual_seed(7)
        indices = torch.randint(0, 256, (32, 1024), dtype=torch.uint8, device='cuda')
        packed = pack_indices(indices)
        inputs = torch.randn(17, 1024, device='cuda', dtype=torch.float16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for bits in (4, 8):
                book = torch.randn(32, 2 ** bits, device='cuda', dtype=torch.float16) * 0.01
                actual = compressed_linear(inputs, packed, book, bits)
                expected = torch.nn.functional.linear(inputs, book.gather(1, indices.long() >> (8 - bits)))
                torch.testing.assert_close(actual, expected, atol=0.006, rtol=0.02)
        torch.cuda.current_stream().wait_stream(stream)


if __name__ == '__main__':
    unittest.main()