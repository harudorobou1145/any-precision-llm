import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as functional


spec = importlib.util.spec_from_file_location(
    'trainable', Path(__file__).resolve().parents[1] / 'any_precision/trainable.py')
trainable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainable)


class TrainableTests(unittest.TestCase):
    def test_bf16_hierarchical_reassignment(self):
        books = {bits: torch.arange(2 ** bits).float()[None, :] / (2 ** bits - 1)
                 for bits in range(2, 9)}
        target = torch.tensor([[0.0, 0.18, 0.51, 0.99]])
        layer = trainable.TrainableAnyPrecisionLinear(torch.zeros(1, 4), books)
        trainable.reassign_bf16_indices(layer, target, chunk_rows=1)
        expected = (books[2][:, :, None] - target[:, None, :]).square().argmin(1)
        self.assertTrue(torch.equal(layer.indices.long() >> 6, expected))
        for bits in range(3, 9):
            left = expected * 2
            expected = left + ((books[bits].gather(1, left + 1) - target).square() <
                               (books[bits].gather(1, left) - target).square()).long()
            self.assertTrue(torch.equal(layer.indices.long() >> (8 - bits), expected))
        self.assertEqual(trainable.reassign_bf16_indices(layer, target), 0)

    def test_full_projection_install_and_reload(self):
        def build():
            model = torch.nn.Module()
            model.config = SimpleNamespace(model_type='qwen3_5')
            model.model = torch.nn.Module()
            model.model.language_model = torch.nn.Module()
            model.model.language_model.layers = torch.nn.ModuleList()
            for branch, names in [('linear_attn', ('in_proj_qkv', 'in_proj_z', 'out_proj')),
                                  ('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj'))]:
                layer = torch.nn.Module()
                layer.mlp = torch.nn.ModuleDict({name: torch.nn.Linear(4, 2)
                                                for name in ('gate_proj', 'up_proj', 'down_proj')})
                setattr(layer, branch, torch.nn.ModuleDict({name: torch.nn.Linear(4, 2) for name in names}))
                model.model.language_model.layers.append(layer)
            return model

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            for folder in ('weights', 'lut_2', 'lut_4', 'lut_8'):
                (cache / folder).mkdir()
            model = build()
            for index, layer in enumerate(model.model.language_model.layers):
                keys = [name for name, module in layer.named_modules() if isinstance(module, torch.nn.Linear)]
                torch.save({key: torch.zeros(2, 1, 4, dtype=torch.uint8) for key in keys},
                           cache / 'weights' / f'l{index}.pt')
                for bits in (2, 4, 8):
                    torch.save({key: torch.randn(2, 1, 2 ** bits) for key in keys},
                               cache / f'lut_{bits}' / f'l{index}.pt')
            trainable.install_trainable_qwen(model, cache, (2, 4, 8))
            modules = [module for module in model.modules()
                       if isinstance(module, trainable.TrainableAnyPrecisionLinear)]
            self.assertEqual(len(modules), 13)
            for bits in (2, 4, 8):
                with trainable.student_precision(model, bits):
                    sum(module(torch.ones(1, 4)).square().mean() for module in modules).backward()
            self.assertTrue(all(module.delta.grad is not None for module in modules))
            restored = trainable.install_trainable_qwen(build(), cache, (2, 4, 8))
            restored.load_state_dict(model.state_dict(), strict=True)
            broken = torch.load(cache / 'weights/l0.pt', weights_only=True)
            broken.pop('linear_attn.out_proj')
            torch.save(broken, cache / 'weights/l0.pt')
            with self.assertRaisesRegex(ValueError, 'coverage'):
                trainable.install_trainable_qwen(build(), cache, (2, 4, 8))

    def test_initial_weights_and_shared_update(self):
        torch.manual_seed(7)
        indices = torch.tensor([[0, 1, 15, 16, 127, 255], [255, 128, 16, 15, 1, 0]], dtype=torch.uint8)
        books = {bits: torch.randn(2, 2 ** bits) for bits in (4, 8)}
        layer = trainable.TrainableAnyPrecisionLinear(indices, books)
        inputs = torch.randn(3, 6, requires_grad=True)
        gradients = []
        outputs = {}
        for bits in (4, 8):
            with trainable.student_precision(layer, bits):
                outputs[bits] = layer(inputs).detach()
                expected = functional.linear(inputs, books[bits].gather(1, indices.long() >> (8 - bits)))
                torch.testing.assert_close(layer(inputs), expected)
                gradients.append(torch.autograd.grad(layer(inputs).square().mean(), layer.delta)[0])
                self.assertGreater(gradients[-1].abs().sum().item(), 0)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        for bits in (4, 8):
            with trainable.student_precision(layer, bits):
                layer(inputs).square().mean().backward()
        torch.testing.assert_close(layer.delta.grad, sum(gradients))
        optimizer.step()
        self.assertEqual(list(dict(layer.named_parameters())), ['delta'])
        for bits in (4, 8):
            with trainable.student_precision(layer, bits):
                self.assertFalse(torch.equal(layer(inputs), outputs[bits]))

    def test_precision_restored_after_error(self):
        layer = trainable.TrainableAnyPrecisionLinear(torch.zeros(1, 2),
                                                      {4: torch.zeros(1, 16), 8: torch.zeros(1, 256)})
        with self.assertRaises(RuntimeError):
            with trainable.student_precision(layer, 4):
                raise RuntimeError('probe')
        self.assertEqual(layer.precision, 8)
        with self.assertRaises(ValueError):
            with trainable.student_precision(layer, 3):
                pass

    def test_fixed_prefix_projection_and_trust_region(self):
        indices = torch.tensor([[0, 64, 128, 192]], dtype=torch.uint8)
        books = {bits: (torch.arange(2 ** bits).float() / (2 ** bits) + 1)[None]
                 for bits in (2, 4, 8)}
        layer = trainable.TrainableAnyPrecisionLinear(indices, books)
        before2 = layer.codebook(2).gather(1, indices.long() >> 6)
        state = {}
        updater = trainable.FixedPrefixUpdater(layer, state, lr=0.1, trust_ratio=0.2, max_fraction=1)
        updater.accumulate(-torch.ones(1, 4))
        metrics = updater.step()
        self.assertGreater(metrics['changed'], 0)
        torch.testing.assert_close(layer.indices >> 6, indices >> 6)
        torch.testing.assert_close(layer.codebook(2).gather(1, layer.indices.long() >> 6), before2)
        self.assertLessEqual(metrics['relative_w4'], 0.2)
        self.assertLessEqual(metrics['relative_w8'], 0.2)
        restricted = trainable.FixedPrefixUpdater(layer, {}, lr=1, trust_ratio=1e-9, max_fraction=1)
        previous = layer.indices.clone()
        restricted.accumulate(-torch.ones(1, 4))
        self.assertEqual(restricted.step()['changed'], 0)
        torch.testing.assert_close(layer.indices, previous)

    def test_index_gradient_capture_and_adam_state(self):
        books = {bits: torch.ones(1, 2 ** bits) for bits in (2, 4, 8)}
        layer = trainable.TrainableAnyPrecisionLinear(torch.zeros(1, 2), books)
        updater = trainable.FixedPrefixUpdater(layer, {})
        layer._prefix_updater = updater
        for bits in (2, 4):
            with trainable.student_precision(layer, bits):
                layer(torch.ones(1, 2)).sum().backward()
        torch.testing.assert_close(updater.gradient, torch.ones(1, 2))
        self.assertEqual(updater.step()['changed'], 0)
        self.assertEqual(updater.state['step'], 1)
        self.assertIsNone(updater.gradient)

    def test_prefix_checkpoint_recompute_and_optimizer_resume(self):
        from torch.utils.checkpoint import checkpoint

        books = {bits: torch.ones(1, 2 ** bits) for bits in (2, 4, 8)}
        layer = trainable.TrainableAnyPrecisionLinear(torch.zeros(1, 2), books)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)
        updater = trainable.FixedPrefixUpdater(layer, {})
        layer._prefix_updater = updater
        with trainable.student_precision(layer, 4):
            checkpoint(layer, torch.ones(1, 2, requires_grad=True), use_reentrant=True).sum().backward()
        torch.testing.assert_close(updater.gradient, torch.ones(1, 2))
        optimizer.step()
        updater.state = optimizer.state[layer.delta].setdefault('prefix_pv', {})
        updater.step()
        saved = io.BytesIO()
        torch.save(optimizer.state_dict(), saved)
        saved.seek(0)
        resumed = torch.optim.AdamW(layer.parameters(), lr=1e-4)
        resumed.load_state_dict(torch.load(saved, weights_only=True))
        restored = resumed.state[layer.delta]['prefix_pv']
        self.assertEqual(restored['step'], 1)
        torch.testing.assert_close(restored['mean'], updater.state['mean'])
        resumed.step()


if __name__ == '__main__':
    unittest.main()