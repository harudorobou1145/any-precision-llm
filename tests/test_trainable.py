import importlib.util
from pathlib import Path
import unittest

import torch
from torch.nn import functional as functional


spec = importlib.util.spec_from_file_location(
    'trainable', Path(__file__).resolve().parents[1] / 'any_precision/trainable.py')
trainable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainable)


class TrainableTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()