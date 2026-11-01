import argparse
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


spec = importlib.util.spec_from_file_location(
    'trainable', Path(__file__).resolve().parents[1] / 'any_precision/trainable.py')
trainable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    torch.manual_seed(7)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='cpu', local_files_only=True)
    trainable.install_trainable_qwen(model, args.cache)
    model.to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    batch = tokenizer('The capital of France is Paris. Two plus two equals four.', return_tensors='pt').to('cuda')
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-4, weight_decay=0)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
    metrics = {'trainable_tensors': len(parameters),
               'trainable_parameters': sum(parameter.numel() for parameter in parameters.values()),
               'loss_before': {}, 'loss_after': {}, 'gradient_norm': {}, 'generated': {}}
    assert len(parameters) == 72
    optimizer.zero_grad(set_to_none=True)
    for bits in (4, 8):
        with trainable.student_precision(model, bits):
            loss = model(**batch, labels=batch.input_ids, use_cache=False).loss
            assert torch.isfinite(loss)
            metrics['loss_before'][bits] = loss.item()
            (loss / 2).backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in parameters.values())
            metrics['gradient_norm'][bits] = sum(parameter.grad.float().norm().item()
                                                for parameter in parameters.values())
    optimizer.step()
    metrics['updated_tensors'] = sum(not torch.equal(initial[name], parameter.detach().cpu())
                                     for name, parameter in parameters.items())
    assert metrics['updated_tensors'] == 72
    with torch.no_grad():
        for bits in (4, 8):
            with trainable.student_precision(model, bits):
                metrics['loss_after'][bits] = model(**batch, labels=batch.input_ids, use_cache=False).loss.item()
                generated = model.generate(**batch, max_new_tokens=8, do_sample=False)
                metrics['generated'][bits] = tokenizer.decode(generated[0, batch.input_ids.shape[1]:])
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format': 'anyprecision-shared-delta-v1', 'base_model': args.model,
                'quantization_cache': args.cache, 'precisions': [4, 8],
                'deltas': {name: parameter.detach().cpu() for name, parameter in parameters.items()},
                'optimizer': optimizer.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    with torch.no_grad():
        for parameter in parameters.values():
            parameter.zero_()
        for name, parameter in parameters.items():
            parameter.copy_(saved['deltas'][name])
        for bits in (4, 8):
            with trainable.student_precision(model, bits):
                restored = model(**batch, labels=batch.input_ids, use_cache=False).loss.item()
                assert abs(restored - metrics['loss_after'][bits]) < 1e-5
    metrics['checkpoint_reload'] = 'passed'
    metrics['peak_cuda_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()