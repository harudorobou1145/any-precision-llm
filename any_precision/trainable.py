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
        updater = getattr(self, '_prefix_updater', None)
        if updater is not None and self.precision == updater.bits and weight.requires_grad:
            weight.register_hook(updater.accumulate)
        bias = None if self.bias is None else self.bias.to(inputs.dtype)
        return functional.linear(inputs, weight, bias)


@torch.no_grad()
def protective_sibling_step(layer, bits, before, columns, apply=False, original=None):
    if not 3 <= bits <= layer.parent_bits or before.shape != (layer.out_features, 1 << bits):
        raise ValueError('Invalid protective sibling codebook')
    after = layer.codebook(bits)
    shift = layer.parent_bits - bits
    selected = layer.indices.index_select(1, columns)
    prefix = selected.long() >> shift
    reference = before.gather(1, prefix if original is None else original.long() >> shift)
    flips = (after.gather(1, prefix ^ 1) - reference).abs() < (after.gather(1, prefix) - reference).abs()
    eligible = flips.sum()
    actual = eligible.new_zeros(())
    if apply:
        layer.indices.index_copy_(1, columns, selected ^ (flips.byte() << shift))
        actual = (selected != layer.indices.index_select(1, columns)).sum()
    return eligible, actual


class ProtectiveOutProj:
    def __init__(self, model, fraction, interval, scope='out_proj'):
        if scope not in ('out_proj', 'all_quantized'):
            raise ValueError(f'Unknown protective scope: {scope}')
        self.layers = [module for name, module in model.named_modules()
                       if isinstance(module, TrainableAnyPrecisionLinear)
                       and (scope == 'all_quantized' or name.endswith('linear_attn.out_proj'))]
        if not self.layers or not 0 < fraction <= 1 or interval < 1:
            raise ValueError(f'Protective flips require quantized layers in {scope}, a valid fraction and interval')
        if any(layer.precisions != list(range(2, 9)) for layer in self.layers):
            raise ValueError('Protective flips require W2 through W8 codebooks')
        self.fraction, self.interval = fraction, interval
        self.steps = self.total = self.positions_total = 0
        self.columns = {}
        self.handles = []

    @torch.no_grad()
    def begin(self):
        self.sample = self.steps % self.interval == 0
        self.bits = range(3, 9)
        self.before = {layer: {bits: layer.codebook(bits) for bits in self.bits} for layer in self.layers}
        if self.sample:
            self.energy = {layer: layer.delta.new_zeros(layer.in_features) for layer in self.layers}
            for layer in self.layers:
                self.handles.append(layer.register_forward_pre_hook(self.observe))

    @torch.no_grad()
    def observe(self, layer, args):
        if layer.precision == 4:
            values = args[0].detach().reshape(-1, layer.in_features).float()
            self.energy[layer].add_(values.square().sum(0))

    @torch.no_grad()
    def finish(self, finite):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        stats = {}
        changed = 0
        positions = 0
        dist = torch.distributed
        if finite:
            for layer in self.layers:
                if self.sample:
                    energy = self.energy[layer]
                    if dist.is_initialized():
                        dist.all_reduce(energy)
                    if not energy.isfinite().all() or energy.max() <= 0:
                        raise RuntimeError('Missing or invalid W4 out_proj activations')
                    count = max(1, int(layer.in_features * self.fraction))
                    self.columns[layer] = energy.argsort(descending=True, stable=True)[:count].contiguous()
                columns = self.columns[layer]
                original = layer.indices.index_select(1, columns)
                if self.sample:
                    stats['sample_positions'] = stats.get('sample_positions', 0) + layer.out_features * len(columns)
                for bits in self.bits:
                    eligible, actual = protective_sibling_step(layer, bits, self.before[layer][bits], columns,
                                                              apply=True, original=original)
                    changed += int(actual)
                    if self.sample:
                        key = f'sample_w{bits}_eligible'
                        stats[key] = stats.get(key, 0) + int(eligible)
                positions += int((original != layer.indices.index_select(1, columns)).sum())
            self.steps += 1
            self.total += changed
            self.positions_total += positions
        self.before = self.energy = None
        return dict(stats, flips=changed, flips_total=self.total, positions_changed=positions,
                positions_changed_total=self.positions_total)


@torch.no_grad()
def reassign_bf16_indices(layer, target, chunk_rows=128):
    if layer.precisions != list(range(2, 9)) or target.shape != layer.indices.shape:
        raise ValueError('Hierarchical reassignment requires all codebooks W2 through W8 and matching targets.')
    changed = 0
    for start in range(0, layer.out_features, chunk_rows):
        stop = min(start + chunk_rows, layer.out_features)
        reference = target[start:stop].to(device=layer.indices.device, dtype=torch.float32)
        if not torch.isfinite(reference).all():
            raise ValueError('Nonfinite BF16 reference weights.')
        book = layer.codebook(2)[start:stop]
        best = (book[:, 0, None] - reference).square()
        indices = torch.zeros_like(reference, dtype=torch.long)
        for candidate in range(1, 4):
            error = (book[:, candidate, None] - reference).square()
            better = error < best
            indices = torch.where(better, candidate, indices)
            best = torch.minimum(best, error)
        for bits in range(3, 9):
            book = layer.codebook(bits)[start:stop]
            left = indices * 2
            error_left = (book.gather(1, left) - reference).square()
            error_right = (book.gather(1, left + 1) - reference).square()
            indices = left + (error_right < error_left).long()
        changed += (layer.indices[start:stop] != indices).sum().item()
        layer.indices[start:stop].copy_(indices.to(torch.uint8))
    return changed


class BF16HierarchicalReassigner:
    def __init__(self, model, model_path):
        import json
        import re
        from safetensors import safe_open

        model_path = Path(model_path)
        weight_map = json.loads((model_path / 'model.safetensors.index.json').read_text())['weight_map']
        self.targets = []
        for name, module in model.named_modules():
            if not isinstance(module, TrainableAnyPrecisionLinear):
                continue
            match = re.search(r'\.layers\.(\d+)\.(.+)$', name.replace('_fsdp_wrapped_module.', ''))
            if match is None:
                raise ValueError(f'Cannot identify BF16 projection: {name}')
            suffix = f'.layers.{match.group(1)}.{match.group(2)}.weight'
            reference_key = 'model.language_model' + suffix
            keys = [reference_key] if reference_key in weight_map else []
            if len(keys) != 1:
                raise ValueError(f'Expected one BF16 reference for {name}')
            key = keys[0]
            with safe_open(model_path / weight_map[key], framework='pt', device='cpu') as reader:
                target = reader.get_tensor(key).to(torch.bfloat16)
            if target.shape != module.indices.shape:
                raise ValueError(f'BF16 reference shape mismatch for {name}')
            self.targets.append((module, target))
        if not self.targets:
            raise ValueError('No quantized projections for hierarchical reassignment.')

    def step(self):
        return sum(reassign_bf16_indices(module, target) for module, target in self.targets)


class FixedPrefixUpdater:
    """PV-style Adam projection with a fixed W2 prefix, not an AQLM reproduction."""

    def __init__(self, layer, state, bits=4, lr=1e-4, trust_ratio=1e-3, max_fraction=0.01):
        if layer.parent_bits != 8 or bits != 4 or 2 not in layer.precisions:
            raise ValueError('Fixed-prefix updates require W2/W4/W8 codebooks.')
        if not 0 < trust_ratio <= 1 or not 0 < max_fraction <= 1 or lr <= 0:
            raise ValueError('Invalid fixed-prefix update limits.')
        self.layer, self.state, self.bits = layer, state, bits
        self.lr, self.trust_ratio, self.max_fraction = lr, trust_ratio, max_fraction
        self.gradient = None

    def accumulate(self, gradient):
        gradient = gradient.detach().float()
        self.gradient = gradient.clone() if self.gradient is None else self.gradient + gradient

    @torch.no_grad()
    def step(self):
        layer = self.layer
        gradient = self.gradient
        self.gradient = None
        if gradient is None or not torch.isfinite(gradient).all():
            raise ValueError('Missing or nonfinite W4 index gradient.')
        state = self.state
        state['step'] = state.get('step', 0) + 1
        if 'mean' not in state:
            state['mean'] = torch.zeros_like(gradient)
            state['variance'] = torch.zeros_like(gradient)
        mean, variance = state['mean'], state['variance']
        mean.mul_(0.9).add_(gradient, alpha=0.1)
        variance.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
        direction = (mean / (1 - 0.9 ** state['step'])) / (
            (variance / (1 - 0.999 ** state['step'])).sqrt() + 1e-8)
        indices = layer.indices.long()
        book4, book8 = layer.codebook(4), layer.codebook(8)
        old4 = book4.gather(1, indices >> 4)
        old8 = book8.gather(1, indices)
        target = old4 - self.lr * direction
        candidate = indices.clone()
        best_error = (old4 - target).square()
        prefix = (indices >> 6) << 6
        for suffix in range(64):
            proposed = prefix + suffix
            value4 = book4.gather(1, proposed >> 4)
            value8 = book8.gather(1, proposed)
            error = (value4 - target).square()
            better = error < best_error
            tied = (error == best_error) & ((value8 - old8).abs() <
                (book8.gather(1, candidate) - old8).abs())
            candidate = torch.where(better | tied, proposed, candidate)
            best_error = torch.minimum(best_error, error)
        change4 = book4.gather(1, candidate >> 4) - old4
        change8 = book8.gather(1, candidate) - old8
        eligible = (candidate != indices) & (best_error < (old4 - target).square())
        selected = torch.nonzero(eligible.flatten(), as_tuple=False).flatten()
        if selected.numel():
            order = torch.argsort(direction.abs().flatten()[selected], descending=True, stable=True)
            selected = selected[order[:max(1, int(indices.numel() * self.max_fraction))]]
            cost4 = change4.flatten()[selected].square().cumsum(0)
            cost8 = change8.flatten()[selected].square().cumsum(0)
            allowed = ((cost4 <= self.trust_ratio ** 2 * old4.square().sum()) &
                       (cost8 <= self.trust_ratio ** 2 * old8.square().sum()))
            selected = selected[allowed]
        updated = indices.clone().flatten()
        updated[selected] = candidate.flatten()[selected]
        updated = updated.reshape_as(indices)
        assert torch.equal(updated >> 6, indices >> 6)
        actual4 = book4.gather(1, updated >> 4) - old4
        actual8 = book8.gather(1, updated) - old8
        layer.indices.copy_(updated.to(torch.uint8))
        return {
            'changed': selected.numel(),
            'relative_w4': (actual4.norm() / old4.norm().clamp_min(1e-12)).item(),
            'relative_w8': (actual8.norm() / old8.norm().clamp_min(1e-12)).item(),
        }


def install_trainable_qwen(model, cache_path, precisions=(4, 8)):
    """Install cached Qwen projections, accepting complete or legacy MLP-only caches."""
    cache_path = Path(cache_path)
    precisions = list(precisions)
    available = sorted(int(path.name[4:]) for path in cache_path.glob('lut_*'))
    if not available or not precisions or not set(precisions).issubset(available):
        raise ValueError('Requested precisions are missing from the quantization cache.')
    parent_bits = max(available)
    if parent_bits not in precisions:
        raise ValueError('Include the parent precision to define the shared codebook.')
    if model.config.model_type != 'qwen3_5':
        raise ValueError('This installer supports the Qwen3.5 adaptation only.')
    layers = model.model.language_model.layers
    replacements = []
    coverage = set()
    for layer_index, layer in enumerate(layers):
        indices = torch.load(cache_path / 'weights' / f'l{layer_index}.pt', weights_only=False)
        luts = {bits: torch.load(cache_path / f'lut_{bits}' / f'l{layer_index}.pt',
                                 weights_only=False) for bits in precisions}
        mlp_keys = {f'mlp.{name}' for name in ('gate_proj', 'up_proj', 'down_proj')}
        attention_keys = ({f'linear_attn.{name}' for name in ('in_proj_qkv', 'in_proj_z', 'out_proj')}
                          if hasattr(layer, 'linear_attn') else
                          {f'self_attn.{name}' for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')})
        keys = set(indices)
        if keys not in (mlp_keys, mlp_keys | attention_keys):
            raise ValueError(f'Incomplete or unsupported projection coverage in layer {layer_index}: {sorted(keys)}')
        coverage.add(keys == mlp_keys)
        if len(coverage) != 1:
            raise ValueError('Mixed MLP-only and full-projection layers in the cache.')
        if any(set(book) != keys for book in luts.values()):
            raise ValueError(f'Codebook/index module mismatch in layer {layer_index}.')
        for key in indices:
            parent_path, name = key.rsplit('.', 1)
            parent = layer.get_submodule(parent_path)
            original = getattr(parent, name)
            index = torch.as_tensor(indices[key])
            if index.shape != (original.out_features, 1, original.in_features):
                raise ValueError(f'Unsupported index shape for layer {layer_index} {key}.')
            codebooks = {bits: torch.as_tensor(luts[bits][key])[:, 0, :] for bits in precisions}
            replacement = TrainableAnyPrecisionLinear(index[:, 0, :], codebooks, original.bias)
            replacements.append((parent, name, replacement.to(original.weight.device)))
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