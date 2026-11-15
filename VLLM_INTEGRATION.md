# Native compressed vLLM integration

This experimental plugin supports the existing group-count-one Qwen3.5 MLP
cache. It does not replace original MOPD's loss, actor update or teacher routing.

## Representation

The rollout stores one parent bitplane tensor per MLP projection (gate/up are
fused), plus W4/W8 centroid tables. Both precisions consume the same parent
indices. The CUDA matmul performs lookup during execution; it never materializes
a full floating-point MLP weight matrix. Prefill is split into at most eight
activation rows per kernel call, so high prefill throughput is not established.

Only the MLP is compressed. Attention, embeddings, vision weights and other
parameters retain their normal representation. The training actor still uses
its existing differentiable per-forward weight expansion and owns a separate
index representation. This is NOT one GPU copy across actor and rollout.

## Installation and scope

Install this repository with the same Python used by MOPD:

```sh
/usr/bin/python -m pip install --no-deps -e .
cd any_precision/modules/kernels
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.6 /usr/bin/python -m pip install --no-build-isolation --no-deps .
```

The tested extension build targets SM 8.6 only. Tested runtime: torch 2.11.0
CUDA 12.8, vLLM 0.25.1, transformers 5.19.0. Requirements: FP16 rollout,
TP=1, eager execution, no LoRA, no prefix caching. Other combinations are not
validated. No BF16 expanded-weight rollout fallback is provided.

In the sibling Open-MOPD local launcher, set `ANYPRECISION_ROLLOUT=vllm`
alongside the existing `ANYPRECISION_CACHE` and teacher configuration. The
default remains `hf`. The experimental vLLM switch disables engine sleep;
therefore actor and rollout coexist in GPU memory. Single-GPU Ray/FSDP training
completed 26 updates with this switch on 2026-11-01.

The loader uses `load_format=anyprecision`, `quantization=anyprecision`, and
`model_loader_extra_config={"cache_path": "..."}`. Actor synchronization
sends fixed base tables plus the shared delta, derives both precision tables,
validates every layer before writing, and leaves bitplanes unchanged. Samples
are grouped by `student_bits` and restored to their original ordering.

## Verified behavior

- Compressed CUDA matmul matches gathered reference weights at W4/W8, including
  more than eight activation rows and a nondefault CUDA stream.
- Real Qwen3.5-0.8B loads 48 fused compressed MLP modules, with 252 MiB of
  parent bitplanes; vLLM reports approximately 1.38 GiB model-loading memory.
  This excludes KV cache, runtime allocations, actor and teachers.
- On one fixed prompt, W4/W8 greedy next tokens match the HF shared-codebook
  model and selected-token log probabilities differ by less than 0.1.
- Real MOPD rollout generates a mixed-precision batch and accepts actor
  codebook synchronization. A subsequent budget experiment completed 26 original
  MOPD updates, saved a final checkpoint and evaluated its compressed W4/W8 paths.

Run `tests/test_vllm_integration.py` with unittest. Full-model checks require
`ANYPRECISION_MODEL_TEST=1`, `ANYPRECISION_MODEL` and `ANYPRECISION_CACHE`.
Set `ANYPRECISION_MOPD_TEST=1` and add the sibling `training/verl` directory
to `PYTHONPATH` to exercise MOPD rollout generation and synchronization.

## Experiment record (2026-11-01)

- Qwen adaptation includes architecture analysis, calibration/gradient handling,
  evaluation support and compatible model/linear wrappers.
- Shared trainable codebook deltas support fixed parent8 indices and W4/W8 views.
- Native vLLM plugin adds packed loading, quantization, precision switching and
  actor-codebook synchronization; CUDA kernels use the current stream.
- Original MOPD ran 26 updates on one 48 GB GPU, batch 8 prompts expanded to
  16 precision-paired responses, learning rate 1e-6, response limit 2048.
- Fixed 256-question GSM8K evaluation, greedy with thinking disabled: strict
  W4 32 -> 35 correct, W8 36 -> 36; relaxed numeric scoring W4 127 -> 131,
  W8 145 -> 144. Relaxed paired bootstrap intervals include zero for both.
- Mean outer step was 203.85 s, including 143.27 s generation; no controlled
  floating-point-versus-compressed throughput benchmark has been performed.

Quality improvement is not established. Actor MLP weight expansion, separate
actor/rollout storage, small-row prefill calls and standalone model export remain
limitations. Teachers used different compute dtypes (W4 FP16, W8 BF16); their
effect has not been isolated. Model weights, datasets and checkpoints are not
included in this repository.

Publication validation: `python -m unittest discover -s tests -p 'test_*.py'`
ran 8 tests, 7 passed and 1 optional full-model test skipped.
`git diff --check` passed. Training and full-model checks were not rerun for
this publication.

## W2 and BF16 adapter (2026-11-09)

The plugin accepts W2 through W8 and BF16 activations. W2 or BF16 uses
bitplane decoding into a cached dense weight followed by floating-point linear
operations; this is not a native INT2 kernel. Codebook synchronization clears
the cache. Each visited precision keeps its own dense weight, increasing memory
use. The previous FP16 W3-W8 packed kernel remains available.

Real Qwen3.5-0.8B seed2-parent8 checks passed for W2/W4/W8: greedy next tokens
match the HF reference and selected-token log probabilities differ by less
than 0.1. Set ANYPRECISION_TEST_BITS=2,4,8 and
ANYPRECISION_TEST_DTYPE=bfloat16 for the optional full-model test.

ANYPRECISION_BENCHMARK=1 additionally compares eight identical short prompts
with 128 generated tokens, including first-use cache decoding. One RTX4090
measurement: W2 vLLM 5.279s versus HF 5.004s; W4 vLLM 3.236s versus HF 5.184s.
These single measurements do not establish a W2 speedup or end-to-end MOPD
throughput improvement. Full W2/BF16 MOPD training is not yet validated.
The benchmark uses eager execution and local callbacks; the latter require
VLLM_ALLOW_INSECURE_SERIALIZATION=1 in the trusted local test environment only.