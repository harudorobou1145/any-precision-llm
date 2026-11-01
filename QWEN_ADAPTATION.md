# Qwen3.5 Adaptation

## Baseline

- Recorded: 2026-10-31
- Fork: https://github.com/harudorobou1145/any-precision-llm
- Upstream: https://github.com/SNU-ARC/any-precision-llm
- Starting commit: ee5288962ad5934c3988ae6c9a73e3aa6cd76058
- Working tree before adaptation: clean (`git status --short`).
- Original origin: https://github.com/SNU-ARC/any-precision-llm.git
- Model: Qwen/Qwen3.5-0.8B, official conditional-generation checkpoint.
- Initial scope: text inputs; MLP-only quantization; seed 4, parent 8.
- No training or MOPD changes are included.

## Implementation

The Qwen loader retains the conditional-generation model and uses BF16.
An explicit architecture configuration selects gate/up/down projections only.
Original non-Qwen loading defaults are retained.

Packed inference and calibration require separate validation. Successful
loading alone does not demonstrate a working quantized Qwen checkpoint.

Calibration now freezes non-target Qwen parameters, disables KV caching, and
avoids an unconditional whole-model BF16 conversion. Squared-gradient
accumulation is unchanged. Packed reconstruction uses the Qwen model class;
LUT buffers are explicitly FP16 and BF16 activations are converted only at
the quantized Linear boundary. Temporary precision is restored on exceptions.

## Validation (2026-10-31)

- Real checkpoint CPU loading: passed; 24 layers and 72 MLP projections.
- Three unittest cases: passed (model selection/meta construction, architecture
	configuration, mocked short/long Linear dtype boundaries).
- Real checkpoint GPU BF16 forward/backward: passed; loss 2.8310482501983643
	on `The capital of France is Paris.`; 72 finite MLP gradients.
- Real checkpoint greedy generation of 8 new tokens: passed.
- Optional optimized recurrent kernels are absent; reference PyTorch paths
	were used. These checks are not performance benchmarks.

## Remaining Gates

- Install quantization dependencies and compile `any_precision_ext`.
- Validate actual CUDA outputs against explicit dequantization for Qwen sizes.
- Run tiny calibration, clustering, packing, and checkpoint reload.
- Verify 4/5/6/7/8-bit outputs, precision switching, quality, and memory.
- A tiny-calibration packed artifact now exists (see below); inference is not
	yet validated.

## Tiny PTQ Run (2026-10-31)

## Full Calibration PTQ Run (2026-10-31)

- Calibration: C4, 100 examples of 512 tokens, random seed 0.
- Quantization: 24 layers, MLP-only, seed 4, parent 8, group count 1.
- Execution: GPU index 2, 4 CPU workers; approximately 68 seconds total.
- Gradient calibration, seed/upscale and packing all completed successfully.
- Output: `/workspace/my_project/outputs/anyprecision_qwen35_08b/mlp_seed4_parent8_s0/calibration_cache/packed/anyprec-(Qwen3.5-0.8B)-w8_orig4-gc1-c4_s100_blk512`.
- Packed inference, quality and memory measurements remain unverified.

### Tiny Run Details

Installed missing quantization dependencies, then ran the real quantize CLI.
The initial run failed on missing `datasets`. After dependency installation,
C4 repository pagination failed on official-site DNS resolution. The C4
loader now downloads the explicitly selected shard with `hf_hub_download`
and loads the local JSON file instead of listing the dataset repository.

- Dataset revision resolved: `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`.
- Shard: `en/c4-train.00000-of-01024.json.gz`.
- Calibration: 2 examples, 128 tokens, random seed 0.
- Quantization: MLP-only, seed 4, parent 8, group count 1, 4 CPU workers.
- Result: gradient calibration, all 24 layers' seed/upscale, and packing passed.
- Output: `/workspace/my_project/outputs/anyprecision_qwen35_08b/mlp_seed4_parent8_s0/smoke_cache/packed/anyprec-(Qwen3.5-0.8B)-w8_orig4-gc1-c4_s2_blk128`.
- This is a smoke-test artifact, not a quality-validated deployment model.
- Actual CUDA extension, reload, generation and bit-switching tests remain open.

Remote `origin` now points to the fork; `upstream` retains the original URL.
No commit, branch creation, or push was performed.

Run focused tests with:

```bash
python -m unittest discover -s tests -v
```