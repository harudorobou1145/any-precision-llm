# Trainable AnyPrecision Qwen

`any_precision.trainable` adds a differentiable training mode using the existing
Qwen3.5 projection quantization cache, without the inference CUDA extension.
Legacy MLP-only caches remain supported. Complete caches also cover
`linear_attn.in_proj_qkv/in_proj_z/out_proj` and `self_attn.q_proj/k_proj/v_proj/o_proj`.
The installer rejects partial projection sets and mixed coverage across layers.

For parent precision P and requested precision b, indices are `I_P >> (P-b)`.
Codebooks are `C_b = C_b_initial + mean_children(delta_P)`. Each layer has one
FP32 trainable parent delta; all initial codebooks and indices are fixed buffers.
This preserves the original PTQ values at initialization and couples updates
across precisions. It is codebook tuning, not full-weight QAT or index learning.
The child mean is a training design choice, not a claim that the initial coarse
centroids equal unweighted means of fine centroids.

```python
from any_precision.trainable import install_trainable_qwen, student_precision

install_trainable_qwen(model, quantization_cache, precisions=(4, 8))
optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=1e-4,
)
optimizer.zero_grad(set_to_none=True)
for bits in (4, 8):
    with student_precision(model, bits):
        loss = model(**batch, labels=batch['input_ids'], use_cache=False).loss
        (loss / 2).backward()
optimizer.step()
```

Load the original local Qwen first, then install the calibrated projections before
constructing an optimizer or wrapping with distributed training. Other model
parameters are frozen. Keep the precision context active through backward for
checkpoint recomputation. This mutable context is sequential, not thread-safe.
Generation must use the same context to actually select the requested precision.
The loader currently requires group count one and inclusion of parent precision.

## Verified Smoke Test

On 2026-10-31, the existing 100 x 512 calibration cache was tested on GPU 2:

- 72 trainable tensors, 50,331,648 FP32 delta parameters.
- Both precisions accumulate gradients before a single AdamW step.
- All 72 tensors changed; all accumulated gradients were finite.
- Short-text loss: 4-bit 3.377897 -> 0.950274; 8-bit 3.250833 -> 0.122544.
- Both precisions generated eight tokens after the update.
- Delta checkpoint restoration reproduced both post-update losses.
- Peak allocated CUDA memory: 4.658 GiB for this short single-sample test.

Run `python tests/smoke_trainable_qwen.py --help` for required local paths.
The checkpoint stores deltas, optimizer and base/cache references, not a standalone
packed inference model. A full training `state_dict` also contains fixed buffers.

These are training plumbing checks on the same short text, not evaluation scores.
Training materializes floating-point weights and integer gather indices; memory
for long sequences must be measured separately. Optimized Qwen attention kernels
were absent, so the test used reference PyTorch implementations.

## Remaining Integration

Ray/FSDP actor construction, precision-specific rollout workers, teacher scoring,
packed inference export and policy synchronization are not wired by this module.
Do not start the existing vLLM MOPD launcher expecting it to load this training
format automatically. No distributed MOPD run has been validated yet.

## Full-Projection Student (2026-11-16)

This supersedes the historical integration status above for HF rollout.
The default Qwen calibration architecture now selects 150 language-model
projections: 72 MLP, 54 linear-attention and 24 full-attention projections.
Embeddings, LM head, convolution, normalization and `in_proj_a/in_proj_b`
remain floating-point, matching the teacher projection scope. Activations
and compute remain floating-point; this is not activation quantization.

Use a separate cache root when switching from MLP-only calibration:

```bash
CUDA_VISIBLE_DEVICES=2 HF_HOME=/cache/huggingface \
HF_HUB_CACHE=/workspace/my_project/hf_cache/hub \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/workspace/miniconda3/envs/anyprecision/bin/python quantize.py \
    /workspace/my_project/models/Qwen3.5-0.8B \
    --seed_precision 2 --parent_precision 8 --mode quantize \
    --cache_dir /data/anyprecision_full_student/cache \
    --dataset c4 --num_examples 100 --seq_len 512 --cpu_count 4 --random_state 0
```

The resulting cache is under `quantized/(Qwen3.5-0.8B)-w8_orig2-gc1-c4_s100_blk512`.
Full WikiText-2 test PPL (1024-token non-overlapping blocks, 296902 scored tokens):
W2 4988.2121, W4 21.9610, W8 19.3838. W2 quality is severely degraded;
successful execution does not establish a usable W2 baseline.

Two-GPU HF MOPD completed three steps with index updates disabled. The saved
checkpoint contains 150 index/codebook modules; all 78 attention deltas changed
and all deltas are finite. Artifacts: `/data/anyprecision_mopd/full_projection_smoke_20261116`.
The Open-MOPD entrypoint is `my_scripts/run_full_student_w2w4.sh` (dry-run by
default). Complete caches are not supported by the MLP-only vLLM loader.