# Trainable AnyPrecision Qwen

`any_precision.trainable` adds a differentiable training mode using the existing
Qwen3.5 MLP quantization cache, without the inference CUDA extension.

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

Load the original local Qwen first, then install the calibrated MLPs before
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