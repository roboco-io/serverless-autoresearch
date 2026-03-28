# Chapter 4: First Experiment — CUDA Errors, SDPA Fallback, and $0.02 of Joy

> **Time**: 1 hour
> **Cost**: $0.06 (3 failed + 1 successful experiment)
> **Key Insight**: Flash Attention 3 doesn't support all GPU architectures — always have a fallback.

## Context
Infrastructure ready. Time to run the first real experiment on SageMaker. What could go wrong? Everything.

## The Prompts

### Attempt 1: SageMaker Profiler Error
```
ml.g7e.2xlarge 스팟이지? 테스트 해 보자.
```
> *Translation: "ml.g7e.2xlarge is Spot, right? Let's test it."*

**Error**: `ValidationException: Profiler is currently not supported with instance type ml.g7e.2xlarge`
**Fix**: Added `disable_profiler=True` to the PyTorch Estimator.

### Attempt 2: Flash Attention 3 CUDA Crash
Same prompt, retry. Job ran but crashed:
```
CUDA error (/build/source/flash-attn/flash_fwd_launch_template.h:192):
no kernel image is available for execution on the device
```
FA3 pre-compiled kernels only support Hopper (sm_90) and Ampere (sm_80/86). The L40S is Ada Lovelace (sm_89) — not supported.

First fix attempt: try/except around FA3 import. Still crashed — `get_kernel()` succeeds but the CUDA kernel fails at runtime.

### Attempt 3: Explicit Capability Check
```python
FA3_SUPPORTED = cap in ((9, 0), (8, 0), (8, 6))
if FA3_SUPPORTED:
    # use FA3
else:
    # PyTorch SDPA fallback
```

The attention forward also needed modification:
```python
if USE_FA3:
    y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
else:
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = y.transpose(1, 2)
```

### Attempt 4: Success!
Result:
```
val_bpb:          1.065437
training_seconds: 300.1
peak_vram_mb:     22805.5
mfu_percent:      20.55
total_tokens_M:   260.6
num_steps:        497
num_params_M:     50.3
```

**$0.02 for the first successful autonomous ML experiment on SageMaker Spot.**

## The Result
Compared with Karpathy's H100 baseline:

| Metric | H100 (original) | L40S (ours) |
|--------|-----------------|-------------|
| val_bpb | 0.998 | 1.065 |
| MFU | 39.8% | 20.5% |
| Peak VRAM | 45 GB | 22.8 GB |
| Tokens (5min) | 500M | 261M |
| Cost | N/A | **$0.02** |

MFU is exactly half because SDPA is ~2x less efficient than FA3. The val_bpb gap (6.7%) is entirely due to processing fewer tokens in the same 5-minute budget.

## Lessons Learned
- **FA3 is architecture-specific** — always check `torch.cuda.get_device_capability()` and provide fallbacks
- **Runtime CUDA errors ≠ import errors** — `get_kernel()` can succeed while the actual kernel fails
- **g7e instances need `disable_profiler=True`** — newer instance types don't support SageMaker Profiler
- **47% VRAM utilization** means room for optimization (DEVICE_BATCH_SIZE can be increased)

### Debugging: FA3 CUDA Kernel Crash
**Symptom**: `CUDA error: no kernel image is available for execution on the device`
**Root Cause**: FA3 pre-compiled kernels don't include Ada Lovelace (sm_89)
**Fix**: Explicit compute capability check + PyTorch SDPA fallback
**Time Lost**: ~30 minutes across 3 attempts

### Running Cost
| Phase | Action | Cost | Cumulative |
|-------|--------|------|-----------|
| Planning | Architecture design | $0.00 | $0.00 |
| Building | Code generation | $0.00 | $0.00 |
| Infra | IAM + S3 + quotas | $0.00 | $0.00 |
| Exp 1 | 3 failed + 1 success (L40S) | $0.06 | $0.06 |
