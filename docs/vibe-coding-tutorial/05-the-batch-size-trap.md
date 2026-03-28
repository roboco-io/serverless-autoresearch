# Chapter 5: The Batch Size Trap — When "More VRAM" Doesn't Mean "Better"

> **Time**: 30 minutes
> **Cost**: $0.07 (4 experiments)
> **Key Insight**: DEVICE_BATCH_SIZE ≠ token throughput. Increase TOTAL_BATCH_SIZE instead.

## Context
Experiment #001 showed only 47% VRAM utilization (22.8GB / 48GB). The obvious optimization: double DEVICE_BATCH_SIZE from 64 to 128 to use that free VRAM. This is a trap.

## The Prompt
```
이제 두번째 실험을 autoresearch로 시작해줘.
```

The pipeline was configured with DEVICE_BATCH_SIZE=128 and submitted 4 parallel candidates.

## What Happened
Results from 4 experiments with DEVICE_BATCH_SIZE=128:

| Candidate | val_bpb | VRAM |
|-----------|---------|------|
| v00 | 1.0815 | 45 GB |
| v01 | 1.0823 | 45 GB |
| v02 | 1.0812 | 45 GB |
| v03 | 1.0811 | 45 GB |

**val_bpb went from 1.065 to 1.081 — it got WORSE!**

VRAM doubled (22.8→45 GB) as expected, but val_bpb worsened by 1.5%.

## Why?
With TOTAL_BATCH_SIZE fixed at 2^19 (524K tokens):
- BS=64: `grad_accum_steps = 524288 / (64 × 2048) = 4`
- BS=128: `grad_accum_steps = 524288 / (128 × 2048) = 2`

Same total tokens per optimizer step. Fewer gradient accumulation steps means less noise in gradients — which sounds good but actually hurt convergence in this setup. The model sees the exact same number of tokens, just in bigger chunks.

**To actually increase throughput, you need to increase TOTAL_BATCH_SIZE (more tokens per optimizer step), not DEVICE_BATCH_SIZE (more tokens per micro-batch).**

## The Result
Reverted to DEVICE_BATCH_SIZE=64 and continued with the parallel evolution pipeline.

## Lessons Learned
- **DEVICE_BATCH_SIZE controls VRAM usage, not training quality** — it determines micro-batch size and gradient accumulation steps
- **TOTAL_BATCH_SIZE controls tokens per optimizer step** — this is what affects convergence
- **Using more VRAM doesn't automatically mean better results** — a common misconception
- **Quick experiments ($0.07) save expensive mistakes** — this trap would have been costly at scale

## Try It Yourself
```python
# This is a common misunderstanding. To verify:
# Case 1: BS=64, TOTAL=2^19 → grad_accum=4, tokens per step=524K
# Case 2: BS=128, TOTAL=2^19 → grad_accum=2, tokens per step=524K (SAME!)
# Case 3: BS=64, TOTAL=2^20 → grad_accum=8, tokens per step=1048K (MORE!)
```

### Running Cost
| Phase | Action | Cost | Cumulative |
|-------|--------|------|-----------|
| Planning | Architecture design | $0.00 | $0.00 |
| Building | Code generation | $0.00 | $0.00 |
| Infra | IAM + S3 + quotas | $0.00 | $0.00 |
| Exp 1 | First success (L40S) | $0.06 | $0.06 |
| Exp 2a | Batch size trap (4 runs) | $0.07 | $0.13 |
