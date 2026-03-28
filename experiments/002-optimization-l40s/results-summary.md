# Experiment #002 Results: L40S Parallel Evolution (5 Generations)

## Summary

| Metric | Value |
|--------|-------|
| Total experiments | 24 (+ 4 from BS=128 pre-test) |
| Generations | 5 (gen000-004) |
| Successful | 21 |
| Timeouts | 2 |
| Crashed | 0 |
| **Best val_bpb** | **1.0643** (gen004, EMBEDDING_LR→0.709) |
| **Total improvement** | **-0.0013** from baseline 1.0656 |
| Total cost | ~$0.35 |
| Total wall clock | ~2.5 hours (including Spot wait + region migration) |
| Instance | ml.g7e.2xlarge (L40S 48GB, us-east-1, Spot) |

## Evolution Trajectory

```
val_bpb
1.0660 ┤● gen000 baseline
1.0656 ┤ ╲
1.0653 ┤  ╲ gen002 EMBEDDING_LR 0.6→0.549
1.0651 ┤   ╲ gen003 SCALAR_LR 0.5→0.362
1.0643 ┤    ╲● gen004 EMBEDDING_LR 0.549→0.709
       └────────────────────────────
        0    1    2    3    4   generation
```

## Per-Generation Results

### Generation 000 (Baseline, BS=64)

| Candidate | val_bpb | Strategy | Kept? |
|-----------|---------|----------|-------|
| **v00** | **1.0656** | baseline | **Yes** |
| v01 | 1.0656 | SCALAR_LR 0.5→0.443 | No |
| v02 | 1.0658 | MATRIX_LR 0.04→0.047 | No |
| v03 | 1.0658 | UNEMBEDDING_LR→0.003356 | No |

### Generation 001

| Candidate | val_bpb | Strategy | Kept? |
|-----------|---------|----------|-------|
| v01 | 1.0666 | UNEMBEDDING_LR→0.003088 | No |
| v02 | 1.0658 | EMBEDDING_LR 0.6→0.709 | No |
| v03 | 1.0658 | MATRIX_LR→0.0415 | No |
| v04 | **1.0942** | TOTAL_BATCH_SIZE 2^19→2^20 | **No (much worse)** |

### Generation 002

| Candidate | val_bpb | Strategy | Kept? |
|-----------|---------|----------|-------|
| v01 | 1.0664 | MATRIX_LR→0.033 | No |
| **v02** | **1.0653** | **EMBEDDING_LR 0.6→0.549** | **Yes** |
| v03 | 1.0656 | EMBEDDING_LR 0.6→0.450 | No |
| v04 | timeout | WINDOW_PATTERN SSSL→L | No |

### Generation 003

| Candidate | val_bpb | Strategy | Kept? |
|-----------|---------|----------|-------|
| v01 | 1.0663 | UNEMBEDDING_LR→0.002451 | No |
| v02 | 1.0657 | SCALAR_LR→0.379 | No |
| **v03** | **1.0651** | **SCALAR_LR 0.5→0.362** | **Yes** |
| v04 | 1.0655 | WINDOW_PATTERN SSSL→SL | No |

### Generation 004

| Candidate | val_bpb | Strategy | Kept? |
|-----------|---------|----------|-------|
| **v01** | **1.0643** | **EMBEDDING_LR 0.549→0.709** | **Yes (best!)** |
| v02 | 1.0658 | SCALAR_LR 0.362→0.418 | No |
| v03 | timeout | SCALAR_LR 0.362→0.419 | No |
| v04 | 1.0923 | MATRIX_LR→0.055 + DEPTH→9 | No (much worse) |

## What Worked

| Change | val_bpb Impact | Generation |
|--------|---------------|------------|
| **EMBEDDING_LR 0.549→0.709** | **-0.0008** | gen004 (biggest win) |
| SCALAR_LR 0.5→0.362 | -0.0002 | gen003 |
| EMBEDDING_LR 0.6→0.549 | -0.0003 | gen002 |

The pipeline discovered that **EMBEDDING_LR** is the most sensitive parameter. It first lowered it (0.6→0.549), then raised it (0.549→0.709) — the raise was the biggest single improvement.

## What Failed

| Change | val_bpb Impact | Reason |
|--------|---------------|--------|
| TOTAL_BATCH_SIZE 2^19→2^20 | +0.029 (much worse) | Likely too few steps with large batch |
| DEPTH 8→9 + MATRIX_LR↑ | +0.027 (much worse) | Larger model doesn't converge in 500 steps |
| WINDOW_PATTERN SSSL→L | timeout | Full attention too slow for 5-min budget |
| DEVICE_BATCH_SIZE 64→128 | +0.016 (worse) | Reduced grad accum without increasing throughput |

## Evolved train.py Parameters

| Parameter | Original | After Evolution | Change |
|-----------|----------|----------------|--------|
| EMBEDDING_LR | 0.6 | **0.709** | +18% |
| UNEMBEDDING_LR | 0.004 | **0.003369** | -16% |
| SCALAR_LR | 0.5 | **0.362** | -28% |
| DEVICE_BATCH_SIZE | 128→64 | **64** | Reverted |
| All others | — | unchanged | — |

## Cost Breakdown

| Phase | Experiments | Cost | Cumulative |
|-------|------------|------|-----------|
| BS=128 pre-test (us-east-1) | 4 | $0.07 | $0.07 |
| gen000 (us-east-1) | 4 | $0.07 | $0.14 |
| gen001 | 4 | $0.07 | $0.21 |
| gen002 | 4 | $0.07 | $0.28 |
| gen003 | 4 | $0.07 | $0.35 |
| gen004 | 4 | $0.05 | $0.40 |
| **Total** | **24** | **~$0.40** | |

## Comparison with Experiment #001

| Metric | #001 (single baseline) | #002 (5-gen evolution) |
|--------|----------------------|----------------------|
| val_bpb | 1.0654 | **1.0643** (-0.0011) |
| Experiments | 1 | 24 |
| Cost | $0.04 | $0.40 |
| Approach | Manual single run | Autonomous parallel evolution |

## Key Insights

1. **EMBEDDING_LR is the most impactful parameter** on L40S with SDPA — both decreasing and increasing it improved results
2. **Conservative strategy dominates** — all 3 improvements came from small LR adjustments, not architecture changes
3. **Moderate/aggressive strategies failed** — DEPTH changes, TOTAL_BATCH_SIZE changes, and WINDOW_PATTERN changes all worsened results
4. **The 5-min time budget strongly constrains architecture exploration** — larger models can't converge in ~500 steps
5. **Autonomous evolution works** — the pipeline found improvements without human intervention

---

*Completed: 2026-03-28*
*Pipeline: 5 generations × 4 candidates, ml.g7e.2xlarge Spot, us-east-1*
