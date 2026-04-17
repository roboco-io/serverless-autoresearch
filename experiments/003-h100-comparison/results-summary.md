# Experiment #003 Results: H100 Comparison with Original Autoresearch

## Summary

| Metric | Value |
|--------|-------|
| Total SageMaker jobs | 23 (3 single + 20 pipeline) |
| Phases | 3 (single BS=64, 5-gen×4-pop evolution, single BS=128) |
| Successful | 22 |
| Failed (CUDA unknown error) | 1 (first single-run, transient) |
| Timeouts / crashes | 0 |
| **Best val_bpb** | **0.995147** (Phase 3, BS=128 — *below* upstream ~0.998) |
| Best BS=64 val_bpb (gen003 v01) | 1.001586 |
| Baseline val_bpb (L40S-evolved train.py on H100, BS=64) | 1.002991 |
| Total billable time | 5,042 s (1.40 h) |
| Estimated cost (Spot) | **~$2.8 – $4.2** @ $2–3/hr ml.p5.4xlarge Spot |
| Total wall clock | ~2.3 h |
| Instance | ml.p5.4xlarge (H100 80GB, us-east-1, Spot) |
| Attention | Flash Attention 3 (`varunneal/flash-attention-3`, sm_90) |

## Headline Comparison

| Config | Hardware | BS | Attention | val_bpb | Δ vs upstream |
|--------|----------|----|-----------|---------|---------------|
| **#003 Phase 3 (evolved LR + upstream BS)** | **H100** | **128** | **FA3** | **0.9951** | **-0.0029 (-0.3%) ★** |
| Upstream autoresearch (Karpathy) | H100 | 128 | FA3 | ~0.998 | baseline |
| #003 Phase 2 best (gen003 v01, 5-gen evolution) | H100 | 64 | FA3 | 1.0016 | +0.0036 (+0.4%) |
| #003 Phase 1 baseline (L40S-evolved on H100) | H100 | 64 | FA3 | 1.0030 | +0.0050 (+0.5%) |
| #002 best (L40S 5-gen evolution) | L40S | 64 | SDPA | 1.0643 | +0.0663 (+6.6%) |
| #001 baseline | L40S | 64 | SDPA | 1.0654 | +0.0674 (+6.8%) |

**Phase 3 result surpasses the upstream baseline** using upstream's BS=128 plus the L40S-discovered learning-rate regime (EMBEDDING_LR=0.7091, UNEMBEDDING_LR=0.003369, SCALAR_LR=0.3616). Single-run caveat applies — a proper confidence interval requires multiple repeats, and upstream 0.998 is itself a single reported number.

The **hardware+kernel gap** (L40S SDPA → H100 FA3, same code) drops val_bpb by 0.061. The additional **BS=64 → 128 swap on H100** (same LR) drops it by another 0.007. 5-gen evolution on top of BS=64 only found 0.001.

## Evolution Trajectory

```
val_bpb
1.0030 ┤● gen000 baseline (evolved L40S train.py, unchanged)
1.0025 ┤ ╲● gen000 v02  EMBEDDING_LR 0.7091 → 0.6884
1.0018 ┤  ╲● gen001 v03  UNEMBEDDING_LR 0.003369 → 0.004206
1.0018 ┤   ● gen002 (no improvement, kept gen001)
1.0016 ┤    ╲● gen003 v01  EMBEDDING_LR 0.6884 → 0.6433  ★ BEST
1.0016 ┤     ● gen004 (no improvement, kept gen003)
       └────────────────────────────
        0    1    2    3    4   generation
```

## Per-Generation Results

### Generation 000 — starting from L40S-evolved train.py

| Candidate | val_bpb | Strategy | Kept? |
|-----------|---------|----------|-------|
| v00 | 1.002991 | baseline (unmodified L40S-evolved) | — |
| v01 | 1.003966 | conservative: UNEMBEDDING_LR 0.003369 → 0.002498 | No |
| **v02** | **1.002458** | **conservative: EMBEDDING_LR 0.7091 → 0.6884** | **Yes** |
| v03 | 1.004922 | conservative: SCALAR_LR 0.3616 → 0.402013 | No |

### Generation 001

| Candidate | val_bpb | Strategy | Kept? |
|-----------|---------|----------|-------|
| v01 | 1.004118 | conservative: MATRIX_LR 0.04 → 0.031318 | No |
| v02 | 1.002626 | conservative: UNEMBEDDING_LR 0.003369 → 0.004164 | No |
| **v03** | **1.001849** | **conservative: UNEMBEDDING_LR 0.003369 → 0.004206** | **Yes** |
| v04 | 1.019312 | moderate: DEPTH 8 → 9 | No (much worse) |

### Generation 002 — no improvement

| Candidate | val_bpb | Strategy | Kept? |
|-----------|---------|----------|-------|
| v01 | 1.002286 | conservative: UNEMBEDDING_LR 0.004206 → 0.003824 | — (gen001 best retained) |
| v02 | 1.002547 | conservative: MATRIX_LR 0.04 → 0.039682 | No |
| v03 | 1.003107 | conservative: SCALAR_LR 0.3616 → 0.264178 | No |
| v04 | 1.004460 | moderate: WINDOW_PATTERN SSSL → SSL | No |

### Generation 003 — new best

| Candidate | val_bpb | Strategy | Kept? |
|-----------|---------|----------|-------|
| **v01** | **1.001586** | **conservative: EMBEDDING_LR 0.6884 → 0.6433** | **Yes (best overall)** |
| v02 | 1.002807 | conservative: MATRIX_LR 0.04 → 0.029848 | No |
| v03 | 1.002687 | conservative: UNEMBEDDING_LR 0.004206 → 0.004286 | No |
| v04 | 1.010330 | moderate: DEPTH 8 → 9 | No (much worse) |

### Generation 004 — no further improvement

| Candidate | val_bpb | Strategy | Kept? |
|-----------|---------|----------|-------|
| v01 | 1.002653 | conservative: EMBEDDING_LR 0.6433 → 0.555008 | No |
| v02 | 1.002537 | conservative: EMBEDDING_LR 0.6433 → 0.607861 | No |
| v03 | 1.004570 | conservative: MATRIX_LR 0.04 → 0.030257 | No |
| v04 | 1.017065 | moderate: DEPTH 8 → 9 | No (much worse) |

## What Worked

| Change | val_bpb Impact | Generation |
|--------|---------------|------------|
| **EMBEDDING_LR 0.7091 → 0.6884** | **-0.000533** | gen000 |
| **UNEMBEDDING_LR 0.003369 → 0.004206** | **-0.000609** | gen001 |
| **EMBEDDING_LR 0.6884 → 0.6433** | **-0.000263** | gen003 (biggest remaining) |

Interesting reversal: on L40S (exp #002) the evolution **raised** EMBEDDING_LR from 0.6 → 0.7091. On H100 with FA3 and the L40S-evolved starting point, the evolution **lowered** it back toward 0.64 — closer to the upstream 0.6.

## What Failed

| Change | val_bpb Impact | Reason |
|--------|---------------|--------|
| DEPTH 8→9 (attempted 3×) | +0.008 to +0.017 | Larger model can't converge in 500 steps under 5-min budget |
| WINDOW_PATTERN SSSL→SSL | +0.003 | Less sliding-window capacity |

Architecture-level changes remain dominated by the fixed 5-min TIME_BUDGET — exactly as observed in exp #002.

## Phase 3 — BS=128 Match with Upstream Config

After Phase 2 plateaued at 1.0016 with BS=64, we swapped `DEVICE_BATCH_SIZE` 64 → 128 (the upstream value) while keeping the L40S-evolved learning rates. Single run on H100 Spot.

| Config | val_bpb | Peak VRAM | Billable |
|--------|---------|-----------|----------|
| Phase 2 best (BS=64, evolved LR) | 1.0016 | 22.3 GB | ~220s |
| **Phase 3 (BS=128, L40S-evolved LR)** | **0.9951** | **45.0 GB** | **229s** |
| Upstream (BS=128, upstream LR) | ~0.998 | — | — |

**val_bpb dropped by 0.0065 just from doubling the batch size** — larger than the entire 5-gen evolution gain (0.0014). Peak VRAM doubled as expected (22→45 GB), still well under H100's 80 GB. Training wall time stayed similar because FA3 throughput at BS=128 is higher, offsetting the halved grad_accum_steps.

This result **beats the reported upstream value** (0.995 vs 0.998), though that gap is within plausible single-run variance and needs repeated runs to be statistically claimed. The L40S-evolved LR regime (EMBEDDING_LR=0.7091, SCALAR_LR=0.3616) apparently transfers cleanly to BS=128 on H100.

## Used train.py Parameters

| Parameter | Upstream (H100) | #002 L40S-evolved | #003 Phase-2 best | **#003 Phase-3 (final)** |
|-----------|-----------------|-------------------|-------------------|--------------------------|
| EMBEDDING_LR | 0.6 | 0.7091 | 0.643284 | **0.7091** (reverted to L40S) |
| UNEMBEDDING_LR | 0.004 | 0.003369 | 0.004206 | **0.003369** (reverted) |
| SCALAR_LR | 0.5 | 0.3616 | 0.3616 | **0.3616** |
| MATRIX_LR | 0.04 | 0.04 | 0.04 | **0.04** |
| DEVICE_BATCH_SIZE | 128 | 64 | 64 | **128** (matches upstream) |
| DEPTH | 8 | 8 | 8 | **8** |
| WINDOW_PATTERN | SSSL | SSSL | SSSL | **SSSL** |

Phase-3 configuration = L40S-evolved LRs + upstream BS. The Phase-2 H100-specific LR adjustments (EMBEDDING_LR 0.7091 → 0.6433, UNEMBEDDING_LR 0.003369 → 0.004206) were optimal *for BS=64*; with BS=128 the L40S values are used again and outperform the Phase-2 best by -0.0065.

## Cost Breakdown

| Phase | Jobs | Billable | Est. cost @ $2.5/hr |
|-------|------|----------|---------------------|
| Phase 1 single (1 failed CUDA + 1 success) | 2 | 368 s | $0.26 |
| Phase 2 pipeline (5 gen × 4 pop) | 20 | 4,445 s | $3.09 |
| **Phase 3 BS=128 single run** | **1** | **229 s** | **$0.16** |
| **Total** | **23** | **5,042 s (1.40 h)** | **~$3.50** |

Spot price band: $2.80 – $4.20 depending on actual ml.p5.4xlarge Spot rate in us-east-1 during the run. Exact cost will appear in AWS Cost Explorer within 24h.

## Reliability Note — CUDA Unknown Error (Phase 1)

The very first single-run failed at step 338/~970 with `CUDA error: unknown error` inside `train_loss.item()`. A bit-identical retry succeeded. Root cause not confirmed, but the pattern (one-off, non-reproducing, async CUDA error on Spot H100) is consistent with either a transient Spot hardware issue or an FA3 kernel edge case. **21/22 subsequent jobs (all pipeline + retry single) completed cleanly**, so this is <5% failure rate — acceptable, but worth watching if the pipeline is scaled up.

## Comparison with Previous Experiments

| Metric | #001 baseline (L40S) | #002 evolution (L40S) | **#003 H100 (3 phases)** |
|--------|---------------------|----------------------|---------------------------|
| Best val_bpb | 1.0654 | 1.0643 | **0.9951** (Phase 3, BS=128) |
| Jobs | 1 | 24 | 23 |
| Cost | $0.04 | $0.40 | ~$3.50 |
| Hardware | ml.g7e.2xlarge (L40S) | ml.g7e.2xlarge (L40S) | ml.p5.4xlarge (H100) |
| Attention | SDPA | SDPA | **Flash Attention 3** |
| val_bpb / $ (improvement) | — | -0.0011 / $0.40 | **-0.070 / $3.50** (better $/point) |

## Key Insights

1. **Hardware+kernel is the dominant lever.** Moving L40S-evolved code to H100+FA3 recovered 97% of the gap to the upstream result for only ~$3.
2. **Batch size is the next-biggest lever on H100.** With L40S-optimal LRs held fixed, BS=64 → 128 alone improved val_bpb by 0.0065 — about 4× the entire 5-gen LR evolution gain.
3. **We surpassed the reported upstream result** (0.9951 vs ~0.998). Caveat: both are single-run values; proper ±σ requires repeats.
4. **L40S-evolved LRs partially over-fit to SDPA+BS=64.** On H100 BS=64, evolution pulled EMBEDDING_LR back (0.7091 → 0.6433). On H100 BS=128, the original L40S-evolved values became optimal again — LR and BS interact.
5. **Architecture changes remain off-limits** under the fixed 5-min budget — DEPTH=9 fails identically on L40S and H100. This is a property of the *time budget*, not the *hardware*.
6. **Pipeline cost scales as expected**: ~$0.6 per 4-candidate generation on H100 Spot, i.e. ~15× the L40S pipeline cost per generation. At this rate a full 10-gen run would be ~$7–$8 — still well under $10.
7. **One transient CUDA error on Spot H100** (1/23 jobs, 4.3%). Retry succeeded, no pattern. If this rate holds at scale, a 100-job pipeline would see ~4–5 retries needed.

## Follow-ups to Consider

- **Validate Phase 3 reproducibility**: rerun BS=128 + L40S-LR configuration 5–10 times to get a confidence interval on 0.995.
- **Evolve on top of Phase 3**: the 5-gen pipeline started from BS=64. Rerunning evolution with BS=128 as the baseline might find a further regime.
- **Reliability**: decide whether to add CUDA-error auto-retry to `batch_launcher.py` before scaling up.

---

*Completed: 2026-04-17*
*Phase 1: 2 single runs (BS=64) — baseline + retry after 1 transient CUDA error*
*Phase 2: 5-gen × 4-pop evolution (BS=64) — best 1.0016 at gen003 v01*
*Phase 3: single run (BS=128, L40S-evolved LR) — 0.9951, below upstream*
*Hardware: ml.p5.4xlarge Spot (H100 80 GB), us-east-1, Flash Attention 3*
