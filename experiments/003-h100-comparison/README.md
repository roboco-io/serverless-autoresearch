# Experiment #003: H100 Comparison with Original Autoresearch

## Goal

Run the L40S-evolved `train.py` on H100 Spot (ml.p5.4xlarge) to measure:

1. **Hardware gap**: How much does H100 + FA3 improve val_bpb vs L40S + SDPA?
2. **Original reproducibility**: Can we reach Karpathy's upstream H100 result (val_bpb ≈ 0.998)?
3. **Transfer of L40S optimizations**: Do the evolved hyperparameters (EMBEDDING_LR=0.709, SCALAR_LR=0.362, etc.) also help on H100 with FA3, or were they L40S/SDPA-specific?

## Setup

| Item | Value |
|------|-------|
| Instance | `ml.p5.4xlarge` (1× H100 80GB) |
| Region | us-east-1 (Virginia) |
| Pricing | Spot |
| AWS profile | `roboco` |
| Spot quota | 4 (approved 2026-03-28) |
| Attention backend | Flash Attention 3 (sm_90 via `varunneal/flash-attention-3`) |
| train.py state | L40S-evolved (EMBEDDING_LR=0.7091, SCALAR_LR=0.3616, UNEMBEDDING_LR=0.003369, DEVICE_BATCH_SIZE=64) |
| Time budget | TIME_BUDGET=300s (5 min training) |

## Phase Plan

| Phase | Description | Jobs | Est. cost |
|-------|-------------|------|-----------|
| 1 | Single H100 baseline (evolved train.py as-is) | 1 | ~$0.5 |
| 2 | Evolution pipeline (5 gen × 4 pop) | 20 | ~$8 |
| **Total** | | **21** | **~$8-10** |

## Baselines for comparison

| Source | val_bpb | Hardware | Notes |
|--------|---------|----------|-------|
| Upstream autoresearch (Karpathy) | ~0.998 | H100, FA3, BS=128 | Vanilla hyperparameters |
| Experiment #002 best (L40S) | 1.0643 | L40S, SDPA, BS=64 | Evolved EMBEDDING_LR=0.709 |

## Running

```bash
# Phase 1: single H100 run (validates setup, ~15 min incl. Spot wait)
make run-single

# Phase 2: full pipeline (~2-3 h total wall clock)
python -m src.pipeline.orchestrator --generations 5 --population 4
```

## Output

- `results-summary.md` — final report with all results
- `generations/003-h100-*/` — per-generation candidate variants and raw CloudWatch metrics (gitignored)
