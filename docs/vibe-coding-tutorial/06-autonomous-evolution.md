# Chapter 6: Autonomous Evolution — Letting the Pipeline Run Free

> **Time**: 2.5 hours (mostly waiting)
> **Cost**: $0.31 (20 experiments across 5 generations)
> **Key Insight**: Conservative LR tuning beats radical architecture changes for short training runs.

## Context
With the batch size lesson learned, it was time to let the pipeline do what autoresearch does best: autonomous experimentation. No more manual intervention — just set the parameters and let it evolve.

## The Prompt
```
이런거 계속 자동으로 반복해서 실험하는게 autoresearch아냐?
omc autoresearch로 실험을 진행해줘.
```
> *Translation: "Isn't autoresearch supposed to keep repeating experiments like this automatically? Run the experiments with omc autoresearch."*

This was the turning point. Stop micromanaging, start auto-researching.

## What Happened

### Generation 0 (Baseline)
4 conservative LR variants. Best: val_bpb=1.0656 (baseline unchanged).

### Generation 1
Mixed results:
- TOTAL_BATCH_SIZE 2^19→2^20: val_bpb=**1.094** (much worse — too few steps with large batch)
- LR tweaks: ~baseline level

### Generation 2 — First Improvement!
- **EMBEDDING_LR 0.6→0.549**: val_bpb=**1.0653** (improved!)
- WINDOW_PATTERN SSSL→L: timeout (full attention too slow for 5-min budget)

### Generation 3
- **SCALAR_LR 0.5→0.362**: val_bpb=**1.0651** (another improvement!)
- WINDOW_PATTERN SSSL→SL: val_bpb=1.0655 (no help)

### Generation 4 — Biggest Win!
- **EMBEDDING_LR 0.549→0.709**: val_bpb=**1.0643** (biggest single improvement!)
- DEPTH 8→9 + MATRIX_LR↑: val_bpb=1.092 (much worse — larger model can't converge)

### Evolution Curve
```
val_bpb
1.0660 ┤● gen0 baseline
1.0653 ┤ ╲ gen2 (EMBED_LR↓)
1.0651 ┤  ╲ gen3 (SCALAR_LR↓)
1.0643 ┤   ╲● gen4 (EMBED_LR↑) ← best!
       └─────────────────────
        0   1   2   3   4
```

## The Result

### What Worked (all conservative LR changes)
| Change | Delta | Generation |
|--------|-------|-----------|
| EMBEDDING_LR 0.549→0.709 | **-0.0008** | gen4 |
| EMBEDDING_LR 0.6→0.549 | -0.0003 | gen2 |
| SCALAR_LR 0.5→0.362 | -0.0002 | gen3 |

### What Failed (all moderate/aggressive changes)
| Change | Delta | Why |
|--------|-------|-----|
| TOTAL_BATCH_SIZE 2^20 | +0.029 | Too few optimization steps |
| DEPTH 9 + MATRIX_LR↑ | +0.027 | Can't converge in 500 steps |
| WINDOW_PATTERN SSSL→L | timeout | Full attention too slow |

### Infrastructure Hiccup
Mid-experiment, the pipeline crashed: `ResourceLimitExceeded` — previous generation's jobs hadn't terminated before the next generation submitted. Fix: added retry logic with exponential backoff.

## Lessons Learned
- **Conservative wins for short training** — the 5-minute budget strongly constrains what changes can show improvement
- **EMBEDDING_LR is the most sensitive parameter** — both lowering and raising it improved results
- **Architecture changes need longer training** — DEPTH/width changes can't converge in ~500 steps
- **Quota collision is real** — add retry logic for `ResourceLimitExceeded`
- **The pipeline works autonomously** — 5 generations of evolution without human intervention

## Try It Yourself
```bash
# Run the full autonomous pipeline
python -m src.pipeline.orchestrator --generations 5 --population 4

# Or with Makefile
make run
```

### Running Cost
| Phase | Action | Cost | Cumulative |
|-------|--------|------|-----------|
| Planning | Architecture | $0.00 | $0.00 |
| Building | Code generation | $0.00 | $0.00 |
| Infra | IAM + S3 + quotas | $0.00 | $0.00 |
| Exp 1 | First success | $0.06 | $0.06 |
| Exp 2a | Batch size trap | $0.07 | $0.13 |
| Exp 2b | 5-gen evolution (20 runs) | $0.31 | $0.44 |
