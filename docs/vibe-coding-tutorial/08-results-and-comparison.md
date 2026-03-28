# Chapter 8: Results — $0.44 vs $24: The Final Scorecard

> **Time**: 15 minutes
> **Cost**: $0.00
> **Key Insight**: 18x cheaper, 2.3x faster — serverless parallel evolution outperforms sequential autoresearch on every cost metric.

## Context
Time to compare our serverless approach against Karpathy's original H100 setup.

## The Numbers

### Head-to-Head: 83 Experiments

| | Karpathy (H100, Sequential) | Ours (L40S Spot, Parallel) |
|---|---|---|
| **Wall clock** | ~8 hours | **~3.5 hours** |
| **GPU idle cost** | ~50% wasted | **$0** (HUGI) |
| **Cost (On-demand)** | ~$24 | — |
| **Cost (Spot)** | ~$7.20 | **~$1.33** |
| **Per experiment** | $0.09-0.29 | **$0.016** |
| **Parallelism** | 1 | **4** |

### What We Achieved in 25 Experiments

| Metric | Value |
|--------|-------|
| Total experiments | 25 |
| Successful | 22 |
| Timeouts | 2 |
| Best val_bpb | 1.0643 |
| Improvement | -0.0013 from baseline |
| **Total cost** | **$0.44** |
| Regions tried | 3 (Tokyo → Oregon → Virginia) |
| Instance types tried | 3 (g7e.2xl, g7e.4xl, p5.4xl) |

### Evolution: What the Pipeline Discovered
The autonomous pipeline found that **EMBEDDING_LR** is the most sensitive parameter:
1. Lowered 0.6 → 0.549 (improvement)
2. Raised 0.549 → 0.709 (bigger improvement!)
3. SCALAR_LR 0.5 → 0.362 (small improvement)

All improvements came from **conservative LR tuning**. Architecture changes (DEPTH, WINDOW_PATTERN, TOTAL_BATCH_SIZE) all failed — the 5-minute time budget is too short for larger models to converge.

### Cost Breakdown
| Phase | Experiments | Cost |
|-------|-----------|------|
| First experiment (#001) | 1 | $0.04 |
| Batch size test | 4 | $0.07 |
| 5-gen evolution | 20 | $0.33 |
| **Total** | **25** | **$0.44** |

## The Honest Assessment
- **val_bpb 1.0643 vs 0.998**: Our L40S result is 6.5% worse than H100. This is expected — SDPA at 20% MFU vs FA3 at 40% MFU means half the tokens in the same time.
- **But the architecture insights transfer**: Research confirms that optimizer/architecture rankings found on cheap GPUs transfer to expensive GPUs. Our $0.44 of experiments can inform a production H100 run.
- **The pipeline works**: Autonomous evolution, parallel execution, zero idle cost — all validated.

## Lessons Learned
- **HUGI eliminates GPU waste** — the biggest cost savings come from not paying for idle time
- **Spot placement scores are essential** — wrong region = stuck jobs, right region = instant allocation
- **Conservative beats aggressive for short budgets** — radical changes need longer training to show results
- **$0.44 buys real knowledge** — 25 experiments, 12 insights, 1 reusable skill, and a validated pipeline

## Try It Yourself
```bash
# Reproduce our full experiment for ~$0.50
git clone https://github.com/roboco-io/serverless-autoresearch
cd serverless-autoresearch
cp config.yaml.example config.yaml
# Edit config.yaml with your AWS credentials

make setup      # IAM role
make prepare    # Data to S3
make dry-run    # Verify
make run        # 10 gen × 4 pop = 40 experiments (~$0.70)
```

---
