# Vibe Coding Tutorial: Running Autoresearch on AWS for $0.44

> How we used conversational AI coding to build a parallel ML experiment pipeline on SageMaker Spot, running 25 autonomous experiments for less than a cup of coffee.

## What You'll Learn

- How to design and build a complete ML pipeline through AI conversation (vibe coding)
- AWS SageMaker Spot Training: setup, Spot capacity, quota management, cost optimization
- The HUGI pattern (Hurry Up and Get Idle) for serverless GPU workloads
- Debugging GPU compatibility issues (Flash Attention 3, CUDA architectures)
- Why cheap GPU experiments can validate expensive production training decisions

## Prerequisites

- AWS account with CLI configured
- Python 3.11+
- Basic understanding of ML training concepts
- Claude Code (or similar AI coding assistant)

## Chapters

| # | Chapter | Time | Cost | Key Takeaway |
|---|---------|------|------|-------------|
| 1 | [The Idea](01-the-idea.md) | 15 min | $0.00 | Start with a deep interview to refine your goal |
| 2 | [Building the Pipeline](02-building-the-pipeline.md) | 45 min | $0.00 | Plan mode prevents wasted work; parallel agents speed up coding |
| 3 | [Infrastructure Adventures](03-infrastructure-adventures.md) | 2 hr | $0.00 | Check Spot placement scores BEFORE choosing a region |
| 4 | [First Experiment](04-first-experiment.md) | 1 hr | $0.06 | FA3 is GPU-specific; always have an SDPA fallback |
| 5 | [The Batch Size Trap](05-the-batch-size-trap.md) | 30 min | $0.07 | DEVICE_BATCH_SIZE ≠ throughput; increase TOTAL_BATCH_SIZE |
| 6 | [Autonomous Evolution](06-autonomous-evolution.md) | 2.5 hr | $0.31 | Conservative LR tuning wins for short training budgets |
| 7 | [Insights & Skills](07-insights-and-skills.md) | 30 min | $0.00 | Turn every lesson into a reusable skill |
| 8 | [Results & Comparison](08-results-and-comparison.md) | 15 min | $0.00 | 18x cheaper, 2.3x faster than original |

## Total

- **Time**: ~8 hours (including all debugging and region migration)
- **Cost**: $0.44 (25 SageMaker Spot experiments)
- **Result**: val_bpb improved from 1.0656 to 1.0643 through autonomous evolution
- **Output**: Working pipeline, 12 documented insights, 1 reusable Claude Code skill

## The Vibe Coding Approach

This tutorial was built entirely through natural language conversation with Claude Code. Every piece of code, every AWS configuration, every debugging session started with a prompt. The prompts are included verbatim — they're the tutorial.

---

_Generated from Claude Code conversation logs (2026-03-27 ~ 2026-03-28)_
_Last generated: 2026-03-28_
_Last log message UUID: conversation end_
