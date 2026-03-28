# Reddit Post Draft

## Title (r/MachineLearning)

```
[P] I vibe-coded Karpathy's autoresearch on AWS for $0.44 — here's the full tutorial so you can do it too
```

## Body

---

**TL;DR**: I built a parallel ML experiment pipeline on SageMaker Spot by talking to an AI for 13 hours. 25 autonomous experiments for $0.44. The whole journey is documented as an **[8-chapter hands-on tutorial](https://github.com/roboco-io/serverless-autoresearch/tree/main/docs/vibe-coding-tutorial)** — follow along step by step and build it yourself. [GitHub](https://github.com/roboco-io/serverless-autoresearch)

---

### What This Actually Is

This is **not** just another repo to star and forget. It's a **learn-by-doing tutorial** that walks you through building a complete ML experiment pipeline from scratch — using only natural language prompts.

Every chapter shows:
- 🎯 **The actual prompt** I typed (in Korean, with English translation)
- 🔧 **What the AI did** — architecture decisions, code generated, tools used
- 💥 **What went wrong** — CUDA crashes, stuck Spot instances, the batch size trap
- 💡 **The insight** — what I learned and what you should know
- 🏃 **Try it yourself** — exact commands to reproduce each step

### The 8 Chapters

| Ch | Title | What You'll Learn | Cost |
|----|-------|------------------|------|
| 1 | [The Idea](https://github.com/roboco-io/serverless-autoresearch/blob/main/docs/vibe-coding-tutorial/01-the-idea.md) | How a deep interview turns a vague idea into a focused architecture | $0 |
| 2 | [Building the Pipeline](https://github.com/roboco-io/serverless-autoresearch/blob/main/docs/vibe-coding-tutorial/02-building-the-pipeline.md) | 23 files generated in one session using parallel AI agents | $0 |
| 3 | [Infrastructure Adventures](https://github.com/roboco-io/serverless-autoresearch/blob/main/docs/vibe-coding-tutorial/03-infrastructure-adventures.md) | Spot capacity hunting across 3 regions, quota wars, the placement score trick | $0 |
| 4 | [First Experiment](https://github.com/roboco-io/serverless-autoresearch/blob/main/docs/vibe-coding-tutorial/04-first-experiment.md) | FA3 CUDA crash → SDPA fallback → first $0.02 success | $0.06 |
| 5 | [The Batch Size Trap](https://github.com/roboco-io/serverless-autoresearch/blob/main/docs/vibe-coding-tutorial/05-the-batch-size-trap.md) | Why doubling batch size made things WORSE (and what to do instead) | $0.07 |
| 6 | [Autonomous Evolution](https://github.com/roboco-io/serverless-autoresearch/blob/main/docs/vibe-coding-tutorial/06-autonomous-evolution.md) | 5 generations of autonomous ML evolution — what worked, what failed | $0.31 |
| 7 | [Insights & Skills](https://github.com/roboco-io/serverless-autoresearch/blob/main/docs/vibe-coding-tutorial/07-insights-and-skills.md) | Turning experiments into reusable knowledge | $0 |
| 8 | [Results](https://github.com/roboco-io/serverless-autoresearch/blob/main/docs/vibe-coding-tutorial/08-results-and-comparison.md) | Final scorecard: 18x cheaper, 2.3x faster than the original | $0 |

**Total: ~8 hours, $0.44, and you'll understand SageMaker Spot, GPU quirks, and autonomous ML research.**

### Why a Tutorial, Not Just Code

I see a lot of "look what I built" posts. But the **how** is always missing. How did you decide on that architecture? What did you try that failed? How much did it actually cost?

This tutorial answers all of that because it's generated from the actual conversation logs. The prompts are real. The failures are real. The $0.44 is real.

Example from Chapter 5 (The Batch Size Trap):

> I doubled DEVICE_BATCH_SIZE from 64 to 128. VRAM usage went from 22GB to 45GB.
> And val_bpb got... **worse**.
>
> Turns out DEVICE_BATCH_SIZE ≠ throughput. With fixed TOTAL_BATCH_SIZE,
> bigger micro-batches just reduce gradient accumulation steps without
> processing more tokens. $0.07 lesson learned.

These are the kinds of mistakes that save you hours when you know about them in advance.

### The Pipeline (for those who want the technical details)

Built on Karpathy's [autoresearch](https://github.com/karpathy/autoresearch), adapted for serverless:

| | Original | This Project |
|---|---|---|
| Execution | 1 experiment at a time | **4 parallel** (Spot) |
| 83 experiments | ~8 hours, ~$24 | **~3.5 hours, ~$1.33** |
| GPU idle cost | ~50% wasted | **$0** (HUGI pattern) |
| GPU required | H100 80GB | **Any** (L40S, A10G, H100) |

**HUGI pattern** (Hurry Up and Get Idle): burst N GPUs for 5 minutes → terminate all → pay $0 until the next generation. The key insight: you're paying for 42 minutes of actual GPU time, not 8 hours of reserved instance.

### What the Pipeline Discovered (Autonomously)

Over 5 generations of evolution, the pipeline found:

**✅ What worked** (all conservative LR changes):
- EMBEDDING_LR 0.6 → 0.709 (biggest win)
- SCALAR_LR 0.5 → 0.362
- EMBEDDING_LR 0.6 → 0.549

**❌ What failed** (all radical changes):
- TOTAL_BATCH_SIZE 2^19 → 2^20 (val_bpb +0.029 worse)
- DEPTH 8 → 9 (can't converge in 500 steps)
- WINDOW_PATTERN SSSL → L (timeout — full attention too slow)

Key finding: **conservative LR tuning beats radical architecture changes when training time is short** (5 min / ~500 steps).

### Hardest Lessons (So You Don't Have To)

1. **Always check `aws ec2 get-spot-placement-scores` before choosing a region.** Same GPU: score 1 in Oregon (stuck 30+ min), score 9 in Virginia (allocated in 2 min). This one command saves hours.

2. **Flash Attention 3 is Hopper-only.** L40S (Ada Lovelace, sm_89) will crash at runtime. Not at import. At runtime. You need explicit compute capability detection + SDPA fallback.

3. **Cheap experiments validate expensive ones.** Research confirms optimizer/architecture rankings found on $0.04 L40S experiments transfer to H100 production training. Use Spot as your hypothesis testbed.

### Get Started

```bash
git clone https://github.com/roboco-io/serverless-autoresearch
cd serverless-autoresearch

# Follow the tutorial from Chapter 1, or jump straight to running:
cp config.yaml.example config.yaml
make setup      # IAM role (~2 min)
make prepare    # Data → S3 (~5 min)
make dry-run    # Verify (free)
make run        # 40 experiments (~$0.70)
```

### Links

- **GitHub**: https://github.com/roboco-io/serverless-autoresearch
- **Start the tutorial**: [Chapter 1: The Idea](https://github.com/roboco-io/serverless-autoresearch/blob/main/docs/vibe-coding-tutorial/01-the-idea.md)
- **Full tutorial index**: [8 chapters](https://github.com/roboco-io/serverless-autoresearch/tree/main/docs/vibe-coding-tutorial)
- **Key insights**: [12 battle-tested lessons](https://github.com/roboco-io/serverless-autoresearch/blob/main/docs/insights.md)
- **Spot capacity guide**: [How to check before you run](https://github.com/roboco-io/serverless-autoresearch/blob/main/docs/spot-capacity-guide.md)

Has anyone else tried running autoresearch on cloud infrastructure? What's your cheapest ML experiment setup? Would love to hear about other approaches.

---

## Title Variants

### For r/aws
```
SageMaker Spot tutorial: 25 GPU training experiments for $0.44 using HUGI pattern — step-by-step guide from zero to autonomous ML pipeline
```

### For r/LocalLLaMA
```
No H100? No problem. Full tutorial: run Karpathy's autoresearch on L40S Spot for $0.44 (8 chapters, learn by doing)
```

### For r/ChatGPTCoding
```
I vibe-coded an entire ML pipeline in 13 hours — here's the 8-chapter tutorial showing every prompt, failure, and $0.44 of experiments
```

### For r/learnmachinelearning
```
Tutorial: Build an autonomous ML experiment pipeline on AWS for under $1 — step by step from idea to results (based on Karpathy's autoresearch)
```

## Posting Schedule

| Day | Subreddit | Focus |
|-----|-----------|-------|
| Tue 9AM EST | r/MachineLearning | Technical depth + tutorial |
| Wed 10AM EST | r/aws | SageMaker Spot patterns |
| Thu 9AM EST | r/LocalLLaMA | Cost efficiency angle |
| Fri 10AM EST | r/ChatGPTCoding | Vibe coding process |
| Sat 10AM EST | r/learnmachinelearning | Learning-focused |
