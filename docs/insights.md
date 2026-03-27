# Serverless Autoresearch — Key Insights

> Lessons learned from running autonomous ML experiments on SageMaker Spot Training.

## 1. Spot Capacity Varies Dramatically by Region

**Discovery:** The same instance type can have Spot placement score 1 (near-impossible) in one region and 9 (instant) in another.

| Region | g7e Score | Result |
|--------|----------|--------|
| us-west-2 | 1-2 | Stuck "Starting" 30+ min |
| us-east-1 | 9 | Allocated in ~2 min |

**Rule:** Always run `aws ec2 get-spot-placement-scores` before choosing a region. See [Spot Capacity Guide](spot-capacity-guide.md).

## 2. Larger Instances Can Be Cheaper on Spot

**Discovery:** g7e.8xlarge ($0.93/hr) was cheaper than g7e.2xlarge ($0.94-$1.82/hr) in us-west-2 because larger instances have less Spot demand.

**Rule:** Check Spot price history for all sizes — don't assume smaller = cheaper.

## 3. DEVICE_BATCH_SIZE ≠ Token Throughput

**Discovery:** Doubling DEVICE_BATCH_SIZE from 64 to 128 with the same TOTAL_BATCH_SIZE **worsened** val_bpb (1.065 → 1.081).

**Why:** With TOTAL_BATCH_SIZE fixed at 2^19, larger DEVICE_BATCH_SIZE reduces gradient accumulation steps (4 → 2) without increasing total tokens processed. It just uses more VRAM for the same work.

**Rule:** To increase throughput, increase TOTAL_BATCH_SIZE (more tokens per optimizer step), not just DEVICE_BATCH_SIZE.

## 4. Flash Attention 3 is GPU-Architecture Specific

**Discovery:** FA3 pre-compiled kernels only support Hopper (sm_90) and Ampere (sm_80/86). Ada Lovelace (sm_89, L40S) is **not supported**, causing runtime CUDA errors.

**Solution:** Explicit compute capability check + PyTorch SDPA fallback. FA2 has community wheels for sm_89.

**Impact:** SDPA gives ~20% MFU vs ~40% with FA3 — half the attention efficiency.

## 5. SageMaker Startup Overhead is Significant

**Discovery:** Each SageMaker Training Job has ~3 min startup overhead (instance allocation + container pull + data download + pip install). For 5-min training jobs, this is **60% overhead**.

**Optimization paths:**
- **Scale up:** Use multi-GPU instance, run N experiments on 1 job (amortize startup)
- **Pre-install deps:** Bake packages into Docker image instead of pip install at runtime
- **Warm pools:** SageMaker warm pools keep instances alive between jobs (but costs money)

## 6. Quota Management is a First-Class Concern

**Discovery:** GPU Spot quotas default to 0 for new instance types. g7e auto-approved within minutes; p5/p6 require manual review (CASE_OPENED, days).

**Rule:** Request quotas in multiple regions upfront. g7e tends to auto-approve; p5+ needs lead time.

## 7. SageMaker Profiler Doesn't Support All Instance Types

**Discovery:** `ml.g7e` instances throw `ValidationException: Profiler is currently not supported` at job creation.

**Fix:** Set `disable_profiler=True` in the PyTorch Estimator.

## 8. The Parallel Evolution Approach Works

**Validated:** The pipeline successfully generates candidates, submits parallel Spot jobs, collects results, and selects the best — all autonomously.

**Cost efficiency:** 4 parallel experiments for $0.066 total, results in ~10 min wall clock (excluding Spot wait time in us-west-2).

## 9. PyArrow Version Matters

**Discovery:** The SageMaker DLC has pyarrow 23.x, but the local environment may have an older version causing `Repetition level histogram size mismatch` when reading parquet files.

**Fix:** Ensure `pyarrow>=21.0.0` in requirements-train.txt.

## 10. config.yaml Should Never Be in Git

**Discovery:** config.yaml contains AWS role ARN, profile, and region — environment-specific and potentially sensitive.

**Rule:** Gitignore config.yaml, provide config.yaml.example as template.
