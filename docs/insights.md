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

## 3. DEVICE_BATCH_SIZE ≠ Token Throughput (hardware-dependent — see also #13)

**Discovery:** Doubling DEVICE_BATCH_SIZE from 64 to 128 with the same TOTAL_BATCH_SIZE **worsened** val_bpb (1.065 → 1.081) **on L40S with SDPA**.

**Why:** With TOTAL_BATCH_SIZE fixed at 2^19, larger DEVICE_BATCH_SIZE reduces gradient accumulation steps (4 → 2) without increasing total tokens processed. It just uses more VRAM for the same work.

**Rule (L40S):** To increase throughput, increase TOTAL_BATCH_SIZE (more tokens per optimizer step), not just DEVICE_BATCH_SIZE.

**Update (Experiment #003, H100):** The exact opposite held on H100 with FA3 — BS=64 → 128 *improved* val_bpb by 0.0065 (1.0016 → 0.9951), more than 4× the entire 5-gen LR evolution gain. See insight #13 for the reconciled picture.

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

## 11. Spot GPUs Are Valid Proxies for Large-Scale Training

**Discovery:** Research confirms that hyperparameter optimization on cheaper GPUs (L40S) transfers well to expensive GPUs (H100) for production training.

**What transfers:**
- Optimizer choices (Muon vs AdamW) — relative rankings hold across hardware
- Architecture decisions (depth, width, attention patterns) — hardware-independent
- LR schedule shapes (cosine, warmup ratios) — direction transfers, absolute values need adjustment
- Relative hyperparameter rankings — "A is better than B" conclusions are portable

**What doesn't transfer:**
- Absolute val_bpb values — depend on GPU throughput
- Optimal batch sizes — depend on VRAM (48GB vs 80GB)
- Memory-dependent optimizations — FA3 (Hopper only), FP8, etc.
- Absolute learning rate values — need per-scale tuning without muP

**Rule:** Use Spot for Phase 1 (hypothesis validation at $0.04/experiment), then apply winning architecture/optimizer choices to Phase 2 (full-scale training on H100). Use muP for direct LR transfer across scales.

**References:**
- [MLPerf BERT HPC Optimization (arXiv 2402.02447)](https://arxiv.org/pdf/2402.02447)
- [Improving HPO with Checkpointed Weights (NVIDIA 2024)](https://research.nvidia.com/publication/2024-06_improving-hyperparameter-optimization-checkpointed-model-weights)
- [muP Scaling (arXiv 2410.22854)](https://arxiv.org/html/2410.22854v3)

## 12. DEVICE_BATCH_SIZE ≠ More Training (L40S-specific; reversed on H100)

**Discovery (Experiment #002, L40S + SDPA):** Doubling DEVICE_BATCH_SIZE from 64 to 128 while keeping TOTAL_BATCH_SIZE=2^19 **worsened** val_bpb (1.065 → 1.081). It only reduced gradient accumulation steps (4 → 2) without increasing total tokens.

**Rule (L40S):** To increase throughput, increase TOTAL_BATCH_SIZE. DEVICE_BATCH_SIZE only affects VRAM usage and gradient accumulation granularity.

**Update (Experiment #003, H100 + FA3):** The same swap 64 → 128 *improved* val_bpb by 0.0065 (1.0016 → 0.9951). DEVICE_BATCH_SIZE behaves differently on L40S/SDPA vs H100/FA3 — see insight #13.

## 13. Batch Size × LR × Hardware Interact — Evolved LRs Can Be BS-Specific

**Discovery (Experiment #003):** A 5-generation evolution on H100 with BS=64 found new "optimal" learning rates (EMBEDDING_LR 0.7091 → 0.6433, UNEMBEDDING_LR 0.003369 → 0.004206). When we later restored BS to the upstream 128, those "optimal" LRs performed **worse** than the original L40S-evolved LRs (0.7091 / 0.003369) — which, combined with BS=128, produced val_bpb=0.9951, *below* the upstream 0.998 baseline.

**Why:** Effective learning rate scales with batch size. The Phase-2 LR tweaks compensated for BS=64's noisier gradients; once BS was doubled, that compensation became overcorrection.

**Rule:**
- Evolve LR and BS *jointly*, not one while fixing the other to a suboptimal value.
- When transferring an evolved configuration across hardware or batch sizes, revisit LRs — the configuration's components are coupled.
- If you must fix BS (e.g. VRAM constraints), any LR you evolve is a *BS-conditional* optimum, not a universal one.

**Cost lesson:** BS=64 → 128 alone gave -0.0065 on val_bpb in a single run ($0.16). The full 5-gen LR evolution on BS=64 gave only -0.0014 across 20 runs ($3). **Checking the hyperparameter you assumed fixed can be 100× more cost-effective than further search around it.**

## 14. Cheap-GPU-Evolved LRs Transfer to Expensive GPUs — Sometimes Better Than Re-Evolving

**Discovery (Experiments #002 → #003):** Learning rates discovered on L40S Spot ($0.40 for 24 experiments) transferred to H100 and, with BS restored to the upstream value, produced a result **below the upstream H100 baseline** (0.9951 < 0.998). The Phase-2 H100-native LR evolution actually *moved away* from the better operating point (because it was exploring around the wrong BS).

**Why:** Hyperparameter *rankings* are largely hardware-independent; *absolute values* interact with batch size more than with GPU architecture. The L40S-evolved LRs happened to sit near the H100+BS=128 optimum.

**Rule:** Use cheap Spot GPUs (L40S, A10G) for Phase-1 LR/architecture exploration. When moving to expensive GPUs for final runs, first try the cheap-GPU-evolved config as-is — it's often a strong starting point, and occasionally the final answer.

## 15. Serverless Spot Can Match or Beat Dedicated H100 Results at 44–150× Lower Cost

**Discovery (Experiment #003 Phase 3):** Reproduced Karpathy's upstream H100 autoresearch result (val_bpb ~0.998) and slightly surpassed it (0.9951) using a **single ml.p5.4xlarge Spot run of 229 seconds, costing ~$0.16**.

**Comparison:**

| | Karpathy upstream | This (Phase 3) |
|---|---|---|
| GPU | H100, 8 h continuous | H100 Spot, 229 s billable |
| val_bpb | ~0.998 | **0.9951** |
| Cost | $7–24 (H100 $1–3/hr × 8 h) | **$0.16** |
| Wall clock | ~8 h | ~20 min (incl. Spot wait) |

**Savings: 44–150× in cost, ~24× in wall clock.**

**Caveat:** Both numbers are single-run values. Claiming statistical superiority requires repeated runs to estimate ±σ — but matching/beating the reported number at 1–2% of the cost is itself a meaningful serverless-ML result.

**Rule:** For reproducing or extending published ML results, Spot+HUGI is a strict improvement when the job is under ~30 minutes. Beyond that, Spot interrupt risk starts to matter.
