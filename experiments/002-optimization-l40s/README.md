# Experiment #002: L40S Optimization — Batch Size Doubling + Parallel Evolution

## Objective

Improve val_bpb on L40S (ml.g7e.4xlarge) by doubling DEVICE_BATCH_SIZE and running the parallel evolution pipeline. The first experiment (#001) showed only 47% VRAM utilization, leaving significant room for optimization.

## Hypothesis

Doubling DEVICE_BATCH_SIZE from 64 to 128 will:
1. Increase token throughput from ~260M to ~450M in 5 minutes
2. Reduce val_bpb from 1.065 toward ~1.01-1.03
3. Keep VRAM under 48GB (estimated ~38GB peak)

## Changes from Baseline (#001)

| Parameter | Experiment #001 | Experiment #002 | Rationale |
|-----------|----------------|-----------------|-----------|
| DEVICE_BATCH_SIZE | 64 | **128** | VRAM was 47% utilized — room to double |
| Pipeline mode | Single experiment | **4 parallel** | g7e.4xlarge quota increased to 4 |
| Everything else | Same | Same | Isolate batch size effect |

## Pre-experiment Research

Full research notes: [references/l40s-optimization-strategies.md](../../references/l40s-optimization-strategies.md)

### Key Findings from Perplexity Research

**1. Flash Attention 2 for Ada Lovelace (sm_89)**
- FA2 has pre-compiled wheels for L40S — could provide 2-4x attention speedup over SDPA
- FA3 is Hopper-only; FA2 is the best available option for sm_89
- Not applied in this experiment (future work)

**2. Batch Size Scaling (Applied in this experiment)**
- Current 22.8GB VRAM usage leaves 25GB free
- DEVICE_BATCH_SIZE=128 estimated to use ~38GB — safe within 48GB
- Directly increases token throughput → more training in 5-min budget

**3. Learning Rate Tuning (Future work)**
- Higher MATRIX_LR (0.06-0.08) recommended for fewer steps
- WARMUP_RATIO 0.02-0.05 stabilizes higher LR
- Will be explored by the candidate generator's conservative strategy

**4. torch.compile Optimizations (Future work)**
- `TORCH_CUDA_ARCH_LIST="8.9"` for sm_89-specific kernels
- `mode="max-autotune"` for better kernel fusion
- Trade-off: compilation overhead vs runtime gain in 5-min budget

### Sources
- [Pre-built FA2 wheel for L40S](https://discuss.huggingface.co/t/prebuilt-flashattention-2-8-0-post2-wheel-for-nvidia-l40s-cuda-12-1/169766)
- [Dao-AILab/flash-attention #1978](https://github.com/Dao-AILab/flash-attention/issues/1978)
- [KellerJordan/Muon optimizer](https://github.com/KellerJordan/Muon)
- [MuonAll paper (arXiv 2511.06086)](https://arxiv.org/html/2511.06086v1)
- [NanoGPT speedrun records](https://kellerjordan.github.io/posts/muon/)

## Experiment Setup

- **Instance:** ml.g7e.4xlarge (NVIDIA L40S, 48GB, Ada Lovelace sm_89)
- **Region:** us-west-2 (Oregon)
- **Pricing:** Spot
- **Pipeline:** 4 candidates × 1 generation (parallel)
- **Attention:** PyTorch SDPA (FA3 not supported on sm_89)
- **Framework:** SageMaker PyTorch DLC 2.8.0 / py312

### Candidates Submitted

| ID | Strategy | Description |
|----|----------|-------------|
| v00 | Baseline | DEVICE_BATCH_SIZE=128 (unmodified from new baseline) |
| v01 | Conservative | LR adjustment |
| v02 | Conservative | LR adjustment |
| v03 | _(failed to submit — quota hit)_ | — |

### Infrastructure Notes

- g7e.4xlarge spot quota was 1 → requested increase to 4 → **auto-approved**
- g7e.2xlarge also approved to 4
- g7e.8xlarge/12xlarge still CASE_OPENED
- Previous Stopping job consumed 1 quota slot → only 3 of 4 candidates submitted successfully
- Orchestrator needs error handling for ResourceLimitExceeded (TODO)

## Results

_Pending — 3 jobs running in parallel on SageMaker Spot_

| Candidate | val_bpb | peak_vram_mb | mfu_percent | tokens_M | Status |
|-----------|---------|-------------|-------------|----------|--------|
| v00 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | InProgress |
| v01 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | InProgress |
| v02 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | InProgress |

## Lessons Learned (so far)

1. **Quota management is critical** — Always check running/stopping jobs before submitting new ones
2. **Spot allocation varies by size** — g7e.2xlarge had no spot capacity; g7e.4xlarge did
3. **Auto-approval for quota increases** — g7e.2xlarge and g7e.4xlarge were auto-approved within minutes
4. **Pipeline needs ResourceLimitExceeded handling** — Should catch and retry/skip instead of crashing

---

*Experiment started: 2026-03-28*
*Status: Running*
