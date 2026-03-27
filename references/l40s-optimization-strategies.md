# Research Notes: L40S Optimization Strategies

> Pre-experiment research for maximizing val_bpb on NVIDIA L40S (48GB, Ada Lovelace sm_89) without Flash Attention 3.

**Sources:** Perplexity AI research (2026-03-28), Karpathy autoresearch community, Dao-AILab/flash-attention, KellerJordan/Muon

---

## Current Baseline (Experiment #001)

| Metric | Value | Gap vs H100 |
|--------|-------|-------------|
| val_bpb | 1.065 | +6.7% (worse) |
| MFU | 20.5% | -19.3pp |
| Peak VRAM | 22.8 GB / 48 GB | 47% utilized |
| Tokens (5min) | 260.6M | 52% of H100 |
| DEVICE_BATCH_SIZE | 64 | 50% of H100 |

**Root cause of gap:** No FA3 on sm_89 → SDPA fallback → 2x slower attention → half the tokens in 5 min.

---

## Strategy 1: Flash Attention 2 for Ada Lovelace (HIGH PRIORITY)

**Finding:** FA2 has pre-compiled wheels for sm_89 (L40S). FA3 is Hopper-only, but **FA2 provides 2-4x attention speedup over SDPA**.

**Action:**
- Install `flash-attn==2.8.0.post2` wheel for CUDA 12.x / PyTorch 2.x / sm_89
- Modify train.py FA detection:
  ```python
  # Priority: FA3 (Hopper) > FA2 (Ampere/Ada) > SDPA (fallback)
  if cap == (9, 0):
      fa3 = ...  # Hopper FA3
  elif cap[0] >= 8:
      from flash_attn import flash_attn_func  # FA2 for sm_80+
  else:
      # SDPA fallback
  ```
- FA2 API differs from FA3: `flash_attn_func(q, k, v, causal=True)` — no `window_size` param in FA2 base

**Expected impact:** MFU 20.5% → 30-35%, tokens 260M → 400-450M

**Risk:** FA2 wheel compatibility with SageMaker DLC PyTorch version. May need `--force-reinstall`.

**References:**
- [Pre-built FA2 wheel for L40S](https://discuss.huggingface.co/t/prebuilt-flashattention-2-8-0-post2-wheel-for-nvidia-l40s-cuda-12-1/169766)
- [Dao-AILab/flash-attention #1978](https://github.com/Dao-AILab/flash-attention/issues/1978)

---

## Strategy 2: Increase DEVICE_BATCH_SIZE (HIGH PRIORITY)

**Finding:** Only 47% VRAM used. Doubling batch size directly increases token throughput.

| DEVICE_BATCH_SIZE | Est. Peak VRAM | Est. Tokens/5min | Est. val_bpb |
|-------------------|----------------|------------------|--------------|
| 64 (current) | 22.8 GB | 260M | 1.065 |
| **128** | ~38 GB | ~450M | ~1.01-1.03 |
| 192 | ~45 GB | ~520M | ~0.99-1.01 |

**Action:** Set `DEVICE_BATCH_SIZE = 128`. If no OOM, try 192.

**Expected impact:** ~1.7x token throughput → significant val_bpb improvement

**Risk:** OOM at 192 (45GB close to 48GB limit). Use 128 as safe default.

---

## Strategy 3: torch.compile Optimizations (MEDIUM PRIORITY)

**Finding:** Ada Lovelace benefits from targeted torch.compile settings.

**Action:**
- Set environment variable: `TORCH_CUDA_ARCH_LIST="8.9"`
- Already using `torch.compile(model, dynamic=False)` — this is good
- Consider adding `mode="max-autotune"` for kernel autotuning (adds compilation time but better runtime)

**Expected impact:** 10-20% MFU improvement from better kernel fusion

**Risk:** Compilation overhead (~30-60s) eats into 5-min budget. First run is slower.

---

## Strategy 4: Learning Rate Tuning (MEDIUM PRIORITY)

**Finding:** With fewer steps (~500), LR schedule matters more. Muon convergence research suggests:

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| MATRIX_LR | 0.04 | 0.06-0.08 | Higher LR for fewer steps |
| EMBEDDING_LR | 0.6 | 0.8-1.0 | Sparse gradients need higher LR |
| WARMUP_RATIO | 0.0 | 0.02-0.05 | Small warmup stabilizes higher LR |
| WARMDOWN_RATIO | 0.5 | 0.4 | Less warmdown = more time at peak LR |
| WEIGHT_DECAY | 0.2 | 0.1 | Less regularization for short runs |

**Expected impact:** 0.01-0.03 val_bpb improvement

**References:**
- [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — muP scaling recommendations
- [NanoGPT speedrun records](https://kellerjordan.github.io/posts/muon/)

---

## Strategy 5: Model Architecture Scaling (LOW PRIORITY)

**Finding:** With 25GB free VRAM, could increase model size.

| Config | DEPTH | n_embd | Params | Est. VRAM | Trade-off |
|--------|-------|--------|--------|-----------|-----------|
| Current | 8 | 512 | 50M | 22.8 GB | Baseline |
| Wider | 8 | 768 | 110M | ~35 GB | Fewer steps, bigger model |
| Deeper | 12 | 512 | 75M | ~30 GB | More layers, same width |

**Caution:** Larger model = fewer steps in 5 min. For short runs, **wider is usually better than deeper** (less sequential dependency). But with only 500 steps, larger models may not converge enough.

**Action:** Only try after strategies 1-4 are exhausted.

---

## Strategy 6: Softcap Tuning (LOW PRIORITY)

**Finding:** Lower softcap (10-12 vs 15) sharpens early distributions.

**Action:** Try `softcap = 12` in train.py forward method.

**Expected impact:** Small (0.005-0.01 val_bpb)

---

## Experiment Plan for #002

### Phase 1: Quick Wins (batch size + LR)
1. `DEVICE_BATCH_SIZE = 128` (double current)
2. `MATRIX_LR = 0.06`
3. `WARMUP_RATIO = 0.03`
4. Run baseline with these changes → measure val_bpb improvement

### Phase 2: FA2 Integration (if available)
5. Install flash-attn 2.x in requirements-train.txt
6. Add FA2 fallback path in train.py
7. Run with FA2 → expect biggest MFU gain

### Phase 3: Full Pipeline
8. Run 10-generation parallel evolution with best configuration
9. Let candidate_generator explore around the new baseline

### Target
- val_bpb: **< 1.02** (closing 50%+ of the H100 gap)
- MFU: **> 30%**
- Tokens: **> 400M** in 5 min

---

## Key References

| Resource | URL | Relevance |
|----------|-----|-----------|
| Flash Attention 2 L40S wheel | [HF Discussion](https://discuss.huggingface.co/t/prebuilt-flashattention-2-8-0-post2-wheel-for-nvidia-l40s-cuda-12-1/169766) | FA2 for sm_89 |
| Dao-AILab FA issues | [#1978](https://github.com/Dao-AILab/flash-attention/issues/1978), [#2376](https://github.com/Dao-AILab/flash-attention/issues/2376) | FA4 sm_89 status |
| Muon optimizer | [GitHub](https://github.com/KellerJordan/Muon) | LR scaling |
| MuonAll paper | [arXiv](https://arxiv.org/html/2511.06086v1) | muP + Muon scaling |
| NanoGPT speedruns | [Blog](https://kellerjordan.github.io/posts/muon/) | Benchmark reference |
| NVIDIA cuTile | [Blog](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/) | Custom FA for Ada |

---

*Research date: 2026-03-28*
