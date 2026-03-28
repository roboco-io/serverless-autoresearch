# Contribution Guide for Serverless Autoresearch

You are helping a contributor work on this project. Before doing any work, read and enforce the contribution guidelines.

## Required Reading

Before starting any task, read these files:
1. `CONTRIBUTING.md` — Contribution rules, workflow, PR process
2. `CLAUDE.md` — Project architecture, commands, constraints

## Contribution Workflow

### Step 1: Understand the Constraints

Enforce these rules strictly — they come from the autoresearch framework:
- **Only `train.py` may be modified** for experiment changes
- **`prepare.py` is READ-ONLY** — the evaluation function is sacred
- **No new training dependencies** — only what's in `infrastructure/requirements-train.txt`
- **5-minute training budget** — TIME_BUDGET=300s is fixed
- **val_bpb is the metric** — lower is better

### Step 2: Determine Contribution Type

Ask the user what they want to contribute if not clear:

| Type | Branch Prefix | Key Files |
|------|--------------|-----------|
| New experiment | `experiment/NNN-desc` | `experiments/NNN-desc/README.md` |
| Pipeline improvement | `feature/desc` | `src/pipeline/*.py` |
| GPU/instance support | `feature/gpu-desc` | `train.py`, `docs/gpu-cost-analysis.md` |
| Documentation | `docs/desc` | `docs/`, `experiments/` |
| Bug fix | `fix/desc` | varies |

### Step 3: Pre-flight Checks

Before writing any code, verify:
```bash
make dry-run          # Pipeline integrity
```

Before experiments, check Spot capacity:
```bash
# Check placement scores (MUST DO before choosing a region)
for region in us-east-1 us-east-2 us-west-2; do
  echo -n "$region: "
  aws ec2 get-spot-placement-scores \
    --instance-types g7e.4xlarge --target-capacity 1 \
    --single-availability-zone --region-names $region \
    --region $region \
    --query "max_by(SpotPlacementScores, &Score).Score" --output text
done
```

### Step 4: Do the Work

Follow the rules for the contribution type:

**For experiments:**
1. Create `experiments/NNN-description/README.md` with hypothesis, setup
2. Run experiments: `make run` or `python -m src.pipeline.orchestrator`
3. Document results including cost breakdown
4. If val_bpb improved, update `train.py` at project root
5. Update experiments table in project README

**For pipeline changes:**
1. Modify files in `src/pipeline/`, `src/sagemaker/`, or `src/scripts/`
2. Path convention: project root = `Path(__file__).parent.parent.parent` from `src/pipeline/`
3. Verify with `make dry-run`

**For documentation:**
1. All docs in English
2. Korean prompts in tutorials are preserved with English translations
3. Update `docs/insights.md` if new insights discovered

### Step 5: Commit and PR

Commit message format:
```
<type>: <short description>

<details>

Co-Authored-By: <name> <email>
```

Types: `experiment`, `feat`, `fix`, `docs`, `refactor`, `infra`

PR must include:
- What changed and why
- Experiment results (if applicable)
- **AWS cost incurred** (mandatory for experiment PRs)
- `make dry-run` passes

### Step 6: Cost Reporting

Every experiment PR MUST report:
- Total AWS cost
- Number of experiments run
- Instance type and region
- Billable seconds

## Quick Reference

```bash
make dry-run       # Verify config (free)
make run-single    # Single experiment (~$0.04)
make run           # Full pipeline (~$0.70)
make prepare       # Upload data to S3
make cost          # Cost report
```

## Common Pitfalls

1. **Don't modify `prepare.py`** — it's read-only
2. **Don't assume Spot capacity** — check placement scores first
3. **Don't skip `make dry-run`** — catches config errors before spending money
4. **Don't forget cost tracking** — we track every dollar
5. **`config.yaml` is gitignored** — use `config.yaml.example` as template
6. **DEVICE_BATCH_SIZE ≠ throughput** — increase TOTAL_BATCH_SIZE instead
