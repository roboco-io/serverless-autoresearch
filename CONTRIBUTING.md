# Contributing to Serverless Autoresearch

Thank you for your interest in contributing! This guide covers how to add experiments, improve the pipeline, and document your findings.

## Quick Start

```bash
git clone https://github.com/roboco-io/serverless-autoresearch
cd serverless-autoresearch
cp config.yaml.example config.yaml
# Edit config.yaml with your AWS credentials
make dry-run  # Verify setup
```

## Types of Contributions

### 1. New Experiments

The most valuable contribution. Run experiments on different GPUs, regions, or with different strategies.

**How to add an experiment:**

1. Create a folder: `experiments/NNN-short-description/`
2. Write `README.md` with: hypothesis, setup, results, lessons learned
3. If val_bpb improves, the evolved `train.py` is the deliverable
4. Update the experiments table in the project README

**Experiment folder structure:**
```
experiments/NNN-your-experiment/
├── README.md           # Hypothesis, setup, results, lessons
└── results-summary.md  # Detailed per-generation data (if pipeline run)
```

### 2. Pipeline Improvements

Improve the evolution pipeline in `src/pipeline/`.

**Rules:**
- `prepare.py` is **read-only** (matches upstream autoresearch constraint)
- `train.py` at project root is the **current best** — don't modify it directly in pipeline PRs
- Pipeline code lives in `src/pipeline/`, `src/sagemaker/`, `src/scripts/`
- Test with `make dry-run` before submitting

### 3. New GPU / Instance Support

Add support for new GPU types or fix compatibility issues.

**Checklist:**
- Update `train.py` FA3/FA2/SDPA detection if needed
- Add pricing to `docs/gpu-cost-analysis.md`
- Add Spot placement score data to `docs/spot-capacity-guide.md`
- Document quota codes in the spot capacity guide

### 4. Documentation

Improve tutorials, add translations, fix errors.

- Tutorials live in `docs/vibe-coding-tutorial/`
- Reference docs in `docs/`
- Research notes in `references/`
- All docs in English (Korean prompts preserved with translations in tutorials)

## Development Workflow

### Branch Naming

```
experiment/NNN-description   # New experiments
feature/short-description    # Pipeline improvements
fix/short-description        # Bug fixes
docs/short-description       # Documentation
```

### Commit Messages

Follow this format:
```
<type>: <short description>

<body with details>

Co-Authored-By: Your Name <email>
```

Types: `experiment`, `feat`, `fix`, `docs`, `refactor`, `infra`

### Pull Request Process

1. Fork the repo and create a branch
2. Make your changes
3. Run `make dry-run` to verify pipeline integrity
4. Update relevant documentation
5. Submit PR with:
   - What you changed and why
   - Experiment results (if applicable)
   - Cost incurred (we track every dollar)

## Code Structure

```
train.py                    # Current best (autoresearch modifies this)
prepare.py                  # Read-only evaluation harness
src/pipeline/               # Evolution pipeline (orchestrator, candidates, etc.)
src/sagemaker/              # SageMaker job wrappers
src/scripts/                # CLI utilities
infrastructure/             # IAM, Docker, requirements
experiments/                # Per-experiment reports
docs/                       # Project documentation
references/                 # Research notes
```

## Key Constraints

These come from the autoresearch framework and cannot be changed:

1. **Only `train.py` is modifiable** for experiments
2. **`prepare.py` is read-only** — evaluation function is the ground truth
3. **No new dependencies** in training (only what's in `infrastructure/requirements-train.txt`)
4. **5-minute training budget** (TIME_BUDGET=300s)
5. **val_bpb is the metric** — lower is better

## Before Running Experiments

1. **Check Spot placement scores** — see [Spot Capacity Guide](docs/spot-capacity-guide.md)
2. **Request GPU quotas** in your target region
3. **Estimate cost** with `make dry-run`
4. **Document everything** — your failures are as valuable as your successes

## Cost Tracking

Every PR that involves AWS usage should include:
- Total cost incurred
- Number of experiments run
- Instance type and region used

We aim to keep experiment costs transparent and reproducible.

## Questions?

Open an issue or start a discussion on GitHub.
