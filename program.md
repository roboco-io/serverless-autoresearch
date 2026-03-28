# Serverless Autoresearch — Parallel Evolutionary Pipeline

A system that runs Karpathy's autoresearch in parallel on SageMaker Spot Training.

## Architecture

Unlike the original autoresearch, which runs experiments sequentially one at a time, this system operates using a **generation-based parallel evolution** approach:

1. **Candidate Generation**: Automatically generate N variants of train.py
2. **Parallel Execution**: Run simultaneously across N SageMaker Spot instances
3. **Selection**: The candidate with the lowest val_bpb becomes the new baseline
4. **Iteration**: Repeat for M generations

## Setup Checklist

1. **Prepare Data**: `python scripts/prepare_s3.py`
2. **IAM Role**: `./infrastructure/setup_iam.sh` → enter role_arn in config.yaml
3. **Docker Image**: `./infrastructure/build_and_push.sh` → enter image_uri in config.yaml
4. **Validate Config**: `python -m pipeline.orchestrator --dry-run`

## Running Experiments

### Dry Run (zero cost)
```bash
python -m pipeline.orchestrator --dry-run
```

### Single Generation Test (~$0.40)
```bash
python -m pipeline.orchestrator --single --population 10
```

### Full Pipeline (~$4)
```bash
python -m pipeline.orchestrator --generations 10 --population 10
```

### Single Experiment Debugging (~$0.04)
```bash
python scripts/run_single.py
```

## The Rules

- Only `train.py` may be modified (model, optimizer, hyperparameters)
- `prepare.py` must not be modified (evaluation functions, data loaders, constants)
- No new dependencies may be added
- Training time: fixed at 5 minutes (TIME_BUDGET=300 seconds)
- Goal: achieve the **lowest val_bpb**

## Candidate Diversity Strategy

Each generation produces candidates with varied strategies:

| Type | Count | Strategy |
|------|-------|----------|
| Conservative | 3 | Fine-tune LR ±10-30% |
| Moderate | 4 | Change DEPTH, ASPECT_RATIO, BATCH_SIZE, WINDOW |
| Aggressive | 2 | Radical combinations (deep-narrow, wide-shallow, high-LR) |
| Crossover | 1 | Combine top-2 ideas from previous generation |

## Cost Model

- Instance: ml.g5.xlarge (A10G 24GB, Ampere)
- Spot price: ~$0.30/hr
- Per experiment: ~$0.04 (8 minutes)
- Per generation (10 candidates): ~$0.40
- Full pipeline (10 gen × 10 pop): ~$4.00

## Output

- `results.tsv`: Full experiment log (TSV format)
- `generations/gen_NNN/`: Per-generation candidate code + results
- Git tags: `gen-NNN-best` — best state for each generation

## OMC Autopilot Integration

Run the full loop autonomously with OMC autopilot:

```
/autopilot

Read program.md and run the full pipeline:
python -m pipeline.orchestrator --generations 10 --population 10

After completion, analyze results.tsv and summarize:
1. Best val_bpb achieved
2. Most impactful changes
3. Cost summary
4. Recommendations for next run
```

## For Education/Demo

After experiments complete, prepare the following:
1. `results.tsv` + `generations/` → visualize the experiment process
2. Git log → trace the evolutionary progression
3. Cost report → demonstrate cloud efficiency
4. Comparison against baseline: 8-hour sequential vs 100-minute parallel
