# Serverless Autoresearch

> Run Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) as a **parallel evolution pipeline** on SageMaker Managed Spot Training.

The original autoresearch runs experiments sequentially on a single GPU — 12 experiments/hour, ~8 hours for 100 experiments. This project leverages cloud horizontal scaling and the **HUGI (Hurry Up and Get Idle)** pattern to complete **100 experiments in ~100 minutes at the same cost (~$4)** with zero GPU idle time.

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="Serverless Autoresearch Architecture" width="100%">
</p>

### Sequential vs Parallel

| | Original autoresearch | Serverless (this repo) |
|---|---|---|
| **Execution** | 1 experiment at a time | **10 experiments in parallel** |
| **100 experiments** | ~8 hours | **~100 minutes** |
| **Cost** | ~$4 (GPU always on) | **~$4** (HUGI: pay only when running) |
| **GPU** | 1x H100 (always occupied) | N x H100 Spot (on-demand burst) |
| **Search strategy** | Greedy (sequential) | **Population-based evolution** |
| **Improvement probability** | 18% per experiment | **86% per generation** |

### HUGI Pattern (Hurry Up and Get Idle)

```
Traditional GPU server:
  ████░░░░████░░░░████░░░░████░░░░  (utilization ~50%, paying 24/7)

HUGI with SageMaker Spot:
  ██████████                          (utilization 100%, $0 when idle)
  ↑ N GPUs burst                 ↑ terminate immediately
```

## Quick Start

### Prerequisites

- AWS CLI configured (`aws configure`)
- Python 3.11+
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)

```bash
pip install boto3 sagemaker pyyaml click
```

### 1. Setup IAM Role

```bash
./infrastructure/setup_iam.sh --profile personal --region ap-northeast-1
# → Copy role ARN to config.yaml
```

### 2. Prepare Data (one-time, ~5 min)

```bash
python scripts/prepare_s3.py --num-shards 10
```

Downloads 10 training shards + validation shard from HuggingFace, trains BPE tokenizer, uploads everything to S3.

### 3. Verify Setup

```bash
python -m pipeline.orchestrator --dry-run
```

### 4. Run Experiments

```bash
# Single generation test (~$0.40, ~10 min)
python -m pipeline.orchestrator --single --population 10

# Full pipeline (~$4, ~100 min)
python -m pipeline.orchestrator --generations 10 --population 10
```

## How It Works

### Generation Loop

Each generation follows 4 steps:

1. **Candidate Generation** — Creates N variants of `train.py` with diverse strategies:

   | Strategy | Count | Description |
   |----------|-------|-------------|
   | Conservative | 3 | LR adjustments (±10-30%) |
   | Moderate | 4 | Architecture changes (depth, width, window, batch) |
   | Aggressive | 2 | Radical combinations (deep-narrow, wide-shallow) |
   | Crossover | 1 | Combine ideas from top-2 of previous generation |

2. **Batch Launch** — Submits all N candidates as parallel SageMaker Spot Training Jobs (async, `wait=False`)

3. **Result Collection** — Polls all jobs until completion, extracts `val_bpb` metric from CloudWatch

4. **Selection** — Best `val_bpb` becomes new baseline, committed with git tag `gen-NNN-best`

### The Rules (same as original autoresearch)

- Only `train.py` can be modified (model architecture, optimizer, hyperparameters)
- `prepare.py` is read-only (evaluation function, data loading, constants)
- No new dependencies allowed
- Fixed 5-minute training time budget (TIME_BUDGET=300s)
- Goal: **lowest val_bpb** (validation bits per byte)

## Project Structure

```
serverless-autoresearch/
├── train.py                    # Training script (agent modifies this)
├── prepare.py                  # Data prep + evaluation (read-only)
├── config.yaml                 # AWS & pipeline configuration
├── program.md                  # Agent instructions
│
├── pipeline/                   # Core pipeline
│   ├── orchestrator.py         # Main evolution loop
│   ├── candidate_generator.py  # Generate N train.py variants
│   ├── batch_launcher.py       # Submit N SageMaker jobs in parallel
│   ├── result_collector.py     # Collect & aggregate results
│   └── selection.py            # Select best, manage git state
│
├── sagemaker/                  # SageMaker wrappers
│   ├── entry_point.py          # Training job entry point
│   └── train_wrapper.py        # Executes train.py, parses results
│
├── infrastructure/             # AWS infrastructure
│   ├── setup_iam.sh            # IAM role creation
│   ├── Dockerfile              # Custom container (optional)
│   └── requirements-train.txt  # Training dependencies
│
├── scripts/                    # Utilities
│   ├── prepare_s3.py           # One-time: data prep + S3 upload
│   ├── run_single.py           # Debug: run single experiment
│   └── cost_report.py          # Cost reporting
│
├── docs/
│   ├── architecture.drawio     # Architecture diagram
│   └── comparison-report.md    # Original vs serverless comparison
│
└── generations/                # Per-generation candidates & results
    └── gen_000/
        ├── candidates/         # train_v01.py ~ train_v10.py
        └── results.json
```

## Configuration

`config.yaml`:

```yaml
aws:
  profile: personal
  region: ap-northeast-1          # Tokyo (H100 Spot available)
  role_arn: "arn:aws:iam::..."

sagemaker:
  instance_type: ml.p5.4xlarge    # H100 80GB
  use_spot: true
  max_run: 900                    # 15 min
  max_wait: 3600                  # 1 hour spot wait
  framework_version: "2.8.0"
  py_version: "py312"

pipeline:
  num_generations: 10
  population_size: 10
  num_conservative: 3
  num_moderate: 4
  num_aggressive: 2
  num_crossover: 1
```

## Cost

| Component | Unit Cost | Qty | Total |
|-----------|-----------|-----|-------|
| ml.p5.4xlarge Spot (8min/exp) | ~$0.04 | 100 | ~$4.00 |
| S3 storage | — | — | ~$0.10 |
| **Total** | | | **~$4.10** |

## OMC Autopilot Integration

Run the full pipeline autonomously with [oh-my-claudecode](https://github.com/nicobailon/oh-my-claudecode):

```
/autopilot

Read program.md and execute:
python -m pipeline.orchestrator --generations 10 --population 10

After completion, analyze results.tsv and summarize findings.
```

## Credits

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — Original sequential autoresearch framework
- [karpathy/nanochat](https://github.com/karpathy/nanochat) — Training codebase that autoresearch is based on

## License

MIT
