# Chapter 2: Building the Pipeline — From Plan to Code in One Session

> **Time**: 45 minutes
> **Cost**: $0.00
> **Key Insight**: Use plan mode to align on architecture before writing a single line of code.

## Context
With the plan approved, we needed to build: a candidate generator, SageMaker job launcher, result collector, and selection module. Plus infrastructure (IAM, Docker, S3 upload scripts).

## The Prompt
The AI entered plan mode and designed the complete file structure. When the user added a twist mid-planning:

```
클라우드의 장점인 병렬 실행과 HUGI(Hurry Up and Get Idle)을 적극적으로 활용해서
실험 시간을 단축시키면서도 높은 성과를 낼 수 있는 구조로 파이프라인을 만들어
```

This redirected the design from sequential to **parallel evolution** — a population-based approach inspired by genetic algorithms.

## What Happened
1. **Plan mode**: The AI explored the upstream autoresearch codebase (train.py, prepare.py, program.md) and the user's existing SageMaker patterns from another project
2. **Architecture redesign**: Sequential → parallel evolution with 4 candidate strategies:
   - Conservative (3): Small LR tweaks
   - Moderate (4): Architecture changes
   - Aggressive (2): Radical experiments
   - Crossover (1): Combine top-2 ideas
3. **Code generation**: 6 tasks created and executed — project structure, training scripts, Docker, SageMaker wrappers, pipeline modules, utility scripts
4. **Parallel execution**: Multiple agents wrote code simultaneously (prepare.py + train.py in parallel, then 5 pipeline modules at once)

## The Result
23 files created in one session:
```
serverless-autoresearch/
├── train.py, prepare.py          # Adapted from upstream
├── src/pipeline/                  # 5 evolution modules
├── src/sagemaker/                 # Job wrappers
├── infrastructure/                # Dockerfile, IAM, requirements
└── src/scripts/                   # CLI utilities
```

Pipeline verified with `make dry-run` — generated candidates, simulated job submission, confirmed all paths work.

## Lessons Learned
- **Plan mode prevents wasted work** — getting alignment before coding saved multiple rewrites
- **Mid-course correction is valuable** — the HUGI suggestion mid-planning improved the architecture significantly
- **Parallel agent execution** — Claude Code spawned multiple agents to write independent modules simultaneously, cutting build time in half
- **Reuse existing patterns** — the AI found and reused SageMaker patterns from the user's other project (topology-efficient-deep-learning)

## Try It Yourself
```bash
git clone https://github.com/roboco-io/serverless-autoresearch
cd serverless-autoresearch
cp config.yaml.example config.yaml
make dry-run  # Verify everything works
```

### Running Cost
| Phase | Action | Cost | Cumulative |
|-------|--------|------|-----------|
| Planning | Deep interview + architecture | $0.00 | $0.00 |
| Building | Code generation (23 files) | $0.00 | $0.00 |
