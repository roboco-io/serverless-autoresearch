# Chapter 1: The Idea — "Can We Run Autoresearch on AWS for Pennies?"

> **Time**: 15 minutes
> **Cost**: $0.00
> **Key Insight**: Start with a clear goal and let the AI interview you to refine it.

## Context
Karpathy released autoresearch — an autonomous ML research framework where AI agents modify training code, run 5-minute experiments, and iterate overnight. But it assumes you have an H100 GPU. We wanted to run it on AWS without owning any GPU.

## The Prompt
```
https://github.com/karpathy/autoresearch 를 사용해서 딥러닝 모델 성능 개선 실험을
재현해 보고 싶어. OMC의 autoresearch를 사용해서 실험을 할 수 있게 셋팅해줘.
추가 컨텍스트가 필요하면 심층 인터뷰를 해줘.
```
> *Translation: "I want to reproduce deep learning model performance improvement experiments using https://github.com/karpathy/autoresearch. Set it up so I can run experiments with OMC's autoresearch. If you need additional context, do a deep interview."*

## What Happened
The AI started a deep interview to understand the requirements:
1. **Computing environment?** → SageMaker Managed Spot Training (serverless, pay per use)
2. **Agent loop?** → OMC autopilot mode (autonomous)
3. **Goal?** → Reproduce experiments first, then create education/demo materials

This interview shaped the entire project direction. Instead of just setting up autoresearch, we designed a **parallel evolution pipeline** — running multiple experiments simultaneously on cheap Spot instances.

## The Result
A clear project plan emerged:
- Parallel evolution pipeline (not sequential like the original)
- SageMaker Spot Training for cost efficiency
- HUGI pattern (Hurry Up and Get Idle) for zero idle cost
- Every experiment documented as a tutorial

## Lessons Learned
- **Always ask for a deep interview** — the AI discovered that SageMaker Spot was the right approach, not EC2 instances or local GPUs
- **"Reproduce" became "improve"** — the interview revealed the real goal was education + cost optimization, not just reproduction
- **The prompt that changes everything**: Adding "추가 컨텍스트가 필요하면 심층 인터뷰를 해줘" ("Do a deep interview if you need more context") turned a vague request into a focused plan

## Try It Yourself
```bash
# Start Claude Code in any project and try:
# "I want to [your goal]. Do a deep interview to refine the approach."
```

### Running Cost
| Phase | Action | Cost | Cumulative |
|-------|--------|------|-----------|
| Planning | Deep interview + architecture design | $0.00 | $0.00 |
