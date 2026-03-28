# Chapter 7: Meta — Turning Experiments into Reusable Knowledge

> **Time**: 30 minutes
> **Cost**: $0.00
> **Key Insight**: Every debugging session is a future tutorial chapter, every lesson is a reusable skill.

## Context
With experiments complete, we had accumulated hard-won knowledge about SageMaker Spot, GPU compatibility, and cost optimization. The question: how to make this knowledge reusable?

## The Prompts

### Documenting Insights
```
지금까지 진행상황에서 얻어진 인사이트를 문서화 한 다음 공식 문서에 따라서 스킬화 해줘.
```
> *Translation: "Document the insights gained from the progress so far, then turn them into a skill following the official documentation."*

12 insights were documented, from "Spot capacity varies 1-9 by region" to "DEVICE_BATCH_SIZE ≠ token throughput."

### Making a Living Skill
```
스킬이 인사이트 문서를 계속 참조하고, 이터레이션마다 업데이트 할 수 있도록 해줘.
```
> *Translation: "Make the skill continuously reference the insights document and be updatable after each iteration."*

Created a Claude Code skill (`sagemaker-spot-training`) with symlinked references — when insights are updated, the skill automatically reflects the latest findings.

### Proxy Training Discovery
```
스팟으로 가설들을 싼 비용으로 빠르게 검증하고 이를 바탕으로 대규모 훈련에 사용할 수 있을까?
```
> *Translation: "Can we quickly validate hypotheses cheaply with Spot, and then use those findings for large-scale training?"*

Research confirmed: architecture/optimizer rankings transfer across GPU types. Cheap Spot experiments ($0.04 each) can validate hypotheses for expensive H100 production runs.

## The Result
- **12 documented insights** in `docs/insights.md`
- **Reusable skill** in claude-skills repo with living references
- **Proxy training validation** — Spot GPUs are valid for hypothesis testing

## Lessons Learned
- **Document as you go** — insights are freshest right after discovery
- **Skills > memories** — a well-structured skill is more useful than scattered notes
- **Symlinks for living documents** — references auto-update without manual sync
- **Cheap experiments validate expensive ones** — architecture rankings transfer across GPU types

### Running Cost
| Phase | Action | Cost | Cumulative |
|-------|--------|------|-----------|
| All experiments | 25 runs on L40S Spot | $0.44 | $0.44 |
| Documentation | Insights + skill creation | $0.00 | $0.44 |

---
