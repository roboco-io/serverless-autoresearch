# Serverless Autoresearch — 병렬 진화 파이프라인

Karpathy의 autoresearch를 SageMaker Spot Training 위에서 병렬 실행하는 시스템.

## Architecture

원본 autoresearch가 순차적으로 실험을 하나씩 실행하는 것과 달리, 이 시스템은 **세대 기반 병렬 진화** 방식으로 동작한다:

1. **후보 생성**: N개의 train.py 변형을 자동 생성
2. **병렬 실행**: N개 SageMaker Spot 인스턴스에서 동시 실행
3. **선택**: 최저 val_bpb 후보가 새 baseline이 됨
4. **반복**: M세대 동안 반복

## Setup Checklist

1. **데이터 준비**: `python scripts/prepare_s3.py`
2. **IAM 역할**: `./infrastructure/setup_iam.sh` → config.yaml에 role_arn 입력
3. **Docker 이미지**: `./infrastructure/build_and_push.sh` → config.yaml에 image_uri 입력
4. **설정 검증**: `python -m pipeline.orchestrator --dry-run`

## Running Experiments

### Dry Run (비용 0)
```bash
python -m pipeline.orchestrator --dry-run
```

### 단일 세대 테스트 (~$0.40)
```bash
python -m pipeline.orchestrator --single --population 10
```

### 전체 파이프라인 (~$4)
```bash
python -m pipeline.orchestrator --generations 10 --population 10
```

### 단일 실험 디버깅 (~$0.04)
```bash
python scripts/run_single.py
```

## The Rules

- `train.py`만 수정 가능 (모델, 옵티마이저, 하이퍼파라미터)
- `prepare.py`는 수정 불가 (평가 함수, 데이터 로더, 상수)
- 의존성 추가 불가
- 학습 시간: 5분 고정 (TIME_BUDGET=300초)
- 목표: **가장 낮은 val_bpb** 달성

## Candidate Diversity Strategy

각 세대에서 다양한 전략의 후보를 생성:

| 유형 | 수량 | 전략 |
|------|------|------|
| Conservative | 3 | LR ±10-30% 미세 조정 |
| Moderate | 4 | DEPTH, ASPECT_RATIO, BATCH_SIZE, WINDOW 변경 |
| Aggressive | 2 | 급진적 조합 (deep-narrow, wide-shallow, high-LR) |
| Crossover | 1 | 이전 Top-2 아이디어 결합 |

## Cost Model

- Instance: ml.g5.xlarge (A10G 24GB, Ampere)
- Spot price: ~$0.30/hr
- Per experiment: ~$0.04 (8분)
- Per generation (10 candidates): ~$0.40
- Full pipeline (10 gen × 10 pop): ~$4.00

## Output

- `results.tsv`: 전체 실험 로그 (TSV 형식)
- `generations/gen_NNN/`: 세대별 후보 코드 + 결과
- Git tags: `gen-NNN-best` — 각 세대 최적 상태

## OMC Autopilot Integration

OMC autopilot으로 전체 루프를 자율 실행:

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

실험 완료 후 다음을 정리:
1. `results.tsv` + `generations/` → 실험 과정 시각화
2. Git log → 진화 과정 추적
3. 비용 리포트 → 클라우드 효율성 입증
4. 원본 대비 비교: 8시간 순차 vs 100분 병렬
