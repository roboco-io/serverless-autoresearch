# Serverless Autoresearch

Karpathy의 [autoresearch](https://github.com/karpathy/autoresearch)를 SageMaker Managed Spot Training 위에서 병렬 실행하는 파이프라인.

## Architecture

- **병렬 진화 파이프라인**: N개 후보를 동시에 SageMaker Spot 인스턴스에서 실행
- **HUGI 패턴**: Burst → GPU 동시 실행 → 완료 후 즉시 종료 → 비용 0
- **GPU**: ml.g5.xlarge (NVIDIA A10G 24GB, Ampere)

## Key Files

| File | Role |
|------|------|
| `train.py` | 에이전트가 수정하는 학습 스크립트 (A10G 적응) |
| `prepare.py` | 데이터/토크나이저 준비 (환경 인식 경로) |
| `pipeline/orchestrator.py` | 메인 진화 루프 |
| `pipeline/candidate_generator.py` | N개 train.py 변형 생성 |
| `pipeline/batch_launcher.py` | SageMaker 작업 병렬 제출 |
| `pipeline/result_collector.py` | 결과 수집/집계 |
| `pipeline/selection.py` | 최적 선택, git 관리 |
| `config.yaml` | AWS/파이프라인 설정 |

## Commands

```bash
# 데이터 준비 (1회)
python scripts/prepare_s3.py

# Docker 이미지 빌드/푸시
./infrastructure/build_and_push.sh

# 단일 실험 테스트
python scripts/run_single.py --dry-run

# 전체 파이프라인 실행
python -m pipeline.orchestrator --generations 10 --population 10

# Dry run (비용 확인만)
python -m pipeline.orchestrator --dry-run
```

## Constraints

- `prepare.py`의 평가 함수(`evaluate_bpb`)는 수정 불가
- `train.py`만 수정 가능 (모델 아키텍처, 옵티마이저, 하이퍼파라미터)
- 의존성 추가 불가 (pyproject.toml의 학습 의존성 고정)
- 학습 시간 예산: 5분 (TIME_BUDGET=300초)

## Tech Stack

- Python 3.11+, PyTorch 2.9.1 (CUDA 12.8)
- SageMaker SDK, boto3
- Docker (ECR)
