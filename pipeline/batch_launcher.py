"""N개 SageMaker Training Job을 동시에 제출하는 모듈."""
import json
import time
from pathlib import Path
from typing import Optional

import boto3
import sagemaker
from sagemaker.estimator import Estimator


def get_session(profile: str, region: str) -> sagemaker.Session:
    """SageMaker 세션 생성."""
    boto_session = boto3.Session(profile_name=profile, region_name=region)
    return sagemaker.Session(boto_session=boto_session)


def launch_batch(
    candidates: list[dict],
    config: dict,
    generation: int,
    session: Optional[sagemaker.Session] = None,
    dry_run: bool = False,
) -> list[dict]:
    """N개 후보를 동시에 SageMaker Training Job으로 제출.

    Args:
        candidates: 후보 목록. 각 항목은 {
            "id": "v01", "train_py_path": "/path/to/train_v01.py",
            "description": "..."
        }
        config: config.yaml 내용
        generation: 현재 세대 번호
        session: SageMaker 세션 (None이면 config에서 생성)
        dry_run: True이면 실제 제출 없이 설정만 출력

    Returns:
        제출된 Job 목록. 각 항목은 {
            "candidate_id": "v01", "job_name": "...",
            "status": "submitted" | "dry_run"
        }
    """
    if session is None:
        session = get_session(config["aws"]["profile"], config["aws"]["region"])

    aws_cfg = config["aws"]
    sm_cfg = config["sagemaker"]
    s3_cfg = config["s3"]

    bucket = s3_cfg.get("bucket") or session.default_bucket()
    role = aws_cfg["role_arn"]
    image_uri = sm_cfg["image_uri"]

    s3_data = f"s3://{bucket}/{s3_cfg['data_prefix']}"
    s3_tokenizer = f"s3://{bucket}/{s3_cfg['tokenizer_prefix']}"

    jobs = []
    timestamp = int(time.time())

    for i, candidate in enumerate(candidates):
        cid = candidate["id"]
        job_name = f"autoresearch-gen{generation:03d}-{cid}-{timestamp}"
        # SageMaker job name 규칙: 최대 63자, [a-zA-Z0-9-]만 허용
        job_name = job_name[:63].rstrip("-")

        print(f"\n[{i+1}/{len(candidates)}] {job_name}")
        print(f"  Candidate: {cid} — {candidate.get('description', 'N/A')}")

        if dry_run:
            cost_est = _estimate_cost(sm_cfg)
            print(f"  [DRY RUN] Instance: {sm_cfg['instance_type']}, "
                  f"Spot: {sm_cfg['use_spot']}, Est. cost: ${cost_est:.3f}")
            jobs.append({
                "candidate_id": cid,
                "job_name": job_name,
                "status": "dry_run",
                "description": candidate.get("description", ""),
            })
            continue

        # 각 후보의 source_dir 준비: train.py + prepare.py를 임시 디렉토리에 복사
        source_dir = _prepare_source_dir(candidate["train_py_path"], generation, cid)

        estimator = Estimator(
            image_uri=image_uri,
            role=role,
            instance_count=1,
            instance_type=sm_cfg["instance_type"],
            use_spot_instances=sm_cfg["use_spot"],
            max_run=sm_cfg["max_run"],
            max_wait=sm_cfg["max_wait"] if sm_cfg["use_spot"] else None,
            sagemaker_session=session,
            source_dir=str(source_dir),
            entry_point="entry_point.py",
            output_path=f"s3://{bucket}/{s3_cfg['output_prefix']}/gen{generation:03d}",
            base_job_name=f"autoresearch-gen{generation:03d}",
            metric_definitions=[
                {"Name": "val_bpb", "Regex": r"val_bpb:\s+([0-9.]+)"},
                {"Name": "peak_vram_mb", "Regex": r"peak_vram_mb:\s+([0-9.]+)"},
                {"Name": "mfu_percent", "Regex": r"mfu_percent:\s+([0-9.]+)"},
                {"Name": "training_seconds", "Regex": r"training_seconds:\s+([0-9.]+)"},
            ],
            environment={
                "SM_CACHE_DIR": "/opt/ml/input/data",
            },
        )

        # 비동기 제출 (wait=False) → 모든 Job이 동시에 실행
        estimator.fit(
            inputs={
                "data": s3_data,
                "tokenizer": s3_tokenizer,
            },
            job_name=job_name,
            wait=False,
            logs=False,
        )

        jobs.append({
            "candidate_id": cid,
            "job_name": job_name,
            "status": "submitted",
            "description": candidate.get("description", ""),
            "estimator": estimator,
        })

        # Rate limiting: SageMaker API 제한 방지
        time.sleep(1)

    print(f"\n{'='*60}")
    submitted = sum(1 for j in jobs if j["status"] == "submitted")
    print(f"Generation {generation}: {submitted}/{len(jobs)} jobs submitted")

    return jobs


def _prepare_source_dir(train_py_path: str, generation: int, candidate_id: str) -> Path:
    """후보의 source_dir을 준비한다.

    각 후보의 train.py + 공유 파일(prepare.py, sagemaker 래퍼)을 임시 디렉토리에 복사.
    """
    import shutil
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "generations" / f"gen_{generation:03d}" / "source" / candidate_id
    source_dir.mkdir(parents=True, exist_ok=True)

    # 후보의 train.py 복사
    shutil.copy2(train_py_path, source_dir / "train.py")
    # 공유 파일 복사
    shutil.copy2(project_root / "prepare.py", source_dir / "prepare.py")
    shutil.copy2(project_root / "sagemaker" / "entry_point.py", source_dir / "entry_point.py")
    shutil.copy2(project_root / "sagemaker" / "train_wrapper.py", source_dir / "train_wrapper.py")

    return source_dir


def _estimate_cost(sm_cfg: dict) -> float:
    """단일 실험의 예상 비용 (USD)."""
    # ml.g5.xlarge spot 가격 기준 (ap-northeast-2)
    spot_prices = {
        "ml.g5.xlarge": 0.30,
        "ml.g5.2xlarge": 0.45,
        "ml.p3.2xlarge": 0.92,
    }
    hourly = spot_prices.get(sm_cfg["instance_type"], 0.50)
    if not sm_cfg["use_spot"]:
        hourly *= 3  # on-demand는 약 3배
    minutes = sm_cfg["max_run"] / 60  # 최대 실행 시간
    estimated_minutes = min(minutes, 8)  # 실제로는 ~8분
    return hourly * estimated_minutes / 60
