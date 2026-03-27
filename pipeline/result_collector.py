"""SageMaker Training Job 결과를 수집하고 집계하는 모듈."""
import json
import time
from pathlib import Path
from typing import Optional

import boto3


def collect_results(
    jobs: list[dict],
    config: dict,
    timeout: int = 1800,
    poll_interval: int = 30,
) -> list[dict]:
    """모든 Job의 완료를 기다리고 결과를 수집.

    Args:
        jobs: launch_batch()의 반환값
        config: config.yaml 내용
        timeout: 최대 대기 시간 (초)
        poll_interval: 폴링 간격 (초)

    Returns:
        결과 목록. 각 항목은 {
            "candidate_id": "v01", "job_name": "...",
            "status": "success" | "crash" | "timeout" | "spot_interrupted",
            "val_bpb": 0.997, "peak_vram_mb": 12000, ...
        }
    """
    # dry_run job은 그대로 반환
    active_jobs = [j for j in jobs if j["status"] == "submitted"]
    dry_jobs = [j for j in jobs if j["status"] == "dry_run"]

    if not active_jobs:
        return dry_jobs

    profile = config["aws"]["profile"]
    region = config["aws"]["region"]
    boto_session = boto3.Session(profile_name=profile, region_name=region)
    sm_client = boto_session.client("sagemaker")
    s3_client = boto_session.client("s3")

    bucket = config["s3"].get("bucket")
    if not bucket:
        import sagemaker
        sm_session = sagemaker.Session(boto_session=boto_session)
        bucket = sm_session.default_bucket()

    start_time = time.time()
    pending = {j["job_name"]: j for j in active_jobs}
    completed = list(dry_jobs)

    print(f"\nWaiting for {len(pending)} jobs (timeout: {timeout}s, poll: {poll_interval}s)")

    while pending and (time.time() - start_time) < timeout:
        for job_name in list(pending.keys()):
            try:
                resp = sm_client.describe_training_job(TrainingJobName=job_name)
                status = resp["TrainingJobStatus"]
            except Exception as e:
                print(f"  Error checking {job_name}: {e}")
                continue

            if status == "Completed":
                job_info = pending.pop(job_name)
                result = _extract_result(job_info, resp, s3_client, bucket)
                completed.append(result)
                print(f"  DONE: {job_info['candidate_id']} — "
                      f"val_bpb={result.get('val_bpb', 'N/A')}")

            elif status == "Failed":
                job_info = pending.pop(job_name)
                failure = resp.get("FailureReason", "Unknown")
                completed.append({
                    **job_info,
                    "status": "crash",
                    "val_bpb": 0.0,
                    "peak_vram_mb": 0.0,
                    "failure_reason": failure,
                })
                print(f"  CRASH: {job_info['candidate_id']} — {failure[:80]}")

            elif status == "Stopped":
                job_info = pending.pop(job_name)
                completed.append({
                    **job_info,
                    "status": "spot_interrupted",
                    "val_bpb": 0.0,
                    "peak_vram_mb": 0.0,
                })
                print(f"  INTERRUPTED: {job_info['candidate_id']}")

        if pending:
            elapsed = int(time.time() - start_time)
            print(f"  [{elapsed}s] {len(pending)} jobs still running...")
            time.sleep(poll_interval)

    # 타임아웃된 Job 처리
    for job_name, job_info in pending.items():
        completed.append({
            **job_info,
            "status": "timeout",
            "val_bpb": 0.0,
            "peak_vram_mb": 0.0,
        })
        print(f"  TIMEOUT: {job_info['candidate_id']}")

    return completed


def _extract_result(
    job_info: dict,
    sm_response: dict,
    s3_client,
    bucket: str,
) -> dict:
    """완료된 Job에서 결과 메트릭을 추출."""
    result = {
        "candidate_id": job_info["candidate_id"],
        "job_name": job_info["job_name"],
        "description": job_info.get("description", ""),
        "status": "success",
    }

    # SageMaker metric definitions에서 최종 값 추출
    metrics = sm_response.get("FinalMetricDataList", [])
    for m in metrics:
        result[m["MetricName"]] = m["Value"]

    # BillableSeconds로 실제 비용 추적
    billable = sm_response.get("BillableTimeInSeconds", 0)
    result["billable_seconds"] = billable

    # S3에서 results.json 다운로드 시도
    output_uri = sm_response.get("OutputDataConfig", {}).get("S3OutputPath", "")
    if output_uri:
        try:
            # SageMaker 출력 경로: s3://bucket/prefix/job_name/output/output.tar.gz
            s3_key = f"{output_uri.replace(f's3://{bucket}/', '')}/{job_info['job_name']}/output/output.tar.gz"
            # results.json은 output/data/results.json에 있음
            # 간단하게 CloudWatch 메트릭만 사용
        except Exception:
            pass

    return result


def save_generation_results(
    results: list[dict],
    generation: int,
    output_dir: Optional[Path] = None,
) -> Path:
    """세대 결과를 로컬 파일로 저장."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "generations" / f"gen_{generation:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # estimator 객체 제거 (직렬화 불가)
    clean_results = []
    for r in results:
        clean = {k: v for k, v in r.items() if k != "estimator"}
        clean_results.append(clean)

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    print(f"Generation {generation} results saved to {results_file}")
    return results_file
