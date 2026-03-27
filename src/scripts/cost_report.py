#!/usr/bin/env python3
"""실험 비용 리포트 생성.

Usage:
    python scripts/cost_report.py [--profile personal]
"""
import json
from pathlib import Path

import click
import yaml
import boto3


PROJECT_ROOT = Path(__file__).parent.parent.parent


@click.command()
@click.option("--profile", default=None)
@click.option("--region", default=None)
@click.option("--limit", default=50, help="최근 N개 Job")
def main(profile, region, limit):
    """SageMaker autoresearch 비용 리포트."""
    config = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    profile = profile or config["aws"]["profile"]
    region = region or config["aws"]["region"]

    session = boto3.Session(profile_name=profile, region_name=region)
    sm = session.client("sagemaker")

    # autoresearch 관련 Job 조회
    resp = sm.list_training_jobs(
        NameContains="autoresearch",
        MaxResults=limit,
        SortBy="CreationTime",
        SortOrder="Descending",
    )

    jobs = resp.get("TrainingJobSummaries", [])
    if not jobs:
        print("No autoresearch jobs found.")
        return

    print(f"{'='*70}")
    print(f"Autoresearch Cost Report (last {len(jobs)} jobs)")
    print(f"{'='*70}")

    total_billable = 0
    total_cost = 0
    status_counts = {}

    for job in jobs:
        name = job["TrainingJobName"]
        status = job["TrainingJobStatus"]
        status_counts[status] = status_counts.get(status, 0) + 1

        # 상세 정보
        detail = sm.describe_training_job(TrainingJobName=name)
        billable = detail.get("BillableTimeInSeconds", 0)
        total_billable += billable
        instance = detail.get("ResourceConfig", {}).get("InstanceType", "?")

        # 비용 추정 (ml.g5.xlarge spot 기준)
        hourly = 0.30 if "spot" in str(detail.get("EnableManagedSpotTraining", False)).lower() or detail.get("EnableManagedSpotTraining") else 1.006
        cost = billable / 3600 * hourly
        total_cost += cost

        print(f"  {name[:50]:50s} | {status:10s} | {billable:5d}s | ${cost:.4f}")

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Total jobs:       {len(jobs)}")
    for s, c in sorted(status_counts.items()):
        print(f"    {s}: {c}")
    print(f"  Total billable:   {total_billable}s ({total_billable/3600:.2f}h)")
    print(f"  Total cost:       ${total_cost:.4f}")
    print(f"  Avg per job:      ${total_cost/len(jobs):.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
