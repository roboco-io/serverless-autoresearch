#!/usr/bin/env python3
"""단일 SageMaker 실험 실행 (디버깅/테스트용).

Usage:
    python scripts/run_single.py [--dry-run] [--wait]
"""
import sys
import time
from pathlib import Path

import click
import yaml

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.batch_launcher import launch_batch, get_session
from pipeline.result_collector import collect_results


@click.command()
@click.option("--train-py", default=None, type=click.Path(exists=True),
              help="사용할 train.py (기본: 프로젝트 루트)")
@click.option("--dry-run", is_flag=True, help="실제 제출 없이 확인")
@click.option("--wait/--no-wait", default=True, help="완료까지 대기")
def main(train_py, dry_run, wait):
    """단일 SageMaker 실험 실행."""
    config = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))

    if train_py is None:
        train_py = str(PROJECT_ROOT / "train.py")

    print("=" * 60)
    print("Single Experiment Runner")
    print("=" * 60)
    print(f"  train.py: {train_py}")
    print(f"  Instance: {config['sagemaker']['instance_type']}")
    print(f"  Spot:     {config['sagemaker']['use_spot']}")
    print(f"  Dry run:  {dry_run}")
    print()

    candidates = [{
        "id": "single",
        "train_py_path": train_py,
        "description": "single experiment test",
    }]

    jobs = launch_batch(candidates, config, generation=999, dry_run=dry_run)

    if dry_run or not wait:
        if jobs:
            print(f"\nJob: {jobs[0].get('job_name', 'N/A')}")
        return

    print("\nWaiting for completion...")
    results = collect_results(jobs, config, timeout=1200, poll_interval=15)

    if results:
        r = results[0]
        print(f"\n{'='*60}")
        print("Results:")
        print(f"  Status:      {r.get('status')}")
        print(f"  val_bpb:     {r.get('val_bpb', 'N/A')}")
        print(f"  peak_vram:   {r.get('peak_vram_mb', 'N/A')} MB")
        print(f"  mfu:         {r.get('mfu_percent', 'N/A')}%")
        print(f"  Billable:    {r.get('billable_seconds', 0)}s")
        cost = r.get("billable_seconds", 0) / 3600 * 0.30
        print(f"  Est. cost:   ${cost:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
