#!/usr/bin/env python3
"""데이터 준비 및 S3 업로드 (1회 실행).

Usage:
    python scripts/prepare_s3.py [--num-shards 10] [--profile personal] [--dry-run]
"""
import os
import subprocess
import sys
from pathlib import Path

import click
import yaml
import boto3


PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = Path.home() / ".cache" / "autoresearch"


@click.command()
@click.option("--num-shards", default=10, help="학습 데이터 샤드 수 (기본 10)")
@click.option("--profile", default=None, help="AWS 프로파일 (기본: config.yaml)")
@click.option("--region", default=None, help="AWS 리전 (기본: config.yaml)")
@click.option("--dry-run", is_flag=True, help="실제 업로드 없이 확인")
def main(num_shards, profile, region, dry_run):
    """데이터 준비 + S3 업로드."""
    config = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    profile = profile or config["aws"]["profile"]
    region = region or config["aws"]["region"]

    print("=" * 60)
    print("Step 1: Local data preparation")
    print("=" * 60)

    # prepare.py 실행 (로컬, GPU 불필요)
    prepare_py = PROJECT_ROOT / "prepare.py"
    if not prepare_py.exists():
        print(f"ERROR: {prepare_py} not found")
        sys.exit(1)

    data_dir = CACHE_DIR / "data"
    tokenizer_dir = CACHE_DIR / "tokenizer"

    if data_dir.exists() and tokenizer_dir.exists():
        shard_count = len(list(data_dir.glob("shard_*.parquet")))
        has_tokenizer = (tokenizer_dir / "tokenizer.pkl").exists()
        if shard_count >= num_shards and has_tokenizer:
            print(f"Data already prepared: {shard_count} shards, tokenizer exists")
        else:
            _run_prepare(prepare_py, num_shards)
    else:
        _run_prepare(prepare_py, num_shards)

    # 확인
    shard_count = len(list(data_dir.glob("shard_*.parquet")))
    print(f"\nLocal data ready:")
    print(f"  Shards: {shard_count} at {data_dir}")
    print(f"  Tokenizer: {tokenizer_dir}")

    print(f"\n{'='*60}")
    print("Step 2: Upload to S3")
    print("=" * 60)

    session = boto3.Session(profile_name=profile, region_name=region)
    s3 = session.client("s3")

    s3_cfg = config["s3"]
    bucket = s3_cfg.get("bucket")
    if not bucket:
        import sagemaker
        sm_session = sagemaker.Session(boto_session=session)
        bucket = sm_session.default_bucket()

    data_prefix = s3_cfg["data_prefix"]
    tokenizer_prefix = s3_cfg["tokenizer_prefix"]

    # 데이터 샤드 업로드
    print(f"\nUploading data shards to s3://{bucket}/{data_prefix}/")
    for shard_file in sorted(data_dir.glob("shard_*.parquet")):
        s3_key = f"{data_prefix}/{shard_file.name}"
        if dry_run:
            print(f"  [DRY RUN] {shard_file.name} → s3://{bucket}/{s3_key}")
        else:
            print(f"  Uploading {shard_file.name}...", end=" ", flush=True)
            s3.upload_file(str(shard_file), bucket, s3_key)
            print("done")

    # 토크나이저 업로드
    print(f"\nUploading tokenizer to s3://{bucket}/{tokenizer_prefix}/")
    for tok_file in tokenizer_dir.iterdir():
        s3_key = f"{tokenizer_prefix}/{tok_file.name}"
        if dry_run:
            print(f"  [DRY RUN] {tok_file.name} → s3://{bucket}/{s3_key}")
        else:
            print(f"  Uploading {tok_file.name}...", end=" ", flush=True)
            s3.upload_file(str(tok_file), bucket, s3_key)
            print("done")

    print(f"\n{'='*60}")
    print("Upload complete!")
    print(f"  Data:      s3://{bucket}/{data_prefix}/")
    print(f"  Tokenizer: s3://{bucket}/{tokenizer_prefix}/")
    print("=" * 60)


def _run_prepare(prepare_py: Path, num_shards: int):
    """prepare.py 실행."""
    print(f"Running prepare.py --num-shards {num_shards} ...")
    result = subprocess.run(
        [sys.executable, str(prepare_py), "--num-shards", str(num_shards)],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("ERROR: prepare.py failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
