"""메인 진화 루프 오케스트레이터.

Usage:
    python -m pipeline.orchestrator --generations 10 --population 10
    python -m pipeline.orchestrator --dry-run
    python -m pipeline.orchestrator --single  # 단일 세대만
"""
import json
import sys
import time
from pathlib import Path

import click
import yaml

from pipeline.candidate_generator import generate_candidates
from pipeline.batch_launcher import launch_batch
from pipeline.result_collector import collect_results, save_generation_results
from pipeline.selection import select_best, update_baseline, log_results_tsv


PROJECT_ROOT = Path(__file__).parent.parent


def load_config(config_path: Path = None) -> dict:
    """config.yaml 로드."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_config(config: dict) -> list[str]:
    """설정 검증. 에러 메시지 리스트 반환."""
    errors = []
    if not config["aws"].get("role_arn"):
        errors.append("aws.role_arn이 설정되지 않았습니다. infrastructure/setup_iam.sh를 실행하세요.")
    return errors


@click.command()
@click.option("--generations", "-g", default=None, type=int, help="세대 수 (기본: config.yaml)")
@click.option("--population", "-p", default=None, type=int, help="세대당 후보 수 (기본: config.yaml)")
@click.option("--dry-run", is_flag=True, help="실제 제출 없이 설정 확인")
@click.option("--single", is_flag=True, help="1세대만 실행")
@click.option("--config-path", type=click.Path(exists=True), default=None)
@click.option("--continue-from", type=int, default=0, help="이전 세대 이후부터 재개")
def main(generations, population, dry_run, single, config_path, continue_from):
    """Serverless Autoresearch — 병렬 진화 파이프라인."""
    config = load_config(Path(config_path) if config_path else None)

    # CLI 옵션으로 config 오버라이드
    if generations is not None:
        config["pipeline"]["num_generations"] = generations
    if population is not None:
        config["pipeline"]["population_size"] = population
    if single:
        config["pipeline"]["num_generations"] = 1

    num_gen = config["pipeline"]["num_generations"]
    pop_size = config["pipeline"]["population_size"]

    print("=" * 60)
    print("Serverless Autoresearch — Parallel Evolution Pipeline")
    print("=" * 60)
    print(f"  Generations:  {num_gen}")
    print(f"  Population:   {pop_size}")
    print(f"  Instance:     {config['sagemaker']['instance_type']}")
    print(f"  Spot:         {config['sagemaker']['use_spot']}")
    print(f"  Dry run:      {dry_run}")
    print()

    # 설정 검증 (dry-run이 아닐 때만)
    if not dry_run:
        errors = validate_config(config)
        if errors:
            for e in errors:
                print(f"ERROR: {e}")
            sys.exit(1)

    # 비용 추정
    est_cost = _estimate_total_cost(config)
    print(f"  Estimated cost: ${est_cost:.2f}")
    print()

    # 진화 루프
    base_train_py = PROJECT_ROOT / "train.py"
    history: list[dict] = []
    current_best_bpb: float | None = None
    total_cost = 0.0

    for gen in range(continue_from, continue_from + num_gen):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"GENERATION {gen:03d}")
        print(f"{'='*60}")

        # 1. 후보 생성
        print("\n[1/4] Generating candidates...")
        candidates = generate_candidates(base_train_py, config, gen, history)
        print(f"  Generated {len(candidates)} candidates")
        for c in candidates:
            print(f"    {c['id']}: {c['description']}")

        # 2. 병렬 제출
        print(f"\n[2/4] Launching {len(candidates)} SageMaker jobs...")
        jobs = launch_batch(candidates, config, gen, dry_run=dry_run)

        # 3. 결과 수집
        if not dry_run:
            print(f"\n[3/4] Collecting results...")
            results = collect_results(jobs, config)
        else:
            results = jobs  # dry_run에서는 job info를 그대로 사용

        # 4. 선택 및 업데이트
        print(f"\n[4/4] Selecting best...")
        save_generation_results(results, gen)

        if not dry_run:
            best = select_best(results, current_best_bpb)
            if best.get("status") == "no_success":
                print(f"  WARNING: {best['message']}")
            else:
                if best.get("improved"):
                    current_best_bpb = best["val_bpb"]
                    update_baseline(best, gen, PROJECT_ROOT)
                print(f"  Best: {best.get('candidate_id', '?')} — "
                      f"val_bpb={best.get('val_bpb', 'N/A')}")

            # 결과 로깅
            log_results_tsv(results, gen)

            # 비용 추적
            gen_cost = sum(r.get("billable_seconds", 0) for r in results) / 3600 * 0.30
            total_cost += gen_cost
            print(f"\n  Generation cost: ${gen_cost:.3f} | Total: ${total_cost:.3f}")

        # 히스토리 업데이트
        history.extend(results)

        elapsed = time.time() - t0
        print(f"  Generation time: {elapsed:.0f}s")

    # 최종 요약
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Generations: {num_gen}")
    print(f"  Total experiments: {len(history)}")
    if current_best_bpb:
        print(f"  Best val_bpb: {current_best_bpb:.6f}")
    print(f"  Total cost: ${total_cost:.3f}")
    print(f"  Results: {PROJECT_ROOT / 'results.tsv'}")


def _estimate_total_cost(config: dict) -> float:
    """전체 파이프라인 예상 비용."""
    pop = config["pipeline"]["population_size"]
    gen = config["pipeline"]["num_generations"]
    # 실험당 ~8분, ml.g5.xlarge spot ~$0.30/hr
    per_exp = 0.04
    return pop * gen * per_exp


if __name__ == "__main__":
    main()
