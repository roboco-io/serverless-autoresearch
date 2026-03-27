"""세대 결과에서 최적 후보를 선택하고 baseline을 업데이트하는 모듈."""
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def select_best(
    results: list[dict],
    current_best_bpb: Optional[float] = None,
) -> dict:
    """가장 낮은 val_bpb를 가진 후보를 선택.

    Args:
        results: result_collector의 결과 목록
        current_best_bpb: 현재 최적 val_bpb (None이면 비교 없이 선택)

    Returns:
        선택된 결과 dict + "improved" 플래그
    """
    successful = [r for r in results if r.get("status") == "success" and r.get("val_bpb", 0) > 0]

    if not successful:
        return {
            "status": "no_success",
            "improved": False,
            "message": "No successful experiments in this generation",
        }

    best = min(successful, key=lambda r: r["val_bpb"])

    if current_best_bpb is not None:
        best["improved"] = best["val_bpb"] < current_best_bpb
        best["delta"] = current_best_bpb - best["val_bpb"]
    else:
        best["improved"] = True
        best["delta"] = 0.0

    return best


def update_baseline(
    best: dict,
    generation: int,
    project_root: Optional[Path] = None,
) -> None:
    """최적 후보의 train.py를 프로젝트 루트로 복사하고 git 커밋."""
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent

    if not best.get("improved", False):
        print(f"Generation {generation}: No improvement (best={best.get('val_bpb', 'N/A')}). "
              "Baseline unchanged.")
        return

    cid = best["candidate_id"]
    source_train = (project_root / "generations" / f"gen_{generation:03d}" /
                    "candidates" / f"train_{cid}.py")

    if not source_train.exists():
        print(f"WARNING: Candidate train.py not found: {source_train}")
        return

    # 프로젝트 루트의 train.py 업데이트
    dest_train = project_root / "train.py"
    shutil.copy2(source_train, dest_train)

    print(f"Generation {generation}: Updated baseline — "
          f"val_bpb={best['val_bpb']:.6f} (delta={best['delta']:.6f})")

    # Git 커밋 + 태그
    try:
        subprocess.run(
            ["git", "add", "train.py"],
            cwd=project_root, check=True, capture_output=True,
        )
        msg = (f"gen{generation:03d}: {cid} — val_bpb={best['val_bpb']:.6f} "
               f"(delta={best['delta']:.6f})\n\n{best.get('description', '')}")
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=project_root, check=True, capture_output=True,
        )
        tag = f"gen-{generation:03d}-best"
        subprocess.run(
            ["git", "tag", tag],
            cwd=project_root, check=True, capture_output=True,
        )
        print(f"  Git: committed and tagged as {tag}")
    except subprocess.CalledProcessError as e:
        print(f"  Git warning: {e.stderr.decode() if e.stderr else str(e)}")


def log_results_tsv(
    results: list[dict],
    generation: int,
    tsv_path: Optional[Path] = None,
) -> None:
    """results.tsv에 세대 결과를 추가."""
    if tsv_path is None:
        tsv_path = Path(__file__).parent.parent.parent / "results.tsv"

    # 헤더 생성 (파일이 없으면)
    if not tsv_path.exists():
        with open(tsv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["generation", "candidate", "val_bpb", "memory_gb", "status", "description"])

    with open(tsv_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for r in results:
            val_bpb = f"{r.get('val_bpb', 0):.6f}" if r.get("val_bpb") else "0.000000"
            vram_mb = r.get("peak_vram_mb", 0)
            memory_gb = f"{vram_mb / 1024:.1f}" if vram_mb else "0.0"
            writer.writerow([
                f"gen{generation:03d}",
                r.get("candidate_id", "?"),
                val_bpb,
                memory_gb,
                r.get("status", "unknown"),
                r.get("description", "")[:100],
            ])
