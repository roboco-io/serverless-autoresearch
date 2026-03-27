"""N개 train.py 변형을 생성하는 모듈.

후보 다양성 전략:
  - Conservative (3): 하이퍼파라미터 미세 조정
  - Moderate (4): 아키텍처 변경
  - Aggressive (2): 급진적 변경
  - Crossover (1): 이전 세대 Top-2 아이디어 결합
"""
import copy
import random
import re
import shutil
from pathlib import Path
from typing import Optional


# 수정 가능한 하이퍼파라미터와 범위 (H100 80GB 기준)
HP_RANGES = {
    "ASPECT_RATIO": (48, 128, int),
    "HEAD_DIM": (64, 256, int),
    "TOTAL_BATCH_SIZE_EXP": (17, 21, int),  # 2^N
    "EMBEDDING_LR": (0.1, 1.2, float),
    "UNEMBEDDING_LR": (0.001, 0.01, float),
    "MATRIX_LR": (0.01, 0.08, float),
    "SCALAR_LR": (0.1, 1.0, float),
    "WEIGHT_DECAY": (0.0, 0.4, float),
    "WARMUP_RATIO": (0.0, 0.1, float),
    "WARMDOWN_RATIO": (0.2, 0.7, float),
    "DEPTH": (4, 16, int),
    "DEVICE_BATCH_SIZE": (32, 256, int),
    "WINDOW_PATTERN": ["L", "SL", "SSL", "SSSL"],
}


def generate_candidates(
    base_train_py: Path,
    config: dict,
    generation: int,
    history: Optional[list[dict]] = None,
) -> list[dict]:
    """현재 best train.py 기반으로 N개 후보 생성.

    Args:
        base_train_py: 현재 best train.py 경로
        config: config.yaml 내용 (pipeline 섹션)
        generation: 현재 세대 번호
        history: 이전 세대 결과 히스토리 (crossover용)

    Returns:
        후보 목록: [{"id": "v01", "train_py_path": "...", "description": "..."}]
    """
    pipe_cfg = config["pipeline"]
    output_dir = Path(__file__).parent.parent / "generations" / f"gen_{generation:03d}" / "candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_code = base_train_py.read_text()
    base_params = _parse_hyperparams(base_code)

    candidates = []
    idx = 1

    # Baseline (세대 0에서만)
    if generation == 0:
        path = output_dir / "train_v00.py"
        shutil.copy2(base_train_py, path)
        candidates.append({
            "id": "v00",
            "train_py_path": str(path),
            "description": "baseline (unmodified)",
        })
        # 나머지 후보 수 조정
        remaining = pipe_cfg["population_size"] - 1
    else:
        remaining = pipe_cfg["population_size"]

    # Conservative 후보
    n_conservative = min(pipe_cfg["num_conservative"], remaining)
    for _ in range(n_conservative):
        cid = f"v{idx:02d}"
        desc, new_code = _make_conservative(base_code, base_params)
        path = output_dir / f"train_{cid}.py"
        path.write_text(new_code)
        candidates.append({"id": cid, "train_py_path": str(path), "description": desc})
        idx += 1

    # Moderate 후보
    n_moderate = min(pipe_cfg["num_moderate"], remaining - n_conservative)
    for _ in range(n_moderate):
        cid = f"v{idx:02d}"
        desc, new_code = _make_moderate(base_code, base_params)
        path = output_dir / f"train_{cid}.py"
        path.write_text(new_code)
        candidates.append({"id": cid, "train_py_path": str(path), "description": desc})
        idx += 1

    # Aggressive 후보
    n_aggressive = min(pipe_cfg["num_aggressive"], remaining - n_conservative - n_moderate)
    for _ in range(n_aggressive):
        cid = f"v{idx:02d}"
        desc, new_code = _make_aggressive(base_code, base_params)
        path = output_dir / f"train_{cid}.py"
        path.write_text(new_code)
        candidates.append({"id": cid, "train_py_path": str(path), "description": desc})
        idx += 1

    # Crossover 후보 (히스토리 있을 때만)
    n_crossover = min(pipe_cfg["num_crossover"], remaining - n_conservative - n_moderate - n_aggressive)
    if history and len(history) >= 2 and n_crossover > 0:
        for _ in range(n_crossover):
            cid = f"v{idx:02d}"
            desc, new_code = _make_crossover(base_code, base_params, history)
            path = output_dir / f"train_{cid}.py"
            path.write_text(new_code)
            candidates.append({"id": cid, "train_py_path": str(path), "description": desc})
            idx += 1

    return candidates


def _parse_hyperparams(code: str) -> dict:
    """train.py에서 하이퍼파라미터 값을 파싱."""
    params = {}
    patterns = {
        "ASPECT_RATIO": r"^ASPECT_RATIO\s*=\s*(\d+)",
        "HEAD_DIM": r"^HEAD_DIM\s*=\s*(\d+)",
        "WINDOW_PATTERN": r'^WINDOW_PATTERN\s*=\s*"([^"]+)"',
        "TOTAL_BATCH_SIZE": r"^TOTAL_BATCH_SIZE\s*=\s*(.+?)(?:\s*#|$)",
        "EMBEDDING_LR": r"^EMBEDDING_LR\s*=\s*([0-9.]+)",
        "UNEMBEDDING_LR": r"^UNEMBEDDING_LR\s*=\s*([0-9.]+)",
        "MATRIX_LR": r"^MATRIX_LR\s*=\s*([0-9.]+)",
        "SCALAR_LR": r"^SCALAR_LR\s*=\s*([0-9.]+)",
        "WEIGHT_DECAY": r"^WEIGHT_DECAY\s*=\s*([0-9.]+)",
        "WARMUP_RATIO": r"^WARMUP_RATIO\s*=\s*([0-9.]+)",
        "WARMDOWN_RATIO": r"^WARMDOWN_RATIO\s*=\s*([0-9.]+)",
        "DEPTH": r"^DEPTH\s*=\s*(\d+)",
        "DEVICE_BATCH_SIZE": r"^DEVICE_BATCH_SIZE\s*=\s*(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, code, re.MULTILINE)
        if match:
            val = match.group(1)
            try:
                if "." in val:
                    params[key] = float(val)
                elif "**" in val:
                    params[key] = eval(val)
                else:
                    params[key] = int(val)
            except (ValueError, SyntaxError):
                params[key] = val
    return params


def _replace_param(code: str, param: str, new_value) -> str:
    """train.py에서 특정 하이퍼파라미터 값을 교체."""
    if param == "WINDOW_PATTERN":
        return re.sub(
            r'^(WINDOW_PATTERN\s*=\s*)"[^"]+"',
            f'\\1"{new_value}"',
            code, count=1, flags=re.MULTILINE,
        )
    elif param == "TOTAL_BATCH_SIZE" and isinstance(new_value, int):
        # 2의 거듭제곱으로 표현
        import math
        exp = int(math.log2(new_value))
        return re.sub(
            r"^(TOTAL_BATCH_SIZE\s*=\s*).+?(\s*#|$)",
            f"\\g<1>2**{exp}\\2",
            code, count=1, flags=re.MULTILINE,
        )
    else:
        formatted = f"{new_value}" if isinstance(new_value, int) else f"{new_value:.4g}"
        return re.sub(
            rf"^({param}\s*=\s*)[^\s#]+",
            f"\\g<1>{formatted}",
            code, count=1, flags=re.MULTILINE,
        )


def _make_conservative(code: str, params: dict) -> tuple[str, str]:
    """하이퍼파라미터 미세 조정 (±10-30%)."""
    # LR 파라미터 중 하나를 랜덤으로 조정
    lr_params = ["EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR"]
    target = random.choice(lr_params)
    current = params.get(target, 0.04)
    factor = random.uniform(0.7, 1.3)
    new_val = round(current * factor, 6)

    desc = f"conservative: {target} {current} → {new_val}"
    new_code = _replace_param(code, target, new_val)
    return desc, new_code


def _make_moderate(code: str, params: dict) -> tuple[str, str]:
    """아키텍처/구조 변경."""
    changes = []
    new_code = code

    choice = random.choice(["depth", "width", "batch", "window", "multi"])

    if choice == "depth":
        current = params.get("DEPTH", 4)
        new_val = random.choice([max(2, current - 1), current + 1, current + 2])
        new_code = _replace_param(new_code, "DEPTH", new_val)
        changes.append(f"DEPTH {current}→{new_val}")

    elif choice == "width":
        current = params.get("ASPECT_RATIO", 64)
        new_val = random.choice([48, 64, 80, 96])
        new_code = _replace_param(new_code, "ASPECT_RATIO", new_val)
        changes.append(f"ASPECT_RATIO {current}→{new_val}")

    elif choice == "batch":
        import math
        current = params.get("TOTAL_BATCH_SIZE", 2**17)
        current_exp = int(math.log2(current))
        new_exp = random.choice([max(15, current_exp - 1), current_exp + 1])
        new_val = 2**new_exp
        new_code = _replace_param(new_code, "TOTAL_BATCH_SIZE", new_val)
        changes.append(f"TOTAL_BATCH_SIZE 2^{current_exp}→2^{new_exp}")

    elif choice == "window":
        patterns = ["L", "SL", "SSL", "SSSL"]
        current = params.get("WINDOW_PATTERN", "SL")
        new_val = random.choice([p for p in patterns if p != current])
        new_code = _replace_param(new_code, "WINDOW_PATTERN", new_val)
        changes.append(f"WINDOW_PATTERN {current}→{new_val}")

    elif choice == "multi":
        # LR + DEPTH 동시 변경
        lr = params.get("MATRIX_LR", 0.04)
        new_lr = round(lr * random.uniform(0.6, 1.5), 6)
        new_code = _replace_param(new_code, "MATRIX_LR", new_lr)
        depth = params.get("DEPTH", 8)
        new_depth = random.choice([max(4, depth - 1), depth + 1])
        new_code = _replace_param(new_code, "DEPTH", new_depth)
        changes.append(f"MATRIX_LR→{new_lr} + DEPTH→{new_depth}")

    desc = f"moderate: {'; '.join(changes)}"
    return desc, new_code


def _make_aggressive(code: str, params: dict) -> tuple[str, str]:
    """급진적 변경."""
    changes = []
    new_code = code

    choice = random.choice(["big_model", "high_lr", "deep_narrow", "wide_shallow"])

    if choice == "big_model":
        new_code = _replace_param(new_code, "DEPTH", 12)
        new_code = _replace_param(new_code, "ASPECT_RATIO", 80)
        new_code = _replace_param(new_code, "DEVICE_BATCH_SIZE", 64)
        changes.append("big model: DEPTH=12 ASPECT=80 BS=64")

    elif choice == "high_lr":
        new_code = _replace_param(new_code, "MATRIX_LR", 0.08)
        new_code = _replace_param(new_code, "EMBEDDING_LR", 1.0)
        new_code = _replace_param(new_code, "WARMDOWN_RATIO", 0.6)
        changes.append("high LR: MATRIX=0.08 EMBED=1.0 WARMDOWN=0.6")

    elif choice == "deep_narrow":
        new_code = _replace_param(new_code, "DEPTH", 16)
        new_code = _replace_param(new_code, "ASPECT_RATIO", 48)
        new_code = _replace_param(new_code, "DEVICE_BATCH_SIZE", 64)
        changes.append("deep narrow: DEPTH=16 ASPECT=48 BS=64")

    elif choice == "wide_shallow":
        new_code = _replace_param(new_code, "DEPTH", 6)
        new_code = _replace_param(new_code, "ASPECT_RATIO", 96)
        new_code = _replace_param(new_code, "DEVICE_BATCH_SIZE", 128)
        changes.append("wide shallow: DEPTH=6 ASPECT=96 BS=128")

    desc = f"aggressive: {'; '.join(changes)}"
    return desc, new_code


def _make_crossover(
    code: str,
    params: dict,
    history: list[dict],
) -> tuple[str, str]:
    """이전 세대 Top-2의 아이디어를 결합."""
    # history에서 성공한 결과 중 Top-2 선택
    successes = [h for h in history if h.get("status") == "success" and h.get("val_bpb", 0) > 0]
    if len(successes) < 2:
        return _make_moderate(code, params)

    top2 = sorted(successes, key=lambda x: x["val_bpb"])[:2]
    desc_parts = [t.get("description", "") for t in top2]

    # 현재 파라미터에서 랜덤으로 2개를 이전 좋은 결과의 방향으로 조정
    new_code = code
    changes = []

    # Conservative + Moderate 조합
    lr = params.get("MATRIX_LR", 0.04)
    new_lr = round(lr * random.uniform(0.8, 1.2), 6)
    new_code = _replace_param(new_code, "MATRIX_LR", new_lr)
    changes.append(f"LR→{new_lr}")

    warmdown = params.get("WARMDOWN_RATIO", 0.5)
    new_warmdown = round(random.uniform(0.3, 0.6), 2)
    new_code = _replace_param(new_code, "WARMDOWN_RATIO", new_warmdown)
    changes.append(f"WARMDOWN→{new_warmdown}")

    desc = f"crossover from [{desc_parts[0][:30]}] + [{desc_parts[1][:30]}]: {'; '.join(changes)}"
    return desc, new_code
