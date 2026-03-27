"""train.py를 SageMaker 환경에서 실행하는 래퍼.

train.py는 모듈이 아닌 스크립트이므로 subprocess로 실행하고
stdout에서 결과를 파싱한다.
"""
import json
import os
import re
import subprocess
import sys


def run_training(code_dir: str) -> dict:
    """train.py를 실행하고 결과를 파싱한다."""
    train_py = os.path.join(code_dir, "train.py")

    if not os.path.exists(train_py):
        print(f"ERROR: train.py not found at {train_py}")
        sys.exit(1)

    print(f"Running training: {train_py}")
    print(f"  Data: {os.environ.get('SM_CHANNEL_DATA', 'default')}")
    print(f"  Tokenizer: {os.environ.get('SM_CHANNEL_TOKENIZER', 'default')}")

    # train.py 실행 (prepare.py는 같은 디렉토리에서 import됨)
    env = os.environ.copy()
    env["PYTHONPATH"] = code_dir + ":" + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, train_py],
        cwd=code_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,  # 15분 타임아웃
    )

    # stdout 전체를 출력 (CloudWatch Logs로 캡처됨)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        print(f"FAIL: train.py exited with code {proc.returncode}")
        return {"status": "crash", "exit_code": proc.returncode}

    # 결과 파싱 (--- 이후 key: value 형식)
    results = parse_results(proc.stdout)
    results["status"] = "success"
    return results


def parse_results(stdout: str) -> dict:
    """train.py stdout에서 결과 메트릭을 파싱한다.

    Expected format after '---':
        val_bpb:          0.997900
        training_seconds: 300.1
        ...
    """
    results = {}
    in_results = False

    for line in stdout.split("\n"):
        line = line.strip()
        if line == "---":
            in_results = True
            continue
        if in_results and ":" in line:
            match = re.match(r"^(\w+):\s+(.+)$", line)
            if match:
                key, value = match.group(1), match.group(2)
                try:
                    results[key] = float(value)
                except ValueError:
                    results[key] = value

    return results
