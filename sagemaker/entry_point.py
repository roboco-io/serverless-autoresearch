#!/usr/bin/env python3
"""SageMaker 학습 작업 엔트리포인트.

SageMaker가 이 스크립트를 실행하며, source_dir의 train.py와 prepare.py를
적절한 환경변수와 함께 실행한다.
"""
import os
import sys

def main():
    # SageMaker 환경변수 설정
    # source_dir의 파일들은 /opt/ml/code/에 복사됨
    code_dir = os.environ.get("SAGEMAKER_SUBMIT_DIRECTORY", "/opt/ml/code")

    # prepare.py가 SageMaker 입력 채널 경로를 사용하도록 설정
    os.environ.setdefault("SM_CHANNEL_DATA",
                          os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data"))
    os.environ.setdefault("SM_CHANNEL_TOKENIZER",
                          os.environ.get("SM_CHANNEL_TOKENIZER", "/opt/ml/input/data/tokenizer"))
    os.environ.setdefault("SM_CACHE_DIR", "/opt/ml/input/data")

    # train_wrapper를 통해 train.py 실행
    sys.path.insert(0, code_dir)
    from train_wrapper import run_training

    results = run_training(code_dir)

    # 결과를 SageMaker 출력 디렉토리에 저장
    import json
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
