#!/bin/bash
set -euo pipefail

# ECR에 학습 컨테이너 빌드 및 푸시
# Usage: ./build_and_push.sh [--profile PROFILE] [--region REGION]

PROFILE="${AWS_PROFILE:-personal}"
REGION="${AWS_REGION:-ap-northeast-2}"
REPO_NAME="autoresearch"
TAG="latest"

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile) PROFILE="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ACCOUNT=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_URI}/${REPO_NAME}:${TAG}"

echo "============================================"
echo "Building autoresearch training container"
echo "  Account:  ${ACCOUNT}"
echo "  Region:   ${REGION}"
echo "  Image:    ${IMAGE_URI}"
echo "============================================"

# ECR 레포 생성 (없으면)
aws ecr describe-repositories --repository-names "$REPO_NAME" \
    --profile "$PROFILE" --region "$REGION" 2>/dev/null || \
aws ecr create-repository --repository-name "$REPO_NAME" \
    --profile "$PROFILE" --region "$REGION"

# ECR 로그인
aws ecr get-login-password --profile "$PROFILE" --region "$REGION" | \
    docker login --username AWS --password-stdin "$ECR_URI"

# 빌드 컨텍스트: 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Docker 빌드
docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${REPO_NAME}:${TAG}" \
    "$PROJECT_DIR"

# 태그 및 푸시
docker tag "${REPO_NAME}:${TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo ""
echo "============================================"
echo "Image pushed: ${IMAGE_URI}"
echo ""
echo "config.yaml의 sagemaker.image_uri를 업데이트하세요:"
echo "  image_uri: ${IMAGE_URI}"
echo "============================================"
