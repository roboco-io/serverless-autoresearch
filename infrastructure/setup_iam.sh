#!/bin/bash
set -euo pipefail

# SageMaker 실행 역할 생성
# Usage: ./setup_iam.sh [--profile PROFILE] [--region REGION]

PROFILE="${AWS_PROFILE:-personal}"
REGION="${AWS_REGION:-ap-northeast-2}"
ROLE_NAME="SageMakerAutoresearchRole"

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile) PROFILE="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ACCOUNT=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)

echo "Creating SageMaker execution role: ${ROLE_NAME}"

# Trust policy
TRUST_POLICY=$(cat <<'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
)

# 역할 생성 (이미 있으면 스킵)
aws iam get-role --role-name "$ROLE_NAME" --profile "$PROFILE" 2>/dev/null || \
aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY" \
    --profile "$PROFILE"

# 필요한 정책 연결
POLICIES=(
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
)

for POLICY in "${POLICIES[@]}"; do
    echo "  Attaching: $(basename "$POLICY")"
    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "$POLICY" \
        --profile "$PROFILE" 2>/dev/null || true
done

ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}"
echo ""
echo "============================================"
echo "Role created: ${ROLE_ARN}"
echo ""
echo "config.yaml의 aws.role_arn을 업데이트하세요:"
echo "  role_arn: ${ROLE_ARN}"
echo "============================================"
