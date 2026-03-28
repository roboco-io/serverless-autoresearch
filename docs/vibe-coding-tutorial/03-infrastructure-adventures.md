# Chapter 3: Infrastructure Adventures — IAM, Regions, and Quota Wars

> **Time**: 2 hours
> **Cost**: $0.00 (infrastructure setup only)
> **Key Insight**: GPU Spot quotas default to 0 — request them BEFORE you need them.

## Context
Code is ready, but AWS infrastructure needs to be configured: IAM roles, S3 data upload, and GPU instance quotas. This turned out to be the most educational part of the journey.

## The Prompts

### Setting Up IAM
The AI ran `infrastructure/setup_iam.sh` which created a SageMaker execution role with S3, ECR, and CloudWatch permissions. First attempt was on the `personal` AWS profile, but later:

```
personal 프로필에 만든 리소스를 모두 삭제하고 roboco 프로필에 실험용 리소스를 생성해줘.
```
> *Translation: "Delete all resources created in the personal profile and create experiment resources in the roboco profile."*

Lesson: decide on the AWS account early.

### The Region Saga
This was the biggest infrastructure lesson:

```
도쿄리전이 아니라 us-west-2를 사용해줘.
```
> *Translation: "Use us-west-2 instead of the Tokyo region."*

→ Moved everything to Oregon.

```
할당량 요청이 가능한건 p5.24xlarge 또는 p5.48xlarge 만 가능하다고 하는데 사실인지 확인해줘.
```
> *Translation: "I heard that quota requests are only possible for p5.24xlarge or p5.48xlarge — verify if that's true."*

→ Investigated: ml.p5.4xlarge (single H100) exists but needs quota approval.

```
웹검색이 아니라 aws cli로 spot 사용 가능한 인스턴스 타입을 us-west-2와 ap-northeast-1 에서 조사해줘
```
> *Translation: "Instead of a web search, use the AWS CLI to investigate which instance types are available for Spot in us-west-2 and ap-northeast-1."*

→ Used `aws service-quotas list-aws-default-service-quotas` to get the definitive list.

### Quota Management
```
us-west-2의 p6, G7e에 대해서도 용량 확보를 요청해줘.
```
> *Translation: "Also request capacity for p6 and G7e in us-west-2."*

Results:
- g7e (L40S): **Auto-approved within minutes** ✓
- p5 (H100): CASE_OPENED (manual review, days) ✗
- p6 (B200/B300): CASE_OPENED ✗

### Discovering Spot Placement Scores
```
현재 스팟 풀의 상태를 조사할 수 있는 방법이 있을까?
```
> *Translation: "Is there a way to investigate the current state of the Spot pool?"*

The AI used Perplexity to discover `aws ec2 get-spot-placement-scores`:
```bash
# us-west-2: Score 1-2 (near impossible)
# us-east-1: Score 9 (instant allocation!)
```

This single command saved hours of stuck jobs.

```
그래. us-east-1에 쿼터 증가 신청도 미리 해줘.
```
> *Translation: "Right. Also submit a quota increase request for us-east-1 in advance."*

All g7e quotas auto-approved in us-east-1. We migrated everything.

## What Happened (Timeline)
- Tried Tokyo → quota needed, requested
- Moved to Oregon → quota needed, requested
- g7e approved but **Spot capacity score 1-2** → jobs stuck 30+ min
- Discovered placement scores → **us-east-1 score 9**
- Migrated to Virginia → **instant allocation**

## The Result
Final setup:
- Region: **us-east-1** (Virginia) — Spot score 9
- Instance: **ml.g7e.2xlarge** (L40S 48GB) — auto-approved quota 4
- Profile: **roboco** (dedicated AWS account)

### Debugging: Spot Instance Won't Start
**Symptom**: Job stuck in "Starting" for 30+ minutes
**Root Cause**: us-west-2 has Spot placement score 1-2 for g7e instances
**Fix**: `aws ec2 get-spot-placement-scores` → switch to us-east-1 (score 9)
**Time Lost**: ~1 hour across multiple attempts

### Debugging: Quota is Zero
**Symptom**: `ResourceLimitExceeded: account-level service limit is 0`
**Root Cause**: New GPU instance types (g7e, p5) have 0 default quota
**Fix**: Request via `aws service-quotas request-service-quota-increase`
**Time Lost**: ~30 minutes (g7e auto-approves; p5 takes days)

## Lessons Learned
- **Check Spot placement scores BEFORE choosing a region** — 30 seconds saves 30 minutes
- **g7e auto-approves, p5 doesn't** — plan GPU instance type accordingly
- **Larger instances can be cheaper on Spot** — g7e.8xlarge was $0.93/hr vs g7e.2xlarge $1.82/hr
- **Keep data in same region** — S3 upload needed per region migration
- **`config.yaml` should never be in git** — contains role ARN and profile

## Try It Yourself
```bash
# 1. Check Spot capacity before choosing a region
for region in us-east-1 us-east-2 us-west-2; do
  echo -n "$region: "
  aws ec2 get-spot-placement-scores \
    --instance-types g7e.4xlarge --target-capacity 1 \
    --single-availability-zone --region-names $region \
    --region $region \
    --query "max_by(SpotPlacementScores, &Score).Score" --output text
done

# 2. Request quotas
./infrastructure/setup_iam.sh --region us-east-1

# 3. Upload data
make prepare
```

### Running Cost
| Phase | Action | Cost | Cumulative |
|-------|--------|------|-----------|
| Planning | Architecture design | $0.00 | $0.00 |
| Building | Code generation | $0.00 | $0.00 |
| Infra | IAM + S3 + quota requests | $0.00 | $0.00 |
