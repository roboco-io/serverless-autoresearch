.PHONY: prepare run run-single dry-run cost setup help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Create IAM role for SageMaker
	./infrastructure/setup_iam.sh

prepare: ## Download data + train tokenizer + upload to S3
	python src/scripts/prepare_s3.py

run: ## Run full pipeline (10 gen x 10 pop)
	python -m src.pipeline.orchestrator --generations 10 --population 10

run-single: ## Run single experiment (debug)
	python src/scripts/run_single.py

dry-run: ## Dry run (no cost, verify config)
	python -m src.pipeline.orchestrator --dry-run --single --population 3

cost: ## Show cost report
	python src/scripts/cost_report.py
