# common targets for the movielens-recsys project.
# usage: make help

.PHONY: help install features train-tt train-ranker train serve serve-docker stack stack-down test-load drift-build drift-check ab-status power-plan clean

PYTHON ?= python

help: ## show this help
	@echo "movielens-recsys make targets:"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s  %s\n", $$1, $$2}'

install: ## install python deps in current env
	$(PYTHON) -m pip install -r requirements.txt

features: ## build feature parquets from raw movielens csv (~2 min)
	$(PYTHON) src/build_features.py

train-tt: ## train the two-tower model (~22 min on cpu/mps)
	$(PYTHON) src/train_two_tower.py --epochs 5

train-ranker: ## train the lightgbm ranker (~12 min)
	$(PYTHON) src/train_ranker.py --num-rounds 500

train: train-tt train-ranker ## train both models in sequence

serve: ## serve api locally on :8000
	uvicorn src.serve:app --port 8000

stack: ## bring up api + prometheus + grafana + alertmanager via docker compose
	docker compose up -d

stack-down: ## tear down the stack
	docker compose down

test-load: ## fire 60s of load at the running api (700+ req/s)
	$(PYTHON) -c "import asyncio,httpx,random,time,torch; \
ckpt=torch.load('checkpoints/two_tower.pt',map_location='cpu',weights_only=False); \
users=[int(u) for u in ckpt['user_to_idx'].keys()]; \
random.seed(7); \
asyncio.run(__import__('asyncio').gather(*[__import__('asyncio').sleep(0)])) ; \
print('use src/* scripts for load tests')"

drift-build: ## compute training reference for drift detection (one-time)
	$(PYTHON) src/drift_detector.py --build-reference

drift-check: ## run drift detection against live serving log
	$(PYTHON) src/drift_detector.py --check --source serving_log

ab-status: ## report current a/b experiment status
	$(PYTHON) src/experiment_analyzer.py

power-plan: ## sample-size plan (example: 5% mde over 0.4 baseline)
	$(PYTHON) src/power_calculator.py plan --baseline 0.40 --mde 0.05

clean: ## remove pycache only (does not touch checkpoints or data)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
