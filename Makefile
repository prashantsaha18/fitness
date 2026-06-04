# FitAI — Developer convenience targets
# Usage: make <target>

.PHONY: help install dev lint format typecheck test test-unit test-integration \
        generate-data train index-content serve benchmark clean

PYTHON  := python3
PKGNAME := fitness-rec-engine

help:                          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-22s\033[0m %s\n",$$1,$$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────
install:                       ## Install package in editable mode
	pip install -e ".[dev,streamlit]"

dev: install                   ## Install + pre-commit hooks
	pre-commit install 2>/dev/null || true

# ── Code quality ──────────────────────────────────────────────────────────────
lint:                          ## Ruff linter (fast, replaces flake8+isort)
	ruff check . --fix

format:                        ## Black formatter
	black .

typecheck:                     ## Mypy static type checking
	mypy api/ ranking/ retrieval/ config/ --ignore-missing-imports

# ── Tests ─────────────────────────────────────────────────────────────────────
test:                          ## All tests with coverage
	pytest tests/ --cov --cov-report=term-missing -v

test-unit:                     ## Fast unit tests only (no I/O)
	pytest tests/unit/ -v --timeout=30

test-integration:              ## Integration tests (mocked services)
	pytest tests/integration/ -v --timeout=60

# ── Data pipeline ─────────────────────────────────────────────────────────────
generate-data:                 ## Generate Kaggle-schema synthetic datasets
	$(PYTHON) -c "from data_pipeline.kaggle.synthetic_datasets import generate_all; \
	              from pathlib import Path; generate_all(Path('data/kaggle_raw'))"

train:                         ## Train DeepFM on Kaggle datasets → ONNX export
	$(PYTHON) scripts/train_kaggle.py

index-content:                 ## Embed + index content catalogue into Qdrant
	$(PYTHON) scripts/index_content.py

seed-db:                       ## Seed NeonDB with synthetic users + content
	$(PYTHON) scripts/seed_data.py

populate-cache:                ## Pre-populate Redis fallback cache
	$(PYTHON) scripts/populate_fallback_cache.py --materialise

# ── Run ───────────────────────────────────────────────────────────────────────
serve:                         ## Start FastAPI recommendation API
	uvicorn api.main:app --reload --port 8000

streamlit:                     ## Start Streamlit frontend
	streamlit run frontend/app.py

# ── Load testing ──────────────────────────────────────────────────────────────
benchmark:                     ## Run 60-second P99 latency benchmark
	$(PYTHON) scripts/benchmark.py --rps 500 --duration 60

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:                     ## Start Qdrant + Redis + Kafka via Docker Compose
	docker compose up -d qdrant redis zookeeper kafka

docker-down:                   ## Stop all Docker services
	docker compose down

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:                         ## Remove generated artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build
	rm -rf data/kaggle_raw data/*.npy
	rm -f artifacts/*.pt artifacts/*.onnx artifacts/*.onnx.data
