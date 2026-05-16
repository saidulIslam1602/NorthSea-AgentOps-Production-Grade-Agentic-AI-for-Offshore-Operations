.DEFAULT_GOAL := help
PYTHON        := python3
PIP           := $(PYTHON) -m pip

# ─── Setup ────────────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install all dependencies (prod + dev) into the active environment
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	pre-commit install

.PHONY: install-prod
install-prod: ## Install production dependencies only (no dev toolchain)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[prod]"

# ─── Docker Compose ───────────────────────────────────────────────────────────

.PHONY: up
up: ## Start the full stack (Postgres, Kafka, Prometheus, Grafana, MLflow, API, consumer)
	docker compose up --build -d

.PHONY: up-infra
up-infra: ## Start only infrastructure services (no API or consumer)
	docker compose up -d postgres zookeeper kafka kafka-ui prometheus grafana otel-collector mlflow

.PHONY: down
down: ## Stop and remove all containers (data volumes are preserved)
	docker compose down

.PHONY: down-volumes
down-volumes: ## Stop all containers AND delete all data volumes (destructive)
	docker compose down -v

.PHONY: logs
logs: ## Tail logs for all running services
	docker compose logs -f

.PHONY: logs-api
logs-api: ## Tail API container logs only
	docker compose logs -f api

# ─── Database ─────────────────────────────────────────────────────────────────

.PHONY: migrate
migrate: ## Apply all pending Alembic migrations (agentops migrate)
	agentops migrate

.PHONY: migrate-sql
migrate-sql: ## Bootstrap schema directly with init.sql (for fresh containers)
	docker compose exec postgres psql -U northsea -d northsea_agentops -f /docker-entrypoint-initdb.d/init.sql

# ─── Data Ingestion ───────────────────────────────────────────────────────────

.PHONY: ingest
ingest: ## Ingest all docs under data/docs/ into the RAG vector store
	agentops ingest

.PHONY: ingest-fresh
ingest-fresh: ## Clear the vector store first, then re-ingest everything
	agentops ingest --clear

# ─── Testing ──────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run unit + integration tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

.PHONY: test-unit
test-unit: ## Run unit tests only (no DB or external services required)
	pytest tests/unit/ -v --cov=src --cov-report=term-missing

.PHONY: test-integration
test-integration: ## Run integration tests (requires running Postgres)
	pytest tests/integration/ -v -m "not requires_kafka" --timeout=60 --tb=short --no-cov

# ─── Code Quality ─────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Run ruff linter and mypy type checker
	ruff check src/ eval/ tests/
	mypy src/ --ignore-missing-imports --no-error-summary

.PHONY: format
format: ## Auto-format all Python files with ruff
	ruff format src/ eval/ tests/

.PHONY: format-check
format-check: ## Check formatting without applying changes (CI mode)
	ruff format --check src/ eval/ tests/

# ─── Evaluation ───────────────────────────────────────────────────────────────

.PHONY: eval
eval: ## Run RAGAS evaluation against the golden test set
	agentops evaluate

.PHONY: eval-detector
eval-detector: ## Train and evaluate anomaly detectors v1 vs v2 on Volve data
	agentops detector-eval

.PHONY: eval-agent
eval-agent: ## Run agent evaluation suite (plan quality, escalation calibration, ECE)
	$(PYTHON) -m eval.agent_eval --output eval/agent_eval_results.json

.PHONY: eval-regression
eval-regression: ## Run agent regression tests (no LLM required)
	$(PYTHON) -m eval.regression_tests

.PHONY: eval-adversarial
eval-adversarial: ## Run prompt injection tests
	$(PYTHON) -m eval.adversarial_tests

# ─── Development Utilities ────────────────────────────────────────────────────

.PHONY: serve
serve: ## Start the FastAPI server locally (hot reload enabled)
	agentops serve --reload

.PHONY: serve-prod
serve-prod: ## Start the FastAPI server in production mode (4 workers)
	agentops serve --workers 4

# ─── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "NorthSea AgentOps — Production-Grade Agentic AI for Offshore Operations"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""
