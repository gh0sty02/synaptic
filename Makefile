.PHONY: dev infra backend frontend up down ingest serve lint lint-fix

dev: up

# Spin up only the infrastructure containers (postgres, redis)
infra:
	docker compose up -d postgres redis

# Run only the backend
backend:
	cd backend && ../.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run only the frontend
frontend:
	cd frontend && pnpm dev

# Run everything: infra in Docker + backend + frontend natively
up: infra
	@echo "Starting backend (port 8000) and frontend (port 3000)..."
	@trap 'kill 0' SIGINT; \
	$(MAKE) backend & \
	$(MAKE) frontend & \
	wait

# Run backend + frontend only (no Docker — assumes infra is already up)
serve:
	@echo "Starting backend (port 8000) and frontend (port 3000)..."
	@trap 'kill 0' SIGINT; \
	$(MAKE) backend & \
	$(MAKE) frontend & \
	wait

# Tear down all Docker containers
down:
	docker compose down

# Apply DB schema to local postgres
migrate:
	PGPASSWORD=synaptic_local psql -h localhost -U synaptic -d synaptic -f backend/db/schema.sql

# Run the StackOverflow ingestion pipeline
ingest:
	cd backend && PYTHONPATH=$(PWD)/backend ../.venv/bin/python ingestion/stackoverflow_loader.py

# Lint the backend with ruff
lint:
	cd backend && ../.venv/bin/ruff check .

# Lint and auto-fix what ruff can fix
lint-fix:
	cd backend && ../.venv/bin/ruff check --fix .

# Run the RAGAS eval suite against the held-out StackOverflow question set.
# Override sample size: make eval SAMPLE_SIZE=1000
SAMPLE_SIZE ?= 50
eval:
	cd backend && PYTHONPATH=$(PWD)/backend ../.venv/bin/python -m evals.ragas_runner --sample-size $(SAMPLE_SIZE)