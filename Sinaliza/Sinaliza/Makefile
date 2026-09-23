.PHONY: help install dev test lint train evaluate docker clean

help: ## Mostra esta ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------- Setup ----------

install: ## Instala dependências Python + Frontend
	pip install -r requirements.txt
	cd frontend && npm install

# ---------- Dev ----------

dev-backend: ## Inicia backend em modo dev
	uvicorn backend.main:app --reload --port 8000

dev-frontend: ## Inicia frontend em modo dev
	cd frontend && npm run dev

dev: ## Inicia backend + frontend (requer 2 terminais — use tmux ou docker)
	@echo "Execute em terminais separados:"
	@echo "  make dev-backend"
	@echo "  make dev-frontend"

# ---------- Pipeline ML ----------

download: ## Download do dataset V-Librasil
	python scripts/download_dataset.py --output data/raw

extract: ## Extrair landmarks com MediaPipe
	python scripts/extract_landmarks.py \
		--input data/raw/v-librasil \
		--output data/landmarks \
		--workers 4

validate: ## Validar landmarks extraídos
	python scripts/validate_landmarks.py --input data/landmarks

build-dataset: ## Construir splits de treino/val/teste
	python scripts/build_dataset.py \
		--input data/landmarks \
		--output data/processed

pipeline: download extract validate build-dataset ## Pipeline completo de dados

# ---------- Treinamento ----------

train-bilstm: ## Treinar modelo Bi-LSTM
	python scripts/train.py --config configs/bilstm.yaml

train-transformer: ## Treinar modelo Transformer
	python scripts/train.py --config configs/transformer.yaml

train-tcn: ## Treinar modelo TCN
	python scripts/train.py --config configs/tcn.yaml

train-all: train-bilstm train-transformer train-tcn ## Treinar todos os modelos

# ---------- Avaliação ----------

evaluate: ## Avaliar modelo (CHECKPOINT=path)
	python scripts/evaluate.py --checkpoint $(CHECKPOINT)

compare: ## Comparar todos os modelos
	python scripts/compare_models.py \
		--checkpoints artifacts/checkpoints/best_*.pt \
		--output artifacts/comparison

# ---------- Testes ----------

test: ## Executar testes
	pytest tests/ -v --tb=short

test-cov: ## Testes com cobertura
	pytest tests/ -v --cov=ml --cov=backend --cov-report=html

# ---------- Docker ----------

docker: ## Build e run com Docker Compose
	docker compose up --build

docker-down: ## Parar containers
	docker compose down

# ---------- Limpeza ----------

clean: ## Limpar artefatos e cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf frontend/dist frontend/.vite

clean-all: clean ## Limpar tudo (incluindo artifacts)
	rm -rf artifacts/
