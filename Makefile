.DEFAULT_GOAL := help

.PHONY: help install train test test-ci run docker-build docker-run lint format setup-git release clean

help: ## Mostra os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependências e configura pre-commit
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install
	cp -n .env.example .env || true
	@echo "✅ Instalação concluída! Edite o arquivo .env com suas configurações."

train: ## Executa o treinamento do modelo
	python -m src.train

evaluate: ## Avalia o modelo treinado
	python -m src.evaluate

test: ## Executa os testes unitários com relatório de cobertura
	pytest tests/ --cov=src --cov=app --cov-report=term-missing -v

test-ci: ## Executa os testes no modo CI (falha se cobertura < 80%)
	pytest tests/ --cov=src --cov=app --cov-report=xml --cov-fail-under=80 -v

run: ## Inicia a API em modo desenvolvimento
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build: ## Faz o build da imagem Docker
	docker build -t passos-magicos-ml .

docker-run: ## Executa o container Docker
	docker run -p 8000:8000 --env-file .env passos-magicos-ml

docker-compose-up: ## Inicia todos os serviços com Docker Compose
	docker-compose up --build -d

docker-compose-down: ## Para todos os serviços Docker Compose
	docker-compose down

lint: ## Executa o linting com ruff
	ruff check src/ app/ tests/

format: ## Formata o código com black
	black src/ app/ tests/

format-check: ## Verifica formatação sem alterar arquivos
	black --check src/ app/ tests/

monitoring: ## Inicia o dashboard de monitoramento (Streamlit)
	streamlit run monitoring/dashboard.py --server.port 8501

drift-report: ## Gera relatório de drift dos dados
	python monitoring/drift_report.py

clean: ## Remove arquivos temporários e de build
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml .coverage
	@echo "✅ Limpeza concluída!"

setup-git: ## Inicializa o repositório Git e configura branches
	git init
	git add .
	git commit -m "feat: initial project scaffold"
	git checkout -b develop
	pre-commit install
	@echo "✅ Git configurado! Crie sua primeira feature branch com: git checkout -b feature/eda"

release: ## Cria uma nova release (uso: make release VERSION=v1.0.0)
	@if [ -z "$(VERSION)" ]; then echo "❌ Informe a versão: make release VERSION=v1.0.0"; exit 1; fi
	git checkout main
	git merge develop
	git tag $(VERSION)
	git push origin main --tags
	@echo "✅ Release $(VERSION) criada!"
