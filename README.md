# Passos Mágicos — Predição de Risco de Defasagem Escolar

<div align="center">
  <img src="https://passosmagicos.org.br/wp-content/uploads/2020/08/logo.png" alt="Passos Mágicos" width="200"/>

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
  [![scikit-learn](https://img.shields.io/badge/sklearn-1.4-orange.svg)](https://scikit-learn.org/)
  [![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

---

## 🎯 Visão Geral

Projeto **MLOps end-to-end** desenvolvido para o **Datathon da Associação Passos Mágicos**, uma ONG que transforma a vida de crianças e jovens de baixa renda por meio da educação no município de Embu-Guaçu (SP).

**Problema:** Identificar estudantes com risco de defasagem escolar antes que o problema se agrave, permitindo intervenção pedagógica preventiva.

**Solução:** Pipeline completa de Machine Learning com API REST em produção, capaz de prever o risco de defasagem com base nos indicadores PEDE (Pesquisa Extensiva do Desenvolvimento Educacional).

### Stack Tecnológica
| Camada | Tecnologia |
|--------|-----------|
| **ML** | scikit-learn · XGBoost · LightGBM · Optuna |
| **API** | FastAPI · Uvicorn · Pydantic |
| **Testes** | pytest · pytest-cov (≥80%) |
| **Monitoramento** | Evidently AI · Streamlit |
| **Deploy** | Docker · Docker Compose |
| **CI/CD** | GitHub Actions |

---

## 📁 Estrutura do Projeto

```
passos_magicos_datathon/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # Lint → Tests → Docker build
│   │   └── cd.yml              # Deploy em tags v*
│   └── pull_request_template.md
├── app/
│   ├── main.py                 # FastAPI app + lifespan model loading
│   ├── routes.py               # Endpoints: /predict, /predict/batch, /health, /model-info
│   └── model/                  # Modelos serializados (.joblib)
├── src/
│   ├── utils.py                # Config (.env), logger JSON, helpers
│   ├── preprocessing.py        # Load, clean, encode, scale, build_target, split
│   ├── feature_engineering.py  # Trend, agg, interaction, phase-age gap, SHAP
│   ├── train.py                # Multi-model CV, Optuna, joblib save
│   └── evaluate.py             # Métricas, ROC, PR, confusion matrix
├── tests/
│   ├── test_preprocessing.py   # Testes unitários de preprocessing
│   ├── test_feature_engineering.py
│   ├── test_model.py           # Loading, prediction format (mocks)
│   └── test_api.py             # TestClient: todos os endpoints
├── monitoring/
│   ├── drift_report.py         # PSI + KS test + Evidently AI
│   └── dashboard.py            # Streamlit dashboard
├── notebooks/
│   └── eda.ipynb               # Análise Exploratória de Dados
├── data/
│   ├── raw/                    # Arquivos PEDE originais (CSV/XLSX)
│   └── processed/              # Dados processados para treino
├── logs/                       # Logs JSON estruturados
├── .env.example                # Template de variáveis de ambiente
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## 🚀 Instruções de Deploy

### Pré-requisitos
- Python 3.10+
- Docker (opcional)
- Git

### 1. Setup Local

```bash
# Clonar o repositório
git clone <seu-repositorio>
cd passos_magicos_datathon

# Instalar dependências e configurar ambiente
make install

# Editar variáveis de ambiente
cp .env.example .env
# Edite .env conforme necessário
```

### 2. Adicionar os Dados

Coloque os arquivos CSV/XLSX do PEDE em:
```
data/raw/PEDE_2022.csv
data/raw/PEDE_2023.csv
data/raw/PEDE_2024.csv
```

### 3. Executar o ETL (normalização dos dados)

```bash
python -m src.etl
# Unifica os arquivos PEDE 2022-2024 em data/processed/pede_unified.csv
# Também gera train_reference.csv e current_production.csv para monitoramento
```

### 4. Treinar o Modelo

```bash
make train
# Treina todos os modelos, otimiza hiperparâmetros com Optuna,
# salva o melhor modelo em app/model/model.joblib
```

### 5. Executar a API

```bash
# Modo desenvolvimento (com reload)
make run

# A API estará disponível em: http://localhost:8000
# Documentação interativa: http://localhost:8000/docs
```

### 6. Deploy com Docker

```bash
# Build da imagem
make docker-build

# Executar container
make docker-run

# Ou com Docker Compose (API + monitoramento)
make docker-compose-up
```

---

## 📡 Exemplos de Chamadas à API

### curl

```bash
# Health check
curl http://localhost:8000/health

# Predição individual
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "IDADE": 14,
    "FASE": "F6",
    "PEDRA": "Ametista",
    "IAA": 7.5,
    "IEG": 6.8,
    "IPS": 8.0,
    "IDA": 7.2,
    "IPP": 6.5,
    "IPV": 7.0,
    "IAN": 6.9,
    "INDE": 7.1
  }'

# Predição em lote
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"students": [{"IDADE": 14, "FASE": "F6", "PEDRA": "Ametista", "IAA": 7.5, "IEG": 6.8, "IPS": 8.0, "IDA": 7.2, "IPP": 6.5, "IPV": 7.0, "IAN": 6.9, "INDE": 7.1}]}'
```

### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8000"

student = {
    "IDADE": 14, "FASE": "F6", "PEDRA": "Ametista",
    "IAA": 7.5, "IEG": 6.8, "IPS": 8.0, "IDA": 7.2,
    "IPP": 6.5, "IPV": 7.0, "IAN": 6.9, "INDE": 7.1
}

# Single prediction
response = requests.post(f"{BASE_URL}/predict", json=student)
result = response.json()

print(f"Risco: {result['student_risk']}")
print(f"Probabilidade: {result['probability']:.1%}")
print(f"Recomendação: {result['recommendation']}")

# Expected output:
# Risco: alto
# Probabilidade: 70.0%
# Recomendação: 📋 Acompanhamento pedagógico prioritário recomendado...
```

---

## 🤖 Pipeline de ML

### 1. EDA (`notebooks/eda.ipynb`)
Análise exploratória: distribuições, nulos, correlações, análise temporal 2022→2024.

### 2. Pré-processamento (`src/preprocessing.py`)
| Etapa | Estratégia |
|-------|-----------|
| Nulos numéricos | Mediana |
| Nulos categóricos | Moda |
| Encoding | OneHotEncoder (categóricas) |
| Scaling | StandardScaler |
| Target | RISCO_DEFASAGEM: INDE < 5.5 ou gap de fase/idade ≥ 2 |
| Split | 70% treino / 15% val / 15% teste (estratificado) |

### 3. Feature Engineering (`src/feature_engineering.py`)
- **Tendência:** INDE_DELTA, INDE_TREND entre anos
- **Agregações:** média, desvio, mín, máx dos indicadores
- **Interações:** INDE×IPS, IDA×IPP, IEG×IAN
- **Defasagem fase/idade:** FASE_GAP, DEFASAGEM_SEVERA

### 4. Treinamento (`src/train.py`)
- Candidatos: LogisticRegression, RandomForest, XGBoost, LightGBM, SVM
- Validação: StratifiedKFold (k=5)
- Otimização: **Optuna** (50 trials por padrão)
- Seleção: melhor Recall na validação cruzada

### 5. Avaliação (`src/evaluate.py`)
- Accuracy, Precision, **Recall**, F1-Score, AUC-ROC
- Curvas: ROC, Precision-Recall, Confusion Matrix

---

## 📊 Métricas e Justificativa

**Métrica principal: Recall**

> No contexto de identificação de estudantes em risco, **deixar de identificar um aluno que precisava de ajuda** (Falso Negativo) é muito mais custoso do que **acionar intervenção desnecessária** (Falso Positivo). Por isso, maximizamos o Recall, garantindo que o máximo de alunos em risco seja identificado.

| Métrica | Meta |
|---------|------|
| Recall | ≥ 0.80 |
| AUC-ROC | ≥ 0.85 |
| F1-Score | ≥ 0.75 |

---

## 🔍 Monitoramento

### Logs
Logs estruturados em JSON disponíveis em `logs/app.log`.

### Dashboard Streamlit
```bash
make monitoring
# Acesse: http://localhost:8501
```

### Relatório de Drift
```bash
make drift-report
# Relatório salvo em monitoring/reports/
```

**Indicadores monitorados:**
- PSI por feature (Population Stability Index)
- KS test (Kolmogorov-Smirnov)
- Relatório HTML com Evidently AI

---

## 🧪 Testes

```bash
# Executar todos os testes com cobertura
make test

# Executar no modo CI (falha se cobertura < 80%)
make test-ci
```

**Cobertura mínima: 80%**

---

## 🔧 Comandos Úteis (Makefile)

```bash
make help           # Lista todos os comandos
make install        # Instala dependências + pre-commit
make train          # Treina o modelo
make test           # Executa testes
make run            # Inicia a API
make docker-build   # Build Docker
make lint           # Linting com ruff
make format         # Formata com black
make setup-git      # Inicializa Git + configura branches
```

---

## 📋 Branches e Fluxo de Trabalho

| Branch | Uso |
|--------|-----|
| `main` | Produção. Protegida. |
| `develop` | Integração. |
| `feature/*` | Novas features. |
| `fix/*` | Correções. |
| `release/*` | Preparação de release. |

```bash
# Fluxo de desenvolvimento
git checkout -b feature/minha-feature develop
# ... trabalhe ...
git commit -m "feat: descrição da feature"
git push origin feature/minha-feature
# Abra Pull Request para develop
```

---

## 📄 Licença

MIT License — Associação Passos Mágicos Datathon
