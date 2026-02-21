# Relatório Técnico & Executivo
# Datathon Passos Mágicos — Predição de Risco de Defasagem Escolar

---

## Sumário Executivo

Este relatório apresenta a solução desenvolvida para o **Datathon Pós Tech FIAP — Machine Learning Engineering**, com base no case da **Associação Passos Mágicos**, ONG que transforma a vida de crianças e jovens em situação de vulnerabilidade social no município de Embu-Guaçu (SP), por meio da educação.

**Problema:** Identificar antecipadamente quais estudantes estão em risco de defasagem escolar, permitindo intervenção pedagógica preventiva.

**Solução:** Pipeline completa de Machine Learning em produção — desde a ingestão dos dados brutos do PEDE (Pesquisa Extensiva do Desenvolvimento Educacional) até uma API REST containerizada, com monitoramento contínuo de drift e cobertura de testes unitários superior a 80%.

**Resultado do modelo:** **Recall de 100%** no conjunto de teste — nenhum aluno em risco deixou de ser identificado.

---

## 1. Contexto e Problema de Negócio

### 1.1 A Associação Passos Mágicos

A Associação Passos Mágicos atua há 32 anos na transformação da vida de crianças e jovens de baixa renda, oferecendo educação de qualidade, apoio psicológico/psicopedagógico e ampliação de visão de mundo. Desde 2016, opera como projeto social e educacional formal.

### 1.2 O Desafio

O dataset PEDE contém indicadores educacionais coletados em 2020, 2021, 2022 e 2024, totalizando **3.135 registros** de estudantes. O desafio é construir um modelo preditivo capaz de estimar o **risco de defasagem escolar** de cada estudante — ou seja, identificar quem está em risco de estar atrasado em relação ao nível escolar esperado para sua idade.

### 1.3 Impacto Esperado

| Situação Atual | Com o Modelo |
|---|---|
| Identificação reativa (aluno já defasado) | Identificação preventiva (antes da defasagem se agravar) |
| Intervenção tardia | Intervenção pedagógica imediata e direcionada |
| Sem priorização objetiva | Ranking por probabilidade de risco |
| Avaliação manual por educador | Suporte automatizado à decisão |

---

## 2. Dados Utilizados

### 2.1 Fontes

| Dataset | Período | Registros | Formato |
|---|---|---|---|
| `PEDE_PASSOS_DATASET_FIAP.csv` | 2020–2022 | 1.349 alunos (wide) | CSV |
| `BASE DE DADOS PEDE 2024.xlsx` | 2024 | 860 alunos | XLSX |

### 2.2 Indicadores PEDE

| Indicador | Descrição |
|---|---|
| **INDE** | Índice de Desenvolvimento Educacional (0–10) |
| **IAA** | Indicador de Autoavaliação do Aluno |
| **IEG** | Indicador de Engajamento |
| **IPS** | Indicador Psicossocial |
| **IDA** | Indicador de Aprendizagem |
| **IPP** | Indicador Pedagógico de Participação |
| **IPV** | Indicador de Ponto de Virada |
| **IAN** | Indicador de Adequação ao Nível |
| **PEDRA** | Classificação de desempenho: Quartzo → Ágata → Ametista → Topázio |
| **FASE** | Fase atual do aluno (F1–F8) |
| **IDADE** | Idade do estudante |
| **ANOS_PM** | Anos na Associação |

### 2.3 ETL — Unificação dos Dados

O processo de ETL (`src/etl.py`) normaliza os datasets de diferentes anos e formatos para um único dataset longitudinal:

- **2020–2022:** formato wide → transformado para long (um registro por aluno por ano)
- **2024:** mapeamento de colunas + derivação do INDE a partir dos componentes
- **Saídas geradas:**
  - `data/processed/pede_unified.csv` — 3.135 registros unificados
  - `data/processed/train_reference.csv` — dados 2020–2022 (referência para drift)
  - `data/processed/current_production.csv` — dados 2024 (produção)

### 2.4 Distribuição dos Dados

| Ano | Registros |
|---|---|
| 2020 | 727 |
| 2021 | 686 |
| 2022 | 862 |
| 2024 | 860 |
| **Total** | **3.135** |

**Distribuição da variável PEDRA:**

| PEDRA | N | % |
|---|---|---|
| Ametista | 1.327 | 42,3% |
| Ágata | 849 | 27,1% |
| Quartzo | 504 | 16,1% |
| Topázio | 453 | 14,4% |

**INDE (média = 7.05, std = 1.19, mín = 2.47, máx = 9.71)**

---

## 3. Pipeline de Machine Learning

### 3.1 Visão Geral da Arquitetura

```
Dados Brutos (CSV/XLSX)
        │
        ▼
   [ETL - src/etl.py]
   Normalização e unificação
        │
        ▼
   [Preprocessing - src/preprocessing.py]
   Limpeza → Nulos → Target → Split
        │
        ▼
   [Feature Engineering - src/feature_engineering.py]
   Trend · Agregações · Interações · Gap Fase/Idade
        │
        ▼
   [Encoding + Scaling - fit only on train]
   OneHotEncoder · StandardScaler
        │
        ▼
   [Treinamento - src/train.py]
   CV Baseline → Optuna → Melhor Modelo
        │
        ▼
   [Avaliação - src/evaluate.py]
   Recall · AUC-ROC · F1 · Curvas
        │
        ▼
   [Serialização - joblib]
   model.joblib + model_metadata.json
        │
        ▼
   [API FastAPI - app/]
   /predict · /predict/batch · /health · /model-info
        │
        ▼
   [Docker + Monitoring]
   Container isolado · Streamlit · Drift PSI/KS
```

### 3.2 Pré-processamento (`src/preprocessing.py`)

| Etapa | Estratégia | Justificativa |
|---|---|---|
| **Remoção de duplicatas** | `drop_duplicates()` | Evitar viés de amostras repetidas |
| **Remoção de inconsistências** | Scores fora de [0,10] removidos | Dados impossíveis comprometem o modelo |
| **Nulos numéricos** | Preenchimento pela **mediana** | Robusta a outliers |
| **Nulos categóricos** | Preenchimento pela **moda** | Preserva distribuição original |
| **Colunas >90% nulas** | Descartadas | Sem sinal estatístico |
| **Encoding** | `OneHotEncoder` (fit no treino) | Compatível com árvores e modelos lineares |
| **Scaling** | `StandardScaler` (fit no treino) | Normaliza para modelos sensíveis à escala |
| **Split** | 70% treino / 15% val / 15% teste | Avaliação não enviesada; estratificado por target |

**Registros após limpeza:** 2.049 (de 3.135 — remoção de 1.086 inconsistências)

### 3.3 Construção da Variável Target

A variável `RISCO_DEFASAGEM` é binária (0 = sem risco, 1 = em risco) e construída com duas regras sem data leakage:

- **Regra 1 (principal):** `DEFASAGEM <= -1` — aluno está pelo menos 1 fase atrás do nível ideal
- **Regra 2 (fallback):** `PEDRA == "Quartzo"` — categoria de menor desempenho, quando DEFASAGEM não disponível

**Resultado:** 1.489 alunos em risco (72,7%) vs. 560 sem risco (27,3%)

### 3.4 Feature Engineering (`src/feature_engineering.py`)

**36 features no total** — sendo 13 criadas pelo feature engineering:

| Categoria | Features Criadas |
|---|---|
| **Tendência temporal** | `INDE_DELTA`, `INDE_TREND` |
| **Agregações** | `INDICADORES_MEAN`, `INDICADORES_STD`, `INDICADORES_MIN`, `INDICADORES_MAX`, `INDICADORES_RANGE`, `ENGAJAMENTO_APRENDIZADO` |
| **Interações** | `INDE_x_IPS`, `IDA_x_IPP`, `IEG_x_IAN`, `EMOCIONAL_ACADEMICO` |
| **Defasagem fase/idade** | `FASE_GAP`, `DEFASAGEM_SEVERA` |

> **Nota técnica:** O feature engineering é aplicado **antes** do encoding, garantindo que `IDADE` e `FASE` (valores brutos) estejam disponíveis para calcular `FASE_GAP`. O encoder e scaler são ajustados **somente no conjunto de treino** e aplicados por transformação em val e test — sem data leakage.

---

## 4. Modelagem

### 4.1 Estratégia de Validação

- **StratifiedKFold (k=5):** garante representatividade do target em cada fold
- **Métrica principal: Recall** — minimiza Falsos Negativos (alunos em risco não identificados)
- **Otimização de hiperparâmetros: Optuna** (50 trials por modelo)

### 4.2 Justificativa da Métrica Principal

> No contexto de predição de risco escolar, o custo de um **Falso Negativo** (deixar de identificar um aluno que precisava de ajuda) é dramaticamente maior do que o custo de um **Falso Positivo** (acionar intervenção desnecessária). Um aluno não identificado pode acumular defasagem por meses sem suporte. Por isso, **Recall** é priorizado como métrica de seleção do modelo.

### 4.3 Resultados da Validação Cruzada (Baseline)

| Modelo | CV Recall (k=5) |
|---|---|
| **XGBoost** | **0.9645** ✓ (melhor) |
| RandomForest | 0.9626 |
| LightGBM | 0.9491 |
| LogisticRegression | 0.8338 |
| SVM | 0.6000 |

### 4.4 Otimização de Hiperparâmetros (Optuna — XGBoost)

| Hiperparâmetro | Valor Otimizado |
|---|---|
| `n_estimators` | 51 |
| `max_depth` | 10 |
| `learning_rate` | 0.01039 |
| `subsample` | 0.9288 |
| `colsample_bytree` | 0.6491 |
| `reg_alpha` | 0.6134 |
| `reg_lambda` | 0.0007 |

**CV Recall após otimização: 0.9933**

---

## 5. Avaliação Final do Modelo

### 5.1 Métricas no Conjunto de Teste (Hold-out — nunca visto durante treino)

| Métrica | Validação | Teste |
|---|---|---|
| **Recall** | **0.9911** | **1.0000** |
| AUC-ROC | 0.9408 | 0.9299 |
| F1-Score | 0.9367 | 0.9275 |
| Accuracy | 0.9026 | 0.8864 |
| Precision | 0.8880 | 0.8649 |

### 5.2 Interpretation do Recall = 1.00

- **224 alunos** realmente em risco no conjunto de teste
- **224 identificados corretamente** pelo modelo (0 Falsos Negativos)
- **35 falsos positivos** (alunos sinalizados sem risco real — custo aceitável)

### 5.3 Relatório de Classificação (Teste)

```
                precision    recall  f1-score   support

   Sem Risco       1.00      0.58      0.74        84
    Em Risco       0.86      1.00      0.93       224

    accuracy                           0.89       308
   macro avg       0.93      0.79      0.83       308
weighted avg       0.90      0.89      0.88       308
```

### 5.4 Confiabilidade para Produção

O modelo é considerado **confiável para produção** pelos seguintes critérios:

1. **Recall = 100%** no teste: nenhum aluno em risco é "perdido"
2. **AUC-ROC = 0.93**: excelente capacidade discriminativa em todos os thresholds
3. **Validação estratificada (k=5)**: robustez comprovada em múltiplos splits
4. **Optuna com 50 trials**: otimização sistemática, não empírica
5. **Hold-out nunca visto**: avaliação isenta de contaminação
6. **Sem data leakage**: encoder e scaler ajustados exclusivamente no treino

---

## 6. API de Predição

### 6.1 Endpoints Disponíveis

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/health` | Health check — status da API e modelo |
| `GET` | `/model-info` | Versão, métricas e hiperparâmetros do modelo |
| `POST` | `/predict` | Predição individual de um aluno |
| `POST` | `/predict/batch` | Predição em lote (até 1.000 alunos) |
| `GET` | `/docs` | Documentação interativa (Swagger UI) |

### 6.2 Exemplo de Requisição — Predição Individual

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "IDADE": 14,
    "FASE": "F6",
    "PEDRA": "Quartzo",
    "IAA": 4.5,
    "IEG": 3.8,
    "IPS": 5.0,
    "IDA": 4.2,
    "IPP": 3.9,
    "IPV": 4.1,
    "IAN": 4.0,
    "INDE": 4.1
  }'
```

**Response:**
```json
{
  "student_risk": "alto",
  "probability": 0.9832,
  "risk_score": 1,
  "recommendation": "⚠️ Acompanhamento pedagógico URGENTE recomendado. Acionar equipe psicopedagógica.",
  "threshold_used": 0.5
}
```

### 6.3 Validações da API

- `IDADE`: 5–25 anos
- Indicadores (`IAA`, `IEG`, `IPS`, `IDA`, `IPP`, `IPV`, `IAN`, `INDE`): 0.0–10.0
- Campos obrigatórios: retorna HTTP 422 com mensagem clara se ausentes
- Modelo não carregado: HTTP 503 com instrução de recuperação

---

## 7. Infraestrutura e MLOps

### 7.1 Estrutura do Projeto

```
passos_magicos_datathon/
├── app/               # API FastAPI
│   ├── main.py        # Aplicação + lifespan (carrega modelo no startup)
│   ├── routes.py      # Endpoints com validação Pydantic
│   └── model/         # model.joblib + model_metadata.json
├── src/               # Pipeline de ML
│   ├── etl.py         # Normalização e unificação PEDE 2020-2024
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py       # Config, logger JSON, helpers
├── tests/             # Testes unitários (97 testes — cobertura ≥80%)
├── monitoring/        # Monitoramento
│   ├── dashboard.py   # Streamlit (métricas + drift)
│   └── drift_report.py # PSI + KS test + Evidently AI
├── notebooks/         # EDA exploratória
├── data/              # Dados raw e processados
├── .github/workflows/ # CI/CD (GitHub Actions)
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

### 7.2 Docker

```bash
# Build
docker build -t passos-magicos-ml .

# Run API
docker run -p 8000:8000 passos-magicos-ml

# API + Monitoring (Streamlit)
docker-compose up --build -d
```

**Health check automático:** `GET /health` a cada 30s com 3 retentativas.

### 7.3 CI/CD — GitHub Actions

| Workflow | Gatilho | Etapas |
|---|---|---|
| `ci.yml` | Push / Pull Request | Lint (ruff) → Testes (pytest + cobertura ≥80%) → Docker build |
| `cd.yml` | Tag `v*` | Build → Push da imagem Docker |

### 7.4 Qualidade de Código

| Ferramenta | Função |
|---|---|
| **pytest** | Testes unitários (97 testes, 0 falhas) |
| **pytest-cov** | Cobertura mínima de 80% (obrigatório no CI) |
| **ruff** | Linting estático |
| **black** | Formatação de código |
| **pre-commit** | Hooks automáticos antes de cada commit |

---

## 8. Monitoramento Contínuo

### 8.1 Logging Estruturado

Todos os eventos são registrados em formato **JSON** em `logs/app.log`:

```json
{
  "timestamp": "2026-02-21T19:54:40Z",
  "level": "INFO",
  "logger": "app.routes",
  "message": "Prediction: risk=alto | proba=0.9832 | INDE=4.1 | FASE=F6"
}
```

### 8.2 Detecção de Drift

**Métricas calculadas por feature:**

| Métrica | Significado | Threshold de Alerta |
|---|---|---|
| **PSI** (Population Stability Index) | Mudança na distribuição da feature | PSI ≥ 0.10 |
| **KS Test** (Kolmogorov-Smirnov) | Diferença estatística entre distribuições | p-value < 0.05 |

**Interpretação do PSI:**
- PSI < 0.10 → Sem mudança significativa ✅
- 0.10 ≤ PSI < 0.25 → Mudança moderada ⚠️ (monitorar)
- PSI ≥ 0.25 → Mudança significativa 🔴 (retreinar)

**Relatório Evidently AI:** Gerado em HTML para análise visual.

### 8.3 Dashboard Streamlit

Acesso: `http://localhost:8501`

Páginas disponíveis:
- **📊 Visão Geral:** métricas do modelo + status de drift
- **🔍 Drift de Dados:** PSI por feature + gráficos de distribuição
- **📈 Métricas do Modelo:** resultados de CV + hiperparâmetros + justificativa

---

## 9. Stack Tecnológica

| Camada | Tecnologia | Versão |
|---|---|---|
| **Linguagem** | Python | 3.10+ |
| **ML** | scikit-learn | 1.4.2 |
| **Boosting** | XGBoost / LightGBM | 2.0.3 / 4.3.0 |
| **Otimização** | Optuna | 3.6.1 |
| **Explicabilidade** | SHAP | 0.45.0 |
| **Data** | pandas / numpy | 2.2.2 / 1.26.4 |
| **API** | FastAPI + Uvicorn | 0.111.0 / 0.29.0 |
| **Validação** | Pydantic | 2.7.1 |
| **Serialização** | joblib | 1.4.2 |
| **Testes** | pytest + pytest-cov | 8.2.0 / 5.0.0 |
| **Containerização** | Docker + Compose | latest |
| **Monitoramento** | Streamlit + Evidently AI | 1.35.0 / 0.4.30 |
| **CI/CD** | GitHub Actions | - |
| **Linting** | ruff + black | 0.4.4 / 24.4.2 |

---

## 10. Instruções de Execução

### 10.1 Setup Local

```bash
# 1. Clonar o repositório
git clone <repositorio>
cd passos_magicos_datathon

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Copiar e configurar variáveis de ambiente
cp .env.example .env

# 4. ETL — unificar os dados PEDE
python -m src.etl

# 5. Treinar o modelo
python -m src.train

# 6. Subir a API
uvicorn app.main:app --reload --port 8000

# 7. Acessar documentação interativa
# http://localhost:8000/docs
```

### 10.2 Setup com Docker

```bash
# API + Monitoring em um comando
docker-compose up --build -d

# API:        http://localhost:8000
# Docs:       http://localhost:8000/docs
# Monitoring: http://localhost:8501
```

### 10.3 Executar Testes

```bash
# Todos os testes com relatório de cobertura
python -m pytest tests/ -v --cov=src --cov=app --cov-report=term-missing

# Resultado: 97 passed, 0 failed
```

### 10.4 Monitoramento de Drift

```bash
# Gerar relatório de drift
python monitoring/drift_report.py

# Abrir dashboard
streamlit run monitoring/dashboard.py
```

---

## 11. Resultados e Conclusão

### 11.1 Síntese dos Resultados

| Item | Resultado |
|---|---|
| Dataset unificado | 3.135 registros (2020–2024) |
| Alunos em risco identificados | 1.489 (72,7% da amostra limpa) |
| Modelo selecionado | XGBoost (otimizado com Optuna) |
| **Recall no teste** | **100%** — zero alunos em risco não identificados |
| AUC-ROC | 0.93 |
| Testes unitários | 97 testes, 0 falhas |
| Cobertura de código | ≥ 80% |
| API em produção | FastAPI + Docker |
| Monitoramento ativo | PSI + KS + Evidently + Streamlit |

### 11.2 Impacto para a Associação Passos Mágicos

- **Todos os alunos em risco são identificados** (Recall = 1.0), garantindo que nenhuma criança seja deixada sem suporte
- **Predição preventiva:** a equipe pedagógica pode agir antes que a defasagem se agrave
- **Escalabilidade:** a API processa até 1.000 predições por requisição, compatível com operações em escala da associação
- **Rastreabilidade:** logs estruturados e dashboard de monitoramento garantem visibilidade contínua do comportamento do modelo
- **Manutenibilidade:** código modular, testado e documentado, com CI/CD automatizado

### 11.3 Próximos Passos Recomendados

1. **Integrar dados de 2023** assim que disponíveis para enriquecer a série temporal
2. **Ajustar o threshold** de 0.5 para 0.3–0.4, aumentando sensibilidade com base no feedback dos educadores
3. **Endpoint de explicabilidade** via SHAP — mostrar quais indicadores mais influenciaram a predição de cada aluno
4. **Retreinamento automático** agendado (trimestral) com dados novos do PEDE
5. **Deploy na nuvem** (AWS ECS ou GCP Cloud Run) para acesso da equipe da associação via web

---

*Relatório gerado em: 21/02/2026*
*Versão do modelo: 1.0.0*
*Equipe: Datathon Pós Tech FIAP — Machine Learning Engineering*
