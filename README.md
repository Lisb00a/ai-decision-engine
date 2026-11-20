# AI Decision Engine

Sistema de apoio à decisão para identificação de riscos de atraso em entregas, utilizando Machine Learning, regras de negócio e API REST.

## Sobre o projeto

O AI Decision Engine foi desenvolvido como projeto de portfólio de Engenharia de IA.

O sistema analisa características de entregas, estima o risco de atraso com um modelo Random Forest e combina essa estimativa com regras operacionais para recomendar ações.

**Todos os dados utilizados são sintéticos.** O projeto é demonstrativo e não representa um sistema validado em operações logísticas reais.

## Tecnologias

- Python
- Pandas e NumPy
- Scikit-learn
- FastAPI e Pydantic
- Pytest
- Joblib

## Funcionalidades

- Geração de 1.000 entregas sintéticas
- Criação de resultados simulados de atraso
- Classificação de risco por regras de negócio
- Treinamento de Random Forest e comparação com DummyClassifier
- Avaliação com Accuracy, Precision, Recall, F1-Score e ROC-AUC
- Motor híbrido que combina Machine Learning e regras
- API REST com validação de entradas
- Testes automatizados

## Resultados experimentais

|## Resultados experimentais

O modelo foi avaliado utilizando 1.000 entregas sintéticas, divididas em:

- **60% para treinamento:** 600 registros
- **20% para validação:** 200 registros
- **20% para teste final:** 200 registros

O threshold de classificação foi selecionado na base de validação, utilizando o F1-Score como critério.

**Threshold selecionado: 0.30**

### Métricas no conjunto de teste

| Métrica | DummyClassifier | Random Forest |
|---|---:|---:|
| Accuracy | 0.6900 | 0.7000 |
| Precision | 0.0000 | 0.5114 |
| Recall | 0.0000 | 0.7258 |
| F1-Score | 0.0000 | 0.6000 |
| ROC-AUC | 0.5000 | 0.8008 |

### Matriz de confusão do Random Forest

| Resultado real | Previsto sem atraso | Previsto com atraso |
|---|---:|---:|
| Sem atraso | 95 | 43 |
| Com atraso | 17 | 45 |

O modelo identificou 45 dos 62 atrasos simulados no conjunto de teste, alcançando um Recall de 72,58%.

Os resultados demonstram o funcionamento do pipeline de Machine Learning, mas não representam desempenho validado em entregas reais.

### Reproduzir a avaliação

```powershell
python -m src.gerar_dados
python -m src.criar_rotulos
python -m src.validacao_final
python -m pytest -v
```

O script de validação salva o modelo em `models/random_forest.joblib` e os resultados em `outputs/metricas_finais.json`.

O motor híbrido utiliza o threshold selecionado na validação para sinalizar risco a partir da probabilidade estimada pelo modelo. A classificação de risco alto mantém um limite operacional demonstrativo de 0.70, combinado às regras de negócio.

**Limitação:** os dados e os resultados de atraso são sintéticos. A política híbrida ainda não foi avaliada separadamente em relação ao classificador isolado.
## Como executar

Criar o ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Gerar dados e treinar o modelo:

```powershell
python -m src.gerar_dados
python -m src.criar_rotulos
python -m src.train
```

Executar o motor híbrido:

```powershell
python -m src.ml_decision_engine
```

Iniciar a API:

```powershell
python -m uvicorn src.api:app --reload
```

Documentação interativa: `http://127.0.0.1:8000/docs`

### Endpoints

- `GET /health` — verifica o funcionamento do serviço
- `POST /decidir` — recebe os dados de uma entrega e retorna a classificação de risco e a ação recomendada

Exemplo de requisição:

```json
{
  "distancia_km": 350,
  "dias_ate_prazo": 1,
  "atrasos_anteriores": 4,
  "clima_adverso": 1,
  "carga_prioritaria": 1
}
```

## Testes

```powershell
python -m pytest -v
```

## Limitações e melhorias futuras

- Os dados e os atrasos foram gerados artificialmente.
- As probabilidades não foram calibradas para uma operação real.
- Os thresholds precisam ser selecionados com uma base de validação separada.
- O motor híbrido precisa ser avaliado em comparação com os modelos individuais.
- Uma aplicação real exigiria dados históricos, monitoramento e validação temporal.

## Objetivo

Demonstrar competências em Python, Machine Learning, desenvolvimento de APIs, testes automatizados e integração entre modelos preditivos e regras de negócio.