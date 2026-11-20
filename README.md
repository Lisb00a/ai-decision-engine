# AI Decision Engine

**Sistema Inteligente de Análise de Risco Logístico | Machine Learning + Business Rules + REST API**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Tests](https://img.shields.io/badge/Tests-37%20Passed-brightgreen)
![CI](https://github.com/Lisb00a/ai-decision-engine/actions/workflows/tests.yml/badge.svg)

## Sobre o projeto

O **AI Decision Engine** é um sistema de apoio à decisão desenvolvido para identificar riscos de atraso em entregas logísticas.

O projeto combina um modelo de Machine Learning baseado em **Random Forest Classifier** com regras operacionais para classificar o risco de uma entrega e recomendar ações preventivas.

O objetivo não é apenas gerar uma previsão, mas transformar a estimativa do modelo em uma decisão interpretável, disponibilizada por uma API REST.

**Este projeto utiliza exclusivamente dados sintéticos e possui finalidade educacional e demonstrativa.** Não representa um sistema validado em operações logísticas reais.

## Arquitetura da solução

O sistema utiliza uma arquitetura híbrida de decisão:

```text
                DADOS DA ENTREGA
                       |
                       v
              VALIDAÇÃO PYDANTIC
                       |
              +--------+--------+
              |                 |
              v                 v
        RANDOM FOREST      REGRAS DE NEGÓCIO
              |                 |
              v                 v
      Probabilidade de     Pontuação de risco
           atraso                |
              |                 |
              v                 v
      Classificação ML     Classificação regras
              |                 |
              +--------+--------+
                       |
                       v
              MOTOR HÍBRIDO
                       |
             Maior nível de risco
                       |
             Regra de prioridade
                       |
                       v
              DECISÃO FINAL
                       |
                       v
                 FASTAPI
                       |
                       v
          RISCO + AÇÃO RECOMENDADA
```

A classificação final considera o maior risco identificado pelo modelo ou pelas regras operacionais.

Quando uma carga é prioritária e o risco combinado é médio, o sistema eleva a classificação para alto.

## Tecnologias utilizadas

- **Python 3.13:** desenvolvimento da aplicação.
- **Pandas e NumPy:** processamento e validação dos dados.
- **Scikit-learn:** treinamento e avaliação do Random Forest.
- **FastAPI:** disponibilização do motor de decisão por API REST.
- **Pydantic:** validação dos dados de entrada.
- **Joblib:** persistência do modelo treinado.
- **Pytest:** testes automatizados.
- **GitHub Actions:** execução automática dos testes em CI.

## Funcionalidades

- Geração de 1.000 entregas sintéticas.
- Simulação de resultados de atraso.
- Treinamento de um classificador Random Forest.
- Comparação com um baseline DummyClassifier.
- Seleção de threshold por F1-score na validação.
- Classificação de risco por regras operacionais.
- Integração entre Machine Learning e regras de negócio.
- Recomendações automáticas para acompanhamento das entregas.
- API REST com validação de entradas e tratamento de erros.
- Persistência do modelo e das métricas.
- Testes automatizados e integração contínua.

## Dados e variáveis

O dataset sintético contém características das entregas utilizadas pelo modelo.

| Variável | Descrição |
|---|---|
| `distancia_km` | Distância da entrega em quilômetros |
| `dias_ate_prazo` | Dias restantes até o prazo |
| `atrasos_anteriores` | Histórico de atrasos |
| `clima_adverso` | Indicador de condição climática adversa |
| `carga_prioritaria` | Indicador de prioridade da carga |
| `atrasou` | Resultado sintético utilizado como variável-alvo |

A variável `atrasou` é gerada por um processo probabilístico simulado, utilizando as características da entrega.

Portanto, as métricas representam a capacidade do modelo de aprender padrões desse processo artificial, não a capacidade comprovada de prever atrasos reais.

## Metodologia de Machine Learning

O conjunto de 1.000 registros foi dividido de forma estratificada:

| Conjunto | Registros | Proporção |
|---|---:|---:|
| Treinamento | 600 | 60% |
| Validação | 200 | 20% |
| Teste | 200 | 20% |

O Random Forest utiliza os seguintes hiperparâmetros:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
```

O threshold foi selecionado exclusivamente no conjunto de validação, utilizando o F1-score como critério.

### Seleção do threshold

| Threshold | F1-score na validação |
|---|---:|
| **0.30** | **0.5882** |
| 0.40 | 0.4793 |
| 0.50 | 0.4615 |

**Threshold escolhido: 0.30**

O conjunto de teste foi reservado para a avaliação final e não participou da seleção do threshold.

## Resultados experimentais

### Comparação com o baseline

| Métrica | DummyClassifier | Random Forest |
|---|---:|---:|
| Accuracy | 0.6900 | **0.7000** |
| Precision | 0.0000 | **0.5114** |
| Recall | 0.0000 | **0.7258** |
| F1-score | 0.0000 | **0.6000** |
| ROC-AUC | 0.5000 | **0.8008** |

O Random Forest apresentou ROC-AUC de **0.8008**, enquanto o baseline alcançou 0.5000.

O Recall de 72,58% indica que o modelo identificou aproximadamente 73% dos atrasos simulados no conjunto de teste.

### Matriz de confusão

| Resultado real | Previsto sem atraso | Previsto com atraso |
|---|---:|---:|
| Sem atraso | 95 | 43 |
| Com atraso | 17 | 45 |

No conjunto de teste:

- 95 entregas sem atraso foram classificadas corretamente.
- 45 atrasos foram identificados corretamente.
- 43 entregas sem atraso foram sinalizadas como atrasadas.
- 17 atrasos não foram identificados.

Os resultados evidenciam o funcionamento do pipeline preditivo, mas também mostram a necessidade de considerar os custos operacionais dos falsos positivos e falsos negativos.

## Motor híbrido de decisão

O sistema combina duas classificações independentes.

### 1. Classificação por Machine Learning

| Probabilidade estimada | Risco |
|---|---|
| Abaixo de 0.30 | BAIXO |
| De 0.30 até abaixo de 0.70 | MEDIO |
| A partir de 0.70 | ALTO |

O limite de 0.70 para risco alto é uma regra operacional demonstrativa.

### 2. Classificação por regras de negócio

| Condição | Pontuação |
|---|---:|
| Prazo de até 2 dias | +3 |
| Três ou mais atrasos anteriores | +2 |
| Clima adverso | +2 |
| Distância superior a 300 km | +1 |

A pontuação determina o risco:

| Pontuação | Classificação |
|---|---|
| 0 a 2 | BAIXO |
| 3 a 4 | MEDIO |
| 5 ou mais | ALTO |

### 3. Decisão final

O motor híbrido seleciona o maior nível de risco entre as duas classificações.

Uma regra adicional eleva o risco de MEDIO para ALTO quando a carga é prioritária.

| Risco final | Ação recomendada |
|---|---|
| BAIXO | Fluxo normal |
| MEDIO | Monitorar entrega |
| ALTO | Priorizar acompanhamento |

Essa estratégia é demonstrativa e ainda precisa ser avaliada separadamente para verificar seu impacto sobre o desempenho operacional.

## Como executar

### 1. Clonar o repositório

```powershell
git clone https://github.com/Lisb00a/ai-decision-engine.git
cd ai-decision-engine
```

### 2. Criar e ativar o ambiente virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```powershell
python -m pip install -r requirements.txt
```

### 4. Reproduzir a avaliação

O repositório contém o dataset sintético e os artefatos do modelo.

Para executar novamente o treinamento e a avaliação final:

```powershell
python -m src.validacao_final
```

O processo atualiza:

- `models/random_forest.joblib`
- `outputs/metricas_finais.json`

### 5. Executar o motor híbrido

```powershell
python -m src.ml_decision_engine
```

### 6. Iniciar a API

```powershell
python -m uvicorn src.api:app --reload
```

A documentação interativa estará disponível em:

http://127.0.0.1:8000/docs

## API REST

### GET `/health`

Verifica o funcionamento do serviço.

Exemplo de resposta:

```json
{
  "status": "ok",
  "service": "AI Decision Engine"
}
```

### POST `/decidir`

Recebe as características de uma entrega e retorna a classificação de risco.

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

A resposta contém os campos:

```text
probabilidade_atraso
risco_ml
risco_regras
risco_final
acao_recomendada
motivos
```

Para esse exemplo, as regras de negócio classificam a entrega como risco ALTO, recomendando priorizar o acompanhamento.

A API também possui validação de entradas e tratamento de falhas, incluindo respostas HTTP 422, 500 e 503 conforme a situação.

## Testes automatizados

O projeto possui **37 testes automatizados**, distribuídos entre:

| Arquivo | Testes |
|---|---:|
| `tests/test_api.py` | 13 |
| `tests/test_decision_engine.py` | 5 |
| `tests/test_ml_decision_engine.py` | 19 |
| **Total** | **37** |

Os testes verificam:

- Funcionamento da API.
- Validação de campos obrigatórios e valores inválidos.
- Tratamento de erros do serviço.
- Classificação pelas regras de negócio.
- Classificação do modelo de Machine Learning.
- Integração entre ML e regras.
- Priorização de cargas.
- Validação de probabilidades.
- Leitura do threshold configurado.

Para executar:

```powershell
python -m pytest -v
```

### Integração contínua

O projeto utiliza GitHub Actions para executar os testes automaticamente em eventos de push e pull request na branch `main`.

Workflow:

`.github/workflows/tests.yml`

[Consultar execuções do GitHub Actions](https://github.com/Lisb00a/ai-decision-engine/actions)

## Limitações

Este projeto foi desenvolvido como prova de conceito e possui limitações importantes:

- Os dados e os resultados de atraso são sintéticos.
- A avaliação utiliza divisão aleatória estratificada, não validação temporal.
- O processo de geração dos rótulos utiliza as mesmas características empregadas no treinamento.
- As probabilidades não foram calibradas para uma operação real.
- O threshold de risco alto foi definido como regra demonstrativa.
- A política híbrida ainda não possui uma avaliação comparativa própria.
- O sistema não foi validado com entregas reais ou integrado a uma operação logística.

## Melhorias futuras

- Incorporar dados históricos reais de entregas.
- Implementar validação temporal.
- Avaliar calibração das probabilidades.
- Comparar o desempenho do motor híbrido com o modelo isolado.
- Implementar monitoramento de desempenho e data drift.
- Adicionar testes específicos para o pipeline de geração de dados.
- Avaliar custos de falsos positivos e falsos negativos.

## Objetivo do portfólio

Demonstrar competências práticas em:

**Python · Machine Learning · Scikit-learn · Feature Engineering · Model Evaluation · Business Rules · FastAPI · REST APIs · Pytest · GitHub Actions · CI/CD**

O AI Decision Engine representa a aplicação de um modelo preditivo em um fluxo de decisão, conectando estimativas estatísticas, regras operacionais e uma interface de serviço.

---

**Projeto desenvolvido para fins educacionais e de portfólio em Engenharia de Inteligência Artificial.**