
import json
import math

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.decision_engine import avaliar_entrega
from src.train import FEATURES


# ==========================================
# CAMINHOS E CONFIGURAÇÕES
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "random_forest.joblib"
METRICS_PATH = ROOT / "outputs" / "metricas_finais.json"

LIMITE_RISCO_ALTO = 0.70

NIVEIS_RISCO = {
    "BAIXO": 0,
    "MEDIO": 1,
    "ALTO": 2
}

ACOES = {
    "BAIXO": "Fluxo normal",
    "MEDIO": "Monitorar entrega",
    "ALTO": "Priorizar acompanhamento"
}


# ==========================================
# CARREGAMENTO DO THRESHOLD
# ==========================================

@lru_cache(maxsize=1)
def carregar_threshold():

    with open(METRICS_PATH, "r", encoding="utf-8") as arquivo:
        metricas = json.load(arquivo)

    try:
        threshold = float(metricas["threshold_escolhido"])
    except (KeyError, TypeError, ValueError) as erro:
        raise ValueError(
            "Threshold ausente ou inválido no arquivo de métricas."
        ) from erro

    if not math.isfinite(threshold):
        raise ValueError(
            "O threshold deve ser um número finito."
        )

    if not 0 <= threshold <= LIMITE_RISCO_ALTO:
        raise ValueError(
            "O threshold deve estar entre 0 e 0.70."
        )

    return threshold


# ==========================================
# CARREGAMENTO DO MODELO
# ==========================================

@lru_cache(maxsize=1)
def carregar_modelo():
    return joblib.load(MODEL_PATH)


# ==========================================
# CLASSIFICAÇÃO POR MACHINE LEARNING
# ==========================================

def classificar_risco_ml(probabilidade, threshold):

    if not math.isfinite(probabilidade):
        raise ValueError(
            "A probabilidade prevista pelo modelo é inválida."
        )

    if not 0 <= probabilidade <= 1:
        raise ValueError(
            "A probabilidade deve estar entre 0 e 1."
        )

    if probabilidade >= LIMITE_RISCO_ALTO:
        return "ALTO"

    if probabilidade >= threshold:
        return "MEDIO"

    return "BAIXO"


# ==========================================
# MOTOR HÍBRIDO DE DECISÃO
# ==========================================

def decidir_entrega(entrega):

    # 1. Carregar modelo e threshold
    modelo = carregar_modelo()
    threshold = carregar_threshold()

    # 2. Preparar dados para o modelo
    dados = pd.DataFrame([entrega])
    dados = dados[FEATURES]

    # 3. Calcular probabilidade de atraso
    probabilidades = modelo.predict_proba(dados)

    probabilidade = float(
        probabilidades[0][1]
    )

    # 4. Classificar risco pelo modelo
    risco_ml = classificar_risco_ml(
        probabilidade,
        threshold
    )

    # 5. Avaliar regras de negócio
    resultado_regras = avaliar_entrega(entrega)

    risco_regras = resultado_regras["risco"]

    # 6. Combinar ML + regras
    # Prevalece o maior nível de risco
    risco_final = max(
        [risco_ml, risco_regras],
        key=lambda risco: NIVEIS_RISCO[risco]
    )

    # 7. Regra adicional para carga prioritária
    if entrega["carga_prioritaria"] == 1:
        if risco_final == "MEDIO":
            risco_final = "ALTO"

    # 8. Selecionar ação recomendada
    acao_recomendada = ACOES[risco_final]

    # 9. Retornar decisão
    return {
        "probabilidade_atraso": round(
            probabilidade, 4
        ),
        "risco_ml": risco_ml,
        "risco_regras": risco_regras,
        "risco_final": risco_final,
        "acao_recomendada": acao_recomendada,
        "motivos": resultado_regras["motivos"]
    }


# ==========================================
# EXECUÇÃO MANUAL
# ==========================================

if __name__ == "__main__":

    exemplo = {
        "distancia_km": 350,
        "dias_ate_prazo": 1,
        "atrasos_anteriores": 4,
        "clima_adverso": 1,
        "carga_prioritaria": 1
    }

    resultado = decidir_entrega(exemplo)

    print("\n=== DECISAO INTELIGENTE ===")

    for chave, valor in resultado.items():
        print(f"{chave}: {valor}")
