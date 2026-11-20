
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.decision_engine import avaliar_entrega
from src.train import FEATURES


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest.joblib"


@lru_cache(maxsize=1)
def carregar_modelo():
    return joblib.load(MODEL_PATH)


def decidir_entrega(entrega):
    modelo = carregar_modelo()

    dados = pd.DataFrame([entrega])
    dados = dados[FEATURES]

    probabilidade = float(
        modelo.predict_proba(dados)[0][1]
    )

    # Classificação por Machine Learning
    if probabilidade >= 0.70:
        risco_ml = "ALTO"
    elif probabilidade >= 0.40:
        risco_ml = "MEDIO"
    else:
        risco_ml = "BAIXO"

    # Classificação pelas regras de negócio
    resultado_regras = avaliar_entrega(entrega)

    niveis = {
        "BAIXO": 0,
        "MEDIO": 1,
        "ALTO": 2
    }

    # Motor híbrido: prevalece o maior risco
    risco_final = max(
        [risco_ml, resultado_regras["risco"]],
        key=lambda risco: niveis[risco]
    )

    # Regra adicional para cargas prioritárias
    if entrega["carga_prioritaria"] == 1:
        if risco_final == "MEDIO":
            risco_final = "ALTO"

    acoes = {
        "BAIXO": "Fluxo normal",
        "MEDIO": "Monitorar entrega",
        "ALTO": "Priorizar acompanhamento"
    }

    return {
        "probabilidade_atraso": round(
            probabilidade, 4
        ),
        "risco_ml": risco_ml,
        "risco_regras": resultado_regras["risco"],
        "risco_final": risco_final,
        "acao_recomendada": acoes[risco_final],
        "motivos": resultado_regras["motivos"]
    }


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
