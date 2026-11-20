
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "entregas.csv"


def criar_rotulos(seed=42):
    df = pd.read_csv(DATA_PATH)

    rng = np.random.default_rng(seed)

    # Pontuação simulada de risco
    score = (
        -2.5
        + 0.005 * df["distancia_km"]
        - 0.25 * df["dias_ate_prazo"]
        + 0.35 * df["atrasos_anteriores"]
        + 0.8 * df["clima_adverso"]
        + 0.3 * df["carga_prioritaria"]
    )

    # Converter score em probabilidade
    probabilidade = 1 / (1 + np.exp(-score))

    # Simular se cada entrega atrasou
    df["atrasou"] = rng.binomial(
        1,
        probabilidade
    )

    df.to_csv(
        DATA_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n=== ROTULOS CRIADOS ===")
    print(f"Total de entregas: {len(df)}")

    print("\nDistribuição dos resultados:")
    print(df["atrasou"].value_counts())

    print("\nPercentual de atrasos:")
    print(
        df["atrasou"].value_counts(
            normalize=True
        ).mul(100).round(2)
    )

    return df


if __name__ == "__main__":
    criar_rotulos()
