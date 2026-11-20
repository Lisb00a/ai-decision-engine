
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "entregas.csv"


def gerar_dados(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "id_entrega": range(1, n + 1),

        "distancia_km": rng.integers(
            5, 501, size=n
        ),

        "dias_ate_prazo": rng.integers(
            0, 11, size=n
        ),

        "atrasos_anteriores": rng.integers(
            0, 6, size=n
        ),

        "clima_adverso": rng.integers(
            0, 2, size=n
        ),

        "carga_prioritaria": rng.integers(
            0, 2, size=n
        )
    })

    DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        DATA_PATH,
        index=False,
        encoding="utf-8"
    )

    print("Base de dados criada com sucesso!")
    print(f"Total de entregas: {len(df)}")

    print("\nPrimeiras cinco entregas:")
    print(df.head())

    print("\nResumo estatístico:")
    print(df.describe())

    return df


if __name__ == "__main__":
    gerar_dados()
