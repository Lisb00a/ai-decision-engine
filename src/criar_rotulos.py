
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# CONFIGURAÇÕES
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "entregas.csv"

COLUNAS_OBRIGATORIAS = [
    "distancia_km",
    "dias_ate_prazo",
    "atrasos_anteriores",
    "clima_adverso",
    "carga_prioritaria"
]


# ==========================================
# VALIDAÇÃO DOS DADOS
# ==========================================

def validar_dados(df):

    if df.empty:
        raise ValueError(
            "O dataset está vazio."
        )

    faltantes = [
        coluna
        for coluna in COLUNAS_OBRIGATORIAS
        if coluna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {faltantes}"
        )

    dados = df[COLUNAS_OBRIGATORIAS]

    if dados.isna().any().any():
        raise ValueError(
            "Existem valores ausentes nas variáveis."
        )

    for coluna in COLUNAS_OBRIGATORIAS:
        if not pd.api.types.is_numeric_dtype(df[coluna]):
            raise ValueError(
                f"A coluna '{coluna}' deve ser numérica."
            )

    if not np.isfinite(
        dados.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Existem valores infinitos no dataset."
        )

    for coluna in [
        "distancia_km",
        "dias_ate_prazo",
        "atrasos_anteriores"
    ]:
        if (df[coluna] < 0).any():
            raise ValueError(
                f"A coluna '{coluna}' contém valores negativos."
            )

    for coluna in [
        "clima_adverso",
        "carga_prioritaria"
    ]:
        if not df[coluna].isin([0, 1]).all():
            raise ValueError(
                f"A coluna '{coluna}' deve conter apenas 0 e 1."
            )

    return df


# ==========================================
# CRIAÇÃO DOS RÓTULOS SINTÉTICOS
# ==========================================

def criar_rotulos(seed=42, sobrescrever=False):

    df = pd.read_csv(DATA_PATH)

    validar_dados(df)

    if "atrasou" in df.columns and not sobrescrever:
        raise ValueError(
            "A coluna 'atrasou' já existe. "
            "Para recriar os rótulos, utilize "
            "sobrescrever=True."
        )

    rng = np.random.default_rng(seed)

    # 1. Calcular score sintético
    score = (
        -2.5
        + 0.005 * df["distancia_km"]
        - 0.25 * df["dias_ate_prazo"]
        + 0.35 * df["atrasos_anteriores"]
        + 0.8 * df["clima_adverso"]
        + 0.3 * df["carga_prioritaria"]
    )

    # 2. Converter score em probabilidade
    probabilidade = 1 / (
        1 + np.exp(-score)
    )

    # 3. Simular o resultado da entrega
    df["atrasou"] = rng.binomial(
        n=1,
        p=probabilidade
    )

    # 4. Salvar dataset atualizado
    df.to_csv(
        DATA_PATH,
        index=False,
        encoding="utf-8"
    )

    # 5. Exibir resultados
    print("\n=== ROTULOS SINTETICOS CRIADOS ===")

    print(
        f"Total de entregas: {len(df)}"
    )

    print("\nDistribuição dos resultados:")
    print(
        df["atrasou"].value_counts()
    )

    print("\nPercentual de atrasos:")
    print(
        df["atrasou"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    return df


if __name__ == "__main__":
    criar_rotulos()
