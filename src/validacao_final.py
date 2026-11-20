
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.train import FEATURES


# ==========================================
# CONFIGURAÇÕES
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "entregas.csv"
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
METRICS_PATH = ROOT / "outputs" / "metricas_finais.json"

RANDOM_STATE = 42
THRESHOLDS = [0.30, 0.40, 0.50]


# ==========================================
# VALIDAÇÃO DOS DADOS
# ==========================================

def validar_dados(df):

    colunas_obrigatorias = list(FEATURES) + ["atrasou"]

    faltantes = [
        coluna
        for coluna in colunas_obrigatorias
        if coluna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {faltantes}"
        )

    if df.empty:
        raise ValueError(
            "O dataset está vazio."
        )

    if df[colunas_obrigatorias].isna().any().any():
        raise ValueError(
            "O dataset contém valores ausentes."
        )

    if df.duplicated().any():
        raise ValueError(
            "O dataset contém linhas duplicadas."
        )

    if not df["atrasou"].isin([0, 1]).all():
        raise ValueError(
            "A coluna 'atrasou' deve conter apenas 0 e 1."
        )

    if df["atrasou"].nunique() != 2:
        raise ValueError(
            "O dataset precisa conter as duas classes."
        )

    for coluna in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[coluna]):
            raise ValueError(
                f"A variável '{coluna}' deve ser numérica."
            )

    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "As variáveis contêm valores infinitos."
        )

    return df


# ==========================================
# CÁLCULO DE MÉTRICAS
# ==========================================

def calcular_metricas(
    y_real,
    probabilidades,
    threshold
):

    y_real = np.asarray(y_real)
    probabilidades = np.asarray(probabilidades)

    if len(y_real) != len(probabilidades):
        raise ValueError(
            "Rótulos e probabilidades possuem tamanhos diferentes."
        )

    if len(y_real) == 0:
        raise ValueError(
            "Não existem registros para avaliação."
        )

    if not np.isfinite(probabilidades).all():
        raise ValueError(
            "Foram encontradas probabilidades inválidas."
        )

    if not np.all(
        (probabilidades >= 0) &
        (probabilidades <= 1)
    ):
        raise ValueError(
            "As probabilidades devem estar entre 0 e 1."
        )

    previsoes = (
        probabilidades >= threshold
    ).astype(int)

    roc_auc = None

    if len(np.unique(y_real)) == 2:
        roc_auc = round(
            float(
                roc_auc_score(
                    y_real,
                    probabilidades
                )
            ),
            4
        )

    return {
        "accuracy": round(
            accuracy_score(y_real, previsoes), 4
        ),
        "precision": round(
            precision_score(
                y_real,
                previsoes,
                zero_division=0
            ),
            4
        ),
        "recall": round(
            recall_score(
                y_real,
                previsoes,
                zero_division=0
            ),
            4
        ),
        "f1": round(
            f1_score(
                y_real,
                previsoes,
                zero_division=0
            ),
            4
        ),
        "roc_auc": roc_auc,
        "matriz_confusao": confusion_matrix(
            y_real,
            previsoes,
            labels=[0, 1]
        ).tolist()
    }


# ==========================================
# VALIDAÇÃO FINAL
# ==========================================

def executar():

    print("\n=== AI DECISION ENGINE ===")
    print("Iniciando validação final...")

    # 1. Carregar e validar dataset
    df = pd.read_csv(DATA_PATH)

    validar_dados(df)

    X = df[FEATURES]
    y = df["atrasou"]

    # 2. Divisão 60% / 20% / 20%
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print("\nDivisão dos dados:")
    print(f"Treino: {len(X_train)}")
    print(f"Validação: {len(X_val)}")
    print(f"Teste: {len(X_test)}")

    # 3. Treinar Random Forest
    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    # 4. Avaliar thresholds na validação
    probabilidades_val = modelo.predict_proba(
        X_val
    )[:, 1]

    resultados_validacao = {
        str(threshold): calcular_metricas(
            y_val,
            probabilidades_val,
            threshold
        )
        for threshold in THRESHOLDS
    }

    # 5. Escolher threshold pelo F1
    melhor_threshold = max(
        THRESHOLDS,
        key=lambda threshold: (
            resultados_validacao[
                str(threshold)
            ]["f1"],
            threshold
        )
    )

    # 6. Avaliação final no teste
    probabilidades_test = modelo.predict_proba(
        X_test
    )[:, 1]

    metricas_finais = calcular_metricas(
        y_test,
        probabilidades_test,
        melhor_threshold
    )

    # 7. Baseline
    baseline = DummyClassifier(
        strategy="prior"
    )

    baseline.fit(X_train, y_train)

    probabilidades_baseline = (
        baseline.predict_proba(X_test)[:, 1]
    )

    metricas_baseline = calcular_metricas(
        y_test,
        probabilidades_baseline,
        0.50
    )

    # 8. Consolidar resultados
    resultado = {
        "divisao": {
            "treino": len(X_train),
            "validacao": len(X_val),
            "teste": len(X_test)
        },
        "thresholds_validacao": resultados_validacao,
        "threshold_escolhido": melhor_threshold,
        "baseline_teste": metricas_baseline,
        "random_forest_teste": metricas_finais
    }

    # 9. Salvar artefatos
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        modelo,
        MODEL_PATH
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            resultado,
            arquivo,
            indent=4
        )

    # 10. Exibir resultados
    print("\nF1 POR THRESHOLD NA VALIDAÇÃO:")

    for threshold in THRESHOLDS:
        f1 = resultados_validacao[
            str(threshold)
        ]["f1"]

        print(
            f"{threshold:.2f}: {f1:.4f}"
        )

    print(
        "\nTHRESHOLD ESCOLHIDO:",
        melhor_threshold
    )

    print("\nBASELINE NO TESTE:")
    print(metricas_baseline)

    print("\nRANDOM FOREST NO TESTE:")
    print(metricas_finais)

    print(
        "\nModelo e métricas salvos com sucesso!"
    )

    return resultado


if __name__ == "__main__":
    executar()
