
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.train import FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "entregas.csv"
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
METRICS_PATH = ROOT / "outputs" / "metricas_finais.json"


def calcular_metricas(y_real, probabilidades, threshold):
    previsoes = (probabilidades >= threshold).astype(int)

    return {
        "accuracy": round(accuracy_score(y_real, previsoes), 4),
        "precision": round(
            precision_score(y_real, previsoes, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_real, previsoes, zero_division=0), 4
        ),
        "f1": round(
            f1_score(y_real, previsoes, zero_division=0), 4
        ),
        "roc_auc": round(
            roc_auc_score(y_real, probabilidades), 4
        ),
        "matriz_confusao": confusion_matrix(
            y_real, previsoes
        ).tolist(),
    }


def executar():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["atrasou"]

    # 60% treino, 20% validação, 20% teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp,
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    modelo.fit(X_train, y_train)

    probabilidades_val = modelo.predict_proba(X_val)[:, 1]

    thresholds = [0.30, 0.40, 0.50]

    resultados_validacao = {
        str(t): calcular_metricas(
            y_val, probabilidades_val, t
        )
        for t in thresholds
    }

    melhor_threshold = max(
        thresholds,
        key=lambda t: (
            resultados_validacao[str(t)]["f1"],
            t,
        ),
    )

    # Avaliação final: o teste não participa da escolha
    probabilidades_test = modelo.predict_proba(X_test)[:, 1]

    metricas_finais = calcular_metricas(
        y_test,
        probabilidades_test,
        melhor_threshold,
    )

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)

    metricas_baseline = calcular_metricas(
        y_test,
        baseline.predict_proba(X_test)[:, 1],
        0.50,
    )

    resultado = {
        "divisao": {
            "treino": len(X_train),
            "validacao": len(X_val),
            "teste": len(X_test),
        },
        "thresholds_validacao": resultados_validacao,
        "threshold_escolhido": melhor_threshold,
        "baseline_teste": metricas_baseline,
        "random_forest_teste": metricas_finais,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(modelo, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, indent=4)

    print("\n=== VALIDACAO FINAL ===")
    print("Treino:", len(X_train))
    print("Validacao:", len(X_val))
    print("Teste:", len(X_test))

    print("\nF1 POR THRESHOLD NA VALIDACAO:")
    for threshold in thresholds:
        f1 = resultados_validacao[str(threshold)]["f1"]
        print(f"{threshold:.2f}: {f1:.4f}")

    print("\nTHRESHOLD ESCOLHIDO:", melhor_threshold)

    print("\nBASELINE NO TESTE:")
    print(metricas_baseline)

    print("\nRANDOM FOREST NO TESTE:")
    print(metricas_finais)

    print("\nModelo e metricas salvos com sucesso!")


if __name__ == "__main__":
    executar()
