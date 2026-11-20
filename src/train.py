
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
    classification_report
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "entregas.csv"
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
METRICS_PATH = ROOT / "outputs" / "metricas.json"

FEATURES = [
    "distancia_km",
    "dias_ate_prazo",
    "atrasos_anteriores",
    "clima_adverso",
    "carga_prioritaria"
]


def avaliar_modelo(modelo, X_test, y_test):
    previsoes = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(
            accuracy_score(y_test, previsoes), 4
        ),
        "precision": round(
            precision_score(y_test, previsoes, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_test, previsoes, zero_division=0), 4
        ),
        "f1": round(
            f1_score(y_test, previsoes, zero_division=0), 4
        ),
        "roc_auc": round(
            roc_auc_score(y_test, probabilidades), 4
        )
    }


def treinar():
    df = pd.read_csv(DATA_PATH)

    if "atrasou" not in df.columns:
        raise ValueError(
            "Coluna 'atrasou' ausente. "
            "Execute python -m src.criar_rotulos"
        )

    X = df[FEATURES]
    y = df["atrasou"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    baseline = DummyClassifier(
        strategy="prior"
    )

    baseline.fit(X_train, y_train)

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    metricas = {
        "baseline": avaliar_modelo(
            baseline, X_test, y_test
        ),
        "random_forest": avaliar_modelo(
            modelo, X_test, y_test
        )
    }

    print("\n=== AI DECISION ENGINE - ML ===")

    print("\nBASELINE:")
    print(metricas["baseline"])

    print("\nRANDOM FOREST:")
    print(metricas["random_forest"])

    print("\nRELATORIO DE CLASSIFICACAO:")
    print(
        classification_report(
            y_test,
            modelo.predict(X_test),
            zero_division=0
        )
    )

    print("\nIMPORTANCIA DAS VARIAVEIS:")
    importancias = pd.Series(
        modelo.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)

    print(importancias)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(modelo, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as arquivo:
        json.dump(
            metricas,
            arquivo,
            indent=4
        )

    print("\nModelo salvo com sucesso!")
    print(f"Arquivo: {MODEL_PATH}")

    return modelo, metricas


if __name__ == "__main__":
    treinar()
