
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.train import FEATURES


ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "entregas.csv")

X = df[FEATURES]
y = df["atrasou"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

modelo = joblib.load(
    ROOT / "models" / "random_forest.joblib"
)

probabilidades = modelo.predict_proba(X_test)[:, 1]

for threshold in [0.3, 0.4, 0.5]:
    previsoes = (probabilidades >= threshold).astype(int)

    print(f"\n=== THRESHOLD {threshold} ===")

    print(
        "Precision:",
        round(precision_score(
            y_test, previsoes, zero_division=0
        ), 4)
    )

    print(
        "Recall:",
        round(recall_score(
            y_test, previsoes, zero_division=0
        ), 4)
    )

    print(
        "F1:",
        round(f1_score(
            y_test, previsoes, zero_division=0
        ), 4)
    )

    print("Matriz de confusão:")
    print(confusion_matrix(y_test, previsoes))
