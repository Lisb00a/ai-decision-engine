
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "entregas.csv"
OUTPUT_PATH = ROOT / "outputs" / "decisoes.csv"


def avaliar_entrega(entrega):
    pontos = 0
    motivos = []

    if entrega["dias_ate_prazo"] <= 2:
        pontos += 3
        motivos.append("Prazo próximo")

    if entrega["atrasos_anteriores"] >= 3:
        pontos += 2
        motivos.append("Histórico de atrasos")

    if entrega["clima_adverso"] == 1:
        pontos += 2
        motivos.append("Clima adverso")

    if entrega["distancia_km"] > 300:
        pontos += 1
        motivos.append("Longa distância")

    if pontos >= 5:
        risco = "ALTO"
        acao = "Priorizar acompanhamento"

    elif pontos >= 3:
        risco = "MEDIO"
        acao = "Monitorar entrega"

    else:
        risco = "BAIXO"
        acao = "Fluxo normal"

    return {
        "pontuacao": pontos,
        "risco": risco,
        "acao": acao,
        "motivos": "; ".join(motivos) or "Sem alertas"
    }


def executar():
    df = pd.read_csv(DATA_PATH)

    decisoes = df.apply(
        lambda linha: avaliar_entrega(linha),
        axis=1
    )

    resultados = pd.concat(
        [
            df,
            pd.DataFrame(decisoes.tolist())
        ],
        axis=1
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    resultados.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n=== AI DECISION ENGINE ===")

    print("\nTotal de entregas:")
    print(len(resultados))

    print("\nDistribuição de risco:")
    print(resultados["risco"].value_counts())

    print("\nPrimeiras decisões:")
    print(
        resultados[
            ["id_entrega", "pontuacao", "risco", "acao"]
        ].head()
    )

    print(f"\nRelatório salvo em: {OUTPUT_PATH}")

    return resultados


if __name__ == "__main__":
    executar()
