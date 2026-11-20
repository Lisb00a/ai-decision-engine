
import pytest
from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


@pytest.fixture
def entrega_valida():
    return {
        "distancia_km": 350,
        "dias_ate_prazo": 1,
        "atrasos_anteriores": 4,
        "clima_adverso": 1,
        "carga_prioritaria": 1
    }


def test_health():
    resposta = client.get("/health")

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_decidir_sucesso(entrega_valida):
    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert "probabilidade_atraso" in dados
    assert "risco_ml" in dados
    assert "risco_regras" in dados
    assert "risco_final" in dados
    assert "acao_recomendada" in dados

    assert 0 <= dados["probabilidade_atraso"] <= 1
    assert dados["risco_final"] in [
        "BAIXO", "MEDIO", "ALTO"
    ]


def test_distancia_negativa(entrega_valida):
    entrega_valida["distancia_km"] = -10

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422


def test_clima_invalido(entrega_valida):
    entrega_valida["clima_adverso"] = 3

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422


def test_campo_obrigatorio_ausente(entrega_valida):
    del entrega_valida["dias_ate_prazo"]

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422


def test_carga_prioritaria_invalida(entrega_valida):
    entrega_valida["carga_prioritaria"] = 2

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422


def test_recomendacao_alto_risco(entrega_valida):
    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert dados["risco_final"] == "ALTO"
    assert (
        dados["acao_recomendada"]
        == "Priorizar acompanhamento"
    )
