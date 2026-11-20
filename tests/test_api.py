
import pytest

from fastapi.testclient import TestClient

from src import api


# ==========================================
# CONFIGURAÇÃO DOS TESTES
# ==========================================

client = TestClient(api.app)


@pytest.fixture
def entrega_valida():
    return {
        "distancia_km": 350,
        "dias_ate_prazo": 1,
        "atrasos_anteriores": 4,
        "clima_adverso": 1,
        "carga_prioritaria": 1
    }


# ==========================================
# TESTE 01 - HEALTH CHECK
# ==========================================

def test_health():
    resposta = client.get("/health")

    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"
    assert resposta.json()["service"] == "AI Decision Engine"


# ==========================================
# TESTE 02 - DECISÃO COM SUCESSO
# ==========================================

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


# ==========================================
# TESTE 03 - DISTÂNCIA NEGATIVA
# ==========================================

def test_distancia_negativa(entrega_valida):
    entrega_valida["distancia_km"] = -10

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422


# ==========================================
# TESTE 04 - CLIMA INVÁLIDO
# ==========================================

def test_clima_invalido(entrega_valida):
    entrega_valida["clima_adverso"] = 3

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422


# ==========================================
# TESTE 05 - CAMPO OBRIGATÓRIO AUSENTE
# ==========================================

def test_campo_obrigatorio_ausente(entrega_valida):
    del entrega_valida["dias_ate_prazo"]

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422


# ==========================================
# TESTE 06 - CARGA PRIORITÁRIA INVÁLIDA
# ==========================================

def test_carga_prioritaria_invalida(entrega_valida):
    entrega_valida["carga_prioritaria"] = 2

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422


# ==========================================
# TESTE 07 - RECOMENDAÇÃO DE ALTO RISCO
# ==========================================

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


# ==========================================
# TESTE 08 - CAMPO DESCONHECIDO
# ==========================================

def test_campo_desconhecido(entrega_valida):
    entrega_valida["campo_inexistente"] = "teste"

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422


# ==========================================
# TESTE 09 - DISTÂNCIA INFINITA
# ==========================================

def test_distancia_infinita(entrega_valida):
    entrega_valida["distancia_km"] = "Infinity"

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422


# ==========================================
# TESTE 10 - ATRASOS NEGATIVOS
# ==========================================

def test_atrasos_negativos(entrega_valida):
    entrega_valida["atrasos_anteriores"] = -1

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422


# ==========================================
# TESTE 11 - MODELO INDISPONÍVEL
# ==========================================

def test_modelo_indisponivel(monkeypatch, entrega_valida):

    def modelo_indisponivel(dados):
        raise FileNotFoundError(
            "Arquivo do modelo não encontrado"
        )

    monkeypatch.setattr(
        api,
        "decidir_entrega",
        modelo_indisponivel
    )

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 503

    assert (
        resposta.json()["detail"]
        == "O modelo de decisão está indisponível."
    )


# ==========================================
# TESTE 12 - ERRO NOS DADOS DO MODELO
# ==========================================

def test_erro_dados_modelo(monkeypatch, entrega_valida):

    def erro_dados(dados):
        raise ValueError(
            "Dados incompatíveis com o modelo"
        )

    monkeypatch.setattr(
        api,
        "decidir_entrega",
        erro_dados
    )

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 422

    assert (
        resposta.json()["detail"]
        == "Não foi possível processar os dados da entrega."
    )


# ==========================================
# TESTE 13 - ERRO INTERNO
# ==========================================

def test_erro_interno(monkeypatch, entrega_valida):

    def erro_inesperado(dados):
        raise RuntimeError(
            "Erro interno simulado"
        )

    monkeypatch.setattr(
        api,
        "decidir_entrega",
        erro_inesperado
    )

    resposta = client.post(
        "/decidir",
        json=entrega_valida
    )

    assert resposta.status_code == 500

    assert (
        resposta.json()["detail"]
        == "Erro interno ao analisar o risco da entrega."
    )
