
from src.decision_engine import avaliar_entrega


def test_risco_alto():
    entrega = {
        "dias_ate_prazo": 1,
        "atrasos_anteriores": 4,
        "clima_adverso": 1,
        "distancia_km": 350
    }

    resultado = avaliar_entrega(entrega)

    assert resultado["pontuacao"] == 8
    assert resultado["risco"] == "ALTO"
    assert resultado["acao"] == "Priorizar acompanhamento"


def test_risco_medio():
    entrega = {
        "dias_ate_prazo": 2,
        "atrasos_anteriores": 0,
        "clima_adverso": 0,
        "distancia_km": 100
    }

    resultado = avaliar_entrega(entrega)

    assert resultado["pontuacao"] == 3
    assert resultado["risco"] == "MEDIO"
    assert resultado["acao"] == "Monitorar entrega"


def test_risco_baixo():
    entrega = {
        "dias_ate_prazo": 8,
        "atrasos_anteriores": 0,
        "clima_adverso": 0,
        "distancia_km": 50
    }

    resultado = avaliar_entrega(entrega)

    assert resultado["pontuacao"] == 0
    assert resultado["risco"] == "BAIXO"
    assert resultado["acao"] == "Fluxo normal"


def test_multiplos_fatores():
    entrega = {
        "dias_ate_prazo": 5,
        "atrasos_anteriores": 3,
        "clima_adverso": 1,
        "distancia_km": 400
    }

    resultado = avaliar_entrega(entrega)

    assert resultado["pontuacao"] == 5
    assert resultado["risco"] == "ALTO"
    assert len(resultado["motivos"].split("; ")) == 3


def test_limite_distancia():
    entrega = {
        "dias_ate_prazo": 8,
        "atrasos_anteriores": 0,
        "clima_adverso": 0,
        "distancia_km": 300
    }

    resultado = avaliar_entrega(entrega)

    assert resultado["pontuacao"] == 0
    assert resultado["risco"] == "BAIXO"
