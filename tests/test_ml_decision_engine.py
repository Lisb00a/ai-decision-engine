
import math

import pytest

from src import ml_decision_engine as engine


# ==========================================
# MODELO SIMULADO
# ==========================================

class ModeloSimulado:
    def __init__(self, probabilidade):
        self.probabilidade = probabilidade

    def predict_proba(self, dados):
        return [[
            1 - self.probabilidade,
            self.probabilidade
        ]]


@pytest.fixture
def entrega_baixo_risco():
    return {
        "distancia_km": 50,
        "dias_ate_prazo": 10,
        "atrasos_anteriores": 0,
        "clima_adverso": 0,
        "carga_prioritaria": 0
    }


@pytest.fixture
def configurar_modelo(monkeypatch):
    def configurar(probabilidade, threshold=0.30):
        monkeypatch.setattr(
            engine,
            "carregar_modelo",
            lambda: ModeloSimulado(probabilidade)
        )
        monkeypatch.setattr(
            engine,
            "carregar_threshold",
            lambda: threshold
        )

    return configurar


# ==========================================
# CLASSIFICAÇÃO DO MACHINE LEARNING
# ==========================================

@pytest.mark.parametrize(
    "probabilidade, risco_esperado",
    [
        (0.10, "BAIXO"),
        (0.29, "BAIXO"),
        (0.30, "MEDIO"),
        (0.50, "MEDIO"),
        (0.69, "MEDIO"),
        (0.70, "ALTO"),
        (0.95, "ALTO")
    ]
)
def test_classificacao_ml(
    probabilidade,
    risco_esperado
):
    resultado = engine.classificar_risco_ml(
        probabilidade,
        0.30
    )

    assert resultado == risco_esperado


# ==========================================
# PROBABILIDADES INVÁLIDAS
# ==========================================

@pytest.mark.parametrize(
    "probabilidade",
    [
        -0.1,
        1.1,
        math.nan,
        math.inf,
        -math.inf
    ]
)
def test_probabilidade_invalida(probabilidade):
    with pytest.raises(ValueError):
        engine.classificar_risco_ml(
            probabilidade,
            0.30
        )


# ==========================================
# RISCO BAIXO
# ==========================================

def test_decisao_baixo_risco(
    entrega_baixo_risco,
    configurar_modelo
):
    configurar_modelo(0.10)

    resultado = engine.decidir_entrega(
        entrega_baixo_risco
    )

    assert resultado["risco_ml"] == "BAIXO"
    assert resultado["risco_regras"] == "BAIXO"
    assert resultado["risco_final"] == "BAIXO"
    assert resultado["acao_recomendada"] == "Fluxo normal"


# ==========================================
# RISCO MÉDIO PELO MODELO
# ==========================================

def test_decisao_medio_por_ml(
    entrega_baixo_risco,
    configurar_modelo
):
    configurar_modelo(0.50)

    resultado = engine.decidir_entrega(
        entrega_baixo_risco
    )

    assert resultado["risco_ml"] == "MEDIO"
    assert resultado["risco_regras"] == "BAIXO"
    assert resultado["risco_final"] == "MEDIO"
    assert resultado["acao_recomendada"] == "Monitorar entrega"


# ==========================================
# RISCO ALTO PELO MODELO
# ==========================================

def test_decisao_alto_por_ml(
    entrega_baixo_risco,
    configurar_modelo
):
    configurar_modelo(0.85)

    resultado = engine.decidir_entrega(
        entrega_baixo_risco
    )

    assert resultado["risco_ml"] == "ALTO"
    assert resultado["risco_final"] == "ALTO"
    assert (
        resultado["acao_recomendada"]
        == "Priorizar acompanhamento"
    )


# ==========================================
# REGRAS PREVALECEM SOBRE O ML
# ==========================================

def test_regras_prevalecem_sobre_ml(
    entrega_baixo_risco,
    configurar_modelo
):
    configurar_modelo(0.10)

    entrega_baixo_risco["dias_ate_prazo"] = 1
    entrega_baixo_risco["atrasos_anteriores"] = 4

    resultado = engine.decidir_entrega(
        entrega_baixo_risco
    )

    assert resultado["risco_ml"] == "BAIXO"
    assert resultado["risco_regras"] == "ALTO"
    assert resultado["risco_final"] == "ALTO"


# ==========================================
# CARGA PRIORITÁRIA
# ==========================================

def test_carga_prioritaria_eleva_risco(
    entrega_baixo_risco,
    configurar_modelo
):
    configurar_modelo(0.50)

    entrega_baixo_risco["carga_prioritaria"] = 1

    resultado = engine.decidir_entrega(
        entrega_baixo_risco
    )

    assert resultado["risco_ml"] == "MEDIO"
    assert resultado["risco_regras"] == "BAIXO"
    assert resultado["risco_final"] == "ALTO"


# ==========================================
# PROBABILIDADE RETORNADA
# ==========================================

def test_probabilidade_arredondada(
    entrega_baixo_risco,
    configurar_modelo
):
    configurar_modelo(0.456789)

    resultado = engine.decidir_entrega(
        entrega_baixo_risco
    )

    assert resultado["probabilidade_atraso"] == 0.4568


# ==========================================
# THRESHOLD REAL
# ==========================================

def test_threshold_configurado():
    engine.carregar_threshold.cache_clear()

    threshold = engine.carregar_threshold()

    assert threshold == pytest.approx(0.30)
    assert 0 <= threshold <= 0.70
