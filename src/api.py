
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.ml_decision_engine import decidir_entrega


# ==========================================
# CONFIGURAÇÃO DE LOGS
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURAÇÃO DA API
# ==========================================

app = FastAPI(
    title="AI Decision Engine",
    description=(
        "API de análise de risco de entregas "
        "com Machine Learning e regras de negócio"
    ),
    version="1.1.0"
)


# ==========================================
# MODELO DE ENTRADA
# ==========================================

class EntregaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distancia_km: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Distância da entrega em quilômetros"
    )

    dias_ate_prazo: int = Field(
        ge=0,
        description="Quantidade de dias até o prazo"
    )

    atrasos_anteriores: int = Field(
        ge=0,
        description="Quantidade de atrasos anteriores"
    )

    clima_adverso: int = Field(
        ge=0,
        le=1,
        description="0 = Não, 1 = Sim"
    )

    carga_prioritaria: int = Field(
        ge=0,
        le=1,
        description="0 = Não, 1 = Sim"
    )


# ==========================================
# ENDPOINT - HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AI Decision Engine"
    }


# ==========================================
# ENDPOINT - ANÁLISE DE RISCO
# ==========================================

@app.post("/decidir")
def decidir(entrega: EntregaRequest):

    inicio = time.perf_counter()

    dados = entrega.model_dump()

    logger.info(
        "Nova análise de risco recebida"
    )

    try:
        resultado = decidir_entrega(dados)

        duracao = time.perf_counter() - inicio

        logger.info(
            "Análise concluída | tempo=%.3fs",
            duracao
        )

        return resultado

    except (FileNotFoundError, OSError) as erro:
        logger.error(
            "Falha ao acessar recurso do modelo | tipo=%s",
            type(erro).__name__
        )

        raise HTTPException(
            status_code=503,
            detail="O modelo de decisão está indisponível."
        ) from erro

    except ValueError as erro:
        logger.warning(
            "Erro ao processar dados da entrega | tipo=%s",
            type(erro).__name__
        )

        raise HTTPException(
            status_code=422,
            detail="Não foi possível processar os dados da entrega."
        ) from erro

    except Exception as erro:
        logger.exception(
            "Erro inesperado durante a análise de risco"
        )

        raise HTTPException(
            status_code=500,
            detail="Erro interno ao analisar o risco da entrega."
        ) from erro
