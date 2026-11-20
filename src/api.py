
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.ml_decision_engine import decidir_entrega


app = FastAPI(
    title="AI Decision Engine",
    description="API de análise de risco de entregas",
    version="1.0.0"
)


class EntregaRequest(BaseModel):
    distancia_km: float = Field(ge=0)
    dias_ate_prazo: int = Field(ge=0)
    atrasos_anteriores: int = Field(ge=0)
    clima_adverso: int = Field(ge=0, le=1)
    carga_prioritaria: int = Field(ge=0, le=1)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AI Decision Engine"
    }


@app.post("/decidir")
def decidir(entrega: EntregaRequest):
    dados = entrega.model_dump()
    return decidir_entrega(dados)
