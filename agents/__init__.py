"""Pacote com os agentes do projeto (heurístico, RL e genético) e o catálogo que os carrega."""

from agents.catalog import (
    LADO_DIREITO,
    LADO_ESQUERDO,
    PREFERENCIAS_LADO,
    AgenteHeuristico,
    AgenteRLCarregado,
    carregar_agente,
    lados_para_jogo,
)

__all__ = [
    "LADO_DIREITO",
    "LADO_ESQUERDO",
    "PREFERENCIAS_LADO",
    "AgenteHeuristico",
    "AgenteRLCarregado",
    "carregar_agente",
    "lados_para_jogo",
]
