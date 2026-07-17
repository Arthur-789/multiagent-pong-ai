"""Catálogo dos tipos de agente disponíveis para avaliação e jogo."""

import os

from config import CAMINHO_MELHOR_CHECKPOINT, GA_CAMINHO_CHECKPOINT

LADO_DIREITO = "first_0"
LADO_ESQUERDO = "second_0"

PREFERENCIAS_LADO = {
    "rl": LADO_DIREITO,
    "genetico": LADO_ESQUERDO,
    "heuristico": None,
}


def lados_para_jogo(tipo):
    lado_agente = PREFERENCIAS_LADO[tipo] or LADO_DIREITO
    lado_humano = LADO_ESQUERDO if lado_agente == LADO_DIREITO else LADO_DIREITO
    return lado_humano, lado_agente


class AgenteHeuristico:
    def __init__(self, agente_id):
        self.agente_id = agente_id

    def resetar_estado(self):
        pass

    def escolher_acao(self, observacao):
        import heuristic_agent

        return heuristic_agent.escolher_acao(observacao, agente_id=self.agente_id)


class AgenteRLCarregado:
    def __init__(self, agente):
        self.agente = agente

    def resetar_estado(self):
        self.agente.resetar_estado()

    def escolher_acao(self, observacao):
        return self.agente.escolher_acao(observacao, explorar=False)


def _exigir_artefato(tipo, caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Artefato treinado de '{tipo}' não encontrado em {caminho}. "
            f"Execute: python main.py train {tipo}"
        )


def carregar_agente(tipo, agente_id):
    if tipo == "rl":
        _exigir_artefato(tipo, CAMINHO_MELHOR_CHECKPOINT)
        from rl_agent import AgenteRL

        agente = AgenteRL()
        agente.carregar(CAMINHO_MELHOR_CHECKPOINT)
        return AgenteRLCarregado(agente)
    if tipo == "genetico":
        _exigir_artefato(tipo, GA_CAMINHO_CHECKPOINT)
        import numpy as np
        from genetic_agent import AgenteGenetico

        return AgenteGenetico(np.load(GA_CAMINHO_CHECKPOINT))
    if tipo == "heuristico":
        return AgenteHeuristico(agente_id)
    raise ValueError(f"Tipo de agente desconhecido: {tipo}")
