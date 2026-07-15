"""Agente de Q-learning tabular com estado derivado exclusivamente da RAM."""

import os
import random
import tempfile
from pathlib import Path

import numpy as np

from config import FATOR_DESCONTO, TAXA_APRENDIZADO, TAXA_APRENDIZADO_FINAL

IDX_BOLA_X = 49
IDX_JOGADOR_Y = 51
IDX_BOLA_Y = 54

ACOES_VALIDAS = (1, 4, 5)  # FIRE, FIRE_RIGHT e FIRE_LEFT
NUM_FAIXAS_BOLA_X = 8
LIMITES_ERRO_VERTICAL = (-32, -16, -4, 4, 16, 32)
NUM_FAIXAS_ERRO = len(LIMITES_ERRO_VERTICAL) + 1
NUM_DIRECOES = 3  # negativa, parada/desconhecida, positiva
NUM_ESTADOS = 1 + NUM_FAIXAS_BOLA_X * NUM_FAIXAS_ERRO * NUM_DIRECOES**2

FORMATO_CHECKPOINT = 2
ALGORITMO = "q_learning_tabular_ram"


def _codigo_direcao(delta):
    return 0 if delta < 0 else 2 if delta > 0 else 1


class AgenteRL:
    def __init__(self):
        self.tabela_q = np.zeros((NUM_ESTADOS, len(ACOES_VALIDAS)), dtype=np.float32)
        self.visitas = np.zeros_like(self.tabela_q, dtype=np.uint32)
        self.passos = 0
        self.epsilon = 1.0
        self.resetar_estado()

    def resetar_estado(self):
        """Descarta o histórico usado para calcular a direção da bola."""
        self._bola_anterior = None

    def extrair_estado(self, observacao_ram):
        """Converte bytes selecionados da RAM em um dos estados discretos."""
        bola_x = int(observacao_ram[IDX_BOLA_X])
        bola_y = int(observacao_ram[IDX_BOLA_Y])
        jogador_y = int(observacao_ram[IDX_JOGADOR_Y])

        if bola_x == 0 or bola_y == 0:
            self._bola_anterior = None
            return 0  # Bola ausente: estado de saque.

        delta_x = 0
        delta_y = 0
        if self._bola_anterior is not None:
            delta_x = bola_x - self._bola_anterior[0]
            delta_y = bola_y - self._bola_anterior[1]
        self._bola_anterior = (bola_x, bola_y)

        faixa_x = min(bola_x * NUM_FAIXAS_BOLA_X // 256, NUM_FAIXAS_BOLA_X - 1)
        faixa_erro = int(
            np.digitize(bola_y - jogador_y, LIMITES_ERRO_VERTICAL)
        )
        direcao_x = _codigo_direcao(delta_x)
        direcao_y = _codigo_direcao(delta_y)

        estado = faixa_x
        estado = estado * NUM_FAIXAS_ERRO + faixa_erro
        estado = estado * NUM_DIRECOES + direcao_x
        estado = estado * NUM_DIRECOES + direcao_y
        return 1 + estado

    def escolher_acao(self, percepcao, explorar=True):
        """Escolhe uma ação por política epsilon-greedy."""
        estado = (
            int(percepcao)
            if isinstance(percepcao, (int, np.integer))
            else self.extrair_estado(percepcao)
        )
        if explorar and random.random() < self.epsilon:
            return random.choice(ACOES_VALIDAS)

        indice_acao = int(np.argmax(self.tabela_q[estado]))
        return ACOES_VALIDAS[indice_acao]

    def treinar_passo(self, estado, acao, recompensa, proximo_estado, terminou):
        """Aplica uma atualização da equação do Q-learning tabular."""
        indice_acao = ACOES_VALIDAS.index(acao)
        q_atual = self.tabela_q[estado, indice_acao]
        melhor_q_futuro = 0.0 if terminou else float(self.tabela_q[proximo_estado].max())
        alvo = recompensa + FATOR_DESCONTO * melhor_q_futuro
        visitas = int(self.visitas[estado, indice_acao]) + 1
        taxa_aprendizado = (
            TAXA_APRENDIZADO_FINAL
            + (TAXA_APRENDIZADO - TAXA_APRENDIZADO_FINAL) / np.sqrt(visitas)
        )
        self.tabela_q[estado, indice_acao] += taxa_aprendizado * (alvo - q_atual)
        self.visitas[estado, indice_acao] = visitas
        self.passos += 1

    def salvar(self, caminho, episodio=None):
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        temporario = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=caminho.parent,
                prefix=f".{caminho.name}.",
                suffix=".tmp",
                delete=False,
            ) as arquivo:
                temporario = Path(arquivo.name)
                np.savez_compressed(
                    arquivo,
                    formato=np.int64(FORMATO_CHECKPOINT),
                    algoritmo=np.array(ALGORITMO),
                    episodio=np.int64(episodio or 0),
                    epsilon=np.float64(self.epsilon),
                    passos=np.int64(self.passos),
                    acoes=np.array(ACOES_VALIDAS, dtype=np.int64),
                    tabela_q=self.tabela_q,
                    visitas=self.visitas,
                )
                arquivo.flush()
                os.fsync(arquivo.fileno())
            os.replace(temporario, caminho)
        except BaseException:
            if temporario is not None:
                temporario.unlink(missing_ok=True)
            raise

    def carregar(self, caminho):
        try:
            with np.load(caminho, allow_pickle=False) as estado:
                formato = int(estado["formato"])
                algoritmo = str(estado["algoritmo"])
                acoes = tuple(int(acao) for acao in estado["acoes"])
                tabela_q = estado["tabela_q"]
                visitas = estado["visitas"]

                if formato != FORMATO_CHECKPOINT or algoritmo != ALGORITMO:
                    raise ValueError("checkpoint não pertence ao agente tabular atual")
                if (
                    acoes != ACOES_VALIDAS
                    or tabela_q.shape != self.tabela_q.shape
                    or visitas.shape != self.visitas.shape
                ):
                    raise ValueError("checkpoint usa ações ou discretização incompatíveis")

                self.tabela_q[:] = tabela_q
                self.visitas[:] = visitas
                self.epsilon = float(estado["epsilon"])
                self.passos = int(estado["passos"])
                episodio = int(estado["episodio"])
        except (KeyError, ValueError) as erro:
            raise ValueError(f"checkpoint tabular inválido: {caminho}") from erro

        self.resetar_estado()
        return episodio
