import unittest
from unittest.mock import Mock

import numpy as np

from evaluate import jogar_partida

NOME_RL = "first_0"
NOME_HEURISTICO = "second_0"

class AmbienteFalso:
    def __init__(self, turnos):
        self.turnos = turnos
        self.turno_atual = None

    def reset(self, seed=None):
        self.seed = seed

    def agent_iter(self):
        for turno in self.turnos:
            self.turno_atual = turno
            yield turno[0]

    def last(self):
        agente, recompensa, pontos_esq, pontos_dir, terminou, truncado = (
            self.turno_atual
        )
        observacao = np.zeros(128, dtype=np.uint8)
        observacao[14] = pontos_esq
        observacao[13] = pontos_dir
        return observacao, recompensa, terminou, truncado, {}

    def step(self, acao):
        pass


class EvaluateTest(unittest.TestCase):
    def test_retorna_placar_oficial_da_ram(self):
        turnos = [
            (NOME_RL, -1, 7, 21, True, False),
            (NOME_HEURISTICO, 1, 7, 21, True, False),
        ]
        env = AmbienteFalso(turnos)
        agente_esq = Mock()
        agente_esq.escolher_acao.return_value = 0
        agente_dir = Mock()
        agente_dir.escolher_acao.return_value = 0

        placar = jogar_partida(env, agente_esq, agente_dir, seed=123)

        self.assertEqual(placar, (7, 21, False))

    def test_reinicia_historico_do_rl_depois_de_cada_ponto(self):
        turnos = [
            (NOME_RL, 0, 0, 0, False, False),
            (NOME_HEURISTICO, 0, 0, 0, False, False),
            (NOME_RL, -1, 0, 1, False, False),
        ]
        env = AmbienteFalso(turnos)
        agente_esq = Mock()
        agente_esq.escolher_acao.return_value = 0
        agente_dir = Mock()
        agente_dir.escolher_acao.return_value = 0

        jogar_partida(env, agente_esq, agente_dir, seed=123)

        self.assertEqual(agente_esq.resetar_estado.call_count, 2)
        # dir agent didn't receive a reward != 0 when it was its turn, 
        # wait, the loop processes rl, then heuristico, then rl.
        # the third turn rl gets reward -1, so it resets state.

    def test_informa_quando_partida_foi_truncada(self):
        turnos = [(NOME_RL, 0, 3, 4, False, True)]
        env = AmbienteFalso(turnos)
        agente_esq = Mock()
        agente_dir = Mock()

        resultado = jogar_partida(env, agente_esq, agente_dir, seed=123)

        self.assertEqual(resultado, (3, 4, True))


if __name__ == "__main__":
    unittest.main()
