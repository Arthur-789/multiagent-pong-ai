import unittest
from unittest.mock import Mock, patch

import numpy as np

from evaluate import NOME_HEURISTICO, NOME_RL, jogar_partida


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
        agente, recompensa, pontos_rl, pontos_heuristico, terminou, truncado = (
            self.turno_atual
        )
        observacao = np.zeros(128, dtype=np.uint8)
        observacao[14] = pontos_rl
        observacao[13] = pontos_heuristico
        return observacao, recompensa, terminou, truncado, {}

    def step(self, acao):
        pass


class EvaluateTest(unittest.TestCase):
    @patch("evaluate.heuristic_agent.escolher_acao", return_value=0)
    def test_retorna_placar_oficial_da_ram(self, _):
        turnos = [
            (NOME_RL, -1, 7, 21, True, False),
            (NOME_HEURISTICO, 1, 7, 21, True, False),
        ]
        env = AmbienteFalso(turnos)
        agente_rl = Mock()
        agente_rl.escolher_acao.return_value = 0

        placar = jogar_partida(env, agente_rl, seed=123)

        self.assertEqual(placar, (7, 21, False))

    @patch("evaluate.heuristic_agent.escolher_acao", return_value=0)
    def test_reinicia_historico_do_rl_depois_de_cada_ponto(self, _):
        turnos = [
            (NOME_RL, 0, 0, 0, False, False),
            (NOME_HEURISTICO, 0, 0, 0, False, False),
            (NOME_RL, -1, 0, 1, False, False),
        ]
        env = AmbienteFalso(turnos)
        agente_rl = Mock()
        agente_rl.escolher_acao.return_value = 0

        jogar_partida(env, agente_rl, seed=123)

        self.assertEqual(agente_rl.resetar_estado.call_count, 2)

    @patch("evaluate.heuristic_agent.escolher_acao", return_value=0)
    def test_informa_quando_partida_foi_truncada(self, _):
        turnos = [(NOME_RL, 0, 3, 4, False, True)]
        env = AmbienteFalso(turnos)
        agente_rl = Mock()

        resultado = jogar_partida(env, agente_rl, seed=123)

        self.assertEqual(resultado, (3, 4, True))


if __name__ == "__main__":
    unittest.main()
