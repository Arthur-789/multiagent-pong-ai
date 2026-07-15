import unittest
from unittest.mock import Mock, patch

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
        _, recompensa = self.turno_atual
        return object(), recompensa, False, False, {}

    def step(self, acao):
        pass


class EvaluateTest(unittest.TestCase):
    @patch("evaluate.heuristic_agent.escolher_acao", return_value=0)
    def test_recompensas_negativas_nao_entram_no_placar(self, _):
        turnos = []
        for _ in range(21):
            turnos.extend(((NOME_RL, -1), (NOME_HEURISTICO, 1)))
        env = AmbienteFalso(turnos)
        agente_rl = Mock()
        agente_rl.escolher_acao.return_value = 0

        placar = jogar_partida(env, agente_rl, seed=123)

        self.assertEqual(placar, (0, 21))
        agente_rl.resetar_estado.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
