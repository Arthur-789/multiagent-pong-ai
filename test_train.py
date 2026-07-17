import unittest
from unittest.mock import patch

import numpy as np

from train import avaliar_para_selecao


class AmbienteValidacaoFalso:
    def __init__(self, resultados):
        self.resultados = resultados
        self.partida = -1
        self.agente_atual = None
        self.acoes = []
        self.fechado = False

    def reset(self, seed=None):
        self.partida += 1
        self.seed = seed

    def agent_iter(self):
        for agente in ("first_0", "second_0", "first_0", "second_0"):
            self.agente_atual = agente
            yield agente

    def last(self):
        pontos_esq, pontos_dir, truncado = self.resultados[self.partida]
        observacao = np.zeros(128, dtype=np.uint8)
        observacao[13] = pontos_esq
        observacao[14] = pontos_dir
        observacao[50] = 10
        observacao[51] = 100
        observacao[54] = 100
        terminou = len(self.acoes) % 4 >= 2
        return observacao, 0, terminou, truncado and terminou, {}

    def step(self, acao):
        self.acoes.append((self.agente_atual, acao))

    def close(self):
        self.fechado = True


class AgenteRLFalso:
    def __init__(self):
        self.valores_explorar = []

    def resetar_estado(self):
        pass

    def escolher_acao(self, observacao, explorar):
        self.valores_explorar.append(explorar)
        return 1


class TrainTest(unittest.TestCase):
    @patch("train.PARTIDAS_VALIDACAO", 3)
    @patch("train.criar_ambiente")
    def test_avalia_rl_contra_baseline_sem_exploracao(self, criar_env):
        env = AmbienteValidacaoFalso(
            [
                (19, 21, False),
                (21, 10, False),
                (3, 4, True),
            ]
        )
        criar_env.return_value = env
        agente_rl = AgenteRLFalso()

        resultado = avaliar_para_selecao(agente_rl)

        self.assertEqual(
            resultado,
            {
                "margem_media": -4.5,
                "pontos_medios": 15.5,
                "taxa_vitoria": 0.5,
            },
        )
        self.assertEqual(agente_rl.valores_explorar, [False, False, False])
        acoes_baseline = [
            acao for agente, acao in env.acoes if agente == "second_0" and acao
        ]
        self.assertEqual(acoes_baseline, [5, 5, 5])
        self.assertTrue(env.fechado)


if __name__ == "__main__":
    unittest.main()
