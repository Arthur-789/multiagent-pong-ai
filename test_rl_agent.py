import unittest

import numpy as np

from config import FATOR_DESCONTO, TAXA_APRENDIZADO
from rl_agent import ACOES_VALIDAS, AgenteRL, NUM_ESTADOS


class AgenteRLTest(unittest.TestCase):
    def _ram(self, bola_x=100, bola_y=100, jogador_y=100):
        ram = np.zeros(128, dtype=np.uint8)
        ram[49] = bola_x
        ram[51] = jogador_y
        ram[54] = bola_y
        return ram

    def test_estado_fica_dentro_da_tabela(self):
        agente = AgenteRL()

        estado = agente.extrair_estado(self._ram(255, 255, 0))

        self.assertGreaterEqual(estado, 1)
        self.assertLess(estado, NUM_ESTADOS)

    def test_bola_ausente_representa_estado_de_saque(self):
        agente = AgenteRL()

        self.assertEqual(agente.extrair_estado(self._ram(bola_x=0)), 0)

    def test_historico_da_ram_altera_direcao_do_estado(self):
        agente = AgenteRL()
        agente.extrair_estado(self._ram(100, 100, 100))

        movendo = agente.extrair_estado(self._ram(101, 102, 100))
        agente.resetar_estado()
        sem_historico = agente.extrair_estado(self._ram(101, 102, 100))

        self.assertNotEqual(movendo, sem_historico)

    def test_atualiza_apenas_valor_q_da_acao_executada(self):
        agente = AgenteRL()
        agente.tabela_q[2] = [2.0, 4.0, 6.0]

        agente.treinar_passo(1, ACOES_VALIDAS[1], 3.0, 2, terminou=False)

        esperado = TAXA_APRENDIZADO * (3.0 + FATOR_DESCONTO * 6.0)
        self.assertAlmostEqual(float(agente.tabela_q[1, 1]), esperado, places=6)
        self.assertEqual(float(agente.tabela_q[1, 0]), 0.0)
        self.assertEqual(float(agente.tabela_q[1, 2]), 0.0)

    def test_transicao_terminal_nao_considera_estado_futuro(self):
        agente = AgenteRL()
        agente.tabela_q[2] = 100.0

        agente.treinar_passo(1, ACOES_VALIDAS[0], -10.0, 2, terminou=True)

        self.assertAlmostEqual(
            float(agente.tabela_q[1, 0]), TAXA_APRENDIZADO * -10.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
