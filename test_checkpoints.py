import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from checkpoints import caminho_checkpoint, checkpoint_mais_recente
from rl_agent import AgenteRL


class CheckpointsTest(unittest.TestCase):
    def test_encontra_maior_episodio_numericamente(self):
        with tempfile.TemporaryDirectory() as diretorio:
            caminho_checkpoint(diretorio, 9).touch()
            esperado = caminho_checkpoint(diretorio, 100)
            esperado.touch()
            (Path(diretorio) / "arquivo_incompleto.tmp").touch()

            self.assertEqual(checkpoint_mais_recente(diretorio), esperado)

    def test_checkpoint_restaura_estado_do_treino(self):
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = caminho_checkpoint(diretorio, 12)
            agente = AgenteRL()
            agente.epsilon = 0.42
            agente.passos = 321
            agente.tabela_q[7, 2] = 8.5
            agente.visitas[7, 2] = 123
            agente.salvar(caminho, episodio=12)

            restaurado = AgenteRL()
            episodio = restaurado.carregar(caminho)

            self.assertEqual(episodio, 12)
            self.assertEqual(restaurado.epsilon, 0.42)
            self.assertEqual(restaurado.passos, 321)
            np.testing.assert_array_equal(agente.tabela_q, restaurado.tabela_q)
            np.testing.assert_array_equal(agente.visitas, restaurado.visitas)

    def test_falha_na_serializacao_nao_cria_checkpoint_parcial(self):
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = caminho_checkpoint(diretorio, 1)
            agente = AgenteRL()

            with patch("rl_agent.np.savez_compressed", side_effect=OSError("falha")):
                with self.assertRaises(OSError):
                    agente.salvar(caminho, episodio=1)

            self.assertFalse(caminho.exists())
            self.assertEqual(list(Path(diretorio).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
