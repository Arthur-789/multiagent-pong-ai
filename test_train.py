import unittest
from unittest.mock import Mock, patch

from train import avaliar_para_selecao


class TrainTest(unittest.TestCase):
    @patch("train.PARTIDAS_VALIDACAO", 3)
    @patch("train.jogar_partida")
    @patch("train.criar_ambiente")
    def test_avaliacao_para_selecao_calcula_metricas(self, criar_env, jogar_partida):
        env = Mock()
        criar_env.return_value = env
        jogar_partida.side_effect = [
            (21, 19, False),
            (10, 21, False),
            (3, 4, True),
        ]

        resultado = avaliar_para_selecao(Mock())

        self.assertEqual(resultado["margem_media"], -4.5)
        self.assertEqual(resultado["pontos_medios"], 15.5)
        self.assertEqual(resultado["taxa_vitoria"], 0.5)
        env.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
