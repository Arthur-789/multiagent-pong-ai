import unittest
from unittest.mock import patch

from agents import LADO_DIREITO, LADO_ESQUERDO, carregar_agente, lados_para_jogo


class JogarTest(unittest.TestCase):
    def test_preserva_lado_de_treinamento_do_adversario(self):
        self.assertEqual(lados_para_jogo("rl"), (LADO_ESQUERDO, LADO_DIREITO))
        self.assertEqual(lados_para_jogo("genetico"), (LADO_DIREITO, LADO_ESQUERDO))
        self.assertEqual(
            lados_para_jogo("heuristico"), (LADO_ESQUERDO, LADO_DIREITO)
        )

    @patch("agents.os.path.exists", return_value=False)
    def test_falha_quando_artefato_padrao_nao_existe(self, _):
        with self.assertRaisesRegex(FileNotFoundError, "python main.py train rl"):
            carregar_agente("rl", LADO_DIREITO)

    @patch("agents.os.path.exists", return_value=False)
    def test_falha_quando_artefato_genetico_padrao_nao_existe(self, _):
        with self.assertRaisesRegex(
            FileNotFoundError, "python main.py train genetico"
        ):
            carregar_agente("genetico", LADO_ESQUERDO)


if __name__ == "__main__":
    unittest.main()
