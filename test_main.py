import unittest
from unittest.mock import patch

import main


class MainTest(unittest.TestCase):
    @patch("main.treinar_rl")
    def test_train_rl_inicia_treinamento_do_tipo_escolhido(self, treinar_rl):
        main.main(["train", "rl"])

        treinar_rl.assert_called_once_with()

    @patch("main.treinar_genetico")
    def test_train_genetico_inicia_treinamento_do_tipo_escolhido(
        self, treinar_genetico
    ):
        main.main(["train", "genetico"])

        treinar_genetico.assert_called_once_with()

    @patch("main.avaliar_agentes")
    def test_eval_compara_os_tipos_escolhidos_com_renderizacao(self, avaliar):
        main.main(["eval", "rl", "genetico", "--render"])

        avaliar.assert_called_once_with("rl", "genetico", render=True)

    def test_eval_rejeita_agentes_treinados_que_exigem_o_mesmo_lado(self):
        for tipo in ("rl", "genetico"):
            with self.subTest(tipo=tipo):
                with self.assertRaises(SystemExit) as erro:
                    main.main(["eval", tipo, tipo])

                self.assertNotEqual(erro.exception.code, 0)

    @patch("main.jogar_contra")
    def test_play_inicia_partida_contra_o_tipo_escolhido(self, jogar_contra):
        main.main(["play", "heuristico"])

        jogar_contra.assert_called_once_with("heuristico")


if __name__ == "__main__":
    unittest.main()
