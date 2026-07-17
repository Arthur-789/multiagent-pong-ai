import unittest
from unittest.mock import Mock, patch, call

import numpy as np

import train_genetic


class AvaliarRallyTest(unittest.TestCase):
    def test_reset_recebe_seed_explicita(self):
        env = Mock()
        env.agent_iter.return_value = iter([])  # rally "vazio" só para o teste de reset
        env.last.return_value = ([0] * 128, 0, True, False, {})

        agente = Mock()
        train_genetic.avaliar_rally(agente, None, "heuristico", env, seed=123)

        env.reset.assert_called_once_with(seed=123)

    def test_seeds_diferentes_nao_repetem_chamada_identica(self):
        env = Mock()
        env.agent_iter.return_value = iter([])
        env.last.return_value = ([0] * 128, 0, True, False, {})
        agente = Mock()

        train_genetic.avaliar_rally(agente, None, "heuristico", env, seed=1)
        train_genetic.avaliar_rally(agente, None, "heuristico", env, seed=2)

        env.reset.assert_has_calls([call(seed=1), call(seed=2)])


class EvalAgenteTest(unittest.TestCase):
    @patch("train_genetic.avaliar_rally")
    @patch("train_genetic.criar_ambiente")
    def test_eval_agente_usa_multiplos_rallies_com_seeds_distintas(self, criar_env, avaliar_rally):
        env = Mock()
        criar_env.return_value = env
        # 3 rallies com fitness diferentes: garante que a média é calculada
        avaliar_rally.side_effect = [10.0, -25.0, 5.0]

        individuo = [0] * 12288
        resultado = train_genetic.eval_agente(individuo, n_rallies=3, seed_base=100)

        # Verifica que cada rally usou uma seed distinta derivada de seed_base
        seeds_usadas = [chamada.kwargs["seed"] for chamada in avaliar_rally.call_args_list]
        self.assertEqual(seeds_usadas, [100, 101, 102])

        media_esperada = (10.0 - 25.0 + 5.0) / 3
        self.assertAlmostEqual(resultado[0], media_esperada + 1000.0)
        env.close.assert_called_once_with()

    @patch("train_genetic.avaliar_rally")
    @patch("train_genetic.criar_ambiente")
    def test_eval_agente_retorna_tupla_para_deap(self, criar_env, avaliar_rally):
        criar_env.return_value = Mock()
        avaliar_rally.return_value = 0.0

        resultado = train_genetic.eval_agente([0] * 12288, n_rallies=1)

        self.assertIsInstance(resultado, tuple)
        self.assertEqual(len(resultado), 1)


class ToolboxConfigTest(unittest.TestCase):
    def test_mutacao_usa_taxa_baixa_por_bit(self):
        mutate_func = train_genetic.toolbox.mutate
        indpb = mutate_func.keywords.get("indpb")
        self.assertIsNotNone(indpb)
        self.assertLess(indpb, 0.001)

    def test_selecao_nao_e_mais_roleta(self):
        select_func = train_genetic.toolbox.select
        nome = getattr(select_func, "func", select_func).__name__
        self.assertNotEqual(nome, "selRoulette")


if __name__ == "__main__":
    unittest.main()
