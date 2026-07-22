import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, call

import numpy as np

import main
from agents import LADO_DIREITO, LADO_ESQUERDO, carregar_agente, lados_para_jogo
from environment import criar_ambiente
from evaluate import jogar_partida
from rl_agent import (
    ACOES_VALIDAS,
    AgenteRL,
    NUM_ESTADOS,
    caminho_checkpoint,
    checkpoint_mais_recente,
)
from config import FATOR_DESCONTO, TAXA_APRENDIZADO, TAXA_APRENDIZADO_FINAL


# =====================================================================
# main.py
# =====================================================================
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

    @patch("main.avaliar_agentes")
    def test_eval_permite_dois_baselines_heuristicos(self, avaliar):
        main.main(["eval", "heuristico", "heuristico"])

        avaliar.assert_called_once_with("heuristico", "heuristico", render=False)

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

    def test_rejeita_aliases_da_interface_antiga(self):
        for alias in ("treinar", "avaliar", "jogar", "assistir"):
            with self.subTest(alias=alias):
                with self.assertRaises(SystemExit) as erro:
                    main.main([alias])

                self.assertNotEqual(erro.exception.code, 0)


# =====================================================================
# environment.py
# =====================================================================
class EnvironmentTest(unittest.TestCase):
    @patch("environment.pong_v3.env")
    def test_configura_probabilidade_de_repetir_acao(self, criar_env):
        env = Mock()
        criar_env.return_value = env

        resultado = criar_ambiente(probabilidade_acao_repetida=0.25)

        self.assertIs(resultado, env)
        env.unwrapped.ale.setFloat.assert_called_once_with(
            b"repeat_action_probability", 0.25
        )


# =====================================================================
# agents.py
# =====================================================================
class AgentsTest(unittest.TestCase):
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


# =====================================================================
# rl_agent.py
# =====================================================================
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

    def test_taxa_de_aprendizado_diminui_com_as_visitas(self):
        agente = AgenteRL()
        acao = ACOES_VALIDAS[0]

        agente.treinar_passo(1, acao, 10.0, 2, terminou=True)
        primeiro_valor = float(agente.tabela_q[1, 0])
        agente.treinar_passo(1, acao, 10.0, 2, terminou=True)

        segunda_taxa = TAXA_APRENDIZADO_FINAL + (
            TAXA_APRENDIZADO - TAXA_APRENDIZADO_FINAL
        ) / np.sqrt(2)
        esperado = primeiro_valor + segunda_taxa * (10.0 - primeiro_valor)
        self.assertAlmostEqual(float(agente.tabela_q[1, 0]), esperado, places=6)
        self.assertEqual(int(agente.visitas[1, 0]), 2)


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


# =====================================================================
# evaluate.py
# =====================================================================
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
        observacao[13] = pontos_esq
        observacao[14] = pontos_dir
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

        self.assertEqual(agente_dir.resetar_estado.call_count, 2)

    def test_informa_quando_partida_foi_truncada(self):
        turnos = [(NOME_RL, 0, 3, 4, False, True)]
        env = AmbienteFalso(turnos)
        agente_esq = Mock()
        agente_dir = Mock()

        resultado = jogar_partida(env, agente_esq, agente_dir, seed=123)

        self.assertEqual(resultado, (3, 4, True))


# =====================================================================
# train.py
# =====================================================================
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
        from train_rl import avaliar_para_selecao

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


# =====================================================================
# train_genetic.py
# =====================================================================
class AvaliarRallyTest(unittest.TestCase):
    def _observacao(self, bola_x, bola_y=100, jogador_y=100):
        observacao = np.zeros(128, dtype=np.uint8)
        observacao[49] = bola_x
        observacao[50] = jogador_y
        observacao[54] = bola_y
        return observacao

    def _ambiente_com_observacoes_genetico(self, observacoes):
        class AmbienteSequencial:
            def __init__(self, observacoes):
                self.turnos = [
                    (nome, observacao)
                    for observacao in observacoes
                    for nome in ("second_0", "first_0")
                ]
                self.turno_atual = None

            def reset(self, seed=None):
                pass

            def agent_iter(self):
                for turno in self.turnos:
                    self.turno_atual = turno
                    yield turno[0]

            def last(self):
                return self.turno_atual[1], 0, False, False, {}

            def step(self, acao):
                pass

        return AmbienteSequencial(observacoes)

    def test_reset_recebe_seed_explicita(self):
        import train_genetic

        env = Mock()
        env.agent_iter.return_value = iter([])  # rally "vazio" só para o teste de reset
        env.last.return_value = ([0] * 128, 0, True, False, {})

        agente = Mock()
        train_genetic.avaliar_rally(agente, None, "heuristico", env, seed=123)

        env.reset.assert_called_once_with(seed=123)

    def test_seeds_diferentes_nao_repetem_chamada_identica(self):
        import train_genetic

        env = Mock()
        env.agent_iter.return_value = iter([])
        env.last.return_value = ([0] * 128, 0, True, False, {})
        agente = Mock()

        train_genetic.avaliar_rally(agente, None, "heuristico", env, seed=1)
        train_genetic.avaliar_rally(agente, None, "heuristico", env, seed=2)

        env.reset.assert_has_calls([call(seed=1), call(seed=2)])

    def test_fitness_nao_premia_reversao_no_lado_direito(self):
        import train_genetic

        env = self._ambiente_com_observacoes_genetico(
            [
                self._observacao(0, 0),
                self._observacao(100),
                self._observacao(101),
                self._observacao(100),
            ]
        )
        agente = Mock()
        agente.escolher_acao.return_value = 1

        fitness = train_genetic.avaliar_rally(
            agente, None, "heuristico", env, seed=123
        )

        self.assertEqual(fitness, 0.0)

    def test_fitness_premia_reversao_no_lado_esquerdo(self):
        import train_genetic

        env = self._ambiente_com_observacoes_genetico(
            [
                self._observacao(0, 0),
                self._observacao(100),
                self._observacao(99),
                self._observacao(100),
            ]
        )
        agente = Mock()
        agente.escolher_acao.return_value = 1

        fitness = train_genetic.avaliar_rally(
            agente, None, "heuristico", env, seed=123
        )

        self.assertEqual(fitness, 5.0)


class EvalAgenteTest(unittest.TestCase):
    @patch("train_genetic.avaliar_rally")
    @patch("train_genetic.criar_ambiente")
    def test_eval_agente_usa_multiplos_rallies_com_seeds_distintas(self, criar_env, avaliar_rally):
        import train_genetic

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
        self.assertAlmostEqual(resultado[0], media_esperada)
        env.close.assert_called_once_with()

    @patch("train_genetic.avaliar_rally")
    @patch("train_genetic.criar_ambiente")
    def test_eval_agente_retorna_tupla_para_deap(self, criar_env, avaliar_rally):
        import train_genetic

        criar_env.return_value = Mock()
        avaliar_rally.return_value = 0.0

        resultado = train_genetic.eval_agente([0] * 12288, n_rallies=1)

        self.assertIsInstance(resultado, tuple)
        self.assertEqual(len(resultado), 1)


class ToolboxConfigTest(unittest.TestCase):
    def test_mutacao_usa_taxa_baixa_por_bit(self):
        import train_genetic

        mutate_func = train_genetic.toolbox.mutate
        indpb = mutate_func.keywords.get("indpb")
        self.assertIsNotNone(indpb)
        self.assertLess(indpb, 0.001)

    def test_selecao_e_roleta_com_fitness_negativo(self):
        import train_genetic

        select_func = train_genetic.toolbox.select
        nome = getattr(select_func, "func", select_func).__name__
        self.assertEqual(nome, "sel_roleta")

        populacao = []
        for fitness in (-25.0, 0.0, 25.0):
            individuo = train_genetic.creator.Individual([0] * 12288)
            individuo.fitness.values = (fitness,)
            populacao.append(individuo)

        with patch("train_genetic.random.random", return_value=0.99):
            selecionado = select_func(populacao, 1)[0]

        self.assertIs(selecionado, populacao[2])


if __name__ == "__main__":
    unittest.main()
