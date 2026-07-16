# Treina o agente de Q-learning tabular praticando devolucoes.

import sys
import random
from collections import deque

import numpy as np

from environment import criar_ambiente
from evaluate import jogar_partida
from rl_agent import AgenteRL
import heuristic_agent
from reward_shaping import RastreadorRecompensa
from checkpoints import caminho_checkpoint, checkpoint_mais_recente
from config import (
    TREINO_EPISODIOS,
    EPSILON_INICIAL,
    EPSILON_FINAL,
    EPSILON_DECAIMENTO,
    CAMINHO_MELHOR_CHECKPOINT,
    DIRETORIO_CHECKPOINTS,
    INTERVALO_VALIDACAO,
    PARTIDAS_VALIDACAO,
    PROBABILIDADE_ACAO_REPETIDA_AVALIACAO,
    RENDER_TREINO,
    RECOMPENSA_APROXIMACAO,
    RECOMPENSA_REBATIDA,
    RECOMPENSA_PONTO,
    PUNICAO_PONTO_SOFRIDO,
    SEED,
    SEED_VALIDACAO,
)

IDX_BOLA_X = 49
IDX_BOLA_Y = 54
IDX_JOGADOR_Y = 51


def avaliar_para_selecao(agente_rl):
    env = criar_ambiente(
        probabilidade_acao_repetida=PROBABILIDADE_ACAO_REPETIDA_AVALIACAO
    )
    margens = []
    pontos = []
    vitorias = 0
    try:
        for partida in range(PARTIDAS_VALIDACAO):
            pontos_rl, pontos_oponente, truncada = jogar_partida(
                env,
                agente_rl,
                seed=SEED_VALIDACAO + partida,
            )
            if truncada:
                continue
            margens.append(pontos_rl - pontos_oponente)
            pontos.append(pontos_rl)
            vitorias += int(pontos_rl > pontos_oponente)
    finally:
        env.close()

    if not margens:
        raise RuntimeError("todas as partidas de validação foram truncadas")

    return {
        "margem_media": sum(margens) / len(margens),
        "pontos_medios": sum(pontos) / len(pontos),
        "taxa_vitoria": vitorias / len(margens),
    }

def treinar(render=RENDER_TREINO):
    random.seed(SEED)
    np.random.seed(SEED)

    render_mode = "human" if render else None
    env = criar_ambiente(render_mode=render_mode)
    agente_rl = AgenteRL()
    ultimo_checkpoint = checkpoint_mais_recente(DIRETORIO_CHECKPOINTS)
    if ultimo_checkpoint is None:
        episodio_inicial = 1
        agente_rl.epsilon = EPSILON_INICIAL
    else:
        ultimo_episodio = agente_rl.carregar(ultimo_checkpoint)
        episodio_inicial = ultimo_episodio + 1
        agente_rl.epsilon = max(
            EPSILON_FINAL,
            EPSILON_INICIAL * EPSILON_DECAIMENTO ** ultimo_episodio,
        )

    recompensas_recentes = deque(maxlen=100)
    rebatidas_recentes = deque(maxlen=100)
    vitorias_recentes = deque(maxlen=100)
    melhor_margem = float("-inf")
    if ultimo_checkpoint is not None:
        try:
            agente_melhor = AgenteRL()
            agente_melhor.carregar(CAMINHO_MELHOR_CHECKPOINT)
        except FileNotFoundError:
            pass
        else:
            melhor_margem = avaliar_para_selecao(agente_melhor)["margem_media"]

    NOME_RL = "first_0"
    NOME_OPONENTE_TREINO = "second_0"

    print("Iniciando treinamento do agente de RL...")
    print(f"Estados da tabela Q: {agente_rl.tabela_q.shape[0]}")
    print(f"Episódios: {TREINO_EPISODIOS}\n")
    if ultimo_checkpoint is not None:
        print(
            f"Retomando do episódio {ultimo_episodio}, checkpoint "
            f"'{ultimo_checkpoint}'.\n"
        )

    for episodio in range(episodio_inicial, TREINO_EPISODIOS + 1):
        env.reset(seed=SEED + episodio)
        agente_rl.resetar_estado()
        recompensa_total = 0
        total_rebatidas = 0
        pontos_feitos = 0
        pontos_sofridos = 0
        ultimo_estado_rl = {}
        ultima_acao_rl = {}
        rastreador = RastreadorRecompensa(NOME_RL)
        rebateu_no_rally = False

        for agente in env.agent_iter():
            observacao, recompensa, terminou, truncado, _ = env.last()

            if agente == NOME_RL:
                fim_do_rally = recompensa != 0
                if recompensa != 0:
                    agente_rl.resetar_estado()
                estado_rl = agente_rl.extrair_estado(observacao)
                bola_x = int(observacao[IDX_BOLA_X])
                bola_y = int(observacao[IDX_BOLA_Y])
                jogador_y = int(observacao[IDX_JOGADOR_Y])
                
                recompensa_treino, rebateu = rastreador.atualizar_recompensa(
                    observacao, recompensa, bola_x, bola_y, jogador_y
                )

                if recompensa > 0:
                    pontos_feitos += 1
                    rebateu_no_rally = False
                elif recompensa < 0:
                    pontos_sofridos += 1
                    rebateu_no_rally = False
                elif rebateu:
                    total_rebatidas += 1
                    rebateu_no_rally = True

                # Aprende com o resultado do passo anterior
                if agente in ultimo_estado_rl:
                    agente_rl.treinar_passo(
                        ultimo_estado_rl[agente],
                        ultima_acao_rl[agente],
                        recompensa_treino,
                        estado_rl,
                        terminou or truncado or fim_do_rally,
                    )
                recompensa_total += recompensa_treino

                # Cada rally e um episodio de treino: o primeiro ponto decide.
                if fim_do_rally:
                    break

                if terminou or truncado:
                    acao = None
                else:
                    acao = agente_rl.escolher_acao(estado_rl, explorar=True)
                    ultimo_estado_rl[agente] = estado_rl
                    ultima_acao_rl[agente] = acao

            elif agente == NOME_OPONENTE_TREINO:
                if terminou or truncado:
                    acao = None
                else:
                    acao = heuristic_agent.escolher_acao(observacao)

            env.step(acao)

        # Decaimento do epsilon (exploração)
        agente_rl.epsilon = max(EPSILON_FINAL, agente_rl.epsilon * EPSILON_DECAIMENTO)
        agente_rl.salvar(
            caminho_checkpoint(DIRETORIO_CHECKPOINTS, episodio),
            episodio=episodio,
        )

        if episodio % INTERVALO_VALIDACAO == 0 or episodio == TREINO_EPISODIOS:
            resultado_validacao = avaliar_para_selecao(agente_rl)
            print(
                f"Validação {episodio}: margem média "
                f"{resultado_validacao['margem_media']:.2f} | "
                f"pontos RL {resultado_validacao['pontos_medios']:.2f} | "
                f"vitórias {resultado_validacao['taxa_vitoria']:.1%}"
            )
            if resultado_validacao["margem_media"] > melhor_margem:
                melhor_margem = resultado_validacao["margem_media"]
                agente_rl.salvar(CAMINHO_MELHOR_CHECKPOINT, episodio=episodio)
                print(f"Novo melhor checkpoint: '{CAMINHO_MELHOR_CHECKPOINT}'.")

        recompensas_recentes.append(recompensa_total)
        rebatidas_recentes.append(total_rebatidas)
        vitorias_recentes.append(pontos_feitos > pontos_sofridos)

        if episodio % 100 == 0 or episodio == episodio_inicial:
            tamanho_janela = len(vitorias_recentes)
            print(
                f"Episódio {episodio}/{TREINO_EPISODIOS} | "
                f"Últimos {tamanho_janela}: "
                f"vitórias {sum(vitorias_recentes) / tamanho_janela:.1%} | "
                f"recompensa média {sum(recompensas_recentes) / tamanho_janela:.1f} | "
                f"rebatidas médias {sum(rebatidas_recentes) / tamanho_janela:.2f} | "
                f"Epsilon: {agente_rl.epsilon:.3f}"
            )

    env.close()
    if episodio_inicial > TREINO_EPISODIOS:
        print(
            f"Treinamento já concluído até o episódio {ultimo_episodio}. "
            "Nenhum episódio novo executado."
        )
    else:
        print(
            "\nTreinamento concluído. Último checkpoint salvo em "
            f"'{caminho_checkpoint(DIRETORIO_CHECKPOINTS, TREINO_EPISODIOS)}'."
        )

if __name__ == "__main__":
    treinar(render="--render" in sys.argv)
