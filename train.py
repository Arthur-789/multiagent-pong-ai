# Treina o agente de RL (Q-Learning com MLP) praticando devolucoes.

import sys
import random
import numpy as np
import torch

from environment import criar_ambiente
from rl_agent import AgenteRL
from training_opponent import OponenteTreino
from config import (
    TREINO_EPISODIOS,
    EPSILON_INICIAL,
    EPSILON_FINAL,
    EPSILON_DECAIMENTO,
    CAMINHO_MODELO,
    RENDER_TREINO,
    RECOMPENSA_APROXIMACAO,
    RECOMPENSA_REBATIDA,
    RECOMPENSA_PONTO,
    PUNICAO_PONTO_SOFRIDO,
    SEED,
)

IDX_BOLA_X = 49
IDX_BOLA_Y = 54
IDX_JOGADOR_Y = 51

def treinar(render=RENDER_TREINO):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    render_mode = "human" if render else None
    env = criar_ambiente(render_mode=render_mode)
    agente_rl = AgenteRL()
    oponente_treino = OponenteTreino()
    agente_rl.epsilon = EPSILON_INICIAL

    NOME_RL = "first_0"
    NOME_OPONENTE_TREINO = "second_0"

    print("Iniciando treinamento do agente de RL...")
    print(f"Dispositivo PyTorch: {agente_rl.dispositivo}")
    print(f"Episódios: {TREINO_EPISODIOS}\n")

    for episodio in range(1, TREINO_EPISODIOS + 1):
        env.reset(seed=SEED + episodio)
        oponente_treino.resetar()
        recompensa_total = 0
        total_rebatidas = 0
        pontos_feitos = 0
        pontos_sofridos = 0
        ultimo_estado_rl = {}
        ultima_acao_rl = {}
        bola_x_anterior = None
        direcao_x_anterior = 0
        distancia_anterior = None
        rebateu_no_rally = False

        for agente in env.agent_iter():
            observacao, recompensa, terminou, truncado, _ = env.last()

            if agente == NOME_RL:
                bola_x = int(observacao[IDX_BOLA_X])
                bola_y = int(observacao[IDX_BOLA_Y])
                jogador_y = int(observacao[IDX_JOGADOR_Y])
                rebateu = False
                if recompensa != 0 or bola_x == 0 or bola_y == 0:
                    bola_x_anterior = None
                    direcao_x_anterior = 0
                    distancia_anterior = None
                elif bola_x_anterior is None:
                    bola_x_anterior = bola_x
                else:
                    direcao_x = bola_x - bola_x_anterior
                    rebateu = direcao_x_anterior > 0 and direcao_x < 0
                    if direcao_x != 0:
                        direcao_x_anterior = direcao_x
                    bola_x_anterior = bola_x

                recompensa_treino = 0.0
                if recompensa > 0:
                    pontos_feitos += 1
                    if rebateu_no_rally:
                        recompensa_treino = RECOMPENSA_PONTO
                    rebateu_no_rally = False
                elif recompensa < 0:
                    recompensa_treino = PUNICAO_PONTO_SOFRIDO
                    pontos_sofridos += 1
                    rebateu_no_rally = False
                elif rebateu:
                    recompensa_treino = RECOMPENSA_REBATIDA
                    total_rebatidas += 1
                    rebateu_no_rally = True

                if direcao_x_anterior > 0:
                    distancia = abs(bola_y - jogador_y)
                    if distancia_anterior is not None:
                        progresso = distancia_anterior - distancia
                        recompensa_treino += RECOMPENSA_APROXIMACAO * progresso
                    distancia_anterior = distancia
                else:
                    distancia_anterior = None

                # Aprende com o resultado do passo anterior
                if agente in ultimo_estado_rl:
                    agente_rl.treinar_passo(
                        ultimo_estado_rl[agente],
                        ultima_acao_rl[agente],
                        recompensa_treino,
                        observacao,
                        terminou or truncado,
                    )
                recompensa_total += recompensa_treino

                if terminou or truncado:
                    acao = None
                else:
                    acao = agente_rl.escolher_acao(observacao, explorar=True)
                    ultimo_estado_rl[agente] = observacao
                    ultima_acao_rl[agente] = acao

            elif agente == NOME_OPONENTE_TREINO:
                if terminou or truncado:
                    acao = None
                else:
                    acao = oponente_treino.escolher_acao(observacao, recompensa)

            env.step(acao)

        # Decaimento do epsilon (exploração)
        agente_rl.epsilon = max(EPSILON_FINAL, agente_rl.epsilon * EPSILON_DECAIMENTO)

        if episodio % 10 == 0 or episodio == 1:
            print(
                f"Episódio {episodio}/{TREINO_EPISODIOS} | "
                f"Recompensa treino: {recompensa_total:.1f} | "
                f"Rebatidas: {total_rebatidas} | "
                f"Placar: {pontos_feitos}x{pontos_sofridos} | "
                f"Epsilon: {agente_rl.epsilon:.3f}"
            )

    env.close()
    agente_rl.salvar(CAMINHO_MODELO)
    print(f"\nTreinamento concluído. Modelo salvo em '{CAMINHO_MODELO}'.")

if __name__ == "__main__":
    treinar(render="--render" in sys.argv)
