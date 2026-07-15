# Treina o agente de RL (Q-Learning com MLP) jogando contra o agente heurístico.

import sys
import random
import numpy as np

from environment import criar_ambiente
from rl_agent import AgenteRL
import heuristic_agent
from config import (
    TREINO_EPISODIOS,
    EPSILON_INICIAL,
    EPSILON_FINAL,
    EPSILON_DECAIMENTO,
    CAMINHO_MODELO,
    RENDER_TREINO,
    SEED,
)

def treinar(render=RENDER_TREINO):
    random.seed(SEED)
    np.random.seed(SEED)

    render_mode = "human" if render else None
    env = criar_ambiente(render_mode=render_mode)
    agente_rl = AgenteRL()
    agente_rl.epsilon = EPSILON_INICIAL

    NOME_RL = "first_0"
    NOME_HEURISTICO = "second_0"

    print("Iniciando treinamento do agente de RL...")
    print(f"Episódios: {TREINO_EPISODIOS}\n")

    for episodio in range(1, TREINO_EPISODIOS + 1):
        env.reset(seed=SEED + episodio)
        recompensa_total = 0
        ultimo_estado_rl = {}
        ultima_acao_rl = {}

        for agente in env.agent_iter():
            observacao, recompensa, terminou, truncado, _ = env.last()

            if agente == NOME_RL:
                # Aprende com o resultado do passo anterior
                if agente in ultimo_estado_rl:
                    agente_rl.treinar_passo(
                        ultimo_estado_rl[agente],
                        ultima_acao_rl[agente],
                        recompensa,
                        observacao,
                        terminou or truncado,
                    )
                recompensa_total += recompensa

                if terminou or truncado:
                    acao = None
                else:
                    acao = agente_rl.escolher_acao(observacao, explorar=True)
                    ultimo_estado_rl[agente] = observacao
                    ultima_acao_rl[agente] = acao

            elif agente == NOME_HEURISTICO:
                if terminou or truncado:
                    acao = None
                else:
                    acao = heuristic_agent.escolher_acao(observacao)

            env.step(acao)

        # Decaimento do epsilon (exploração)
        agente_rl.epsilon = max(EPSILON_FINAL, agente_rl.epsilon * EPSILON_DECAIMENTO)

        if episodio % 10 == 0 or episodio == 1:
            print(
                f"Episódio {episodio}/{TREINO_EPISODIOS} | "
                f"Recompensa RL: {recompensa_total:.1f} | "
                f"Epsilon: {agente_rl.epsilon:.3f}"
            )

    env.close()
    agente_rl.salvar(CAMINHO_MODELO)
    print(f"\nTreinamento concluído. Modelo salvo em '{CAMINHO_MODELO}'.")

if __name__ == "__main__":
    treinar(render="--render" in sys.argv)