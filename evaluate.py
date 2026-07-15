# Compara o agente heurístico com o agente de RL treinado, jogando várias partidas entre eles e reportando estatísticas finais.

import sys

from environment import criar_ambiente
from rl_agent import AgenteRL
import heuristic_agent
from config import PARTIDAS_AVALIACAO, CAMINHO_MODELO, RENDER_AVALIACAO, SEED

NOME_RL = "first_0"
NOME_HEURISTICO = "second_0"

def jogar_partida(env, agente_rl, seed):
    # Joga uma partida completa e retorna a recompensa total de cada agente
    env.reset(seed=seed)
    recompensa_rl = 0
    recompensa_heuristico = 0

    for agente in env.agent_iter():
        observacao, recompensa, terminou, truncado, _ = env.last()

        if agente == NOME_RL:
            recompensa_rl += recompensa
            acao = None if (terminou or truncado) else agente_rl.escolher_acao(observacao, explorar=False)
        else:
            recompensa_heuristico += recompensa
            acao = None if (terminou or truncado) else heuristic_agent.escolher_acao(observacao)

        env.step(acao)

    return recompensa_rl, recompensa_heuristico

def avaliar(render=RENDER_AVALIACAO):
    render_mode = "human" if render else None
    env = criar_ambiente(render_mode=render_mode)

    agente_rl = AgenteRL()
    agente_rl.carregar(CAMINHO_MODELO)

    vitorias_heuristico = 0
    vitorias_rl = 0
    empates = 0
    pontuacoes_rl = []
    pontuacoes_heuristico = []

    print(f"Avaliando agentes em {PARTIDAS_AVALIACAO} partidas...\n")

    for partida in range(1, PARTIDAS_AVALIACAO + 1):
        pontos_rl, pontos_heuristico = jogar_partida(env, agente_rl, seed=SEED + partida)
        pontuacoes_rl.append(pontos_rl)
        pontuacoes_heuristico.append(pontos_heuristico)

        if pontos_rl > pontos_heuristico:
            vitorias_rl += 1
            resultado = "RL venceu"
        elif pontos_heuristico > pontos_rl:
            vitorias_heuristico += 1
            resultado = "Heurístico venceu"
        else:
            empates += 1
            resultado = "Empate"

        print(f"Partida {partida}/{PARTIDAS_AVALIACAO}: {resultado} "
              f"(RL {pontos_rl:.0f} x {pontos_heuristico:.0f} Heurístico)")

    env.close()
    imprimir_resultados(vitorias_heuristico, vitorias_rl, empates,
                         pontuacoes_heuristico, pontuacoes_rl)

def imprimir_resultados(vitorias_heuristico, vitorias_rl, empates,
                         pontuacoes_heuristico, pontuacoes_rl):
    total = vitorias_heuristico + vitorias_rl + empates
    media_heuristico = sum(pontuacoes_heuristico) / total
    media_rl = sum(pontuacoes_rl) / total

    print("\nResultados")
    print("Agente de busca heurística:")
    print(f"  Vitórias: {vitorias_heuristico}")
    print("Agente de aprendizado por reforço:")
    print(f"  Vitórias: {vitorias_rl}")
    print(f"Empates: {empates}")
    print("\nPontuação média:")
    print(f"  Heurístico: {media_heuristico:.2f}")
    print(f"  RL: {media_rl:.2f}")
    print("\nTaxa de vitória:")
    print(f"  Heurístico: {vitorias_heuristico / total:.1%}")
    print(f"  RL: {vitorias_rl / total:.1%}")

if __name__ == "__main__":
    avaliar(render="--render" in sys.argv)