# Compara o agente heurístico com o agente de RL treinado, jogando várias partidas entre eles e reportando estatísticas finais.

import sys

import pygame

from environment import criar_ambiente
from jogar import resolver_modelo
from rl_agent import AgenteRL
import heuristic_agent
from config import (
    PARTIDAS_AVALIACAO,
    PROBABILIDADE_ACAO_REPETIDA_AVALIACAO,
    RENDER_AVALIACAO,
    FPS_JOGO,
    SEED,
)

NOME_RL = "first_0"
NOME_HEURISTICO = "second_0"
IDX_PLACAR_RL = 14
IDX_PLACAR_HEURISTICO = 13

def jogar_partida(env, agente_rl, seed, relogio=None):
    # Joga uma partida completa e retorna o placar oficial e se houve truncamento.
    env.reset(seed=seed)
    agente_rl.resetar_estado()
    pontos_rl = 0
    pontos_heuristico = 0
    partida_truncada = False

    for agente in env.agent_iter():
        observacao, recompensa, terminou, truncado, _ = env.last()
        pontos_rl = int(observacao[IDX_PLACAR_RL])
        pontos_heuristico = int(observacao[IDX_PLACAR_HEURISTICO])
        partida_truncada = partida_truncada or truncado

        if agente == NOME_RL:
            if recompensa != 0:
                agente_rl.resetar_estado()
            acao = None if (terminou or truncado) else agente_rl.escolher_acao(observacao, explorar=False)
        else:
            acao = None if (terminou or truncado) else heuristic_agent.escolher_acao(observacao, agente_id=agente)

        env.step(acao)
        if relogio is not None and agente == NOME_HEURISTICO:
            relogio.tick(FPS_JOGO)

    return pontos_rl, pontos_heuristico, partida_truncada

def avaliar(modelo=None, render=RENDER_AVALIACAO):
    render_mode = "human" if render else None
    env = criar_ambiente(
        render_mode=render_mode,
        probabilidade_acao_repetida=PROBABILIDADE_ACAO_REPETIDA_AVALIACAO,
    )
    relogio = pygame.time.Clock() if render else None

    agente_rl = AgenteRL()
    caminho_modelo = resolver_modelo(modelo)
    agente_rl.carregar(caminho_modelo)

    vitorias_heuristico = 0
    vitorias_rl = 0
    empates = 0
    truncamentos = 0
    pontuacoes_rl = []
    pontuacoes_heuristico = []

    print(f"Modelo: {caminho_modelo}")
    print(f"Avaliando agentes em {PARTIDAS_AVALIACAO} partidas...\n")

    for partida in range(1, PARTIDAS_AVALIACAO + 1):
        pontos_rl, pontos_heuristico, partida_truncada = jogar_partida(
            env, agente_rl, seed=SEED + partida, relogio=relogio
        )

        if partida_truncada:
            truncamentos += 1
            resultado = "Truncada"
        elif pontos_rl > pontos_heuristico:
            vitorias_rl += 1
            resultado = "RL venceu"
        elif pontos_heuristico > pontos_rl:
            vitorias_heuristico += 1
            resultado = "Heurístico venceu"
        else:
            empates += 1
            resultado = "Empate"

        if not partida_truncada:
            pontuacoes_rl.append(pontos_rl)
            pontuacoes_heuristico.append(pontos_heuristico)

        print(f"Partida {partida}/{PARTIDAS_AVALIACAO}: {resultado} "
              f"(RL {pontos_rl:.0f} x {pontos_heuristico:.0f} Heurístico)")

    env.close()
    imprimir_resultados(vitorias_heuristico, vitorias_rl, empates, truncamentos,
                         pontuacoes_heuristico, pontuacoes_rl)

def imprimir_resultados(vitorias_heuristico, vitorias_rl, empates, truncamentos,
                         pontuacoes_heuristico, pontuacoes_rl):
    total = vitorias_heuristico + vitorias_rl + empates
    if total == 0:
        print(f"\nNenhuma partida completa. Truncamentos: {truncamentos}")
        return
    media_heuristico = sum(pontuacoes_heuristico) / total
    media_rl = sum(pontuacoes_rl) / total

    print("\nResultados")
    print("Agente de busca heurística:")
    print(f"  Vitórias: {vitorias_heuristico}")
    print("Agente de aprendizado por reforço:")
    print(f"  Vitórias: {vitorias_rl}")
    print(f"Empates: {empates}")
    print(f"Truncamentos: {truncamentos}")
    print("\nPontuação média:")
    print(f"  Heurístico: {media_heuristico:.2f}")
    print(f"  RL: {media_rl:.2f}")
    print("\nTaxa de vitória:")
    print(f"  Heurístico: {vitorias_heuristico / total:.1%}")
    print(f"  RL: {vitorias_rl / total:.1%}")

if __name__ == "__main__":
    modelo = next((arg for arg in sys.argv[1:] if not arg.startswith("--")), None)
    avaliar(modelo=modelo, render="--render" in sys.argv)
