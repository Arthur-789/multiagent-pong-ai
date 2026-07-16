# Compara dois agentes treinados (ou heurísticos), jogando várias partidas entre eles e reportando estatísticas finais.

import sys
import pygame
import numpy as np

from environment import criar_ambiente
from rl_agent import AgenteRL
from genetic_agent import AgenteGenetico
import heuristic_agent
from config import (
    PARTIDAS_AVALIACAO,
    PROBABILIDADE_ACAO_REPETIDA_AVALIACAO,
    RENDER_AVALIACAO,
    FPS_JOGO,
    SEED,
)

IDX_PLACAR_ESQ = 14
IDX_PLACAR_DIR = 13

LADO_DIREITO = "first_0"
LADO_ESQUERDO = "second_0"

PREFERENCIAS = {
    "rl": LADO_DIREITO,
    "genetico": LADO_ESQUERDO,

    "heuristico": None
}

class AgenteHeuristicoWrapper:
    def __init__(self, agente_id):
        self.agente_id = agente_id

    def resetar_estado(self):
        pass

    def escolher_acao(self, observacao):
        return heuristic_agent.escolher_acao(observacao, agente_id=self.agente_id)

class AgenteRLWrapper:
    def __init__(self, agente):
        self.agente = agente

    def resetar_estado(self):
        self.agente.resetar_estado()

    def escolher_acao(self, observacao):
        return self.agente.escolher_acao(observacao, explorar=False)

def instanciar_agente(tipo, agente_id):
    if tipo == "rl":
        agente = AgenteRL()
        agente.carregar("checkpoints_tabular/melhor_qtable.npz")
        return AgenteRLWrapper(agente)
    elif tipo == "genetico":
        cromossomo = np.load("melhor_cromossomo.npy")
        return AgenteGenetico(cromossomo)
    elif tipo == "heuristico":
        return AgenteHeuristicoWrapper(agente_id)
    else:
        raise ValueError(f"Agente desconhecido: {tipo}")

def jogar_partida(env, esq_agent, dir_agent, seed, relogio=None):
    env.reset(seed=seed)
    esq_agent.resetar_estado()
    dir_agent.resetar_estado()
    
    pontos_esq = 0
    pontos_dir = 0
    partida_truncada = False

    for nome_agente in env.agent_iter():
        observacao, recompensa, terminou, truncado, _ = env.last()
        pontos_esq = int(observacao[IDX_PLACAR_ESQ])
        pontos_dir = int(observacao[IDX_PLACAR_DIR])
        partida_truncada = partida_truncada or truncado

        if terminou or truncado:
            acao = None
        else:
            agente_atual = esq_agent if nome_agente == LADO_ESQUERDO else dir_agent
            
            if recompensa != 0:
                agente_atual.resetar_estado()
            
            acao = agente_atual.escolher_acao(observacao)

        env.step(acao)
        if relogio is not None and nome_agente == LADO_ESQUERDO:
            relogio.tick(FPS_JOGO)

    return pontos_esq, pontos_dir, partida_truncada

def alocar_lados(agente1, agente2):
    if agente1 not in PREFERENCIAS:
        raise ValueError(f"Agente desconhecido: {agente1}. Opções: rl, genetico, heuristico.")
    if agente2 not in PREFERENCIAS:
        raise ValueError(f"Agente desconhecido: {agente2}. Opções: rl, genetico, heuristico.")

    p1 = PREFERENCIAS[agente1]
    p2 = PREFERENCIAS[agente2]

    if p1 is not None and p2 is not None and p1 == p2:
        raise ValueError(f"Não é possível colocar {agente1} contra {agente2} pois ambos requerem o lado {p1}")

    if p1 == LADO_DIREITO or p2 == LADO_ESQUERDO:
        esq_type, dir_type = agente2, agente1
    else:
        esq_type, dir_type = agente1, agente2
        
    return (esq_type, instanciar_agente(esq_type, LADO_ESQUERDO),
            dir_type, instanciar_agente(dir_type, LADO_DIREITO))

def avaliar(agente1_type, agente2_type, render=RENDER_AVALIACAO):
    print(f"Alocando {agente1_type} e {agente2_type}...")
    esq_type, esq_agent, dir_type, dir_agent = alocar_lados(agente1_type, agente2_type)
    
    print(f"Esquerda ({LADO_ESQUERDO}): {esq_type}")
    print(f"Direita ({LADO_DIREITO}): {dir_type}\n")

    render_mode = "human" if render else None
    env = criar_ambiente(
        render_mode=render_mode,
        probabilidade_acao_repetida=PROBABILIDADE_ACAO_REPETIDA_AVALIACAO,
    )
    relogio = pygame.time.Clock() if render else None

    vitorias_esq = 0
    vitorias_dir = 0
    empates = 0
    truncamentos = 0
    pontuacoes_esq = []
    pontuacoes_dir = []

    print(f"Avaliando agentes em {PARTIDAS_AVALIACAO} partidas...\n")

    for partida in range(1, PARTIDAS_AVALIACAO + 1):
        pontos_esq, pontos_dir, partida_truncada = jogar_partida(
            env, esq_agent, dir_agent, seed=SEED + partida, relogio=relogio
        )

        if partida_truncada:
            truncamentos += 1
            resultado = "Truncada"
        elif pontos_esq > pontos_dir:
            vitorias_esq += 1
            resultado = f"{esq_type} (Esq) venceu"
        elif pontos_dir > pontos_esq:
            vitorias_dir += 1
            resultado = f"{dir_type} (Dir) venceu"
        else:
            empates += 1
            resultado = "Empate"

        if not partida_truncada:
            pontuacoes_esq.append(pontos_esq)
            pontuacoes_dir.append(pontos_dir)

        print(f"Partida {partida}/{PARTIDAS_AVALIACAO}: {resultado} "
              f"({esq_type} {pontos_esq:.0f} x {pontos_dir:.0f} {dir_type})")

    env.close()
    
    total = vitorias_esq + vitorias_dir + empates
    if total == 0:
        print(f"\nNenhuma partida completa. Truncamentos: {truncamentos}")
        return
        
    media_esq = sum(pontuacoes_esq) / total
    media_dir = sum(pontuacoes_dir) / total

    print("\nResultados")
    print(f"Esquerda ({esq_type}):")
    print(f"  Vitórias: {vitorias_esq}")
    print(f"  Pontuação média: {media_esq:.2f}")
    print(f"  Taxa de vitória: {vitorias_esq / total:.1%}")
    
    print(f"\nDireita ({dir_type}):")
    print(f"  Vitórias: {vitorias_dir}")
    print(f"  Pontuação média: {media_dir:.2f}")
    print(f"  Taxa de vitória: {vitorias_dir / total:.1%}")
    
    print(f"\nEmpates: {empates}")
    print(f"Truncamentos: {truncamentos}")

if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if len(args) != 2:
        print("Uso: python evaluate.py <agente1> <agente2> [--render]")
        print("Opções: rl, genetico, heuristico")
        sys.exit(1)
        
    avaliar(args[0], args[1], render="--render" in sys.argv)
