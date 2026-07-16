import sys
import random
import numpy as np
from deap import base, creator, tools, algorithms

from genetic_agent import AgenteGenetico
from environment import criar_ambiente
import heuristic_agent
from config import (
    RECOMPENSA_PONTO, PUNICAO_PONTO_SOFRIDO, RECOMPENSA_REBATIDA, RECOMPENSA_APROXIMACAO
)

IDX_BOLA_X = 49
IDX_BOLA_Y = 54
IDX_OPONENTE_Y = 50 # Raquete Direita (Genético)

def avaliar_rally(agente, oponente, oponente_tipo, env):
    """
    Avalia o agente genético em 1 rally contra o oponente especificado.
    Retorna o fitness_total do rally (usando reward shaping copiado de train.py).
    """
    env.reset()
    if oponente_tipo == "rl" and oponente is not None:
        oponente.resetar_estado()

    fitness_total = 0.0
    bola_x_anterior = None
    direcao_x_anterior = 0
    distancia_anterior = None
    
    for nome_agente in env.agent_iter():
        observacao, recompensa, terminou, truncado, _ = env.last()

        if nome_agente == "second_0": # Genético (Direita)
            bola_x = int(observacao[IDX_BOLA_X])
            bola_y = int(observacao[IDX_BOLA_Y])
            jogador_y = int(observacao[IDX_OPONENTE_Y])
            
            recompensa_shape = 0.0
            rebateu = False

            if recompensa != 0 or bola_x == 0 or bola_y == 0:
                bola_x_anterior = None
                direcao_x_anterior = 0
                distancia_anterior = None
            elif bola_x_anterior is None:
                bola_x_anterior = bola_x
            else:
                direcao_x = bola_x - bola_x_anterior
                # O Genético está na direita, então a bola se aproxima dele quando direcao_x > 0
                # Ele rebate se a bola estava indo para a direita (>0) e agora vai para a esquerda (<0)
                rebateu = direcao_x_anterior > 0 and direcao_x < 0
                if direcao_x != 0:
                    direcao_x_anterior = direcao_x
                bola_x_anterior = bola_x

            if recompensa > 0:
                recompensa_shape = RECOMPENSA_PONTO
            elif recompensa < 0:
                recompensa_shape = PUNICAO_PONTO_SOFRIDO
            elif rebateu:
                recompensa_shape = RECOMPENSA_REBATIDA

            # Distância de aproximação quando a bola vem na direção da raquete (direita, >0)
            if direcao_x_anterior > 0:
                distancia = abs(bola_y - jogador_y)
                if distancia_anterior is not None:
                    progresso = distancia_anterior - distancia
                    recompensa_shape += RECOMPENSA_APROXIMACAO * progresso
                distancia_anterior = distancia
            else:
                distancia_anterior = None

            fitness_total += recompensa_shape
            
            if terminou or truncado:
                acao = None
            else:
                acao = agente.escolher_acao(observacao)
        else: # Oponente (Esquerda, first_0)
            if terminou or truncado:
                acao = None
            else:
                if oponente_tipo == "rl" and oponente is not None:
                    acao = oponente.escolher_acao(observacao, explorar=False)
                else:
                    acao = heuristic_agent.escolher_acao(observacao, agente_id=nome_agente)

        env.step(acao)

        # Fim do rally
        if recompensa != 0:
            break
            
    return fitness_total

def eval_agente(individuo):
    agente = AgenteGenetico(individuo)
    env = criar_ambiente()
    
    # Avalia apenas contra o Agente Heurístico (Baseline)
    fitness_heuristico = avaliar_rally(agente, None, "heuristico", env)
    
    env.close()
    
    # Ajustar para Roleta: garantir que seja > 0 (Shift constante de segurança)
    # A roleta não aceita negativos. Como é 1 rally, usamos +1000.0.
    return (fitness_heuristico + 1000.0,)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 12288)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_agente)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01) # Mutação FlipBit
toolbox.register("select", tools.selRoulette) # Seleção Roleta

def treinar(pop_size=20, n_gen=10):
    random.seed(42)
    np.random.seed(42)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1) # Elitismo (HallOfFame)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("Iniciando Treinamento do Agente Genético...")
    # Executamos o algoritmo EA simple
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    # Opcional: salvar o melhor indivíduo
    melhor_ind = hof[0]
    np.save("melhor_cromossomo.npy", np.array(melhor_ind, dtype=np.uint8))
    print("Treinamento finalizado. Melhor fitness absoluto (sem offset):", melhor_ind.fitness.values[0] - 1000.0)
    
if __name__ == "__main__":
    treinar()
