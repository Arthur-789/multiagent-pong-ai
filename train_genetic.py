import sys
import random
import numpy as np
from deap import base, creator, tools, algorithms

from genetic_agent import AgenteGenetico
from environment import criar_ambiente
from rl_agent import AgenteRL
import heuristic_agent
from reward_shaping import RastreadorRecompensa

IDX_BOLA_X = 49
IDX_BOLA_Y = 54
IDX_OPONENTE_Y = 50 # Raquete Direita (Genético)

def eval_agente(individuo):
    agente = AgenteGenetico(individuo)
    
    # Sorteia oponente base: RL ou Heurístico
    oponente_tipo = random.choice(["rl", "heuristico"])
    
    if oponente_tipo == "rl":
        oponente = AgenteRL()
        try:
            oponente.carregar("checkpoints_tabular/melhor_qtable.npz")
        except:
            pass
    else:
        oponente = None # heuristic_agent não tem estado

    env = criar_ambiente()
    env.reset()
    if oponente_tipo == "rl":
        oponente.resetar_estado()

    fitness_total = 0.0
    rastreador = RastreadorRecompensa("second_0")
    
    # 1 rally
    for nome_agente in env.agent_iter():
        observacao, recompensa, terminou, truncado, _ = env.last()

        if nome_agente == "second_0": # Genético
            bola_x = int(observacao[IDX_BOLA_X])
            bola_y = int(observacao[IDX_BOLA_Y])
            jogador_y = int(observacao[IDX_OPONENTE_Y])
            
            recompensa_shape, _ = rastreador.atualizar_recompensa(
                observacao, recompensa, bola_x, bola_y, jogador_y
            )

            fitness_total += recompensa_shape
            
            if terminou or truncado:
                acao = None
            else:
                acao = agente.escolher_acao(observacao)
        else: # Oponente (first_0)
            if terminou or truncado:
                acao = None
            else:
                if oponente_tipo == "rl":
                    acao = oponente.escolher_acao(observacao, explorar=False)
                else:
                    acao = heuristic_agent.escolher_acao(observacao, agente_id=nome_agente)

        env.step(acao)

        if recompensa != 0:
            break
            
    env.close()
    
    return (fitness_total,)

# Wrapper seguro para roleta que lida com fitness negativos
def selRouletteSafe(individuals, k):
    min_fit = min(ind.fitness.values[0] for ind in individuals)
    orig_fitness = [ind.fitness.values[0] for ind in individuals]
    
    shift = 0.0
    if min_fit <= 0:
        shift = abs(min_fit) + 1.0
        
    for ind in individuals:
        ind.fitness.values = (ind.fitness.values[0] + shift,)
        
    chosen = tools.selRoulette(individuals, k)
    
    for i, ind in enumerate(individuals):
        ind.fitness.values = (orig_fitness[i],)
        
    return chosen

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 12288)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_agente)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01) # Mutação FlipBit
toolbox.register("select", selRouletteSafe) # Seleção Roleta segura

def treinar(pop_size=20, n_gen=10):
    random.seed(42)
    np.random.seed(42)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1) # Elitismo
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("Iniciando Treinamento do Agente Genético...")
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, 
                                   stats=stats, halloffame=hof, verbose=True)
    
    melhor_ind = hof[0]
    np.save("melhor_cromossomo.npy", np.array(melhor_ind, dtype=np.uint8))
    print("Treinamento finalizado. Melhor fitness absoluto:", melhor_ind.fitness.values[0])
    
if __name__ == "__main__":
    treinar()
