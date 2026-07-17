import sys
import argparse
import random
import numpy as np
from deap import base, creator, tools

from genetic_agent import AgenteGenetico
from environment import criar_ambiente
import heuristic_agent
from config import (
    RECOMPENSA_PONTO, PUNICAO_PONTO_SOFRIDO, RECOMPENSA_REBATIDA, RECOMPENSA_APROXIMACAO,
    SEED,
    GA_POP_SIZE, GA_N_GEN, GA_CXPB, GA_MUTPB, GA_RALLIES_POR_AVALIACAO, GA_CAMINHO_CHECKPOINT,
)

IDX_BOLA_X = 49
IDX_BOLA_Y = 54
IDX_OPONENTE_Y = 50

def avaliar_rally(agente, oponente, oponente_tipo, env, seed=None):
    """
    Avalia o agente genético em 1 rally contra o oponente especificado.
    Retorna o fitness_total do rally (usando reward shaping copiado de train.py).
    """
    env.reset(seed=seed)
    if oponente_tipo == "rl" and oponente is not None:
        oponente.resetar_estado()

    fitness_total = 0.0
    bola_x_anterior = None
    direcao_x_anterior = 0
    distancia_anterior = None
    
    for nome_agente in env.agent_iter():
        observacao, recompensa, terminou, truncado, _ = env.last()

        if nome_agente == "second_0": # Genético (Esquerda)
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
                # O Genético (second_0) fica no lado esquerdo da tela, então a bola aproxima dele quando direcao_x > 0
                # Ele rebate se a bola estava se aproximando (>0) e passa a se afastar (<0)
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

            # Distância de aproximação quando a bola vem na direção da raquete (direcao_x > 0)
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
        else: # Oponente (Direita, first_0)
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

def eval_agente(individuo, n_rallies=GA_RALLIES_POR_AVALIACAO, seed_base=SEED):
    """
    Avalia o individuo em varios rallies com seeds diferentes e retorna a media.
    Isso evita tanto o determinismo total quanto o ruido excessivo de uma unica seed
    aleatoria, que dificultaria comparar individuos de forma justa entre geracoes.
    """
    agente = AgenteGenetico(individuo)
    env = criar_ambiente()

    fitness_total = 0.0
    for i in range(n_rallies):
        seed = seed_base + i
        fitness_total += avaliar_rally(agente, None, "heuristico", env, seed=seed)

    env.close()

    fitness_medio = fitness_total / n_rallies

    # Ajustar para garantir que seja > 0 (Shift constante de segurança).
    # Necessário pois a seleção por roleta lida mal com fitness negativo.
    return (fitness_medio + 1000.0,)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 12288)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_agente)
toolbox.register("mate", tools.cxTwoPoint)
# indpb=0.01 em 12288 bits gera ~123 flips por individuo mutado
# flipar o bit mais significativo de um peso muda seu valor quase de -1.0 para +1.0 de uma vez).
# 1/12288 mantem em media ~1 bit alterado por individuo mutado, uma perturbacao muito mais suave.
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / 12288)
# Selecao por torneio mantem pressao seletiva mesmo quando o fitness fica quase plano (ex.: varios individuos empatados em 1000).
# A Roleta praticamente vira selecao uniforme quando a variancia do fitness e pequena, perdendo elitismo.
toolbox.register("select", tools.selTournament, tournsize=3)

def treinar(pop_size=GA_POP_SIZE, n_gen=GA_N_GEN, cxpb=GA_CXPB, mutpb=GA_MUTPB,
            checkpoint_path=GA_CAMINHO_CHECKPOINT):
    random.seed(42)
    np.random.seed(42)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)  # Elitismo (HallOfFame)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # Reporta o fitness "real" (subtraindo o offset de +1000 usado so para manter os valores positivos durante a selecao), facilitando leitura do log.
    stats.register("avg", lambda vals: np.mean(vals) - 1000.0)
    stats.register("std", np.std)
    stats.register("min", lambda vals: np.min(vals) - 1000.0)
    stats.register("max", lambda vals: np.max(vals) - 1000.0)

    print("Iniciando Treinamento do Agente Genético...")
    print(f"pop_size={pop_size} n_gen={n_gen} cxpb={cxpb} mutpb={mutpb} "
          f"rallies_por_avaliacao={GA_RALLIES_POR_AVALIACAO}\n")

    # Avalia a populacao inicial (geracao 0) manualmente para poder salvar checkpoint e permitir acompanhar evolucao desde o inicio.
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    print(logbook.stream)
    _salvar_checkpoint(hof[0], checkpoint_path)

    for gen in range(1, n_gen + 1):
        # Seleciona e clona a proxima geracao
        descendentes = toolbox.select(pop, len(pop))
        descendentes = list(map(toolbox.clone, descendentes))

        # Crossover
        for filho1, filho2 in zip(descendentes[::2], descendentes[1::2]):
            if random.random() < cxpb:
                toolbox.mate(filho1, filho2)
                del filho1.fitness.values
                del filho2.fitness.values

        # Mutação
        for mutante in descendentes:
            if random.random() < mutpb:
                toolbox.mutate(mutante)
                del mutante.fitness.values

        # Reavalia apenas indivíduos modificados
        invalidos = [ind for ind in descendentes if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalidos)
        for ind, fit in zip(invalidos, fitnesses):
            ind.fitness.values = fit

        # Elitismo explícito: garante que o melhor indivíduo já visto sobreviva mesmo se toda a nova geração regredir.
        melhor_anterior = toolbox.clone(hof[0]) if len(hof) > 0 else None
        pop[:] = descendentes
        hof.update(pop)
        if melhor_anterior is not None and melhor_anterior.fitness.values[0] > max(ind.fitness.values[0] for ind in pop):
            pior_idx = min(range(len(pop)), key=lambda i: pop[i].fitness.values[0])
            pop[pior_idx] = melhor_anterior

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalidos), **record)
        print(logbook.stream)

        _salvar_checkpoint(hof[0], checkpoint_path)

    melhor_ind = hof[0]
    print("\nTreinamento finalizado. Melhor fitness absoluto (sem offset):",
          melhor_ind.fitness.values[0] - 1000.0)
    print(f"Cromossomo salvo em '{checkpoint_path}'.")


def _salvar_checkpoint(individuo, caminho):
    np.save(caminho, np.array(individuo, dtype=np.uint8))


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Treina o agente genético do Pong via DEAP."
    )
    parser.add_argument(
        "--pop-size", type=int, default=GA_POP_SIZE,
        help=f"Tamanho da população (padrão: config.GA_POP_SIZE = {GA_POP_SIZE})",
    )
    parser.add_argument(
        "--n-gen", type=int, default=GA_N_GEN,
        help=f"Número de gerações (padrão: config.GA_N_GEN = {GA_N_GEN})",
    )
    parser.add_argument(
        "--cxpb", type=float, default=GA_CXPB,
        help=f"Probabilidade de crossover por par de indivíduos (padrão: config.GA_CXPB = {GA_CXPB})",
    )
    parser.add_argument(
        "--mutpb", type=float, default=GA_MUTPB,
        help=f"Probabilidade de mutação por indivíduo (padrão: config.GA_MUTPB = {GA_MUTPB})",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default=GA_CAMINHO_CHECKPOINT,
        help=f"Caminho onde salvar o melhor cromossomo (padrão: config.GA_CAMINHO_CHECKPOINT = '{GA_CAMINHO_CHECKPOINT}')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    treinar(
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        checkpoint_path=args.checkpoint_path,
    )
