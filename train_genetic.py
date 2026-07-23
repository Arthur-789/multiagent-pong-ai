import random
import bisect
import numpy as np
from deap import base, creator, tools

from genetic_agent import ACOES_VALIDAS, AgenteGenetico, TAMANHO_CROMOSSOMO
from environment import criar_ambiente
import heuristic_agent
from config import (
    RECOMPENSA_PONTO, PUNICAO_PONTO_SOFRIDO, RECOMPENSA_REBATIDA, RECOMPENSA_APROXIMACAO,
    SEED,
    GA_POP_SIZE, GA_N_GEN, GA_CXPB, GA_MUTPB, GA_PARTIDAS_POR_AVALIACAO,
    GA_CAMINHO_CHECKPOINT, PROBABILIDADE_ACAO_REPETIDA_AVALIACAO,
)

IDX_BOLA_X = 49
IDX_BOLA_Y = 54
IDX_OPONENTE_Y = 50

def _preparar_diagnostico(diagnostico):
    if diagnostico is None:
        return

    diagnostico.setdefault("acoes", {acao: 0 for acao in ACOES_VALIDAS})
    for campo in (
        "rebatidas",
        "pontos_marcados",
        "pontos_sofridos",
        "fitness_ponto",
        "fitness_rebatida",
        "fitness_aproximacao",
    ):
        diagnostico.setdefault(campo, 0.0 if campo.startswith("fitness_") else 0)


def _avaliar(agente, oponente, oponente_tipo, env, seed, encerrar_apos_ponto,
             diagnostico=None):
    """
    Avalia o agente genético contra o oponente especificado.
    Retorna o Fitness acumulado com o mesmo reward shaping do treino de RL.
    """
    _preparar_diagnostico(diagnostico)
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
                # O Genético (second_0) fica no lado esquerdo da tela, então a bola
                # se aproxima dele quando direcao_x < 0 e se afasta após o rebote.
                rebateu = direcao_x_anterior < 0 and direcao_x > 0
                if direcao_x != 0:
                    direcao_x_anterior = direcao_x
                bola_x_anterior = bola_x

            if recompensa > 0:
                recompensa_shape = RECOMPENSA_PONTO
                if diagnostico is not None:
                    diagnostico["pontos_marcados"] += 1
                    diagnostico["fitness_ponto"] += recompensa_shape
            elif recompensa < 0:
                recompensa_shape = PUNICAO_PONTO_SOFRIDO
                if diagnostico is not None:
                    diagnostico["pontos_sofridos"] += 1
                    diagnostico["fitness_ponto"] += recompensa_shape
            elif rebateu:
                recompensa_shape = RECOMPENSA_REBATIDA
                if diagnostico is not None:
                    diagnostico["rebatidas"] += 1
                    diagnostico["fitness_rebatida"] += recompensa_shape

            # Distância de aproximação quando a bola vem na direção da raquete (direcao_x < 0)
            if direcao_x_anterior < 0:
                distancia = abs(bola_y - jogador_y)
                if distancia_anterior is not None:
                    progresso = distancia_anterior - distancia
                    recompensa_aproximacao = RECOMPENSA_APROXIMACAO * progresso
                    recompensa_shape += recompensa_aproximacao
                    if diagnostico is not None:
                        diagnostico["fitness_aproximacao"] += recompensa_aproximacao
                distancia_anterior = distancia
            else:
                distancia_anterior = None

            fitness_total += recompensa_shape
            
            if terminou or truncado:
                acao = None
            else:
                acao = agente.escolher_acao(observacao)
                if diagnostico is not None:
                    diagnostico["acoes"][acao] += 1
        else: # Oponente (Direita, first_0)
            if terminou or truncado:
                acao = None
            else:
                if oponente_tipo == "rl" and oponente is not None:
                    acao = oponente.escolher_acao(observacao, explorar=False)
                else:
                    acao = heuristic_agent.escolher_acao(observacao, agente_id=nome_agente)

        env.step(acao)

        # No modo rally, o resultado do ponto chega primeiro ao agente que
        # marcou. Mantemos o loop até o Genético processar a recompensa
        # correspondente, inclusive a punição por ponto sofrido.
        if encerrar_apos_ponto and nome_agente == "second_0" and recompensa != 0:
            break
            
    return fitness_total


def avaliar_rally(agente, oponente, oponente_tipo, env, seed=None, diagnostico=None):
    """Avalia um único rally e encerra ao contabilizar o ponto."""
    return _avaliar(
        agente, oponente, oponente_tipo, env, seed,
        encerrar_apos_ponto=True, diagnostico=diagnostico,
    )


def avaliar_partida(agente, oponente, oponente_tipo, env, seed=None,
                    diagnostico=None):
    """Avalia uma partida completa, preservando estados após cada ponto."""
    return _avaliar(
        agente, oponente, oponente_tipo, env, seed,
        encerrar_apos_ponto=False, diagnostico=diagnostico,
    )


def diagnosticar_individuo(individuo, n_partidas=GA_PARTIDAS_POR_AVALIACAO,
                           seed_base=SEED):
    """Resume o comportamento de um indivíduo sem alterar seu Fitness."""
    agente = AgenteGenetico(individuo)
    env = criar_ambiente(
        probabilidade_acao_repetida=PROBABILIDADE_ACAO_REPETIDA_AVALIACAO
    )
    diagnostico = {}
    fitness_total = 0.0

    for i in range(n_partidas):
        fitness_total += avaliar_partida(
            agente, None, "heuristico", env, seed=seed_base + i,
            diagnostico=diagnostico,
        )

    env.close()
    diagnostico["partidas"] = n_partidas
    diagnostico["fitness_medio"] = fitness_total / n_partidas
    return diagnostico


def _imprimir_diagnostico_melhor(individuo):
    diagnostico = diagnosticar_individuo(individuo)
    acoes = diagnostico["acoes"]
    partidas = diagnostico["partidas"]
    print(
        "  diagnostico_melhor: "
        f"fitness_medio={diagnostico['fitness_medio']:.4f}; "
        f"acoes={{FIRE: {acoes[1]}, FIRE_RIGHT: {acoes[4]}, FIRE_LEFT: {acoes[5]}}}; "
        f"rebatidas={diagnostico['rebatidas']}; "
        f"pontos={diagnostico['pontos_marcados']}/{diagnostico['pontos_sofridos']}; "
        "componentes_por_partida="
        f"(ponto={diagnostico['fitness_ponto'] / partidas:.4f}, "
        f"rebatida={diagnostico['fitness_rebatida'] / partidas:.4f}, "
        f"aproximacao={diagnostico['fitness_aproximacao'] / partidas:.4f})"
    )

def eval_agente(individuo, n_partidas=GA_PARTIDAS_POR_AVALIACAO, seed_base=SEED):
    """
    Avalia o indivíduo em partidas completas e retorna o Fitness médio.
    """
    agente = AgenteGenetico(individuo)
    env = criar_ambiente(
        probabilidade_acao_repetida=PROBABILIDADE_ACAO_REPETIDA_AVALIACAO
    )

    fitness_total = 0.0
    for i in range(n_partidas):
        seed = seed_base + i
        fitness_total += avaliar_partida(agente, None, "heuristico", env, seed=seed)

    env.close()

    fitness_medio = fitness_total / n_partidas

    # O fitness permanece real; a seleção transforma esses valores em pesos
    # positivos apenas no momento de montar a roleta.
    return (fitness_medio,)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, TAMANHO_CROMOSSOMO
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_agente)
toolbox.register("mate", tools.cxTwoPoint)
# indpb=0.01 em 6144 bits gera ~61 flips por individuo mutado
# flipar o bit mais significativo de um peso muda seu valor quase de -1.0 para +1.0 de uma vez).
# 4/6144 altera em media ~4 bits por individuo mutado, aumentando a exploracao.
toolbox.register("mutate", tools.mutFlipBit, indpb=4.0 / TAMANHO_CROMOSSOMO)
def sel_roleta(populacao, k, epsilon=1e-6):
    """Seleciona ``k`` indivíduos com reposição pela roleta."""
    if not populacao:
        return []

    menor_fitness = min(ind.fitness.values[0] for ind in populacao)
    pesos = [
        ind.fitness.values[0] - menor_fitness + epsilon
        for ind in populacao
    ]
    acumulados = list(np.cumsum(pesos))
    total = acumulados[-1]

    selecionados = []
    for _ in range(k):
        alvo = random.random() * total
        indice = bisect.bisect_left(acumulados, alvo)
        selecionados.append(populacao[min(indice, len(populacao) - 1)])
    return selecionados


toolbox.register("select", sel_roleta)

def treinar(pop_size=GA_POP_SIZE, n_gen=GA_N_GEN, cxpb=GA_CXPB, mutpb=GA_MUTPB,
            checkpoint_path=GA_CAMINHO_CHECKPOINT):
    random.seed(42)
    np.random.seed(42)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)  # Elitismo (HallOfFame)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # Reporta o fitness real; a transformação positiva existe apenas na roleta.
    stats.register("fitness_medio", np.mean)
    stats.register("variacao_fitness", np.std)
    stats.register("pior_fitness", np.min)

    print("Iniciando Treinamento do Agente Genético...")
    print("Parâmetros do treinamento:")
    print(f"  tamanho_populacao={pop_size}")
    print(f"  numero_geracoes={n_gen}")
    print(f"  taxa_crossover={cxpb}")
    print(f"  taxa_mutacao={mutpb}")
    print(f"  partidas_por_individuo={GA_PARTIDAS_POR_AVALIACAO}\n")

    # Avalia a populacao inicial (geracao 0) manualmente para poder salvar checkpoint e permitir acompanhar evolucao desde o inicio.
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    logbook = tools.Logbook()
    logbook.header = [
        "geracao",
        "individuos_avaliados",
        "fitness_medio",
        "variacao_fitness",
        "pior_fitness",
        "melhor_fitness_geracao",
        "melhor_fitness_historico",
    ]
    melhor_fitness_geracao = max(ind.fitness.values[0] for ind in pop)
    record = stats.compile(pop)
    logbook.record(
        geracao=0,
        individuos_avaliados=len(pop),
        melhor_fitness_geracao=melhor_fitness_geracao,
        melhor_fitness_historico=hof[0].fitness.values[0],
        **record,
    )
    print(logbook.stream)
    _imprimir_diagnostico_melhor(hof[0])
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

        # Captura o melhor resultado produzido nesta geração antes do elitismo.
        melhor_fitness_geracao = max(ind.fitness.values[0] for ind in descendentes)

        # Elitismo explícito: garante que o melhor indivíduo já visto sobreviva mesmo se toda a nova geração regredir.
        melhor_anterior = toolbox.clone(hof[0]) if len(hof) > 0 else None
        melhor_fitness_historico_anterior = hof[0].fitness.values[0]
        pop[:] = descendentes
        hof.update(pop)
        if melhor_anterior is not None and melhor_anterior.fitness.values[0] > max(ind.fitness.values[0] for ind in pop):
            pior_idx = min(range(len(pop)), key=lambda i: pop[i].fitness.values[0])
            pop[pior_idx] = melhor_anterior

        record = stats.compile(pop)
        logbook.record(
            geracao=gen,
            individuos_avaliados=len(invalidos),
            melhor_fitness_geracao=melhor_fitness_geracao,
            melhor_fitness_historico=hof[0].fitness.values[0],
            **record,
        )
        print(logbook.stream)
        if hof[0].fitness.values[0] > melhor_fitness_historico_anterior:
            _imprimir_diagnostico_melhor(hof[0])

        _salvar_checkpoint(hof[0], checkpoint_path)

    melhor_ind = hof[0]
    print("\nTreinamento finalizado. Melhor fitness:",
          melhor_ind.fitness.values[0])
    print(f"Cromossomo salvo em '{checkpoint_path}'.")


def _salvar_checkpoint(individuo, caminho):
    np.save(caminho, np.array(individuo, dtype=np.uint8))
