# Ambiente
ENV_NAME = "pong_v3"
OBS_TYPE = "ram"  # Cada agente recebe os 128 bytes da RAM do Atari
MAX_CYCLES = 20000  # Limite de passos por partida

# Treinamento do agente de RL
TREINO_EPISODIOS = 20000  # Cada episodio corresponde a um rally
TAXA_APRENDIZADO = 0.1  # Alpha inicial da atualização da tabela Q
TAXA_APRENDIZADO_FINAL = 0.01  # Piso do alpha por par estado-acao
FATOR_DESCONTO = 0.99  # Gamma do Q-Learning
EPSILON_INICIAL = 1.0  # Exploração inicial (100%)
EPSILON_FINAL = 0.05  # Exploração mínima
EPSILON_DECAIMENTO = 0.9998  # Atinge o minimo perto do episodio 15000
CAMINHO_MODELO = "tabela_q.npz"  # Tabela Q treinada
DIRETORIO_CHECKPOINTS = "checkpoints_tabular"  # Checkpoints separados dos DQN antigos
CAMINHO_MELHOR_CHECKPOINT = "checkpoints_tabular/melhor_qtable.npz"
INTERVALO_VALIDACAO = 500
PARTIDAS_VALIDACAO = 5
RECOMPENSA_APROXIMACAO = 0.02  # Sinal auxiliar por pixel aproximado
RECOMPENSA_REBATIDA = 5.0  # Recompensa imediata ao devolver a bola
RECOMPENSA_PONTO = 25.0  # Recompensa rara por vencer o oponente de treino
PUNICAO_PONTO_SOFRIDO = -25.0  # Peso simétrico ao ponto marcado

# Treinamento do agente genético
GA_POP_SIZE = 50  # Tamanho da população
GA_N_GEN = 30  # Número de gerações
GA_CXPB = 0.6  # Probabilidade de crossover por par de indivíduos
GA_MUTPB = 0.3  # Probabilidade de mutação por indivíduo
GA_RALLIES_POR_AVALIACAO = 5  # Rallies (com seeds diferentes) usados para calcular o fitness de 1 indivíduo
GA_CAMINHO_CHECKPOINT = "melhor_cromossomo.npy"  # Melhor cromossomo encontrado

# Avaliação
PARTIDAS_AVALIACAO = 20  # Quantas partidas rodar na comparação final
PROBABILIDADE_ACAO_REPETIDA_AVALIACAO = 0.25
SEED_VALIDACAO = 10000

# Visualização (Se True, abre uma janela mostrando os agentes jogando)
RENDER_TREINO = False
RENDER_AVALIACAO = False
FPS_JOGO = 90  # Limite de FPS ao jogar

# Geral
SEED = 42
