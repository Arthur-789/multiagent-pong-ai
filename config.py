# Ambiente
ENV_NAME = "pong_v3"
OBS_TYPE = "ram"  # Cada agente recebe os 128 bytes da RAM do Atari
MAX_CYCLES = 20000  # Limite de passos por partida

# Treinamento do agente de RL
TREINO_EPISODIOS = 800  # Número de episódios de treinamento
TAXA_APRENDIZADO = 0.1  # Alpha da atualização da tabela Q
FATOR_DESCONTO = 0.95  # Gamma do Q-Learning
EPSILON_INICIAL = 1.0  # Exploração inicial (100%)
EPSILON_FINAL = 0.05  # Exploração mínima
EPSILON_DECAIMENTO = 0.98  # Multiplicado a cada episódio
CAMINHO_MODELO = "tabela_q.npz"  # Tabela Q treinada
DIRETORIO_CHECKPOINTS = "checkpoints_tabular"  # Checkpoints separados dos DQN antigos
RECOMPENSA_APROXIMACAO = 0.02  # Sinal auxiliar por pixel aproximado
RECOMPENSA_REBATIDA = 5.0  # Recompensa imediata ao devolver a bola
RECOMPENSA_PONTO = 25.0  # Recompensa rara por vencer o oponente de treino
PUNICAO_PONTO_SOFRIDO = -25.0  # Peso simétrico ao ponto marcado

# Avaliação
PARTIDAS_AVALIACAO = 20  # Quantas partidas rodar na comparação final

# Visualização (Se True, abre uma janela mostrando os agentes jogando)
RENDER_TREINO = False
RENDER_AVALIACAO = False
FPS_JOGO = 90  # Limite de FPS ao jogar

# Geral
SEED = 42
