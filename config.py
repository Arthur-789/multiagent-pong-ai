# Ambiente
ENV_NAME = "pong_v3"
OBS_TYPE = "ram"          # Cada agente recebe os 128 bytes da RAM do Atari
MAX_CYCLES = 2000         # Limite de passos por partida

# Treinamento do agente de RL
TREINO_EPISODIOS = 100        # Número de episódios de treinamento
TAXA_APRENDIZADO = 1e-3       # Taxa de aprendizagem da MLP
FATOR_DESCONTO = 0.95         # Gamma do Q-Learning
EPSILON_INICIAL = 1.0         # Exploração inicial (100%)
EPSILON_FINAL = 0.05          # Exploração mínima
EPSILON_DECAIMENTO = 0.98     # Multiplicado a cada episódio
TAMANHO_CAMADA_OCULTA = 64    # Neurônios na camada oculta da MLP
CAMINHO_MODELO = "modelo_rl.pt"  # Onde o modelo treinado é salvo e carregado

# Avaliação
PARTIDAS_AVALIACAO = 20   # Quantas partidas rodar na comparação final

# Visualização (Se True, abre uma janela mostrando os agentes jogando)
RENDER_TREINO = False
RENDER_AVALIACAO = False

# Geral
SEED = 42