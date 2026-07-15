# Ambiente
ENV_NAME = "pong_v3"
OBS_TYPE = "ram"  # Cada agente recebe os 128 bytes da RAM do Atari
MAX_CYCLES = 20000  # Limite de passos por partida

# Treinamento do agente de RL
TREINO_EPISODIOS = 400  # Número de episódios de treinamento
TAXA_APRENDIZADO = 1e-3  # Taxa de aprendizagem da MLP
FATOR_DESCONTO = 0.95  # Gamma do Q-Learning
EPSILON_INICIAL = 1.0  # Exploração inicial (100%)
EPSILON_FINAL = 0.05  # Exploração mínima
EPSILON_DECAIMENTO = 0.98  # Multiplicado a cada episódio
TAMANHO_CAMADA_OCULTA = 64  # Neurônios na camada oculta da MLP
CAMINHO_MODELO = "modelo_rl.pt"  # Onde o modelo treinado é salvo e carregado
TAMANHO_MEMORIA = 10000  # Transições mantidas para quebrar correlações
TAMANHO_LOTE = 64  # Amostras usadas em cada atualização da rede
INTERVALO_TREINO = 4  # Atualiza a rede a cada N passos do ambiente
INTERVALO_REDE_ALVO = 1000  # Sincroniza a rede-alvo a cada N passos
RECOMPENSA_APROXIMACAO = 0.05  # Por pixel aproximado enquanto a bola vem
RECOMPENSA_REBATIDA = 5.0  # Recompensa imediata ao devolver a bola
RECOMPENSA_PONTO = 25.0  # Recompensa rara por vencer o oponente de treino
PUNICAO_PONTO_SOFRIDO = -10.0  # Punição por deixar a bola passar

# Avaliação
PARTIDAS_AVALIACAO = 20  # Quantas partidas rodar na comparação final

# Visualização (Se True, abre uma janela mostrando os agentes jogando)
RENDER_TREINO = False
RENDER_AVALIACAO = False
FPS_JOGO = 90  # Limite de FPS ao jogar

# Geral
SEED = 42
