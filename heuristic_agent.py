# Agente baseado em estado, objetivo e busca heurística.

# Estratégia (busca gulosa / greedy search):
#  1. Ler a posição Y da própria raquete e da bola na RAM.
#  2. Definir o objetivo: alinhar a raquete com a altura da bola.
#  3. Escolher a ação que mais aproxima a raquete da altura da bola.

# Isso é uma busca heurística simples: a cada passo, avalia-se uma função de "distância até o objetivo" e escolhe-se a ação que minimiza
# essa distância (comportamento tipo greedy best-first, sem árvore de busca porque o espaço de ações é pequeno).

IDX_JOGADOR_Y = 51
IDX_OPONENTE_Y = 50
IDX_BOLA_X = 49
IDX_BOLA_Y = 54
IDX_PLACAR_JOGADOR = 14
IDX_PLACAR_OPONENTE = 13

NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
FIRE_RIGHT = 4
FIRE_LEFT = 5

TOLERANCIA_ALINHAMENTO = 3  # em pixels de RAM

def extrair_posicoes(observacao_ram):
    # Extrai a posição Y da raquete do jogador, da raquete do oponente e da bola
    jogador_y = int(observacao_ram[IDX_JOGADOR_Y])
    oponente_y = int(observacao_ram[IDX_OPONENTE_Y])
    bola_y = int(observacao_ram[IDX_BOLA_Y])
    return jogador_y, oponente_y, bola_y

def escolher_acao(observacao_ram):
    # Regra heurística:
    # Calcula a diferença (dy) entre a altura da bola e a altura da raquete
    # Se a raquete já está alinhada, apenas confirma o saque
    # Se estiver desalinhada, move a raquete na direção da bola
    jogador_y, oponente_y, bola_y = extrair_posicoes(observacao_ram)

    dy = bola_y - oponente_y

    alinhado = abs(dy) <= TOLERANCIA_ALINHAMENTO

    if alinhado:
        return FIRE

    # Y maior = mais para baixo na tela, então dy > 0 significa que a
    # bola está abaixo da raquete e precisamos descer (ação LEFT)
    if dy > 0:
        return FIRE_LEFT

    return FIRE_RIGHT
