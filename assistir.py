"""Exibe o modelo treinado jogando contra o oponente de treino."""

import sys

import pygame

from config import CAMINHO_MODELO, FPS_JOGO, SEED
from environment import criar_ambiente
from jogar import resolver_modelo
from rl_agent import AgenteRL
from training_opponent import OponenteTreino

NOME_MODELO = "first_0"
NOME_OPONENTE = "second_0"


def _verificar_saida():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise KeyboardInterrupt
    if pygame.key.get_pressed()[pygame.K_ESCAPE]:
        raise KeyboardInterrupt


def assistir(modelo=CAMINHO_MODELO):
    caminho_modelo = resolver_modelo(modelo)
    env = criar_ambiente(render_mode="human")
    agente_modelo = AgenteRL()
    agente_modelo.carregar(caminho_modelo)
    oponente_treino = OponenteTreino()
    relogio = pygame.time.Clock()

    print(f"Modelo contra oponente de treino: {caminho_modelo}")
    print(f"Dispositivo PyTorch: {agente_modelo.dispositivo}")
    print(f"Limite: {FPS_JOGO} FPS. Pressione ESC para sair.")

    partida = 1
    try:
        while True:
            env.reset(seed=SEED if partida == 1 else None)
            oponente_treino.resetar()
            placar = {NOME_MODELO: 0, NOME_OPONENTE: 0}

            for agente in env.agent_iter():
                observacao, recompensa, terminou, truncado, _ = env.last()
                placar[agente] += recompensa

                if terminou or truncado:
                    acao = None
                elif agente == NOME_MODELO:
                    acao = agente_modelo.escolher_acao(observacao, explorar=False)
                else:
                    acao = oponente_treino.escolher_acao(observacao, recompensa)

                env.step(acao)
                if agente == NOME_OPONENTE:
                    _verificar_saida()
                    relogio.tick(FPS_JOGO)

            print(
                f"Partida {partida}: modelo {placar[NOME_MODELO]:.0f} x "
                f"{placar[NOME_OPONENTE]:.0f} oponente"
            )
            partida += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    assistir(sys.argv[1] if len(sys.argv) > 1 else CAMINHO_MODELO)
