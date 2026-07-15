"""Joga Pong humano vs modelo treinado."""

import os
import sys

import pygame

from checkpoints import checkpoint_mais_recente
from config import CAMINHO_MODELO, DIRETORIO_CHECKPOINTS, FPS_JOGO, SEED
from environment import criar_ambiente
from rl_agent import AgenteRL

NOME_HUMANO = "second_0"
NOME_MODELO = "first_0"
INTERVALO_MOVIMENTO_HUMANO = 3


def resolver_modelo(argumento):
    if not argumento:
        return checkpoint_mais_recente(DIRETORIO_CHECKPOINTS) or CAMINHO_MODELO
    if os.path.exists(argumento):
        return argumento
    if not argumento.endswith(".npz"):
        candidato = f"{argumento}.npz"
        if os.path.exists(candidato):
            return candidato
    return argumento


def _ler_acao_humana(permitir_movimento=True):
    pygame.event.pump()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise KeyboardInterrupt

    teclas = pygame.key.get_pressed()

    if teclas[pygame.K_ESCAPE]:
        raise KeyboardInterrupt

    if permitir_movimento and (teclas[pygame.K_w] or teclas[pygame.K_UP]):
        return 4
    if permitir_movimento and (teclas[pygame.K_s] or teclas[pygame.K_DOWN]):
        return 5
    return 1


def jogar(modelo=None):
    caminho_modelo = resolver_modelo(modelo)
    render_mode = "human"
    env = criar_ambiente(render_mode=render_mode)

    agente_modelo = AgenteRL()
    agente_modelo.carregar(caminho_modelo)

    print(f"Jogando contra o modelo: {caminho_modelo}")
    print("Controles: W/↑ sobe, S/↓ desce, saque automático, ESC sai.")

    try:
        env.reset(seed=SEED)
        agente_modelo.resetar_estado()
        env.render()
        relogio = pygame.time.Clock()
        fps = FPS_JOGO
        passos_humano = 0
        while True:
            for agente in env.agent_iter():
                observacao, recompensa, terminou, truncado, _ = env.last()

                if agente == NOME_HUMANO:
                    passos_humano += 1
                    permitir_movimento = (
                        passos_humano % INTERVALO_MOVIMENTO_HUMANO == 0
                    )
                    acao = None if (terminou or truncado) else _ler_acao_humana(
                        permitir_movimento
                    )
                else:
                    acao = None if (terminou or truncado) else agente_modelo.escolher_acao(
                        observacao, explorar=False
                    )

                env.step(acao)
                if agente == NOME_HUMANO:
                    # Um ciclo do Atari termina após os dois agentes agirem.
                    relogio.tick(fps)

            env.reset()
            agente_modelo.resetar_estado()
            env.render()
            passos_humano = 0
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    jogar(sys.argv[1] if len(sys.argv) > 1 else None)
