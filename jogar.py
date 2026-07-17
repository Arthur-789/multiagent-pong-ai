"""Joga Pong humano contra um dos tipos de agente disponíveis."""

import pygame

from agents import (
    carregar_agente,
    lados_para_jogo,
)
from config import (
    FPS_JOGO,
    SEED,
)
from environment import criar_ambiente

INTERVALO_MOVIMENTO_HUMANO = 3


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


def jogar(tipo):
    lado_humano, lado_agente = lados_para_jogo(tipo)
    agente_modelo = carregar_agente(tipo, lado_agente)
    render_mode = "human"
    env = criar_ambiente(render_mode=render_mode)

    print(f"Jogando contra o agente: {tipo}")
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

                if agente == lado_humano:
                    passos_humano += 1
                    permitir_movimento = (
                        passos_humano % INTERVALO_MOVIMENTO_HUMANO == 0
                    )
                    acao = None if (terminou or truncado) else _ler_acao_humana(
                        permitir_movimento
                    )
                else:
                    acao = None if (terminou or truncado) else agente_modelo.escolher_acao(observacao)

                env.step(acao)
                if agente == lado_humano:
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
