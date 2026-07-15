"""Adversario controlado usado para ensinar o agente de RL a rebater."""

IDX_OPONENTE_Y = 50
IDX_BOLA_Y = 54

FIRE = 1
FIRE_RIGHT = 4
FIRE_LEFT = 5

TOLERANCIA_ALINHAMENTO = 3
PASSOS_ANTECIPACAO = 10


class OponenteTreino:
    """Antecipa a trajetoria da bola para devolve-la continuamente."""

    def __init__(self):
        self.resetar()

    def resetar(self):
        self._bola_y_anterior = None

    def escolher_acao(self, observacao_ram, recompensa=0):
        if recompensa != 0:
            self.resetar()

        oponente_y = int(observacao_ram[IDX_OPONENTE_Y])
        bola_y = int(observacao_ram[IDX_BOLA_Y])

        if bola_y == 0 or self._bola_y_anterior is None:
            velocidade_y = 0
        else:
            velocidade_y = bola_y - self._bola_y_anterior
        self._bola_y_anterior = None if bola_y == 0 else bola_y

        alvo_y = bola_y + PASSOS_ANTECIPACAO * velocidade_y
        dy = alvo_y - oponente_y

        if abs(dy) <= TOLERANCIA_ALINHAMENTO:
            return FIRE
        if dy > 0:
            return FIRE_LEFT
        return FIRE_RIGHT
