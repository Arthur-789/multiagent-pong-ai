from config import (
    RECOMPENSA_PONTO, PUNICAO_PONTO_SOFRIDO, RECOMPENSA_REBATIDA, RECOMPENSA_APROXIMACAO
)

class RastreadorRecompensa:
    def __init__(self, agente_id):
        self.agente_id = agente_id
        self.bola_x_anterior = None
        self.direcao_x_anterior = 0
        self.distancia_anterior = None
        
        # first_0 (Left) -> ball moves left (< 0) towards it
        # second_0 (Right) -> ball moves right (> 0) towards it
        self.direcao_em_minha_direcao = -1 if agente_id == "first_0" else 1

    def atualizar_recompensa(self, observacao, recompensa_ambiente, bola_x, bola_y, jogador_y):
        recompensa_shape = 0.0
        rebateu = False

        if recompensa_ambiente != 0 or bola_x == 0 or bola_y == 0:
            self.bola_x_anterior = None
            self.direcao_x_anterior = 0
            self.distancia_anterior = None
        elif self.bola_x_anterior is None:
            self.bola_x_anterior = bola_x
        else:
            direcao_x = bola_x - self.bola_x_anterior
            
            # Se a bola estava vindo na minha direção e agora vai na direção oposta
            indo_para_mim = (self.direcao_x_anterior * self.direcao_em_minha_direcao > 0)
            indo_embora = (direcao_x * self.direcao_em_minha_direcao < 0)
            
            rebateu = indo_para_mim and indo_embora
            
            if direcao_x != 0:
                self.direcao_x_anterior = direcao_x
            self.bola_x_anterior = bola_x

        if recompensa_ambiente > 0:
            recompensa_shape = RECOMPENSA_PONTO
        elif recompensa_ambiente < 0:
            recompensa_shape = PUNICAO_PONTO_SOFRIDO
        elif rebateu:
            recompensa_shape = RECOMPENSA_REBATIDA

        # Se a bola estiver vindo na minha direção
        if self.direcao_x_anterior * self.direcao_em_minha_direcao > 0:
            distancia = abs(bola_y - jogador_y)
            if self.distancia_anterior is not None:
                progresso = self.distancia_anterior - distancia
                recompensa_shape += RECOMPENSA_APROXIMACAO * progresso
            self.distancia_anterior = distancia
        else:
            self.distancia_anterior = None

        return recompensa_shape, rebateu
