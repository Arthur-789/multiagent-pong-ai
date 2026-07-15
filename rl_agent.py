# Agente de aprendizado por reforço: Q-Learning com aproximação por MLP.

# Usamos uma MLP (Multilayer Perceptron) simples para aproximar a função Q(estado, ação).
# A MLP recebe o vetor de 128 bytes e retorna um valor Q estimado para cada uma das 6 ações possíveis.

# O treinamento segue a regra clássica do Q-Learning: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
# Porém, em vez de atualizar uma tabela, treinamos a MLP para que sua saída Q(s, a) se aproxime do valor-alvo (r + gamma * max Q(s', ·)).

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from config import (
    TAXA_APRENDIZADO,
    FATOR_DESCONTO,
    TAMANHO_CAMADA_OCULTA,
    TAMANHO_MEMORIA,
    TAMANHO_LOTE,
    INTERVALO_TREINO,
    INTERVALO_REDE_ALVO,
)

NUM_ACOES = 6  # Total de ações possíveis
ACOES_VALIDAS = (1, 4, 5)  # FIRE, FIRE_RIGHT e FIRE_LEFT

class RedeQ(nn.Module):

    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(128, TAMANHO_CAMADA_OCULTA),
            nn.ReLU(),
            nn.Linear(TAMANHO_CAMADA_OCULTA, NUM_ACOES),
        )

    def forward(self, x):
        return self.rede(x)

class AgenteRL:

    def __init__(self):
        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rede = RedeQ().to(self.dispositivo)
        self.rede_alvo = RedeQ().to(self.dispositivo)
        self.rede_alvo.load_state_dict(self.rede.state_dict())
        self.rede_alvo.eval()
        self.otimizador = torch.optim.Adam(self.rede.parameters(), lr=TAXA_APRENDIZADO)
        self.funcao_perda = nn.SmoothL1Loss()
        self.memoria = deque(maxlen=TAMANHO_MEMORIA)
        self.passos = 0
        self.epsilon = 1.0

    def _observacao_para_tensor(self, observacao_ram):
        # Normaliza os bytes para o intervalo 0-1
        vetor = np.array(observacao_ram, dtype=np.float32) / 255.0
        return torch.from_numpy(vetor).unsqueeze(0).to(self.dispositivo)

    def escolher_acao(self, observacao_ram, explorar=True):
        """Escolhe ação com política epsilon-greedy."""
        if explorar and random.random() < self.epsilon:
            return random.choice(ACOES_VALIDAS)

        with torch.no_grad():
            estado = self._observacao_para_tensor(observacao_ram)
            valores_q = self.rede(estado)[0]
            melhor_indice = torch.argmax(valores_q[list(ACOES_VALIDAS)]).item()
            return ACOES_VALIDAS[melhor_indice]

    def treinar_passo(self, estado, acao, recompensa, proximo_estado, terminou):
        self.memoria.append(
            (
                np.array(estado, dtype=np.uint8, copy=True),
                acao,
                recompensa,
                np.array(proximo_estado, dtype=np.uint8, copy=True),
                terminou,
            )
        )
        self.passos += 1

        if len(self.memoria) < TAMANHO_LOTE or self.passos % INTERVALO_TREINO != 0:
            return

        lote = random.sample(self.memoria, TAMANHO_LOTE)
        estados, acoes, recompensas, proximos_estados, terminos = zip(*lote)

        estados_t = torch.from_numpy(
            np.stack(estados).astype(np.float32) / 255.0
        ).to(self.dispositivo)
        proximos_estados_t = torch.from_numpy(
            np.stack(proximos_estados).astype(np.float32) / 255.0
        ).to(self.dispositivo)
        acoes_t = torch.tensor(
            acoes, dtype=torch.int64, device=self.dispositivo
        ).unsqueeze(1)
        recompensas_t = torch.tensor(
            recompensas, dtype=torch.float32, device=self.dispositivo
        )
        terminos_t = torch.tensor(
            terminos, dtype=torch.float32, device=self.dispositivo
        )

        q_atual = self.rede(estados_t).gather(1, acoes_t).squeeze(1)

        with torch.no_grad():
            proximo_q = self.rede_alvo(proximos_estados_t)[:, list(ACOES_VALIDAS)]
            proximo_q_max = proximo_q.max(dim=1).values
            alvo = recompensas_t + (1 - terminos_t) * FATOR_DESCONTO * proximo_q_max

        perda = self.funcao_perda(q_atual, alvo)

        self.otimizador.zero_grad()
        perda.backward()
        self.otimizador.step()

        if self.passos % INTERVALO_REDE_ALVO == 0:
            self.rede_alvo.load_state_dict(self.rede.state_dict())

    def salvar(self, caminho):
        torch.save(self.rede.state_dict(), caminho)

    def carregar(self, caminho):
        estado = torch.load(caminho, map_location=self.dispositivo)
        self.rede.load_state_dict(estado)
        self.rede_alvo.load_state_dict(self.rede.state_dict())
        self.rede.eval()
