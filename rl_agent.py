# Agente de aprendizado por reforço: Q-Learning com aproximação por MLP.

# Usamos uma MLP (Multilayer Perceptron) simples para aproximar a função Q(estado, ação).
# A MLP recebe o vetor de 128 bytes e retorna um valor Q estimado para cada uma das 6 ações possíveis.

# O treinamento segue a regra clássica do Q-Learning: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
# Porém, em vez de atualizar uma tabela, treinamos a MLP para que sua saída Q(s, a) se aproxime do valor-alvo (r + gamma * max Q(s', ·)).

import random
import numpy as np
import torch
import torch.nn as nn

from config import (
    TAXA_APRENDIZADO,
    FATOR_DESCONTO,
    TAMANHO_CAMADA_OCULTA,
)

NUM_ACOES = 6  # Total de ações possíveis

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
        self.rede = RedeQ()
        self.otimizador = torch.optim.Adam(self.rede.parameters(), lr=TAXA_APRENDIZADO)
        self.funcao_perda = nn.MSELoss()
        self.epsilon = 1.0

    def _observacao_para_tensor(self, observacao_ram):
        # Normaliza os bytes para o intervalo 0-1
        vetor = np.array(observacao_ram, dtype=np.float32) / 255.0
        return torch.from_numpy(vetor).unsqueeze(0)

    def escolher_acao(self, observacao_ram, explorar=True):
        """Escolhe ação com política epsilon-greedy."""
        if explorar and random.random() < self.epsilon:
            return random.randint(0, NUM_ACOES - 1)

        with torch.no_grad():
            estado = self._observacao_para_tensor(observacao_ram)
            valores_q = self.rede(estado)
            return int(torch.argmax(valores_q, dim=1).item())

    def treinar_passo(self, estado, acao, recompensa, proximo_estado, terminou):
        # Executa uma atualização de Q-Learning
        estado_t = self._observacao_para_tensor(estado)
        proximo_estado_t = self._observacao_para_tensor(proximo_estado)

        valores_q = self.rede(estado_t)
        q_atual = valores_q[0, acao]

        with torch.no_grad():
            proximo_q_max = self.rede(proximo_estado_t).max().item()
            alvo = recompensa + (0 if terminou else FATOR_DESCONTO * proximo_q_max)

        perda = self.funcao_perda(q_atual, torch.tensor(alvo, dtype=torch.float32))

        self.otimizador.zero_grad()
        perda.backward()
        self.otimizador.step()

    def salvar(self, caminho):
        torch.save(self.rede.state_dict(), caminho)

    def carregar(self, caminho):
        self.rede.load_state_dict(torch.load(caminho))
        self.rede.eval()