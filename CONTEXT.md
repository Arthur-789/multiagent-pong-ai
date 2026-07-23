# ggMultiagent Pong AI

Um ambiente de Pong com agentes de aprendizado competindo entre si e contra um baseline heurístico, usando a biblioteca PettingZoo.

## Language

**Baseline Opponent**:
Um oponente estático usado exclusivamente para servir como referência ("saco de pancadas") durante a fase de treinamento de outros agentes.
_Avoid_: Oponente principal, adversário de treino dinâmico.

**Learning Agent**:
Um agente que atualiza seu comportamento durante sua fase de treinamento (ex: AgenteRL, AgenteGenetico).
_Avoid_: Agente treinável, modelo.

**Agent Type**:
A técnica que define como um Learning Agent aprende e age. Os tipos treináveis são `rl` e `genetico`.
_Avoid_: Nome do modelo, arquivo do modelo.

**Best Trained Artifact**:
O artefato padrão que preserva o Learning Agent de melhor desempenho obtido na execução de treinamento mais recente de um Agent Type. Ele não representa necessariamente o melhor resultado histórico entre execuções independentes.
_Avoid_: Modelo padrão, último checkpoint.

**Evaluation Phase**:
O script ou processo final onde os Learning Agents já treinados competem entre si (ou contra o Baseline) para medirmos a performance e generalização de cada técnica.
_Avoid_: Teste, Validação final.

**Fitness**:
A pontuação real que mede o desempenho de um indivíduo do AgenteGenetico durante a avaliação em rallies. Nesta alteração, seu cálculo permanece inalterado: combina pontos, punições, rebatidas e aproximação.
_Avoid_: Peso da roleta, pontuação deslocada.

**Objetivo do Rally**:
Vencer o rally é o objetivo primário do Learning Agent; o número de rebatidas é o critério secundário para comparar vitórias ou outros resultados equivalentes.
_Avoid_: Tempo vivo como objetivo, sobreviver a qualquer custo.

**Rebatida**:
O retorno da bola pelo Learning Agent depois que ela se aproxima da sua raquete e passa a se afastar dela.
_Avoid_: Ação de movimento, ponto ganho.

**Detecção de Rebatida**:
A identificação de uma Rebatida pela mudança de direção horizontal da bola, interpretada de forma relativa ao lado ocupado pelo Learning Agent.
_Avoid_: Recompensa de rebatida, detecção absoluta independente do lado.

**Espaço de Ações de Aprendizagem**:
O conjunto de ações que um Learning Agent pode escolher durante o treinamento. Para o RL e o AgenteGenetico, esse conjunto é `FIRE`, `FIRE_RIGHT` e `FIRE_LEFT`.
_Avoid_: Todas as ações disponíveis no ambiente, ação de saque como direção da bola.

**Peso de Seleção**:
O valor positivo derivado do Fitness somente para calcular a probabilidade de seleção na roleta.
_Avoid_: Fitness ajustado, fitness +1000.

**Sides (Lados da Tela)**:
No ambiente PettingZoo Pong, o mapeamento real na memória RAM (51 para direita, 50 para esquerda) determina que `first_0` representa o **Lado Direito** e `second_0` representa o **Lado Esquerdo**.
_Avoid_: Chamar `first_0` de esquerdo, inverter a ordem dos lados em variáveis e scripts.
