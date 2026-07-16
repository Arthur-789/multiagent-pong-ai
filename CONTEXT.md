# Multiagent Pong AI

Um ambiente de Pong com agentes de aprendizado competindo entre si e contra um baseline heurístico, usando a biblioteca PettingZoo.

## Language

**Baseline Opponent**:
Um oponente estático usado exclusivamente para servir como referência ("saco de pancadas") durante a fase de treinamento de outros agentes.
_Avoid_: Oponente principal, adversário de treino dinâmico.

**Learning Agent**:
Um agente que atualiza seu comportamento durante sua fase de treinamento (ex: AgenteRL, AgenteGenetico).
_Avoid_: Agente treinável, modelo.

**Evaluation Phase**:
O script ou processo final onde os Learning Agents já treinados competem entre si (ou contra o Baseline) para medirmos a performance e generalização de cada técnica.
_Avoid_: Teste, Validação final.

**Sides (Lados da Tela)**:
No ambiente PettingZoo Pong, o mapeamento real na memória RAM (51 para direita, 50 para esquerda) determina que `first_0` representa o **Lado Direito** e `second_0` representa o **Lado Esquerdo**. 
_Avoid_: Chamar `first_0` de esquerdo, inverter a ordem dos lados em variáveis e scripts.
