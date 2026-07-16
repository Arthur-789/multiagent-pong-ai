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
