# Usar o mesmo espaço de ações no RL e no genético

O AgenteGenetico usará somente `FIRE`, `FIRE_RIGHT` e `FIRE_LEFT`, o mesmo espaço de ações do AgenteRL. A representação genética passará de 128 × 6 pesos (12.288 bits) para 128 × 3 pesos (6.144 bits), porque ações do ambiente que não fazem parte do espaço de aprendizagem não devem consumir capacidade evolutiva nem criar políticas diferentes entre os agentes. Checkpoints antigos deixam de ser compatíveis e o treinamento deve começar do zero.

## Considered Options

- **Manter seis saídas e mascarar três**: rejeitada porque preserva genes sem efeito e mantém uma representação incompatível com o espaço de ações efetivamente usado pelo RL.
