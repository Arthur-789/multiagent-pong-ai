# Usar roleta na seleção do algoritmo genético

O algoritmo genético usa seleção por roleta porque essa é a técnica exigida no ensino da disciplina. O `Fitness` permanece como a pontuação real do indivíduo; durante a seleção, calculamos `Peso de Seleção` subtraindo o menor fitness da população e somando `epsilon = 1e-6`, preservando as diferenças relativas sem o achatamento causado pelo deslocamento fixo de `+1000`. O elitismo continua sendo garantido separadamente pelo Hall of Fame.

## Considered Options

- **Torneio**: rejeitado por não corresponder à técnica solicitada pelo professor.
- **Roleta com `fitness + 1000`**: rejeitada porque torna fitnesses próximos quase indistinguíveis na probabilidade de seleção.
