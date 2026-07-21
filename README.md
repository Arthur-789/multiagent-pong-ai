# Atari Pong: Busca Heurística vs. Q-Learning vs. Algoritmo Genético

Projeto acadêmico de Inteligência Artificial que compara três
paradigmas de agentes inteligentes no ambiente Pong do
PettingZoo (Atari), usando observações em modo RAM (vetor de
128 bytes).

O objetivo não é obter o melhor agente possível, mas sim comparar:

1. **Agente de busca heurística** — decide ações com base em
   estado, objetivo e regras de alinhamento com a bola.
2. **Agente de aprendizado por reforço (Q-Learning tabular)** — aprende
   a jogar por tentativa e erro, atualizando diretamente uma tabela Q.
3. **Agente genético (Algoritmo Genético)** — utiliza um algoritmo
   genético clássico (seleção, crossover, mutação, elitismo, via DEAP)
   para otimizar uma rede neural de uma camada a partir de um
   cromossomo binário de 12.288 bits.

## Sobre o jogo

Marcar um ponto concede +1 de recompensa para o jogador e -1 de recompensa para o oponente.
Os saques são cronometrados: se o jogador não sacar a bola dentro de 2 segundos após recebê-la, ele recebe -1 ponto e o cronômetro é reiniciado. Isso impede que um jogador atrase a partida indefinidamente.

## Instalação

### Linux ou macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
AutoROM --accept-license
```

### Windows

O pacote `multi-agent-ale-py` usado pelo PettingZoo para os jogos Atari não publica wheels pré-compiladas para Windows. Isso faz o `pip install` tentar compilar o pacote do zero, o que normalmente falha no Windows.

A forma mais simples e confiável de rodar o projeto no Windows é
usando o WSL2:

```powershell
wsl --install
```

Reinicie o computador se solicitado. Depois, abra o terminal Ubuntu (WSL) e execute:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential cmake zlib1g-dev

cd /mnt/c/Users/SEU_USUARIO/... # Complete com o local onde está o projeto

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
AutoROM --accept-license
```

## Dependências

- `Python`
- `pettingzoo[atari]` - ambiente multiagente Atari
- `autorom` - baixa as ROMs do Atari automaticamente
- `numpy` - tabela Q, discretização e checkpoints
- `deap` - framework de algoritmos genéticos usado pelo agente genético

## Como treinar

```bash
python main.py train rl
```

Isso treina o agente de RL contra o mesmo agente heurístico usado na avaliação,
que move a raquete para reduzir a distância vertical até a bola. O RL recebe recompensa
por aproximar a raquete enquanto a bola vem, recompensa imediata ao rebater,
uma recompensa muito maior ao marcar e uma punição forte ao sofrer um ponto.
Pontos sem uma rebatida anterior não geram recompensa gratuita. Os valores
ficam em `config.py`. Ao final de cada episódio, um checkpoint é salvo em
`checkpoints_tabular/qtable_episodio_XXXXXX.npz`. A gravação é atômica: um arquivo
temporário é concluído antes de substituir o destino. Ao aumentar
`TREINO_EPISODIOS`, o treino retoma automaticamente o checkpoint de maior
número. O progresso e o número de rebatidas são impressos no terminal a cada
10 episódios.

### Treinamento do agente genético

```bash
python main.py train genetico
```

Os hiperparâmetros do algoritmo evolutivo (população, gerações, taxas de
crossover/mutação, rallies por avaliação e caminho do checkpoint) ficam
centralizados em **`config.py`**, junto com os demais parâmetros do projeto.

O treinamento do agente genético é feito separadamente, pois utiliza o framework
DEAP ao invés do loop de treinamento do Q-Learning. O algoritmo genético otimiza
um cromossomo binário de 12.288 bits que codifica uma rede neural de uma camada.
Cada indivíduo da população é avaliado contra o agente heurístico em vários rallies
com seeds diferentes, recebendo as mesmas recompensas do agente de RL, e o fitness
final é a média entre esses rallies. Usar várias seeds evita que o ambiente
reproduza sempre o mesmo cenário inicial, o que tornaria os indivíduos
indistinguíveis entre si. O fitness é deslocado em +1000 para que a seleção
funcione mesmo com valores negativos.

O treinamento usa os seguintes parâmetros do algoritmo evolutivo:

- **População**: 50 indivíduos (padrão)
- **Gerações**: 200 (padrão)
- **Avaliação**: 5 rallies por indivíduo (média), cada um com uma seed diferente
- **Crossover**: dois pontos (taxa de 60% por padrão)
- **Mutação**: flip bit com probabilidade de ~1/12288 por bit (~1 bit esperado por indivíduo mutado), taxa de 30% por indivíduo (padrão)
- **Seleção**: torneio (tamanho 3)
- **Elitismo**: o melhor indivíduo (HallOfFame) é preservado e reinserido na população caso toda a geração seguinte regrida

A cada geração, o melhor cromossomo encontrado até então é salvo em
`melhor_cromossomo.npy` (`GA_CAMINHO_CHECKPOINT`), permitindo interromper o
treino sem perder o progresso.

## Como avaliar

```bash
python main.py eval <agente1> <agente2> [--render]
```

Carrega os artefatos padrão dos tipos treinados e joga várias partidas entre
os agentes, exibindo ao final vitórias, pontuação média e taxa de vitória.

Os valores aceitos para `<agente1>` e `<agente2>` são:

- `rl` — agente de Q-Learning
- `genetico` — agente genético
- `heuristico` — agente heurístico

Como RL e genético foram treinados em lados fixos, `eval rl rl` e
`eval genetico genetico` são rejeitados. O baseline se adapta aos dois lados,
portanto `eval heuristico heuristico` é válido.

Exemplos:

```bash
python main.py eval genetico heuristico
python main.py eval genetico rl
python main.py eval rl genetico --render
```

A flag `--render` abre uma janela visualizando as partidas.

Quando o melhor artefato treinado de um tipo não existe no caminho definido em
`config.py`, `eval` informa o caminho ausente e orienta a executar
`python main.py train <tipo>`; nenhum checkpoint alternativo é carregado.

## Como jogar

```bash
python main.py play <rl|genetico|heuristico>
```

Controles:

- `W` ou `↑` sobe
- `S` ou `↓` desce
- o saque é automático
- `ESC` sai

No modo jogável, o agente permanece no lado para o qual foi treinado; por isso,
o humano joga à direita contra o genético e à esquerda contra os demais. A
execução respeita o limite `FPS_JOGO` definido em `config.py`.
Assim como em `eval`, a ausência do melhor artefato treinado produz um erro com
a instrução de treinamento, sem fallback para outro checkpoint.

## Configurações

Todos os parâmetros ajustáveis (número de episódios de treino,
número de partidas de avaliação, taxa de aprendizado, epsilon,
população e gerações do algoritmo genético, etc.) estão centralizados
em **`config.py`**.

## Estrutura dos arquivos

| Arquivo             | Responsabilidade                                              |
|----------------------|-----------------------------------------------------------------|
| `config.py`          | Configurações do projeto       |
| `environment.py`     | Criação do ambiente PettingZoo Pong (modo RAM)                |
| `heuristic_agent.py` | Agente de busca heurística baseado em posições da RAM           |
| `rl_agent.py`        | Agente de RL: estado discreto, tabela Q, Q-Learning e checkpoints tabulares |
| `genetic_agent.py`   | Agente genético: decodificação de cromossomo e seleção de ação   |
| `agents.py`          | Tipos de agente, artefatos padrão e preferências de lado          |
| `train.py`           | Loop de treinamento do agente de RL                              |
| `train_genetic.py`   | Treinamento do agente genético via algoritmo evolutivo (DEAP)    |
| `evaluate.py`        | Compara dois agentes e imprime as estatísticas finais            |
| `melhor_cromossomo.npy` | Melhor cromossomo encontrado pelo algoritmo genético          |
| `jogar.py`           | Permite jogar contra qualquer tipo de agente                     |
| `main.py`            | Única interface de linha de comando do projeto                   |
| `requirements.txt`   | Dependências do projeto                                          |

## Explicação resumida dos agentes

### Agente heurístico (`heuristic_agent.py`)

Lê a posição vertical (Y) da própria raquete e da bola diretamente de
índices específicos da memória RAM do Atari. A partir dessas posições,
calcula a diferença de altura e escolhe a ação que alinha a raquete com a bola.
Faz uma busca gulosa simples: se a bola está acima, sobe; se está abaixo, desce;
se já está alinhada, confirma o saque.

### Agente de RL (`rl_agent.py`)

Implementa Q-Learning tabular. O ambiente entrega os 128 bytes da RAM e o
agente faz feature engineering somente sobre essa percepção. Ele usa a posição
X da bola (RAM 49), a posição Y da própria raquete (RAM 51) e a posição Y da
bola (RAM 54). Esses valores formam um estado discreto com:

- 8 faixas horizontais para a bola;
- 7 faixas para o erro vertical entre bola e raquete;
- 3 direções horizontais e 3 verticais, obtidas comparando a RAM atual com a anterior;
- 1 estado específico para bola ausente/saque.

O resultado são 505 estados. A tabela possui três ações por estado: `FIRE`,
`FIRE_RIGHT` e `FIRE_LEFT`, totalizando 1.515 valores Q. Nenhum pixel ou dado
externo ao vetor RAM é usado. A atualização aplicada após cada transição é:

```
Q(s,a) = Q(s,a) + alpha * (recompensa + gamma * max(Q(s',a')) - Q(s,a))
```

Durante o treinamento, o agente usa uma política epsilon-greedy
(explora ações aleatórias no início, e gradualmente passa a usar
mais a melhor ação registrada na tabela conforme o epsilon decai).

### Agente genético (`genetic_agent.py`)

Utiliza um algoritmo genético para otimizar os pesos de uma rede neural de
uma camada. O cromossomo é representado como um vetor binário de 12.288 bits,
onde cada peso é codificado com 16 bits em ponto fixo, resultando em 768 pesos
que formam uma matriz de dimensão (128, 6). O agente recebe os 128 bytes da
RAM do Atari, normaliza cada byte dividindo por 255.0 e realiza um produto
escalar com a matriz de pesos. As 6 saídas resultantes correspondem às ações
do jogo (`NOOP`, `FIRE`, `RIGHT`, `LEFT`, `FIRE_RIGHT`, `FIRE_LEFT`), sendo
escolhida a de maior pontuação.

O algoritmo genético é executado pelo script `train_genetic.py`, que usa a
biblioteca DEAP com seleção por torneio, crossover de dois pontos e mutação por
flip bit. A avaliação de cada indivíduo é a média de vários rallies (com seeds
diferentes) contra o agente heurístico, com o mesmo sistema de reward shaping
utilizado no treino do agente de RL. O melhor cromossomo encontrado é salvo em
`melhor_cromossomo.npy` a cada geração.
