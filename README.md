# Atari Pong: Busca Heurística vs. Q-Learning

Projeto acadêmico de Inteligência Artificial que compara dois
paradigmas de agentes inteligentes no ambiente Pong do
PettingZoo (Atari), usando observações em modo RAM (vetor de
128 bytes).

O objetivo não é obter o melhor agente possível, mas sim comparar:

1. **Agente de busca heurística** — decide ações com base em
   estado, objetivo e regras de alinhamento com a bola.
2. **Agente de aprendizado por reforço (Q-Learning + MLP)** — aprende
   a jogar através de tentativa e erro, aproximando a função Q com
   uma rede neural simples (MLP).

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
- `torch` - implementação da MLP usada como aproximador de Q
- `numpy` - manipulação de vetores

As redes e os tensores usam CUDA automaticamente quando uma GPU compatível e
uma instalação do PyTorch com suporte a CUDA estão disponíveis. Caso contrário,
o projeto continua usando a CPU.

## Como treinar

```bash
python main.py treinar
```

Isso treina o agente de RL contra um adversário controlado que antecipa a
trajetória da bola e tenta devolvê-la continuamente. O RL recebe recompensa
por aproximar a raquete enquanto a bola vem, recompensa imediata ao rebater,
uma recompensa muito maior ao marcar e uma punição forte ao sofrer um ponto.
Pontos sem uma rebatida anterior não geram recompensa gratuita. Os valores
ficam em `config.py`. Ao final, o modelo
treinado é salvo em `modelo_rl.pt`. O progresso e o número de rebatidas são
impressos no terminal a cada 10 episódios.

## Como avaliar

```bash
python main.py avaliar
```

Carrega o modelo treinado e joga várias partidas do agente de RL contra o
agente heurístico, exibindo ao final as vitórias, a pontuação média e a taxa de
vitória de cada agente.

## Como jogar

```bash
python main.py jogar modelo_rl.pt
```

Ou diretamente:

```bash
python jogar.py modelo_rl.pt
```

Controles:

- `W` ou `↑` sobe
- `S` ou `↓` desce
- `SPACE` serve
- `ESC` sai

No modo jogável, a execução respeita o limite `FPS_JOGO` definido em
`config.py`, para que a velocidade não dependa do desempenho da CPU ou do tempo
de inferência do modelo.

## Como assistir ao modelo

Para visualizar um modelo jogando contra o mesmo oponente usado no treino:

```bash
python main.py assistir modelo_rl.pt
```

Esse modo não atualiza a rede Q nem executa etapas de treinamento. Cada partida
vai até o encerramento normal do Pong em 21 pontos, imprime o placar, reinicia
automaticamente e respeita o limite `FPS_JOGO` definido em `config.py`. Use
`ESC` para fechar.

Como o placar do Pong não é estritamente de soma zero por causa
da penalidade de saque demorado, é possível uma partida terminar com
os dois agentes tendo recompensa negativa.

## Configurações

Todos os parâmetros ajustáveis (número de episódios de treino,
número de partidas de avaliação, taxa de aprendizado, epsilon,
etc.) estão centralizados em **`config.py`**.

## Estrutura dos arquivos

| Arquivo             | Responsabilidade                                              |
|----------------------|-----------------------------------------------------------------|
| `config.py`          | Configurações do projeto       |
| `environment.py`     | Criação do ambiente PettingZoo Pong (modo RAM)                |
| `heuristic_agent.py` | Agente de busca heurística baseado em posições da RAM           |
| `training_opponent.py` | Adversário controlado usado para treinar devoluções            |
| `rl_agent.py`        | Agente de RL: MLP + lógica de Q-Learning                |
| `train.py`           | Loop de treinamento do agente de RL                              |
| `evaluate.py`        | Compara os dois agentes e imprime as estatísticas finais         |
| `main.py`            | Ponto de entrada (`treinar` ou `avaliar`)                        |
| `escanear_ram.py`    | Ferramenta de diagnóstico: descobre quais índices da RAM mudam de valor |
| `requirements.txt`   | Dependências do projeto                                          |

## Explicação resumida dos agentes

### Agente heurístico (`heuristic_agent.py`)

Lê a posição vertical (Y) da própria raquete e da bola diretamente de
índices específicos da memória RAM do Atari. A partir dessas posições,
calcula a diferença de altura e escolhe a ação que alinha a raquete com a bola.
Faz uma busca gulosa simples: se a bola está acima, sobe; se está abaixo, desce;
se já está alinhada, confirma o saque.

### Agente de RL (`rl_agent.py`)

Implementa Q-Learning clássico, mas em vez de uma tabela Q
(inviável para 128 bytes de estado), usa uma MLP pequena (uma
camada oculta) que recebe o vetor de RAM normalizado e retorna um
valor Q estimado para cada uma das ações. A política usa apenas as três
ações úteis para este treino: `FIRE`, `FIRE_RIGHT` e `FIRE_LEFT`. As
transições ficam em uma memória de experiências e são sorteadas em lotes
para reduzir a correlação entre quadros consecutivos. Uma segunda rede,
atualizada periodicamente, calcula o alvo do Q-Learning:

```
alvo = recompensa + gamma * max(Q(próximo_estado))
```

Durante o treinamento, o agente usa uma política epsilon-greedy
(explora ações aleatórias no início, e gradualmente passa a usar
mais a rede treinada conforme o epsilon decai).
