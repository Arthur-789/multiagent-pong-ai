## **Abertura**

`[TELA: README.md aberto, topo do repositório]`

**Pessoa A:** "Oi, pessoal. Esse vídeo é a apresentação do nosso projeto do Estudo Dirigido de IA: um comparativo entre três tipos diferentes de Agentes de IA, usando o jogo Pong do Atari pela biblioteca PettingZoo. Nossa dupla é composta por eu, \[inserir nome da Pessoa A\], e \[inserir nome da pessoa B\].   
A ideia central não é achar 'o agente mais forte', é comparar três paradigmas diferentes de inteligência artificial jogando exatamente o mesmo jogo, com as mesmas regras."

**Pessoa B:** "Isso. A gente implementou um agente de busca heurística, um agente de aprendizado por reforço com Q-Learning tabular, e um agente evolutivo com algoritmo genético. Os três recebem a mesma fonte de informação, que é a RAM do Atari, e a gente vai mostrar exatamente como cada um funciona, como foi treinado, e como eles se saem um contra o outro."

---

## **O que foi delegado a IA generativa e como foi validado**

`[TELA: heuristic_agent.py, rl_agent.py e genetic_agent.py abertos, alternando entre eles]`

**Pessoa A:** "Antes de entrar no código, vale explicar como a gente usou IA generativa nesse projeto. A gente não pediu pra IA gerar o projeto inteiro de uma vez. A gente delegou tarefas pequenas e específicas: por exemplo, 'como estruturar um agente de busca heurística lendo a posição da bola e da raquete na RAM', ou 'como implementar uma tabela de Q-Learning pra esse tipo de ambiente', ou 'como decodificar um cromossomo binário em pesos de uma rede'. A IA gerou as primeiras versões e trouxe as ideias iniciais de como implementar cada agente."

**Pessoa B:** "E a partir daí a gente leu, entendeu, rodou e testou cada trecho gerado. A gente tem um arquivo de testes automatizados, o `test.py`, que cobre desde o parsing da linha de comando até a lógica de alocação de lado dos jogadores e de checkpoints. Além disso, a gente rodava o jogo de verdade, com `--render`, pra confirmar visualmente se o comportamento batia com o que a gente esperava daquele agente."

**Pessoa A:** "E teve bastante ajuste nosso em cima do que a IA trouxe: por exemplo, o desenho da recompensa intermediária, a forma como o estado é discretizado no RL, a calibração da taxa de mutação no genético. Isso tudo a gente pensou, testou e corrigiu depois de entender o que tinha sido gerado. Então a IA deu o esqueleto inicial, mas o entendimento, a validação e a evolução de cada parte foi nossa."

---

## **Funcionamento do ambiente: regras, estados, observações, ações, término**

`[TELA: environment.py aberto]`

**Pessoa B:** "Vamos falar do ambiente. A gente usa o `pong_v3` do PettingZoo Atari, e aqui no `environment.py` a gente configura ele com `obs_type='ram'`. Isso quer dizer que, a cada passo, o agente não recebe os pixels da tela, ele recebe um vetor de 128 posições, cada uma um valor de 0 a 255, que é a memória RAM crua do Atari naquele instante."

`[TELA: trocar para o jogo rodando com --render, ex: python main.py eval heuristico heuristico --render]`

**Pessoa A:** "E as regras do jogo em si são as regras clássicas do Pong: cada ponto marcado dá mais um pro jogador que marcou e menos um pro adversário. Só que tem uma regra extra importante: o saque é cronometrado. Se o jogador não sacar em até 2 segundos depois de receber a bola, ele perde um ponto automaticamente e o cronômetro reinicia. Isso existe pra evitar que alguém trave a partida sem jogar."

**Pessoa B:** "O ambiente é multiagente: tem o lado `first_0`, que é a direita, e o `second_0`, que é a esquerda. E a condição de término de cada episódio pode ser natural, quando a partida chega ao fim pelo placar do jogo, ou por truncamento, quando bate o limite máximo de passos que a gente configurou, o `MAX_CYCLES`."

**Pessoa A:** "E ligando com a teoria: mesmo sendo um ambiente Atari, com aparência de 'caixa preta', o estado real do jogo, posição da bola, das raquetes, placar, está todo ali dentro desses 128 bytes. Cada agente do projeto decide por conta própria quanto desse vetor ele efetivamente usa como percepção."

---

## **Formato técnico da conexão entre agente e ambiente**

`[TELA: heuristic_agent.py, com os índices de RAM em destaque]`

**Pessoa A:** "Tecnicamente, a conexão segue a interface padrão do PettingZoo: um loop com `env.agent_iter()`, onde a gente chama `env.last()` pra pegar observação, recompensa e status de término, e `env.step(acao)` pra mandar a ação escolhida. A entrada é sempre esse vetor de 128 inteiros, de 0 a 255\. A saída é sempre um número inteiro, representando uma das 6 ações possíveis do Pong: `NOOP`, `FIRE`, `RIGHT`, `LEFT`, `FIRE_RIGHT` e `FIRE_LEFT`. E a recompensa nativa do ambiente é o \+1 e \-1 por ponto, que vem direto do `env.last()`."

**Pessoa B:** "E aqui que fica interessante: cada agente lê esse vetor de 128 bytes de um jeito diferente. O heurístico usa só 4 índices: posição Y da própria raquete, posição Y do oponente, posição X e Y da bola. O RL usa esses mesmos índices, mas transforma isso num estado discreto, bem menor. E o genético usa o vetor inteiro, os 128 bytes, normalizado, dividido por 255, direto como entrada de uma camada linear."

`[TELA: mostrar rapidamente rl_agent.py na função de extração de estado, depois genetic_agent.py na função de escolha de ação]`

**Pessoa A:** "E além da recompensa nativa do jogo, a gente usa reward shaping no treino do RL e do genético. No `config.py` a gente define recompensas extras: por aproximar a raquete da bola, por rebater a bola, uma recompensa maior por vencer o rally, e uma punição por perder o rally. Isso é reward shaping: sinais auxiliares que aceleram o aprendizado sem mudar o objetivo final, que continua sendo vencer o jogo."

---

## **Funcionamento de cada agente e seu treinamento, busca ou evolução**

### **Agente de busca heurística**

`[TELA: heuristic_agent.py]`

**Pessoa B:** "O agente heurístico é o mais simples dos três. A cada passo, ele calcula a diferença vertical entre a posição da bola e a posição da própria raquete. Se a bola está acima, ele sobe. Se está abaixo, ele desce. E quando já está alinhado dentro de uma margem de tolerância, ele confirma o saque. Não tem aprendizado nenhum aqui, é um comportamento fixo, mas é uma decisão informada. Na teoria, isso é um agente baseado em objetivo, guiado por uma função de avaliação simples, que nesse caso é a distância vertical até a bola."

### **Agente de Q-Learning tabular**

`[TELA: rl_agent.py, na atualização da tabela Q; depois train_rl.py]`

**Pessoa A:** "O segundo é o agente de Q-Learning tabular. A regra de atualização é a clássica do Q-Learning:

`Q(s,a) = Q(s,a) + alpha vezes (recompensa + gamma vezes o máximo de Q(s',a') menos Q(s,a))`

A gente discretiza o estado combinando a faixa horizontal da bola, em 8 faixas, o erro vertical entre bola e raquete, em 7 faixas, e a direção horizontal e vertical da bola, 3 por 3\. Isso dá 505 estados possíveis no total."

**Pessoa B:** "E pro treino a gente usa só 3 ações: `FIRE`, `FIRE_RIGHT` e `FIRE_LEFT`. Multiplicando 505 estados por 3 ações, a tabela Q tem 1.515 valores. A exploração é feita com política epsilon-greedy: o agente começa com um epsilon alto, explorando bastante o ambiente, e esse epsilon vai decaindo aos poucos até um piso mínimo, então ele vai explorando cada vez menos conforme aprende. Cada rally é tratado como um episódio de treino."

**Pessoa A:** "Ligando com a teoria: isso é aprendizado por reforço model-free, ou seja, o agente não modela a dinâmica do ambiente, ele aprende só por tentativa e erro, usando diferença temporal. E o balanço entre explorar e explotar é controlado justamente por esse decaimento do epsilon."

### **Agente genético**

`[TELA: genetic_agent.py, depois train_genetic.py]`

**Pessoa B:** "O terceiro é o agente genético. A representação é um cromossomo binário de 12.288 bits. Cada peso é codificado com 16 bits em ponto fixo, o que dá 768 pesos no total, organizados numa matriz de 128 por 6\. O agente pega a RAM normalizada, os 128 bytes divididos por 255, faz um produto escalar com essa matriz, e escolhe a ação com maior pontuação, o argmax."

**Pessoa A:** "O ciclo evolutivo é feito com a biblioteca DEAP. O fitness de cada indivíduo é a recompensa média, usando o mesmo reward shaping do RL, em vários rallies com seeds diferentes. A seleção é por roleta: como o fitness pode ser negativo, subtraímos o menor fitness da população e somamos um epsilon mínimo para formar pesos positivos. Assim, preservamos as diferenças relativas entre os indivíduos."

**Pessoa B:** "O crossover é de dois pontos, com uma taxa configurável. A mutação é flip bit, com probabilidade baixa por bit, calibrada para alterar em média poucos bits por indivíduo mutado. Isso é importante porque um único bit de peso alto pode mudar o valor quase inteiro, de menos um a mais um, então mutar demais destrói o indivíduo. E a gente usa elitismo: o melhor indivíduo de toda a execução fica guardado num Hall of Fame e é reinserido se a geração toda regredir."

**Pessoa A:** "Na teoria, isso é uma busca estocástica populacional, guiada por seleção natural simulada, sem gradiente. O que faz sentido aqui porque o fitness, que depende do agente jogando contra o heurístico, não é uma função diferenciável de forma direta em relação aos pesos."

---

## **Evolução visual do agente durante o desenvolvimento**

`[TELA: comparação lado a lado - agente no início do treino (epsilon alto / gerações iniciais) vs. agente já treinado]`

**Pessoa B:** "Vale mostrar como o comportamento evoluiu durante o desenvolvimento. No começo do treino, tanto no RL quanto no genético, o comportamento é bem errático: a raquete se move de um jeito que parece quase aleatório, o agente perde saque, não consegue se alinhar com a bola. No RL isso é esperado, porque o epsilon está alto, então ele está explorando bastante. No genético, as primeiras gerações têm pesos praticamente aleatórios, então o comportamento também é bem ruim."

**Pessoa A:** "Conforme o treino avança, dá pra ver a raquete começando a acompanhar a trajetória da bola de forma mais consistente, rebatendo mais vezes seguidas, perdendo menos pontos por saque. Ainda sobra alguma dificuldade em ângulos mais extremos e um pouco de oscilação perto da borda da tela, mas a melhora do início pro fim do treino é bem visível."

---

## **Formulação computacional do problema**

`[TELA: config.py]`

**Pessoa B:** "Resumindo a formulação computacional dos três agentes. No heurístico, o estado é a posição Y da bola e da raquete, lida direto, contínua; a ação são duas, subir ou descer, mais o saque; o objetivo é minimizar a distância vertical até a bola; e o que existe no lugar de recompensa é uma heurística, a própria função de distância."

**Pessoa A:** "No RL, o estado são os 505 estados discretos que a gente já comentou; a ação são 3, `FIRE`, `FIRE_RIGHT` e `FIRE_LEFT`; o objetivo é maximizar o retorno acumulado; e a recompensa é o reward shaping: aproximação, rebatida e ponto. No genético, o 'estado' é o vetor RAM completo, os 128 bytes; a ação são as 6 possíveis do jogo; o objetivo é maximizar o fitness médio; e o fitness é a média do mesmo reward shaping, calculada em vários rallies."

**Pessoa B:** "E isso reflete três formulações teóricas diferentes: o heurístico é busca com função heurística explícita; o RL é um processo de decisão sequencial, com estado, ação, recompensa e transição; e o genético é uma formulação evolutiva, com indivíduo, cromossomo, fitness, população e geração."

---

## **Principais parâmetros e hiperparâmetros**

`[TELA: config.py, seção de hiperparâmetros]`

**Pessoa A:** "Passando pelos hiperparâmetros principais, que ficam centralizados no `config.py`. No RL, a gente tem a taxa de aprendizado, o alpha, e um piso mínimo pra ele; o fator de desconto, gamma, que a gente deixou em 0.99, priorizando bastante recompensas futuras; o epsilon inicial, final e a taxa de decaimento, que controlam a exploração; e os pesos do reward shaping: recompensa de aproximação, de rebatida, de ponto, e a punição por ponto sofrido."

**Pessoa B:** "No genético, a gente tem o tamanho da população, 50 indivíduos; o número de gerações, 200; a taxa de crossover, 60%; a taxa de mutação, 30% de chance por indivíduo; e o número de rallies por avaliação de fitness, que é 5, cada um com uma seed diferente."

**Pessoa A:** "E na prática, o efeito de cada um: um alpha maior aprende mais rápido, mas fica mais instável. Um gamma alto faz o agente 'pensar mais à frente'. Um epsilon alto demais atrasa a convergência, mas evita que o agente fique preso num comportamento ruim cedo demais. E população e gerações maiores no genético aumentam a chance de achar uma solução melhor, só que custam mais tempo, porque cada indivíduo precisa jogar vários rallies pra ter o fitness calculado."

---

## **Protocolo de avaliação, métricas e comparação com referência**

`[TELA: terminal, python main.py eval <agente1> <agente2>]`

**Pessoa B:** "Pra avaliação, a gente tem o `evaluate.py`. Ele roda um número fixo de partidas entre dois agentes usando seeds determinísticas, e no final reporta número de vitórias, pontuação média e taxa de vitória de cada lado."

**Pessoa A:** "E o agente heurístico funciona como nossa estratégia de referência, nosso baseline. Tanto o RL quanto o genético foram treinados e avaliados contra ele, então dá pra medir se cada técnica realmente aprendeu algo melhor do que essa heurística simples, além de comparar RL contra genético diretamente."

`[TELA: rodar ao vivo pelo menos duas comparações: eval genetico heuristico e eval rl genetico]`

**Pessoa B:** "Rodando aqui ao vivo: RL contra heurístico, taxa de vitória \[inserir número\], pontuação média \[inserir número\]. Genético contra heurístico, taxa de vitória \[inserir número\], pontuação média \[inserir número\]. E RL contra genético, taxa de vitória \[inserir número\], pontuação média \[inserir número\]."

**Pessoa A:** "\[Comentário sobre o que esses números mostram. Por exemplo, qual paradigma generalizou melhor ou teve resultado mais consistente entre as partidas. Preencher com a leitura real dos resultados obtidos.\]"

---

## **Limitações, falhas, melhorias possíveis e reprodução do experimento**

`[TELA: README.md, seção de instalação e execução]`

**Pessoa A:** "Sobre limitações: o Q-Learning tabular tem o estado discretizado em só 505 estados, então ele perde granularidade fina da posição real da bola e da raquete. O agente genético usa uma rede de uma única camada linear, então a capacidade representacional dele é limitada. E tanto o RL quanto o genético foram treinados sempre contra o mesmo oponente heurístico fixo, o que pode gerar um certo overfitting a esse estilo específico de jogo. Eles podem não generalizar tão bem contra um estilo de jogo diferente."

**Pessoa B:** "Como melhorias possíveis: dá pra usar uma rede neural com mais camadas no genético, aumentar a granularidade dos estados no Q-Learning, ou até migrar pra uma abordagem com aproximação de função em vez de tabela. Também dá pra treinar contra oponentes variados, ou usar self-play, e aumentar a população e o número de gerações do genético."

**Pessoa A:** "E pra quem quiser reproduzir o experimento, o processo é: criar um ambiente virtual, instalar as dependências do `requirements.txt`, aceitar a licença das ROMs com `AutoROM --accept-license`, e depois rodar `python main.py train rl` e `python main.py train genetico` pra treinar, `python main.py eval` pra avaliar, e `python main.py play` pra jogar contra qualquer um dos agentes."

---

## **Resultado final: o agente realizando a tarefa**

`[TELA: gravação ao vivo de uma partida completa, ex: python main.py eval genetico rl --render ou python main.py play rl]`

**Pessoa B:** "Vamos fechar mostrando uma partida completa. \[Narrar ao vivo o que está acontecendo: quando o agente rebate, quando marca ponto, como ele se posiciona ao longo da partida.\]"

**Pessoa A:** "\[Comentar o placar final da partida mostrada.\] E é isso que resume o projeto: três formas bem diferentes de pensar o mesmo problema: busca, aprendizado por reforço e evolução. Todas chegando a um agente capaz de jogar Pong de forma competente."

---

## **Como foram descobertas as posições da RAM do Atari**

`[TELA: repositório github.com/mila-iqia/atari-representation-learning, arquivo atariari/benchmark/ram_annotations.py]`

**Pessoa B:** "Por último, uma pergunta importante: como a gente descobriu o que cada posição da RAM do Atari representa? Porque diferente de um ambiente com estado explícito, a RAM crua não vem documentada, então é preciso descobrir, de alguma forma, o que cada um dos 128 bytes significa."

**Pessoa A:** "Pra isso a gente se apoiou no repositório AtariARI, o Atari Annotated RAM Interface, do grupo de pesquisa Mila, disponível em `github.com/mila-iqia/atari-representation-learning`. Esse trabalho é ligado ao artigo 'Unsupervised State Representation Learning in Atari', publicado na NeurIPS de 2019\. Os autores fizeram engenharia reversa do código-fonte de vários jogos Atari e anotaram, jogo por jogo, quais posições da RAM correspondem a variáveis de estado relevantes: posição da bola, posição de cada jogador, placar, e assim por diante. Esse mapeamento fica no arquivo `ram_annotations.py` do repositório."

**Pessoa B:** "A gente consultou essas anotações especificamente pro Pong e usou elas como ponto de partida pros índices que a gente usa no projeto. Por exemplo, a posição X e Y da bola, a posição Y de cada raquete e o placar de cada lado. E validou isso na prática: rodando o ambiente com `--render` e comparando o valor lido daquela posição da RAM com o que estava acontecendo visualmente na tela, pra confirmar que, por exemplo, o byte apontado como posição Y do jogador realmente varia de forma coerente com o movimento da raquete."

**Pessoa A:** "Então o processo foi esse: usar uma fonte de anotação de RAM já validada pela comunidade de pesquisa, e depois confirmar empiricamente rodando o próprio ambiente."

---

## **Encerramento**

**Pessoa B:** "Então é isso. A gente mostrou o ambiente, os três agentes, como cada um foi treinado, os resultados da avaliação e as limitações do projeto."

**Pessoa A:** "O código completo está no repositório no GitHub, com o passo a passo pra reproduzir tudo. Obrigado por assistir."

---

## **Checklist de gravação de tela**

* \[ \] README.md (abertura)  
* \[ \] `environment.py`  
* \[ \] Jogo rodando com `--render` (regras gerais)  
* \[ \] `heuristic_agent.py`  
* \[ \] `rl_agent.py` (estado discreto \+ atualização Q)  
* \[ \] `train_rl.py` (loop de treino, reward shaping)  
* \[ \] `genetic_agent.py`  
* \[ \] `train_genetic.py` (seleção, crossover, mutação, elitismo)  
* \[ \] `config.py` (hiperparâmetros)  
* \[ \] Comparativo visual: agente início do treino vs. agente treinado  
* \[ \] `evaluate.py` \+ execução real de `python main.py eval ...` com números finais  
* \[ \] Partida final completa mostrando o agente jogando  
* \[ \] Repositório `atariari/benchmark/ram_annotations.py` no GitHub (mila-iqia)
