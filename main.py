# python main.py treinar -> treina o agente de RL
# python main.py avaliar -> compara os dois agentes
# python main.py jogar [modelo] -> joga contra o modelo

import sys

from train import treinar
from evaluate import avaliar
from config import RENDER_TREINO, RENDER_AVALIACAO
from jogar import jogar

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("treinar", "avaliar", "jogar"):
        print("Uso: python main.py [treinar|avaliar|jogar] [--render] [modelo.pt]")
        return

    # A flag --render força a visualização. Sem a flag, usamos o valor
    # padrão configurado em config.py (RENDER_TREINO / RENDER_AVALIACAO).
    forcar_render = "--render" in sys.argv
    modelo = next((arg for arg in sys.argv[2:] if not arg.startswith("--")), None)

    if sys.argv[1] == "treinar":
        treinar(render=forcar_render or RENDER_TREINO)
    elif sys.argv[1] == "jogar":
        jogar(modelo=modelo)
    else:
        avaliar(render=forcar_render or RENDER_AVALIACAO)

if __name__ == "__main__":
    main()
