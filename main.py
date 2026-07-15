# python main.py treinar -> treina o agente de RL
# python main.py avaliar -> compara os dois agentes

import sys

from train import treinar
from evaluate import avaliar
from config import RENDER_TREINO, RENDER_AVALIACAO

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("treinar", "avaliar"):
        print("Uso: python main.py [treinar|avaliar] [--render]")
        return

    # A flag --render força a visualização. Sem a flag, usamos o valor
    # padrão configurado em config.py (RENDER_TREINO / RENDER_AVALIACAO).
    forcar_render = "--render" in sys.argv

    if sys.argv[1] == "treinar":
        treinar(render=forcar_render or RENDER_TREINO)
    else:
        avaliar(render=forcar_render or RENDER_AVALIACAO)

if __name__ == "__main__":
    main()