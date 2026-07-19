"""Interface de linha de comando do projeto."""

import argparse


def treinar_rl():
    from train_rl import treinar

    treinar()


def treinar_genetico():
    from train_genetic import treinar

    treinar()


def avaliar_agentes(tipo1, tipo2, render=False):
    from evaluate import avaliar

    avaliar(tipo1, tipo2, render=render)


def jogar_contra(tipo):
    from jogar import jogar

    jogar(tipo)


def criar_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py", description="Treina, avalia e joga Pong com agentes."
    )
    subcomandos = parser.add_subparsers(dest="comando", required=True)

    train_parser = subcomandos.add_parser("train", help="Treina um agente.")
    train_parser.add_argument("tipo", choices=("rl", "genetico"))

    eval_parser = subcomandos.add_parser("eval", help="Avalia dois agentes.")
    tipos_avaliaveis = ("rl", "genetico", "heuristico")
    eval_parser.add_argument("tipo1", choices=tipos_avaliaveis)
    eval_parser.add_argument("tipo2", choices=tipos_avaliaveis)
    eval_parser.add_argument("--render", action="store_true")

    play_parser = subcomandos.add_parser("play", help="Joga contra um agente.")
    play_parser.add_argument("tipo", choices=tipos_avaliaveis)

    return parser


def main(argv=None):
    parser = criar_parser()
    args = parser.parse_args(argv)

    if args.comando == "eval" and args.tipo1 == args.tipo2 != "heuristico":
        parser.error(f"dois agentes '{args.tipo1}' exigem o mesmo lado")

    try:
        if args.comando == "train" and args.tipo == "rl":
            treinar_rl()
        elif args.comando == "train":
            treinar_genetico()
        elif args.comando == "eval":
            avaliar_agentes(args.tipo1, args.tipo2, render=args.render)
        elif args.comando == "play":
            jogar_contra(args.tipo)
    except FileNotFoundError as erro:
        parser.error(str(erro))


if __name__ == "__main__":
    main()
