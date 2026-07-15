import re
from pathlib import Path


PADRAO_CHECKPOINT = re.compile(r"^qtable_episodio_(\d+)\.npz$")


def caminho_checkpoint(diretorio, episodio):
    return Path(diretorio) / f"qtable_episodio_{episodio:06d}.npz"


def checkpoint_mais_recente(diretorio):
    diretorio = Path(diretorio)
    if not diretorio.is_dir():
        return None

    encontrados = []
    for caminho in diretorio.iterdir():
        correspondencia = PADRAO_CHECKPOINT.fullmatch(caminho.name)
        if correspondencia and caminho.is_file():
            encontrados.append((int(correspondencia.group(1)), caminho))

    return max(encontrados, default=(None, None), key=lambda item: item[0])[1]
