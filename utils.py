from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_data(
    caminho_npz: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """
    Carrega vetores de valores e pesos do .npz e retorna também o caminho do
    CSV filtrado correspondente (mesmo prefixo, extensão .csv).
    """
    npz = np.load(caminho_npz)
    valores = npz["valores"].astype(float)
    pesos = npz["pesos"].astype(float)
    indices = npz["index"]
    caminho_csv = Path(str(caminho_npz).replace(".npz", ".csv"))
    return valores, pesos, indices, caminho_csv


def save_data(
    prefixo_saida: Path,
    df_sel: pd.DataFrame,
    resumo: dict,
) -> None:
    """
    Salva a seleção em CSV e um resumo em JSON usando o prefixo informado.
    """
    prefixo_saida.parent.mkdir(parents=True, exist_ok=True)
    df_sel.to_csv(f"{prefixo_saida}.csv", index=False)
    import json

    with open(f"{prefixo_saida}.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
