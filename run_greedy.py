import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CAPACIDADE_PADRAO,
    INPUT_PREPROCESSADO,
    OUTPUT_PREFIXO_GULOSO,
)
from utils import load_data, save_data


def mochila_gulosa(
    valores: np.ndarray, pesos: np.ndarray, capacidade: float
) -> list[int]:
    """
    Seleciona itens pela maior razão valor/peso até atingir a capacidade.
    Estratégia: ordena por eficiência (valor por hora) e inclui
    enquanto couber.
    """
    if capacidade <= 0:
        return []
    ordem = np.argsort(-(valores / pesos))
    peso_total = 0.0
    escolhidos = []
    for i in ordem:
        if peso_total + pesos[i] <= capacidade:
            escolhidos.append(i)
            peso_total += pesos[i]
    return escolhidos


def executar() -> None:
    parser = argparse.ArgumentParser(description="Executar Mochila Gulosa")
    parser.add_argument("--npz", type=Path, default=INPUT_PREPROCESSADO)
    parser.add_argument("--capacidade", type=float, default=CAPACIDADE_PADRAO)
    parser.add_argument(
        "--prefixo-saida",
        type=Path,
        default=OUTPUT_PREFIXO_GULOSO,
    )
    args = parser.parse_args()

    # Carrega vetores numéricos (valor, peso) e o CSV correspondente
    valores, pesos, _, caminho_csv = load_data(args.npz)
    # Executa heurística gulosa
    idx_rel = mochila_gulosa(valores, pesos, args.capacidade)
    # Converte índices relativos (no NPZ) para absolutos (no CSV filtrado)
    idx_abs = np.arange(len(valores))[idx_rel]

    df = pd.read_csv(caminho_csv)
    df_sel = df.iloc[idx_abs].copy()

    resumo = {
        "algoritmo": "guloso",
        "n_processos": int(len(df_sel)),
        "horas_total": float(df_sel["peso_horas"].sum()),
        "valor_total": float(df_sel["valor"].sum()),
        "capacidade": float(args.capacidade),
    }
    # Persiste seleção e resumo
    save_data(args.prefixo_saida, df_sel, resumo)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    executar()
