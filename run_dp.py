import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    INPUT_PREPROCESSADO,
    OUTPUT_PREFIXO_DP,
    CAPACIDADE_PADRAO,
    RESOLUCAO_PADRAO,
)
from utils import load_data, save_data


def mochila_dp(
    valores: np.ndarray, pesos: np.ndarray, capacidade: float, resolucao: float
) -> list[int]:
    """
    Programação Dinâmica 0-1 (discretizando horas pela resolução informada).
    Retorna índices relativos dos itens selecionados.
    """
    pesos_discretos = (np.round(pesos / resolucao)).astype(int)
    capacidade_discreta = int(np.floor(capacidade / resolucao))
    n = len(valores)

    tabela = np.zeros((n + 1, capacidade_discreta + 1), dtype=float)
    escolhas = np.zeros((n + 1, capacidade_discreta + 1), dtype=bool)

    for i in range(1, n + 1):
        peso_i = pesos_discretos[i - 1]
        valor_i = valores[i - 1]
        anterior = tabela[i - 1]
        atual = tabela[i]
        atual[:] = anterior
        if peso_i <= capacidade_discreta:
            candidato = np.full(capacidade_discreta + 1, -np.inf, dtype=float)
            candidato[peso_i:] = anterior[:-peso_i] + valor_i
            melhor = candidato > atual
            atual[melhor] = candidato[melhor]
            escolhas[i, melhor] = True

    selecionados = []
    restante = capacidade_discreta
    for i in range(n, 0, -1):
        if escolhas[i, restante]:
            selecionados.append(i - 1)
            restante -= pesos_discretos[i - 1]
    selecionados.reverse()
    return selecionados


def filtrar_itens(
    valores: np.ndarray, pesos: np.ndarray, top_k: int | None, modo: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Opcionalmente reduz o universo aos top_k por valor/razão."""
    n = len(valores)
    idx_orig = np.arange(n)
    if not top_k or top_k >= n:
        return valores, pesos, idx_orig
    if modo == "value":
        ordem = np.argsort(-valores)
    elif modo == "ratio":
        ordem = np.argsort(-(valores / pesos))
    else:
        return valores, pesos, idx_orig
    escolher = ordem[:top_k]
    return valores[escolher], pesos[escolher], idx_orig[escolher]


def executar() -> None:
    parser = argparse.ArgumentParser(description="Executar Mochila por DP")
    parser.add_argument("--npz", type=Path, default=INPUT_PREPROCESSADO)
    parser.add_argument("--capacidade", type=float, default=CAPACIDADE_PADRAO)
    parser.add_argument("--resolucao", type=float, default=RESOLUCAO_PADRAO)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--modo-filtro", choices=["none", "value", "ratio"], default="ratio"
    )
    parser.add_argument(
        "--prefixo-saida",
        type=Path,
        default=OUTPUT_PREFIXO_DP,
    )
    args = parser.parse_args()

    valores, pesos, idx_map, caminho_csv = load_data(args.npz)
    valores_f, pesos_f, idx_f = filtrar_itens(
        valores,
        pesos,
        args.top_k if args.top_k > 0 else None,
        args.modo_filtro,
    )
    idx_rel = mochila_dp(valores_f, pesos_f, args.capacidade, args.resolucao)
    idx_abs = idx_f[idx_rel]

    df = pd.read_csv(caminho_csv)
    df_sel = df.iloc[idx_abs].copy()

    resumo = {
        "algoritmo": "dp",
        "n_processos": int(len(df_sel)),
        "horas_total": float(df_sel["peso_horas"].sum()),
        "valor_total": float(df_sel["valor"].sum()),
        "capacidade": float(args.capacidade),
        "resolucao": float(args.resolucao),
        "top_k": int(args.top_k),
        "modo_filtro": args.modo_filtro,
    }
    save_data(args.prefixo_saida, df_sel, resumo)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    executar()
