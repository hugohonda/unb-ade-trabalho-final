import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    INPUT_PREPROCESSADO,
    OUTPUT_PREFIXO_AG,
    CAPACIDADE_PADRAO,
    POPULACAO,
    GERACOES,
    TAXA_CRUZAMENTO,
    TAXA_MUTACAO,
    SEMENTE,
)
from utils import load_data, save_data


def mochila_ag(
    valores: np.ndarray,
    pesos: np.ndarray,
    capacidade: float,
    populacao: int,
    geracoes: int,
    taxa_cruzamento: float,
    taxa_mutacao: float,
    semente: int,
) -> list[int]:
    """Algoritmo Genético simples para Mochila (0-1)."""
    rnd = random.Random(semente)
    n = len(valores)

    def aptidao(bitset):
        peso = 0.0
        valor = 0.0
        for i, b in enumerate(bitset):
            if b:
                peso += pesos[i]
                if peso > capacidade:
                    return -1e12
                valor += valores[i]
        return valor

    def individuo_aleatorio():
        p = min(0.5, capacidade / (float(pesos.sum()) + 1e-9))
        return [1 if rnd.random() < p else 0 for _ in range(n)]

    def cruzar(a, b):
        if rnd.random() > taxa_cruzamento:
            return a[:], b[:]
        c = rnd.randrange(1, n)
        return a[:c] + b[c:], b[:c] + a[c:]

    def mutar(ind):
        for i in range(n):
            if rnd.random() < taxa_mutacao:
                ind[i] ^= 1
        return ind

    populacao_atual = [individuo_aleatorio() for _ in range(populacao)]
    melhor = max(populacao_atual, key=aptidao)
    melhor_fit = aptidao(melhor)

    for _ in range(geracoes):
        selecionados = []
        for _ in range(populacao):
            a, b = rnd.randrange(populacao), rnd.randrange(populacao)
            sa = populacao_atual[a]
            sb = populacao_atual[b]
            selecionados.append(sa if aptidao(sa) >= aptidao(sb) else sb)
        proxima = []
        for i in range(0, populacao, 2):
            p1 = selecionados[i]
            p2 = selecionados[i + 1 if i + 1 < populacao else 0]
            f1, f2 = cruzar(p1, p2)
            proxima.append(mutar(f1))
            proxima.append(mutar(f2))
        cand_melhor = max(proxima, key=aptidao)
        cand_fit = aptidao(cand_melhor)
        if cand_fit >= melhor_fit:
            melhor = cand_melhor[:]
            melhor_fit = cand_fit
        else:
            pior_idx = min(range(populacao), key=lambda k: aptidao(proxima[k]))
            proxima[pior_idx] = melhor[:]
        populacao_atual = proxima

    return [i for i, b in enumerate(melhor) if b]


def executar() -> None:
    parser = argparse.ArgumentParser(description="Executar Mochila via AG")
    parser.add_argument("--npz", type=Path, default=INPUT_PREPROCESSADO)
    parser.add_argument("--capacidade", type=float, default=CAPACIDADE_PADRAO)
    parser.add_argument("--pop", type=int, default=POPULACAO)
    parser.add_argument("--gens", type=int, default=GERACOES)
    parser.add_argument("--cxpb", type=float, default=TAXA_CRUZAMENTO)
    parser.add_argument("--mutpb", type=float, default=TAXA_MUTACAO)
    parser.add_argument("--seed", type=int, default=SEMENTE)
    parser.add_argument(
        "--prefixo-saida",
        type=Path,
        default=OUTPUT_PREFIXO_AG,
    )
    args = parser.parse_args()

    valores, pesos, _, caminho_csv = load_data(args.npz)
    idx_rel = mochila_ag(
        valores,
        pesos,
        args.capacidade,
        args.pop,
        args.gens,
        args.cxpb,
        args.mutpb,
        args.seed,
    )
    idx_abs = np.arange(len(valores))[idx_rel]

    df = pd.read_csv(caminho_csv)
    df_sel = df.iloc[idx_abs].copy()

    resumo = {
        "algoritmo": "ag",
        "n_processos": int(len(df_sel)),
        "horas_total": float(df_sel["peso_horas"].sum()),
        "valor_total": float(df_sel["valor"].sum()),
        "capacidade": float(args.capacidade),
        "pop": int(args.pop),
        "gens": int(args.gens),
        "cxpb": float(args.cxpb),
        "mutpb": float(args.mutpb),
        "seed": int(args.seed),
    }
    save_data(args.prefixo_saida, df_sel, resumo)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    executar()
