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
    """Algoritmo Genético para Mochila (0-1) com operadores corretos."""
    rnd = random.Random(semente)
    n = len(valores)

    def aptidao(bitset):
        """Calcula aptidão: valor total se viável, 0 se inviável."""
        peso_total = sum(pesos[i] for i, b in enumerate(bitset) if b)
        if peso_total > capacidade:
            return 0.0  # Solução inválida
        return sum(valores[i] for i, b in enumerate(bitset) if b)

    def individuo_aleatorio():
        """Gera indivíduo aleatório respeitando capacidade."""
        individuo = [0] * n
        peso_atual = 0.0

        # Ordena itens por razão valor/peso
        razoes = [(i, valores[i] / pesos[i]) for i in range(n)]
        razoes.sort(key=lambda x: x[1], reverse=True)

        # Adiciona itens aleatoriamente respeitando capacidade
        for i, _ in razoes:
            if rnd.random() < 0.3 and peso_atual + pesos[i] <= capacidade:
                individuo[i] = 1
                peso_atual += pesos[i]

        return individuo

    def selecao_torneio(pop, tamanho_torneio=3):
        """Seleção por torneio."""
        selecionados = []
        for _ in range(len(pop)):
            torneio = rnd.sample(pop, min(tamanho_torneio, len(pop)))
            vencedor = max(torneio, key=aptidao)
            selecionados.append(vencedor[:])
        return selecionados

    def cruzar(p1, p2):
        """Crossover uniforme que mantém viabilidade."""
        if rnd.random() > taxa_cruzamento:
            return p1[:], p2[:]

        f1, f2 = [0] * n, [0] * n
        peso_f1, peso_f2 = 0.0, 0.0

        for i in range(n):
            if rnd.random() < 0.5:
                # Herda de p1
                if p1[i] and peso_f1 + pesos[i] <= capacidade:
                    f1[i] = 1
                    peso_f1 += pesos[i]
                if p2[i] and peso_f2 + pesos[i] <= capacidade:
                    f2[i] = 1
                    peso_f2 += pesos[i]
            else:
                # Herda de p2
                if p2[i] and peso_f1 + pesos[i] <= capacidade:
                    f1[i] = 1
                    peso_f1 += pesos[i]
                if p1[i] and peso_f2 + pesos[i] <= capacidade:
                    f2[i] = 1
                    peso_f2 += pesos[i]

        return f1, f2

    def mutar(ind):
        """Mutação que mantém viabilidade."""
        peso_atual = sum(pesos[i] for i, b in enumerate(ind) if b)

        for i in range(n):
            if rnd.random() < taxa_mutacao:
                if ind[i] == 1:
                    # Remove item
                    ind[i] = 0
                    peso_atual -= pesos[i]
                else:
                    # Adiciona item se cabe
                    if peso_atual + pesos[i] <= capacidade:
                        ind[i] = 1
                        peso_atual += pesos[i]

        return ind

    # Inicialização
    populacao_atual = [individuo_aleatorio() for _ in range(populacao)]
    melhor = max(populacao_atual, key=aptidao)
    melhor_fit = aptidao(melhor)

    # Evolução
    for geracao in range(geracoes):
        # Seleção
        selecionados = selecao_torneio(populacao_atual)

        # Crossover e mutação
        proxima_geracao = []
        for i in range(0, len(selecionados), 2):
            p1 = selecionados[i]
            p2 = selecionados[i + 1] if i + 1 < len(selecionados) else selecionados[0]

            f1, f2 = cruzar(p1, p2)
            f1 = mutar(f1)
            f2 = mutar(f2)

            proxima_geracao.extend([f1, f2])

        # Elitismo: mantém o melhor
        proxima_geracao = proxima_geracao[:populacao]
        cand_melhor = max(proxima_geracao, key=aptidao)
        if aptidao(cand_melhor) > melhor_fit:
            melhor = cand_melhor[:]
            melhor_fit = aptidao(melhor)

        # Substitui pior indivíduo pelo melhor
        pior_idx = min(
            range(len(proxima_geracao)), key=lambda k: aptidao(proxima_geracao[k])
        )
        proxima_geracao[pior_idx] = melhor[:]

        populacao_atual = proxima_geracao

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
