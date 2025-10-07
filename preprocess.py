import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import INPUT_SITAF, OUTPUT_SIFAF, INPUT_ESTATISTICAS


def _mapear_tipo_tributo(valor_str: str) -> str:
    """
    Mapeia a descrição detalhada do imposto em `estatisticas-processos.csv`
    para um tipo de tributo agregado compatível com o dataset principal.
    Regras simples por prefixo/palavra-chave para
    ICMS, ISS, IPVA, IPTU, ITCD, ITBI, OUTROS.
    """
    if not isinstance(valor_str, str):
        return "OUTROS"
    s = valor_str.upper()
    if s.startswith("ICMS") or "ICMS" in s:
        return "ICMS"
    if s.startswith("ISS") or " IMPOSTO SOBRE SERVI" in s or "ISS -" in s:
        return "ISS"
    if s.startswith("IPVA") or "VEÍCULOS" in s or "VEICULOS" in s:
        return "IPVA"
    if s.startswith("IPTU") or "PREDIAL" in s or "TERRITORIAL" in s:
        return "IPTU"
    if s.startswith("ITCD") or "CAUSA MORTIS" in s or "ITCMD" in s:
        return "ITCD"
    if s.startswith("ITBI") or "INTER VIVOS" in s or "TRANSMISSÃO" in s:
        return "ITBI"
    return "OUTROS"


def carregar_fatores_por_tributo(caminho_estatisticas: Path) -> dict:
    """
    Lê `estatisticas-processos.csv`, agrega por tipo de tributo mapeado e calcula
    a mediana de `media_prazo`. Normaliza por mediana global para obter
    multiplicadores robustos e coerentes.
    """
    df_est = pd.read_csv(caminho_estatisticas)
    # Coluna de tipo de tributo consolidado
    df_est["tipo_tributo"] = df_est["DETIPOIMPOSTO"].map(_mapear_tipo_tributo)

    # Mediana do prazo por tributo
    med_por_tipo = (
        df_est.groupby("tipo_tributo")["media_prazo"].median().dropna()
    )
    if med_por_tipo.empty:
        raise ValueError("Não foi possível calcular a mediana do prazo por tipo de tributo")

    med_global = med_por_tipo.median()
    if med_global <= 0 or np.isnan(med_global):
        raise ValueError("Não foi possível calcular a mediana global do prazo")

    fatores = (med_por_tipo / med_global).to_dict()
    # Garante presença das chaves esperadas
    for k in ["ICMS", "ISS", "IPVA", "ITCD", "OUTROS"]:
        fatores.setdefault(k, 1.0)
    return fatores


def calcular_pesos(df: pd.DataFrame) -> pd.Series:
    """
    Calcula o esforço operacional (em horas) por processo com a heurística:
    - base 1.0h + 0.5h * log10(valor_total_corrigido)
    - multiplicador por tipo de tributo, quando disponível
    """
    valores_corrigidos = df["valor_total_corrigido"].astype(float).clip(lower=0.0)
    termo_log = np.log10(valores_corrigidos.replace(0, np.nan)).fillna(0).clip(lower=0)
    horas_base = 1.0 + 0.5 * termo_log

    # Deriva fatores a partir das estatísticas observadas
    fatores_observados = carregar_fatores_por_tributo(INPUT_ESTATISTICAS)

    multiplicador = 1.0
    if "tipo_receita" in df.columns:
        multiplicador = df["tipo_receita"].map(fatores_observados).fillna(1.0)
    elif "descricao_receita" in df.columns:
        multiplicador = df["descricao_receita"].map(fatores_observados).fillna(1.0)

    peso_horas = (horas_base * multiplicador).clip(lower=0.25)
    return peso_horas


def preprocessar_dados(caminho_csv: Path, prefixo_saida: Path) -> dict:
    """
    Lê o CSV original, deriva colunas de valor e peso_horas e salva artefatos
    prontos para os algoritmos (CSV filtrado + NPZ com vetores numéricos).
    """
    df = pd.read_csv(caminho_csv, sep=";", decimal=",")

    # Conversão de data (se existir)
    if "data_constituicao" in df.columns:
        df["data_constituicao"] = pd.to_datetime(
            df["data_constituicao"], format="%d/%m/%Y", errors="coerce"
        )

    # Benefício (valor) e custo (peso em horas)
    df["valor"] = df["valor_total_corrigido"].astype(float).clip(lower=0.0)
    df["peso_horas"] = calcular_pesos(df)

    df_alg = df[(df["valor"] > 0) & (df["peso_horas"] > 0)].reset_index(drop=True)

    vetor_valores = df_alg["valor"].to_numpy(dtype=float)
    vetor_pesos = df_alg["peso_horas"].to_numpy(dtype=float)

    # Salvando saídas (CSV e NPZ) para consumo pelos runners
    prefixo_saida.parent.mkdir(parents=True, exist_ok=True)
    df_alg.to_csv(f"{prefixo_saida}.csv", index=False)
    np.savez_compressed(
        f"{prefixo_saida}.npz",
        valores=vetor_valores,
        pesos=vetor_pesos,
        index=df_alg.index.values,
    )

    meta = {
        "linhas_entrada": int(len(df)),
        "linhas_saida": int(len(df_alg)),
        "csv_saida": f"{prefixo_saida}.csv",
        "npz_saida": f"{prefixo_saida}.npz",
        "colunas": list(df_alg.columns),
    }
    with open(f"{prefixo_saida}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def executar() -> None:
    parser = argparse.ArgumentParser(
        description="Pré-processar dados para seleção via Mochila (Knapsack)"
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=INPUT_SITAF,
        help="Caminho do CSV de entrada (separador ';' e decimal ',')",
    )
    parser.add_argument(
        "--prefixo-saida",
        type=Path,
        default=OUTPUT_SIFAF,
        help="Prefixo de saída (.csv, .npz, .meta.json)",
    )
    args = parser.parse_args()

    meta = preprocessar_dados(args.entrada, args.prefixo_saida)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    executar()
