import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DIAS_UTEIS,
    HORAS_DIA,
    INPUT_ESTATISTICAS,
    INPUT_SITAF,
    OUTPUT_SIFAF,
)


def _normalizar(txt: str) -> str:
    up = txt.upper()
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", up)
        if unicodedata.category(ch) != "Mn"
    )


def _mapear_tipo_tributo(valor_str: str) -> str:
    """
    Mapeia a descrição detalhada do imposto em `estatisticas-processos.csv`
    para um tipo de tributo agregado compatível com o dataset principal.
    Regras simples por prefixo/palavra-chave para
    ICMS, ISS, IPVA, IPTU, ITCD, ITBI, OUTROS.
    """
    if not isinstance(valor_str, str):
        return "OUTROS"
    s = _normalizar(valor_str)

    # Regras por palavra-chave (com e sem acento, já normalizado)
    if s.startswith("ICMS") or "ICMS" in s:
        return "ICMS"
    if (
        s.startswith("ISS")
        or "IMPOSTO SOBRE SERVI" in s
        or "SERVICO" in s
        or "ISSQN" in s
        or "ISS -" in s
    ):
        return "ISS"
    if (
        s.startswith("IPVA")
        or "VEICULO" in s
        or "AUTOMOT" in s
        or "MOTO" in s
        or "CARRO" in s
    ):
        return "IPVA"
    if (
        s.startswith("IPTU")
        or "PREDIAL" in s
        or "TERRITORIAL" in s
        or "URBANO" in s
        or "IMPOSTO PREDIAL" in s
        or "IMPOSTO TERRITORIAL" in s
    ):
        return "IPTU"
    if (
        s.startswith("ITCD")
        or "ITCMD" in s
        or "CAUSA MORTIS" in s
        or "DOACAO" in s
        or "HERANCA" in s
    ):
        return "ITCD"
    if (
        s.startswith("ITBI")
        or "INTER VIVOS" in s
        or "TRANSMISSAO" in s
        or "IMOVEIS" in s
        or "IMOVEL" in s
        or "BENS IMOVEIS" in s
    ):
        return "ITBI"
    return "OUTROS"


def carregar_fatores_por_tributo(caminho_estatisticas: Path) -> tuple[dict, float]:
    """
    Lê `estatisticas-processos.csv`, mapeia tipos de tributo e calcula
    a mediana de prazo em horas úteis a partir de `media_prazo` (dias):
    horas_uteis = media_prazo * (DIAS_UTEIS / 365) * HORAS_DIA.
    Normaliza por mediana global para obter multiplicadores robustos.
    """
    df_est = pd.read_csv(caminho_estatisticas)
    # Coluna de tipo de tributo consolidado
    df_est["tipo_tributo"] = df_est["DETIPOIMPOSTO"].map(_mapear_tipo_tributo)

    # Converte `media_prazo` (em dias corridos) para horas úteis
    # Aproximação: proporção de dias úteis no ano multiplicada por 8h/dia
    fator_dias_para_horas = (float(DIAS_UTEIS) / 365.0) * float(HORAS_DIA)
    df_est["media_prazo_horas_uteis"] = (
        pd.to_numeric(df_est["media_prazo"], errors="coerce").clip(lower=0)
        * fator_dias_para_horas
    )

    # Mediana do prazo (em horas úteis) por tributo
    med_por_tipo = (
        df_est.groupby("tipo_tributo")["media_prazo_horas_uteis"].median().dropna()
    )
    if med_por_tipo.empty:
        raise ValueError(
            "Não foi possível calcular a mediana do prazo por tipo de tributo"
        )

    med_global = med_por_tipo.median()
    if med_global <= 0 or np.isnan(med_global):
        raise ValueError("Não foi possível calcular a mediana global do prazo")

    # Usa horas medianas absolutas por tipo (em horas úteis)
    horas_por_tipo = med_por_tipo.to_dict()
    # Garante presença das chaves esperadas (fallback: mediana global)
    for k in ["ICMS", "ISS", "IPVA", "IPTU", "ITCD", "ITBI", "OUTROS"]:
        horas_por_tipo.setdefault(k, float(med_global))
    return horas_por_tipo, float(med_global)


def calcular_pesos(df: pd.DataFrame) -> pd.Series:
    """
    Calcula o esforço operacional (em horas) SEM depender do valor monetário.
    Usa horas úteis medianas por tipo de tributo, calibradas para um alvo
    global (por exemplo, 8h por processo), preservando diferenças relativas
    entre tipos via multiplicadores observados.
    """
    # Horas medianas por tipo e mediana global (em horas úteis)
    horas_por_tributo, mediana_global_horas = carregar_fatores_por_tributo(
        INPUT_ESTATISTICAS
    )

    # Mapeia as categorias do dataset principal para tipos consolidados
    if "descricao_receita" in df.columns:
        tipos_consolidados = df["descricao_receita"].apply(_mapear_tipo_tributo)
    elif "tipo_receita" in df.columns:
        tipos_consolidados = df["tipo_receita"].apply(_mapear_tipo_tributo)
    else:
        tipos_consolidados = pd.Series(["OUTROS"] * len(df), index=df.index)

    # Horas por registro: horas medianas do tipo; fallback: mediana global
    peso_horas = (
        tipos_consolidados.map(horas_por_tributo)
        .fillna(float(mediana_global_horas))
        .astype(float)
    )
    peso_horas = peso_horas.clip(lower=0.25)
    return peso_horas


def preprocessar_dados(caminho_csv: Path, prefixo_saida: Path) -> dict:
    """
    Lê o CSV original, deriva colunas de valor e peso_horas e salva artefatos
    prontos para os algoritmos (CSV filtrado + NPZ com vetores numéricos).
    """
    # Lê o CSV bruto com separador ";" e decimal "," (formatação PT-BR)
    df = pd.read_csv(caminho_csv, sep=";", decimal=",")

    # Valida presença das colunas mínimas necessárias
    if "valor_total_corrigido" not in df.columns:
        raise KeyError("Coluna obrigatória 'valor_total_corrigido' ausente do dataset")

    # Conversão de data (se existir)
    if "data_constituicao" in df.columns:
        df["data_constituicao"] = pd.to_datetime(
            df["data_constituicao"],
            format="%d/%m/%Y",
            errors="coerce",
        )

    # Benefício (valor monetário) e custo (peso em horas)
    df["valor"] = df["valor_total_corrigido"].astype(float).clip(lower=0.0)
    df["peso_horas"] = calcular_pesos(df)

    # Remove linhas inválidas (valor/peso não positivos) e recompõe índices
    df_alg = df[(df["valor"] > 0) & (df["peso_horas"] > 0)].reset_index(drop=True)

    # Constrói vetores numéricos para consumo direto pelos algoritmos
    vetor_valores = df_alg["valor"].to_numpy(dtype=float)
    vetor_pesos = df_alg["peso_horas"].to_numpy(dtype=float)

    # Persistência: CSV filtrado, NPZ compacto com arrays e índices
    prefixo_saida.parent.mkdir(parents=True, exist_ok=True)
    df_alg.to_csv(f"{prefixo_saida}.csv", index=False)
    np.savez_compressed(
        f"{prefixo_saida}.npz",
        valores=vetor_valores,
        pesos=vetor_pesos,
        index=df_alg.index.values,
    )

    # Metadados para rastreabilidade e auditoria
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
    # Sem calibração global: usa horas medianas absolutas por tributo
    args = parser.parse_args()

    meta = preprocessar_dados(args.entrada, args.prefixo_saida)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    executar()
