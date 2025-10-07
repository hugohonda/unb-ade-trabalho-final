import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from config import DIAS_UTEIS, HORAS_DIA


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
    with open(f"{prefixo_saida}.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)


def system_metrics() -> Dict[str, Any]:
    """Retorna métricas simples do ambiente de execução."""
    metrics: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    return metrics


def format_brl(valor: float) -> str:
    """Formata número monetário em BRL (R$ 1.234,56) sem depender de locale."""
    try:
        s = f"{float(valor):,.2f}"
        s = s.replace(",", "_").replace(".", ",").replace("_", ".")
        return f"R$ {s}"
    except Exception:
        return f"R$ {valor}"


def build_summary(
    *,
    algorithm: str,
    inputs: Dict[str, Any],
    params: Dict[str, Any],
    df_candidates: pd.DataFrame,
    df_selected: pd.DataFrame,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Monta um resumo padronizado para as saídas dos runners."""
    n_candidates = int(len(df_candidates))
    n_selected = int(len(df_selected))

    hours_total = float(df_selected["peso_horas"].sum()) if n_selected else 0.0
    value_total = float(df_selected["valor"].sum()) if n_selected else 0.0

    # Conversões de esforço
    days_business_total = hours_total / float(HORAS_DIA) if hours_total else 0.0
    months_business_total = (
        days_business_total / (float(DIAS_UTEIS) / 12.0) if days_business_total else 0.0
    )

    # Médias úteis
    avg_hour_value = value_total / hours_total if hours_total > 0 else 0.0
    avg_value_per_process = value_total / n_selected if n_selected > 0 else 0.0
    avg_hours_per_process = hours_total / n_selected if n_selected > 0 else 0.0
    avg_business_days_per_process = (
        avg_hours_per_process / float(HORAS_DIA) if avg_hours_per_process > 0 else 0.0
    )
    selection_rate = (n_selected / n_candidates) if n_candidates > 0 else 0.0

    return {
        "algorithm": algorithm,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "params": params,
        "counts": {
            "n_candidates": n_candidates,
            "n_selected": n_selected,
            "selection_rate": round(selection_rate, 6),
        },
        "totals": {
            "hours_total": round(hours_total, 2),
            "days_business_total": round(days_business_total, 2),
            "months_business_total": round(months_business_total, 2),
            "value_total_brl": format_brl(value_total),
        },
        "metrics": {
            "avg_hour_value_brl": format_brl(avg_hour_value),
            "avg_value_per_process_brl": format_brl(avg_value_per_process),
            "avg_hours_per_process": round(avg_hours_per_process, 2),
            "avg_business_days_per_process": round(avg_business_days_per_process, 2),
        },
        "runtime": {
            "seconds": float(elapsed_seconds),
            **system_metrics(),
        },
    }
