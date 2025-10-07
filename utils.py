import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

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
    with open(f"{prefixo_saida}.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)


def system_metrics() -> Dict[str, Any]:
    """Retorna métricas simples do ambiente de execução."""
    metrics: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    # Tenta capturar memória pico (Linux)
    try:
        import resource  # type: ignore

        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss: KiB em Linux
        rss_mb = float(usage.ru_maxrss) / 1024.0
        metrics["max_rss_mb"] = rss_mb
    except Exception:
        pass
    return metrics


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
    return {
        "algorithm": algorithm,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "inputs": {
            k: (str(v) if isinstance(v, (Path,)) else v) for k, v in inputs.items()
        },
        "params": params,
        "counts": {
            "n_candidates": int(len(df_candidates)),
            "n_selected": int(len(df_selected)),
        },
        "totals": {
            "hours_total": float(df_selected["peso_horas"].sum())
            if not df_selected.empty
            else 0.0,
            "value_total": float(df_selected["valor"].sum())
            if not df_selected.empty
            else 0.0,
        },
        "runtime": {
            "seconds": float(elapsed_seconds),
            **system_metrics(),
        },
    }
