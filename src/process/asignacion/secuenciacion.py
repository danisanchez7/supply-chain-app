"""Secuenciación de órdenes dentro de cada proveedor.

Cada proveedor se modela como una máquina que procesa sus órdenes en serie.
Para minimizar el tiempo de finalización ponderado Σ w_j C_j la regla óptima
en una máquina es WSPT (Smith, 1956): ordenar por w_j / p_j descendente,
con w_j = prioridad y p_j = minutos de la orden.
"""

from __future__ import annotations

import pandas as pd


def secuenciar_wspt(asignacion: pd.DataFrame) -> pd.DataFrame:
    """Devuelve la asignación con ``secuencia``, ``inicio_min`` y ``fin_min`` por proveedor."""
    df = asignacion.copy()
    df["ratio_wspt"] = df["prioridad"] / df["minutos_totales_de_la_orden"]
    df = df.sort_values(["id", "ratio_wspt"], ascending=[True, False])
    df["secuencia"] = df.groupby("id").cumcount() + 1
    df["fin_min"] = df.groupby("id")["minutos_totales_de_la_orden"].cumsum()
    df["inicio_min"] = df["fin_min"] - df["minutos_totales_de_la_orden"]
    return df.reset_index(drop=True)


def tiempo_ponderado(secuencia: pd.DataFrame) -> float:
    """Σ prioridad · tiempo de finalización (métrica de time-to-market)."""
    return float((secuencia["prioridad"] * secuencia["fin_min"]).sum())
