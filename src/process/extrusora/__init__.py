"""Módulo 2.1 — Análisis de la extrusora (KPI kWh/ton por turno)."""

from __future__ import annotations

import pandas as pd

from process.config import TURNO_INICIO_MANANA, TURNO_INICIO_NOCHE
from process.extrusora.analisis import curvas_temperatura_optimas, test_manana_vs_noche
from process.extrusora.anomalias import detectar_anomalias, ranking_turnos_anomalos
from process.extrusora.limpieza import limpiar_extrusora


def asignar_turno(df: pd.DataFrame, col_fecha: str = "fecha") -> pd.DataFrame:
    """Agrega columnas de turno de 12 h.

    - ``turno``: 'Mañana' (06:00–17:59) o 'Noche' (18:00–05:59).
    - ``fecha_turno``: fecha en que *inició* el turno. Un registro a las 02:00
      del día 15 pertenece al turno Noche del día 14.
    - ``turno_id``: identificador único, p. ej. '2022-11-14_Noche'.
    """
    out = df.copy()
    hora = out[col_fecha].dt.hour
    es_manana = (hora >= TURNO_INICIO_MANANA) & (hora < TURNO_INICIO_NOCHE)
    out["turno"] = pd.Categorical(
        es_manana.map({True: "Mañana", False: "Noche"}), categories=["Mañana", "Noche"]
    )
    out["fecha_turno"] = (out[col_fecha] - pd.Timedelta(hours=TURNO_INICIO_MANANA)).dt.normalize()
    out["turno_id"] = out["fecha_turno"].dt.strftime("%Y-%m-%d") + "_" + out["turno"].astype(str)
    return out


def resumen_por_turno(df: pd.DataFrame, col_kpi: str = "kpi") -> pd.DataFrame:
    """Estadísticos del KPI por turno (requiere haber llamado ``asignar_turno``)."""
    return (
        df.groupby(["turno_id", "fecha_turno", "turno"], observed=True)[col_kpi]
        .agg(n="count", media="mean", mediana="median", std="std")
        .reset_index()
        .sort_values("fecha_turno")
    )


__all__ = [
    "asignar_turno",
    "curvas_temperatura_optimas",
    "detectar_anomalias",
    "limpiar_extrusora",
    "ranking_turnos_anomalos",
    "resumen_por_turno",
    "test_manana_vs_noche",
]
