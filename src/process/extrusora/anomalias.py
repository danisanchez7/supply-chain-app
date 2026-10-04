"""Detección de anomalías en el proceso de extrusión."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def detectar_anomalias(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Usa IsolationForest sobre las curvas de temperatura para detectar anomalías."""
    cols_temp = [c for c in df.columns if c.startswith("temp_barril_")]
    out = df.copy()

    # Rellenar NaN temporales si los hubiera (solo para que el modelo pueda predecir)
    X = out[cols_temp].fillna(out[cols_temp].median())

    iso = IsolationForest(contamination=contamination, random_state=42)
    out["anomalia"] = iso.fit_predict(X) == -1
    return out


def ranking_turnos_anomalos(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve los turnos ordenados por % de registros anómalos."""
    if "anomalia" not in df.columns:
        df = detectar_anomalias(df)

    return (
        df.groupby(["turno_id", "fecha_turno", "turno"], observed=True)["anomalia"]
        .agg(pct_anomalias="mean", total_registros="count")
        .reset_index()
        .sort_values("pct_anomalias", ascending=False)
    )
