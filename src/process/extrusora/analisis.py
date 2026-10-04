"""Pruebas estadísticas: eficiencia entre turnos y curvas de temperatura."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols


def test_manana_vs_noche(df: pd.DataFrame) -> dict[str, Any]:
    """Prueba si hay diferencia significativa en el KPI entre turnos.

    - Usa Mann-Whitney U para una comparación global no paramétrica.
    - Usa OLS controlando por el efecto del producto.
    """
    m = df[df["turno"] == "Mañana"]["kpi"].dropna()
    n = df[df["turno"] == "Noche"]["kpi"].dropna()

    mw = mannwhitneyu(m, n)

    mod = ols("kpi ~ C(turno) + C(producto)", data=df).fit()
    p_noche = mod.pvalues.get("C(turno)[T.Noche]", None)
    coef_noche = mod.params.get("C(turno)[T.Noche]", None)

    return {
        "mann_whitney_pvalue": float(mw.pvalue),
        "ols_pvalue_noche": float(p_noche) if p_noche is not None else None,
        "ols_coef_noche": float(coef_noche) if coef_noche is not None else None,
        "media_manana": float(m.mean()),
        "media_noche": float(n.mean()),
        "mediana_manana": float(m.median()),
        "mediana_noche": float(n.median()),
    }


def curvas_temperatura_optimas(df: pd.DataFrame, n_clusters: int = 5) -> dict[str, Any]:
    """Agrupa las temperaturas en clústeres y prueba si alguno consume menos energía."""
    cols_temp = [c for c in df.columns if c.startswith("temp_barril_")]
    valid = df.dropna(subset=cols_temp + ["md1_power", "md2_power"]).copy()

    X = StandardScaler().fit_transform(valid[cols_temp])
    km = KMeans(n_clusters=n_clusters, random_state=42)
    valid["cluster_temp"] = km.fit_predict(X)

    valid["consumo_total"] = valid["md1_power"] + valid["md2_power"]

    grupos = [g["consumo_total"].values for _, g in valid.groupby("cluster_temp")]
    kw = kruskal(*grupos)

    resumen = (
        valid.groupby("cluster_temp")
        .agg(
            consumo_medio=("consumo_total", "mean"),
            kpi_medio=("kpi", "mean"),
            temp_barril_2_medio=("temp_barril_2", "mean"),
            temp_barril_11_medio=("temp_barril_11", "mean"),
            n=("consumo_total", "count"),
        )
        .reset_index()
    )

    return {
        "kruskal_pvalue": float(kw.pvalue),
        "resumen_clusters": resumen,
        "df_con_clusters": valid,
    }


def detectar_anomalias_isolation_forest(
    df: pd.DataFrame, contamination: float = 0.05
) -> pd.DataFrame:
    """Aplica Isolation Forest para detectar anomalías de consumo energético.

    Identifica comportamientos atípicos en el consumo (kW) y la eficiencia (kpi)
    sin requerir reglas duras manuales.
    """
    df_out = df.copy()
    features = ["md1_power", "md2_power", "kpi"]
    valid = df_out.dropna(subset=features)

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)

    # -1 para anomalía, 1 para normal
    preds = iso.fit_predict(valid[features])

    df_out.loc[valid.index, "anomalia_iso"] = preds == -1
    return df_out
