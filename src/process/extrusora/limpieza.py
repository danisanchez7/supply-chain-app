"""Limpieza de datos de la extrusora."""

import pandas as pd


def limpiar_extrusora(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra registros inválidos, paradas de máquina y outliers extremos.

    - Elimina filas con NaN en variables críticas (KPI, consumo, alimento).
    - Filtra periodos donde la máquina no está operando o está en transición
      (alimento_de_polvo_rot <= 0.05, consumos <= 0).
    - Elimina outliers extremos en KPI que resultan de dividir por valores
      muy cercanos a 0.
    """
    cols_criticas = ["kpi", "alimento_de_polvo_rot", "md1_power", "md2_power"]
    out = df.dropna(subset=cols_criticas).copy()

    # Filtro de operación
    mask_operando = (
        (out["alimento_de_polvo_rot"] > 0.05) & (out["md1_power"] > 0) & (out["md2_power"] > 0)
    )
    out = out[mask_operando]

    # Filtro IQR para quitar la "cola" superior de outliers matemáticos (KPIs de miles)
    q1 = out["kpi"].quantile(0.25)
    q3 = out["kpi"].quantile(0.75)
    iqr = q3 - q1
    # Usamos un umbral laxo (3x IQR en vez de 1.5x) para no quitar anomalías reales,
    # solo errores de sensor o divisiones por ~0.
    lim_sup = q3 + 3 * iqr
    lim_inf = max(0, q1 - 3 * iqr)

    out = out[(out["kpi"] >= lim_inf) & (out["kpi"] <= lim_sup)]
    return out.reset_index(drop=True)
