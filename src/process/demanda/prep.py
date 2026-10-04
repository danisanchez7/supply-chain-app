"""Preparación de datos para el pronóstico de demanda jerárquico."""

from __future__ import annotations

import pandas as pd

from process.data import load_calendar, load_sales


def preparar_demanda_tiendas() -> pd.DataFrame:
    """Agrega las ventas diarias a nivel de tienda.

    M5 tiene 30.490 series a nivel producto-tienda (item_id, store_id).
    Para el modelo jerárquico y de forma eficiente para la memoria,
    agrupamos las ventas por tienda (10 tiendas).
    """
    df = load_sales(long=False)
    d_cols = [c for c in df.columns if c.startswith("d_")]
    # Agrupar por store_id y cat_id (como nivel de producto para el panel data)
    agg = df.groupby(["store_id", "cat_id"], observed=True)[d_cols].sum().reset_index()

    # Formato largo para series de tiempo
    df_long = agg.melt(id_vars=["store_id", "cat_id"], var_name="d", value_name="ventas")

    # Unir con el calendario
    cal = load_calendar()
    df_long = df_long.merge(cal[["d", "date", "wday", "month", "event_name_1"]], on="d", how="left")
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long.sort_values(["store_id", "cat_id", "date"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    return df_long


def crear_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Crea variables temporales (lags, promedios móviles) y de calendario."""
    if lags is None:
        lags = [1, 7, 14, 28]

    out = df.copy()

    # Features de calendario
    out["dia_semana"] = out["date"].dt.dayofweek
    out["es_fin_semana"] = out["dia_semana"].isin([5, 6]).astype(int)
    out["hay_evento"] = out["event_name_1"].notna().astype(int)

    # Lags por tienda y producto
    for lag in lags:
        out[f"lag_{lag}"] = out.groupby(["store_id", "cat_id"], observed=True)["ventas"].shift(lag)

    # Promedios móviles
    out["rolling_mean_7"] = out.groupby(["store_id", "cat_id"], observed=True)["ventas"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    out["rolling_mean_28"] = out.groupby(["store_id", "cat_id"], observed=True)["ventas"].transform(
        lambda x: x.shift(1).rolling(28).mean()
    )

    # Al usar lags, perdemos los primeros 'max(lags)' días
    # Asegurarse de no eliminar filas por culpa de event_name_1 (que es mayormente nulo)
    if "event_name_1" in out.columns:
        out = out.drop(columns=["event_name_1"])
        
    return out.dropna().reset_index(drop=True)
