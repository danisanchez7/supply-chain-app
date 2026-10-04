"""Funciones de lectura de los datos crudos.

Reglas:
- Solo se LEE de ``data/raw`` (nunca se sobrescribe).
- Aquí se hace limpieza *mínima* y determinística: nombres de columnas,
  tipos de datos y valores inválidos. La limpieza analítica (outliers,
  imputación, features) va en los módulos de cada problema.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from process.config import ASIGNACION_RAW, DEMANDA_RAW, EXTRUSORA_RAW


def _snake(col: str) -> str:
    """'Temp Barril 2 ' -> 'temp_barril_2'."""
    import re
    import unicodedata

    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode()
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col.strip().lower())
    return col.strip("_")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=_snake)


# ---------------------------------------------------------------------------
# Módulo 1 — Asignación de órdenes de trabajo
# ---------------------------------------------------------------------------
@dataclass
class AsignacionData:
    """Contenedor de las 4 tablas del problema de asignación."""

    ordenes: pd.DataFrame  # Órdenes de trabajo a asignar
    proveedores: pd.DataFrame  # Proveedor x tipo de producto + clasificación
    historial: pd.DataFrame  # Unidades fabricadas por proveedor x material
    capacidad: pd.DataFrame  # Minutos disponibles por proveedor


def load_asignacion() -> AsignacionData:
    """Carga las tablas del Módulo 1 con columnas en snake_case.

    Columnas resultantes:
    - ordenes: orden, unidades_de_la_op, material, denominacion_tipo_de_producto,
      sam, prioridad, minutos_totales_de_la_orden
    - proveedores: id, descripcion_clasificacion_proveedor,
      denominacion_tipo_de_producto
    - historial: id, material, unidades_fabricadas
    - capacidad: id, capacidad_disponible
    """
    ordenes = _clean_columns(pd.read_csv(ASIGNACION_RAW / "Ordenes_de_trabajo.csv"))
    ordenes["prioridad"] = ordenes["prioridad"].astype("int8")
    ordenes["unidades_de_la_op"] = ordenes["unidades_de_la_op"].astype("int64")

    proveedores = _clean_columns(pd.read_csv(ASIGNACION_RAW / "Lista_de_proveedores.csv"))
    historial = _clean_columns(pd.read_csv(ASIGNACION_RAW / "Historial_de_ordenes.csv"))
    capacidad = _clean_columns(pd.read_csv(ASIGNACION_RAW / "Capacidad_de_proveedores.csv"))

    return AsignacionData(ordenes, proveedores, historial, capacidad)


# ---------------------------------------------------------------------------
# Módulo 2.1 — Extrusora
# ---------------------------------------------------------------------------
# Valores de error exportados desde Excel / historian del PLC
EXTRUSORA_NA_VALUES = ["Bad", "#DIV/0!", "#VALUE!"]


def load_extrusora() -> pd.DataFrame:
    """Carga el histórico de la extrusora.

    - Convierte 'Bad', '#DIV/0!', '#VALUE!' en NaN.
    - Parsea ``fecha`` (formato '29-Oct-22 12:42:00') y la usa ordenada.
    - Normaliza nombres: 'Temp Barril 2' -> 'temp_barril_2', 'KPI' -> 'kpi'.
    """
    df = pd.read_csv(EXTRUSORA_RAW / "historico_extrusora.csv", na_values=EXTRUSORA_NA_VALUES)
    df = _clean_columns(df)
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d-%b-%y %H:%M:%S")
    df = df.rename(columns={"producto_extruder": "producto"})
    df["producto"] = df["producto"].astype("category")
    return df.sort_values("fecha").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Módulo 2.2 — Demanda (estructura M5)
# ---------------------------------------------------------------------------
_ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_calendar() -> pd.DataFrame:
    return pd.read_parquet(DEMANDA_RAW / "calendar.parquet")


def load_sell_prices() -> pd.DataFrame:
    """Ocupa muy poca memoria gracias a Parquet."""
    return pd.read_parquet(DEMANDA_RAW / "sell_prices.parquet")


def load_sales(long: bool = False) -> pd.DataFrame:
    """Ventas diarias producto-tienda (30.490 series x 1.913 días).

    Args:
        long: si True devuelve formato largo (id, ..., d, ventas), que es el
            formato natural para modelos tipo LightGBM. Ojo: ~58M filas.
    """
    path = DEMANDA_RAW / "sales_train_validation.parquet"
    df = pd.read_parquet(path)

    if long:
        df = df.melt(id_vars=_ID_COLS, var_name="d", value_name="ventas")
        df["d"] = df["d"].astype("category")
    return df
