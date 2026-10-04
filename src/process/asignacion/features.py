"""Features del problema de asignación: elegibilidad y puntaje de proveedores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from process.config import PESO_CLASIFICACION
from process.data.loaders import AsignacionData


def experiencia_proveedor_material(data: AsignacionData) -> pd.DataFrame:
    """Unidades históricas fabricadas por (proveedor, material)."""
    return (
        data.historial.groupby(["id", "material"], as_index=False)["unidades_fabricadas"]
        .sum()
        .rename(columns={"unidades_fabricadas": "unidades_historicas"})
    )


def score_proveedores(
    data: AsignacionData,
    peso_clasificacion: float = 0.5,
    peso_experiencia: float = 0.5,
) -> pd.DataFrame:
    """Puntaje 0–1 de cada proveedor = clasificación + experiencia.

    - Clasificación: se mapea con ``config.PESO_CLASIFICACION`` y se normaliza.
    - Experiencia: ``log1p(unidades históricas totales)`` normalizado. Se usa el
      volumen total porque los materiales de las órdenes actuales no aparecen
      en el historial (no hay match a nivel material).
    """
    clasif = data.proveedores.drop_duplicates("id")[["id", "descripcion_clasificacion_proveedor"]]
    clasif = clasif.rename(columns={"descripcion_clasificacion_proveedor": "clasificacion"})

    exp = data.historial.groupby("id", as_index=False)["unidades_fabricadas"].sum()
    exp = exp.rename(columns={"unidades_fabricadas": "unidades_historicas"})

    df = data.capacidad.merge(clasif, on="id", how="left").merge(exp, on="id", how="left")
    df["unidades_historicas"] = df["unidades_historicas"].fillna(0)

    max_peso = max(PESO_CLASIFICACION.values())
    df["score_clasificacion"] = df["clasificacion"].map(PESO_CLASIFICACION).fillna(0) / max_peso
    log_exp = np.log1p(df["unidades_historicas"])
    df["score_experiencia"] = log_exp / log_exp.max() if log_exp.max() > 0 else 0.0

    total = peso_clasificacion + peso_experiencia
    df["score"] = (
        peso_clasificacion * df["score_clasificacion"] + peso_experiencia * df["score_experiencia"]
    ) / total
    return df


def proveedores_elegibles(data: AsignacionData, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pares (orden, proveedor) válidos.

    Un par es válido si el proveedor fabrica el tipo de producto de la orden,
    tiene capacidad > 0 y la orden cabe completa en su capacidad.
    """
    if scores is None:
        scores = score_proveedores(data)
    prov = data.proveedores[["id", "denominacion_tipo_de_producto"]].merge(
        scores[["id", "capacidad_disponible", "score"]], on="id"
    )
    prov = prov[prov["capacidad_disponible"] > 0]
    pares = data.ordenes.merge(prov, on="denominacion_tipo_de_producto", how="inner")
    pares = pares[pares["minutos_totales_de_la_orden"] <= pares["capacidad_disponible"]]
    return pares.reset_index(drop=True)
