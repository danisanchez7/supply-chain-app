"""Tests del sistema de asignación con un caso pequeño y controlado."""

import pandas as pd
import pytest

from process.asignacion import resolver_asignacion, secuenciar_wspt, tiempo_ponderado


@pytest.fixture
def caso():
    # 4 órdenes, 2 proveedores (A: 100 min, B: 60 min)
    # o4 y o5 comparten material M -> deben ir al mismo proveedor
    pares = pd.DataFrame(
        {
            "orden": [1, 1, 2, 3, 3, 4, 4, 5, 5],
            "material": [10, 10, 20, 30, 30, 99, 99, 99, 99],
            "id": ["A", "B", "A", "A", "B", "A", "B", "A", "B"],
            "prioridad": [3, 3, 1, 2, 2, 3, 3, 3, 3],
            "minutos_totales_de_la_orden": [50, 50, 60, 40, 40, 30, 30, 20, 20],
            "score": [0.9, 0.2, 0.9, 0.9, 0.2, 0.9, 0.2, 0.9, 0.2],
        }
    )
    capacidad = pd.Series({"A": 100, "B": 60})
    return pares, capacidad


def test_restricciones(caso):
    pares, cap = caso
    res = resolver_asignacion(pares, cap)
    a = res.asignacion

    assert a["orden"].is_unique  # cada orden a un solo proveedor
    usados = a.groupby("id")["minutos_totales_de_la_orden"].sum()
    assert (usados <= cap[usados.index]).all()  # capacidad
    assert a.loc[a["material"] == 99, "id"].nunique() <= 1  # un material -> un proveedor
    # material 99: o todas sus órdenes o ninguna
    assert a["orden"].isin([4, 5]).sum() in (0, 2)


def test_optimo_prioridad(caso):
    pares, cap = caso
    res = resolver_asignacion(pares, cap)
    # Óptimo: o1,o3 (90) + o4,o5 (50) = 140 <= 160 -> prioridad 3+2+3+3 = 11
    assert res.objetivo_prioridad == 11


def test_wspt_minimiza_tiempo_ponderado():
    a = pd.DataFrame(
        {
            "id": ["A"] * 3,
            "orden": [1, 2, 3],
            "prioridad": [1, 3, 2],
            "minutos_totales_de_la_orden": [10, 10, 10],
        }
    )
    seq = secuenciar_wspt(a)
    assert seq["orden"].tolist() == [2, 3, 1]
    assert seq["fin_min"].tolist() == [10, 20, 30]
    assert tiempo_ponderado(seq) == 3 * 10 + 2 * 20 + 1 * 30
