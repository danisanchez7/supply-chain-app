"""Escenarios what-if sobre el sistema de asignación."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from process.asignacion.features import proveedores_elegibles, score_proveedores
from process.asignacion.modelo import resolver_asignacion
from process.asignacion.secuenciacion import secuenciar_wspt, tiempo_ponderado
from process.data.loaders import AsignacionData


@dataclass
class Escenario:
    nombre: str = "base"
    capacidad_factor: float = 1.0  # multiplica la capacidad de todos
    capacidad_override: dict[int, float] = field(default_factory=dict)  # id -> minutos
    excluir_proveedores: list[int] = field(default_factory=list)
    prioridad_override: dict[int, int] = field(default_factory=dict)  # orden -> 1..3
    peso_clasificacion: float = 0.5
    peso_experiencia: float = 0.5
    tolerancia: float = 0.0


@dataclass
class ResultadoEscenario:
    escenario: Escenario
    secuencia: pd.DataFrame
    kpis: dict


def _aplicar(data: AsignacionData, esc: Escenario) -> AsignacionData:
    capacidad = data.capacidad.copy()
    capacidad["capacidad_disponible"] = capacidad["capacidad_disponible"] * esc.capacidad_factor
    for p, minutos in esc.capacidad_override.items():
        capacidad.loc[capacidad["id"] == p, "capacidad_disponible"] = minutos
    capacidad.loc[capacidad["id"].isin(esc.excluir_proveedores), "capacidad_disponible"] = 0

    ordenes = data.ordenes.copy()
    for o, prio in esc.prioridad_override.items():
        ordenes.loc[ordenes["orden"] == o, "prioridad"] = prio

    return AsignacionData(ordenes, data.proveedores, data.historial, capacidad)


def correr_escenario(
    data: AsignacionData, esc: Escenario | None = None, **kwargs
) -> ResultadoEscenario:
    """Ejecuta features -> MILP -> secuenciación y devuelve KPIs."""
    esc = esc or Escenario()
    d = _aplicar(data, esc)
    scores = score_proveedores(d, esc.peso_clasificacion, esc.peso_experiencia)
    pares = proveedores_elegibles(d, scores)
    cap = scores.set_index("id")["capacidad_disponible"]

    res = resolver_asignacion(pares, cap, tolerancia=esc.tolerancia, **kwargs)
    seq = secuenciar_wspt(res.asignacion)

    o = d.ordenes
    kpis = {
        "escenario": esc.nombre,
        "estado": res.estado,
        "ordenes_asignadas": len(seq),
        "ordenes_total": len(o),
        "pct_ordenes": len(seq) / len(o),
        "prioridad_asignada": res.objetivo_prioridad,
        "prioridad_total": float(o["prioridad"].sum()),
        "minutos_asignados": float(seq["minutos_totales_de_la_orden"].sum()),
        "capacidad_total": float(cap.sum()),
        "proveedores_usados": int(seq["id"].nunique()),
        "score_medio": float(seq["score"].mean()) if len(seq) else 0.0,
        "tiempo_ponderado": tiempo_ponderado(seq),
    }
    return ResultadoEscenario(esc, seq, kpis)


def comparar_escenarios(data: AsignacionData, escenarios: list[Escenario], **kw) -> pd.DataFrame:
    return pd.DataFrame([correr_escenario(data, e, **kw).kpis for e in escenarios])
