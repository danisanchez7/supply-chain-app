"""Modelo MILP de asignación de órdenes a proveedores (PuLP 4 + HiGHS).

Optimización lexicográfica en dos etapas:

1. Maximizar  Σ prioridad_o · x[o,p]   (órdenes entregadas ponderadas).
2. Fijando la etapa 1 (con tolerancia opcional), maximizar
   Σ score_p · x[o,p]  (preferir proveedores con más experiencia/clasificación).

Restricciones:
- Cada orden se asigna a lo sumo a un proveedor.
- Minutos asignados a cada proveedor <= capacidad disponible.
- Un material -> un único proveedor, y todas sus órdenes van juntas
  (x[o,p] = y[m,p] para toda orden o del material m; Σ_p y[m,p] <= 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp


@dataclass
class ResultadoAsignacion:
    asignacion: pd.DataFrame  # filas de `pares` con x = 1
    estado: str
    objetivo_prioridad: float
    objetivo_score: float


def resolver_asignacion(
    pares: pd.DataFrame,
    capacidad: pd.Series,
    tolerancia: float = 0.0,
    time_limit: int = 120,
    msg: bool = False,
) -> ResultadoAsignacion:
    """Resuelve el MILP.

    Args:
        pares: salida de ``proveedores_elegibles`` (orden, material, id, prioridad,
            minutos_totales_de_la_orden, score).
        capacidad: minutos disponibles indexados por id de proveedor.
        tolerancia: ignorado (se usa optimización lexicográfica estricta en 1 etapa).
        time_limit: segundos máximos para el solver.
    """
    pares = pares.reset_index(drop=True)
    idx = list(pares.index)
    # PuLP 4 no acepta escalares numpy: trabajar con tipos nativos de Python
    minutos = [float(v) for v in pares["minutos_totales_de_la_orden"]]
    prioridad = [float(v) for v in pares["prioridad"]]
    score = [float(v) for v in pares["score"]]
    cap = {p: float(c) for p, c in capacidad.items()}

    prob = pulp.LpProblem("asignacion", pulp.LpMaximize)
    x = prob.add_variable_dict("x", idx, cat="Binary")

    # 1) Cada orden a lo sumo a un proveedor
    for _, rows in pares.groupby("orden").indices.items():
        prob += pulp.lpSum(x[i] for i in rows) <= 1

    # 2) Capacidad por proveedor
    for p, rows in pares.groupby("id").indices.items():
        prob += pulp.lpSum(minutos[i] * x[i] for i in rows) <= cap[p]

    # 3) Un material -> un proveedor, todas sus órdenes juntas
    #    (solo hace falta modelarlo cuando el material tiene > 1 orden)
    n_ordenes_mat = pares.groupby("material")["orden"].nunique()
    for m in n_ordenes_mat[n_ordenes_mat > 1].index:
        gm = pares[pares["material"] == m]
        y = {p: prob.add_variable(f"y_{m}_{p}", cat="Binary") for p in gm["id"].unique()}
        prob += pulp.lpSum(y.values()) <= 1
        for i, p in zip(gm.index, gm["id"], strict=True):
            prob += x[i] - y[p] == 0
        # Si un proveedor no es elegible para alguna orden del material, no puede tomarlo
        ordenes_m = set(gm["orden"])
        for p, gp in gm.groupby("id"):
            if set(gp["orden"]) != ordenes_m:
                prob += y[p] == 0

    # Arranque en caliente (warm start) con la heurística greedy
    g_asignadas = set(asignacion_greedy(pares, capacidad).index)
    for i in idx:
        x[i].setInitialValue(1.0 if i in g_asignadas else 0.0)

    solver = pulp.HiGHS(msg=msg, timeLimit=time_limit, warmStart=True)

    # Optimización lexicográfica en 1 sola etapa:
    # Usar escala exponencial para las prioridades (10^prioridad) asegura que el solver
    # prefiera estrictamente entregar una orden de prioridad 3 antes que cualquier cantidad de prioridad 1.
    obj = pulp.lpSum(((10 ** prioridad[i]) + 0.0001 * score[i]) * x[i] for i in idx)
    prob.setObjective(obj)
    stats = prob.solve(solver)
    estado = stats.status.name

    sel = [i for i in idx if (x[i].value() or 0) > 0.5]
    asignacion = pares.loc[sel].reset_index(drop=True)
    return ResultadoAsignacion(
        asignacion=asignacion,
        estado=estado,
        objetivo_prioridad=float(asignacion["prioridad"].sum()),
        objetivo_score=float(asignacion["score"].sum()),
    )


def asignacion_greedy(pares: pd.DataFrame, capacidad: pd.Series) -> pd.DataFrame:
    """Heurística de arranque: asigna por bloque de material respetando restricciones."""
    restante = capacidad.astype(float).to_dict()
    asignadas: list[int] = []
    
    # Agrupar por material para asegurar que todas las órdenes del mismo material
    # vayan al mismo proveedor, evitando romper la restricción 3.
    pares_g = pares.copy()
    pares_g["orden_idx"] = pares_g.index
    
    mat_aggs = pares_g.groupby("material").agg({
        "prioridad": "sum",
        "minutos_totales_de_la_orden": "sum",
        "orden": "nunique"
    }).sort_values(["prioridad", "minutos_totales_de_la_orden"], ascending=[False, True])
    
    for material, row_m in mat_aggs.iterrows():
        minutos = row_m["minutos_totales_de_la_orden"]
        n_ordenes = row_m["orden"]
        
        cands = pares_g[pares_g["material"] == material]
        
        # Un proveedor solo es elegible si puede tomar TODAS las órdenes del material
        prov_validos = []
        for p, gp in cands.groupby("id"):
            if len(gp) == n_ordenes and restante[p] >= minutos:
                prov_validos.append((p, gp["score"].mean(), gp["orden_idx"].tolist()))
                
        # Ordenar por mejor score
        prov_validos.sort(key=lambda x: x[1], reverse=True)
        
        if prov_validos:
            p_elegido, _, indices = prov_validos[0]
            restante[p_elegido] -= minutos
            asignadas.extend(indices)
            
    return pares.loc[asignadas].reset_index(drop=True)
