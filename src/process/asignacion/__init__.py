"""Módulo 1 — Sistema de decisión para asignar órdenes de trabajo a talleres.

Flujo:
1. ``features``: elegibilidad orden-proveedor y puntaje de proveedor.
2. ``modelo``: MILP lexicográfico (prioridad -> calidad del proveedor).
3. ``secuenciacion``: orden de producción por proveedor (WSPT).
4. ``escenarios``: what-if.
"""

from process.asignacion.escenarios import Escenario, comparar_escenarios, correr_escenario
from process.asignacion.features import (
    experiencia_proveedor_material,
    proveedores_elegibles,
    score_proveedores,
)
from process.asignacion.modelo import asignacion_greedy, resolver_asignacion
from process.asignacion.secuenciacion import secuenciar_wspt, tiempo_ponderado

__all__ = [
    "Escenario",
    "asignacion_greedy",
    "comparar_escenarios",
    "correr_escenario",
    "experiencia_proveedor_material",
    "proveedores_elegibles",
    "resolver_asignacion",
    "score_proveedores",
    "secuenciar_wspt",
    "tiempo_ponderado",
]
