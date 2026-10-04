"""Configuración central del proyecto: rutas y constantes.

Todas las rutas se resuelven desde la raíz del repositorio, de modo que el
código funciona igual desde notebooks, scripts o tests sin importar el
directorio de trabajo actual.
"""

from pathlib import Path

# --- Rutas base -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # Datos originales (inmutables, nunca se editan)
INTERIM_DIR = DATA_DIR / "interim"  # Datos intermedios / limpios
PROCESSED_DIR = DATA_DIR / "processed"  # Datasets finales para modelar

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Datos crudos por problema ---------------------------------------------
# Módulo 1: asignación de órdenes de trabajo a talleres
ASIGNACION_RAW = RAW_DIR / "asignacion_ordenes"
# Módulo 2 - punto 1: extrusora (KPI energético kWh/ton)
EXTRUSORA_RAW = RAW_DIR / "extrusora"
# Módulo 2 - punto 2: pronóstico de demanda (dataset tipo M5)
DEMANDA_RAW = RAW_DIR / "demanda_m5"

# --- Constantes de negocio --------------------------------------------------
RANDOM_STATE = 42

# Importancia de la clasificación de proveedores (mayor = más importante).
# SUPUESTO: confirmar con el negocio.
PESO_CLASIFICACION = {
    "Confianza": 4,
    "Acompañamiento": 3,
    "Potenciales": 2,
    "Oportunidad": 1,
}

# Turnos de 12 h de la extrusora: Mañana 06:00–17:59, Noche 18:00–05:59
TURNO_INICIO_MANANA = 6
TURNO_INICIO_NOCHE = 18

# Horizonte de pronóstico de demanda (días)
HORIZONTE_DEMANDA = 28


def ensure_dirs() -> None:
    """Crea las carpetas de salida si no existen."""
    for d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
