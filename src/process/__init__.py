"""Paquete de análisis: asignación de órdenes, extrusora y pronóstico de demanda."""

__version__ = "0.1.0"


def main() -> None:
    from process.config import RAW_DIR

    print(f"process {__version__} — datos crudos en: {RAW_DIR}")
