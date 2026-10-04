# Supply Chain Optimization System

Plataforma integral de soporte a la toma de decisiones diseñada para resolver problemas complejos de manufactura y distribución. Combina **Operations Research (Optimización MILP)**, **Machine Learning (LightGBM)** y una **Interfaz Multimodal (IA Copiloto)** para simular y ejecutar estrategias operativas.

## Características Principales

| Módulo | Tecnología Core | Descripción |
|---|----------|-------|
| **1. Asignación Óptima (MILP)** | `PuLP` + `HiGHS` | Asigna miles de órdenes a talleres maximizando la prioridad entregada. Respeta restricciones estrictas de capacidad y exclusividad de materiales por proveedor. Supera a heurísticas tradicionales evitando cuellos de botella subóptimos. |
| **2. Secuenciación en Planta** | Regla `WSPT` | Secuencia la producción dentro de cada taller utilizando el tiempo de procesamiento más corto ponderado, minimizando el Time to Market global. |
| **3. Pronóstico de Demanda** | `LightGBM` + `MAPIE` | Forecast jerárquico a nivel tienda utilizando series de tiempo. Proyecta hasta 90 días capturando estacionalidad y macrotendencias. Emplea Conformal Prediction para entregar intervalos matemáticamente garantizados (90%) para el cálculo de Stock de Seguridad. |
| **4. Asistente IA (Copiloto)** | Simulación `GenAI` | Motor de inferencia integrado en la interfaz gráfica que lee los KPIs matemáticos en tiempo real y sugiere planes de acción estratégicos (Insights). |

---

## Arquitectura y Almacenamiento

* **Estructura Modular:** Lógica matemática aislada en la carpeta `src/process/`, permitiendo que el proyecto funcione como una librería de Python instalable y altamente escalable.
* **Compresión Parquet:** Para cumplir con los límites de GitHub (100 MB) y reducir los tiempos de carga en memoria a milisegundos, los pesados datasets en CSV fueron migrados a `.parquet` (`115MB -> 31MB`). 

> **Nota para Paso a Producción:** Las bases de datos se incluyen temporalmente de manera local en `data/raw/` para que la app corra *out-of-the-box* tras clonar (MVP). En un despliegue real, se conectará el `src/process/data/loaders.py` a un bucket S3, GCS o un Data Warehouse.

---

## Cómo correrlo localmente (Guía Rápida)

Este proyecto utiliza **[`uv`](https://docs.astral.sh/uv/)**, un gestor de dependencias ultrarrápido escrito en Rust.

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/supply-chain-optimization.git
cd supply-chain-optimization
```

### 2. Instalar el Entorno y Dependencias
```bash
uv sync
```
*Esto creará automáticamente el entorno virtual (`.venv`) e instalará `streamlit`, `lightgbm`, `pulp`, `mapie` y demás librerías en segundos.*

### 3. Lanzar la Plataforma
```bash
uv run streamlit run app.py
```
*Se abrirá una pestaña en tu navegador web por defecto con el Dashboard Interactivo.*

---

## Despliegue con Docker (Opcional)

Si deseas empaquetar la aplicación para un entorno Cloud (AWS, Azure, GCP) utilizando el `Dockerfile` provisto (que debes crear o tener disponible en la raíz):

```bash
docker build -t supply-chain-app .
docker run -p 8501:8501 supply-chain-app
```
Accede a `http://localhost:8501`.

---

## Pruebas y Calidad de Código

El repositorio cuenta con estrictos estándares de formato y testing:

```bash
uv run pytest                # Ejecutar tests de integridad de datos y modelos
uv run ruff check . --fix    # Linter automático
uv run ruff format .         # Formateador de código
```
