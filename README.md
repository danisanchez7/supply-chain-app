# Supply Chain & Manufacturing Optimization Dashboard

Plataforma integral de soporte a la toma de decisiones diseñada para resolver problemas complejos de manufactura, logística y eficiencia energética. Combina **Investigación de Operaciones (Optimización MILP)** y **Machine Learning (Clustering, Árboles de Decisión y Detección de Anomalías)** para diagnosticar cuellos de botella y prescribir estrategias operativas rentables.

## Lo que aborda este proyecto (Business Value)

Este sistema fue construido con una mentalidad orientada al negocio, resolviendo tres grandes "Dolores Operativos" (Pain Points) comunes en la industria:

1. **Saturación de Capacidad (Logística):** Resuelve el desafío de asignar miles de órdenes a proveedores cuando la capacidad instalada es insuficiente, garantizando matemáticamente que los clientes más importantes reciban su producto primero (Maximizando el Time-to-Market).
2. **Volatilidad Energética (Piso de Planta):** Erradica las prácticas empíricas ("a sentimiento") de los operadores de maquinaria industrial. Encuentra estadísticamente la configuración de temperatura exacta en el PLC que minimiza la factura eléctrica mensual por tonelada producida.
3. **Complejidad del Portafolio (Retail):** Protege contra roturas de stock en catálogos gigantescos cruzados por cientos de tiendas. Escala el pronóstico de demanda modelando el comportamiento jerárquico para lograr planificaciones robustas ante estacionalidades cruzadas.

---

## Características Core (Módulos Algorítmicos)

| Módulo | Algoritmos / Tecnología | Descripción Técnica |
|---|----------|-------|
| **1. Asignación (MILP)** | `PuLP` + Solver `HiGHS` | Utiliza Programación Lineal Entera Mixta (MILP) para encontrar la asignación óptima global (Quién fabrica qué) respetando exclusividades de material. Apoyado por heurísticas "Greedy" para un *Warm Start*. |
| **2. Secuenciación (WSPT)** | Regla Matemática `WSPT` | Una vez asignado el trabajo, la regla de Smith (Weighted Shortest Processing Time) ordena la fila de cada proveedor minimizando el tiempo de finalización ponderado. |
| **3. Eficiencia (Extrusora)** | `Isolation Forest` + `K-Means` + `ANOVA` | Filtra ruido de sensores IoT mediante Detección de Anomalías, clusteriza las "recetas" operativas con K-Means, y valida la significancia del ahorro energético con la prueba de Kruskal-Wallis. |
| **4. Demanda (Panel Data)** | `LightGBM` (Gradient Boosting) | Modela decenas de series de tiempo simultáneamente (Nivel 8: Tienda x Categoría) en un único modelo de Panel Data masivo para capturar efectos cruzados (Festivos, Lags temporales). |

---

## Estructura del Proyecto (Arquitectura)

El proyecto sigue una arquitectura de diseño escalable y acoplada a las mejores prácticas de la ingeniería de software. Separa completamente el *frontend* interactivo (Dashboard) de la lógica *backend* matemática, empaquetando esta última (`src/process`) como si fuera una librería instalable.

```text
supply-chain-app/
├── app.py                      # Frontend (Dashboard interactivo en Streamlit)
├── pyproject.toml              # Definición de paquete y dependencias (gestor: uv)
├── uv.lock                     # Lockfile para garantizar builds reproducibles
├── README.md                   # Documentación principal
│
├── data/                       # Base de datos local (ETL)
│   └── raw/                    # Datasets particionados (Logística, Demanda M5, Sensores Extrusora)
│
├── docs/                       # Documentación teórica
│   └── enunciados/             # Requisitos y casos de negocio originales (Jupyter Notebooks)
│
├── notebooks/                  # Entornos interactivos de experimentación
│   ├── 01_asignacion_ordenes.ipynb
│   ├── 02_extrusora_turnos.ipynb
│   └── 03_pronostico_demanda.ipynb
│
├── tests/                      # Suite de pruebas unitarias (Pytest)
│
└── src/process/                # LÓGICA DE NEGOCIO Y MODELOS CORE
    ├── asignacion/             # Motor MILP (Constraints, Función Objetivo) y heurística WSPT
    ├── demanda/                # Feature Engineering y modelos jerárquicos LightGBM
    ├── extrusora/              # Análisis de energía, limpieza de telemetría IoT y tests estadísticos
    ├── data/                   # Data Loaders (Conectores a las fuentes CSV/Parquet)
    └── config.py               # Variables globales del sistema
```

---

## Cómo correrlo localmente (Guía Rápida)

Este proyecto utiliza **[`uv`](https://docs.astral.sh/uv/)**, el estándar moderno ultrarrápido (escrito en Rust) para la gestión de entornos de Python.

### 1. Clonar el Repositorio
```bash
git clone https://github.com/danisanchez7/supply-chain-app.git
cd supply-chain-app
```

### 2. Sincronizar Dependencias
```bash
uv sync
```
*(Esto crea automáticamente el entorno virtual (`.venv`) e instala todas las librerías pesadas (LightGBM, PuLP, Pandas, Streamlit) en cuestión de segundos, aislando el proyecto de tu sistema global).*

### 3. Lanzar el Dashboard
```bash
uv run streamlit run app.py
```
*(Se abrirá automáticamente la aplicación interactiva de diagnóstico en tu navegador web predeterminado).*

---

## Despliegue con Docker (Listo para Cloud)

El proyecto está empaquetado y listo para ser desplegado en servicios de la nube (AWS EC2, Google Cloud Run, Azure App Service) garantizando portabilidad aislando el entorno:

```bash
# 1. Construir la imagen de Docker
docker build -t supply-chain-app .

# 2. Levantar el contenedor
docker run -p 8501:8501 supply-chain-app
```
*Accede a `http://localhost:8501` en tu navegador.*

---

## Pruebas y Calidad de Código

El repositorio cuenta con estrictos estándares de formato, tipado y testing.

```bash
uv run pytest                # Ejecutar tests de integridad de módulos lógicos
uv run ruff check . --fix    # Linter para detectar bad smells en el código
uv run ruff format .         # Formateador automático (Estilo PEP8)
```
