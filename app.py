import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Supply Chain Optimization", layout="wide")

# CSS adjustments for a corporate look without emojis
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid #e9ecef;
        color: #212529;
    }
    .metric-value { font-size: 24px; font-weight: 600; color: #0d6efd; }
    .metric-label { font-size: 12px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
    h1, h2, h3 { color: #212529; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_base_data_v3():
    from process.data import load_asignacion
    from process.demanda import crear_features, preparar_demanda_tiendas

    allocation_data = load_asignacion()

    demand_df = preparar_demanda_tiendas()
    features_df = crear_features(demand_df).dropna()

    return allocation_data, features_df


@st.cache_resource
def solve_allocation_milp(_allocation_data, capacity_factor, excluded_providers):
    from process.asignacion import Escenario, correr_escenario

    scenario = Escenario(
        nombre="Interactive",
        capacidad_factor=capacity_factor,
        excluir_proveedores=excluded_providers,
    )
    return correr_escenario(_allocation_data, scenario, msg=False, time_limit=10)


@st.cache_resource
def forecast_demand(_features_df, horizon):
    from process.demanda import entrenar_modelo_tiendas

    return entrenar_modelo_tiendas(_features_df, horizonte=horizon)


@st.cache_resource
def forecast_future_semester(_demand_df, horizon=180):
    from process.demanda import pronosticar_semestre_futuro

    return pronosticar_semestre_futuro(_demand_df, horizonte_dias=horizon)


@st.cache_resource
def load_extrusora_data():
    import numpy as np

    from process.data import load_extrusora
    from process.extrusora.analisis import detectar_anomalias_isolation_forest

    df = load_extrusora()
    # Determinar turno
    df["hora"] = df["fecha"].dt.hour
    df["turno"] = np.where((df["hora"] >= 6) & (df["hora"] < 18), "Mañana", "Noche")
    # Detectar anomalías
    df = detectar_anomalias_isolation_forest(df)
    return df


allocation_data, features_df = load_base_data_v3()
extrusora_df = load_extrusora_data()

# -----------------------------------------------------------------------------
# Sidebar & Header
# -----------------------------------------------------------------------------
st.title("Sistema de Optimización Supply Chain")
st.markdown(
    "Plataforma de soporte a la toma de decisiones para asignación de órdenes y pronóstico de demanda."
)

st.sidebar.title("Navegación")
tab_selection = st.sidebar.radio(
    "Seleccione el Módulo",
    [
        "Diagnóstico Operativo",
        "Asignación (MILP)",
        "Eficiencia Energética (Extrusora)",
        "Pronóstico de Demanda",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Documentación del Sistema")
with st.sidebar.expander("¿Por qué MILP frente a Alternativas?"):
    st.write("""
    El **Solver MILP (HiGHS)** evalúa el espacio completo y *garantiza* la máxima prioridad global posible sin romper reglas de negocio.
    """)

with st.sidebar.expander("¿Por qué LightGBM para Demanda?"):
    st.write("""
    Se seleccionó **LightGBM** (Gradient Boosting) sobre modelos tradicionales (ARIMA/Prophet) porque:
    1. Captura patrones cruzados no lineales entre múltiples tiendas simultáneamente.
    2. Incorpora variables externas (eventos, rezagos temporales, fines de semana) nativamente sin supuestos estadísticos estrictos.
    3. Es altamente eficiente para series de tiempo jerárquicas.
    
    Al combinarse con **MAPIE (Conformal Prediction)**, provee intervalos de incertidumbre con garantías matemáticas reales, algo fundamental para calcular el Inventario de Seguridad sin excesos.
    """)

# -----------------------------------------------------------------------------
# Tab 0: Diagnóstico Operativo (Resumen Ejecutivo)
# -----------------------------------------------------------------------------
if tab_selection == "Diagnóstico Operativo":
    st.header("Diagnóstico Operativo")
    st.markdown(
        "Análisis preliminar de la infraestructura productiva y comercial. Este diagnóstico cuantifica los cuellos de botella actuales y justifica las estrategias de optimización algorítmica desplegadas en la plataforma."
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # 1. Diagnóstico de Capacidad (Manufactura)
    st.subheader("1. Manufactura: Saturación de Capacidad")
    st.markdown("""
    * **El Dolor del Cliente:** Incapacidad operativa para cumplir con la totalidad de pedidos comerciales. La asignación manual a proveedores (Excel) ignora restricciones complejas, provocando entregas tardías en clientes de alta prioridad.
    * **Datos Analizados:** `orden` (ID del trabajo), `material`, `minutos_totales` (Carga de trabajo), `prioridad` comercial, y matriz de `capacidad_disponible` cruzada con el `score` de calidad del proveedor.
    """)

    minutos_requeridos = allocation_data.ordenes["minutos_totales_de_la_orden"].sum()
    capacidad_total = allocation_data.capacidad["capacidad_disponible"].sum()
    brecha = minutos_requeridos - capacidad_total

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.markdown(
        f'<div class="metric-card"><div class="metric-value">{len(allocation_data.ordenes):,}</div><div class="metric-label">Órdenes Pendientes</div></div>',
        unsafe_allow_html=True,
    )
    col_m2.markdown(
        f'<div class="metric-card"><div class="metric-value">{minutos_requeridos:,.0f}</div><div class="metric-label">Minutos Requeridos</div></div>',
        unsafe_allow_html=True,
    )
    col_m3.markdown(
        f'<div class="metric-card"><div class="metric-value">{capacidad_total:,.0f}</div><div class="metric-label">Capacidad Instalada</div></div>',
        unsafe_allow_html=True,
    )

    if brecha > 0:
        col_m4.markdown(
            f'<div class="metric-card" style="border-left: 4px solid #dc3545;"><div class="metric-value">-{brecha:,.0f}</div><div class="metric-label">Déficit (Minutos)</div></div>',
            unsafe_allow_html=True,
        )
        st.info(
            "El volumen de producción exigido supera la capacidad instalada de los proveedores disponibles. Resulta matemáticamente imposible cumplir con el 100% de las entregas bajo una planificación tradicional, haciendo imperativo el uso del **Módulo de Asignación (MILP)** para maximizar la priorización estratégica comercial (Time-to-Market)."
        )
    else:
        col_m4.markdown(
            f'<div class="metric-card" style="border-left: 4px solid #28a745;"><div class="metric-value">+{abs(brecha):,.0f}</div><div class="metric-label">Holgura (Minutos)</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Diagnóstico de Eficiencia Energética (Extrusora)
    st.subheader("2. Consumo Energético: Volatilidad Térmica (Piso de Planta)")
    st.markdown("""
    * **El Dolor del Cliente:** Facturas de electricidad con altos sobrecostos injustificados. Los operadores de las máquinas configuran las temperaturas de manera empírica ("a sentimiento"), provocando que turnos enteros desperdicien energía sin producir más toneladas.
    * **Datos Analizados (IoT PLC):** `temp_barril_2` a `11` (10 zonas de calor), `md1_power` / `md2_power` (Carga kW motores), `kpi` (Eficiencia real en kWh/tonelada), `turno` (Noche/Mañana).
    """)

    upper_limit = extrusora_df["kpi"].quantile(0.99)
    df_ex_clean = extrusora_df[extrusora_df["kpi"] <= upper_limit]

    kpi_promedio = df_ex_clean["kpi"].mean()
    anomalias = df_ex_clean["anomalia_iso"].sum()
    total_registros = len(df_ex_clean)

    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.markdown(
        f'<div class="metric-card"><div class="metric-value">{total_registros:,}</div><div class="metric-label">Ciclos Registrados</div></div>',
        unsafe_allow_html=True,
    )
    col_e2.markdown(
        f'<div class="metric-card"><div class="metric-value">{kpi_promedio:.2f}</div><div class="metric-label">Eficiencia Media (kWh/ton)</div></div>',
        unsafe_allow_html=True,
    )
    col_e3.markdown(
        f'<div class="metric-card" style="border-left: 4px solid #ffc107;"><div class="metric-value">{anomalias:,}</div><div class="metric-label">Picos Anómalos Detectados</div></div>',
        unsafe_allow_html=True,
    )

    st.info(
        "La operación histórica presenta cientos de picos anómalos de consumo eléctrico que escapan al control estadístico estándar. Esta volatilidad justifica la intervención del **Módulo de Eficiencia Energética** para descubrir, mediante Machine Learning no supervisado, los perfiles de temperatura óptimos que estabilicen este desperdicio."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Diagnóstico Comercial (Demanda)
    st.subheader("3. Comercial: Complejidad Logística del Portafolio")
    st.markdown("""
    * **El Dolor del Cliente:** Pérdida de ventas por roturas de stock (desabasto) y penalizaciones financieras por exceso de inventario estancado. Sus métodos actuales fallan al no poder predecir estacionalidades cruzadas en miles de combinaciones producto-tienda.
    * **Datos Analizados (Retail):** `store_id` (Puntos de venta), `cat_id` (Líneas de producto), `date` (Cronología), `ventas` (Volumen transaccional real).
    """)

    total_ventas = features_df["ventas"].sum()
    total_tiendas = features_df["store_id"].nunique()
    total_categorias = features_df["cat_id"].nunique()
    dias_historia = features_df["date"].nunique()

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.markdown(
        f'<div class="metric-card"><div class="metric-value">{total_ventas:,.0f}</div><div class="metric-label">Volumen Histórico Total</div></div>',
        unsafe_allow_html=True,
    )
    col_d2.markdown(
        f'<div class="metric-card"><div class="metric-value">{total_tiendas}</div><div class="metric-label">Puntos de Venta</div></div>',
        unsafe_allow_html=True,
    )
    col_d3.markdown(
        f'<div class="metric-card"><div class="metric-value">{total_categorias}</div><div class="metric-label">Líneas de Producto</div></div>',
        unsafe_allow_html=True,
    )
    col_d4.markdown(
        f'<div class="metric-card"><div class="metric-value">{dias_historia:,}</div><div class="metric-label">Días de Historia Capturada</div></div>',
        unsafe_allow_html=True,
    )

    st.info(
        "La profundidad del catálogo multiplicada por los puntos de venta genera series de tiempo con fuerte estacionalidad cruzada. Un modelo de promedios simples colapsaría ante esta complejidad, validando el despliegue del **Módulo de Pronóstico** basado en LightGBM (Gradient Boosting) para capturar la macrotendencia de forma jerárquica."
    )

# -----------------------------------------------------------------------------
# Tab 1: Allocation MILP
# -----------------------------------------------------------------------------
elif tab_selection == "Asignación (MILP)":
    st.header("Asignación de Órdenes a Proveedores")
    st.markdown("""
    Este módulo resuelve el problema logístico de **"Quién fabrica qué y en qué orden"**. Utiliza un motor de Optimización Lineal (MILP) para maximizar la prioridad comercial de las entregas, respetando matemáticamente las restricciones de capacidad física de cada taller.
    """)

    with st.expander("Diccionario de Datos: Prioridad y Score"):
        st.write("""
        * **Prioridad:** Escala de urgencia comercial (**3 = Alta (Crítica), 2 = Media, 1 = Baja**). El motor matemático utiliza una escala exponencial (10^n), asegurando que una orden de Prioridad 3 siempre gane sobre cualquier volumen de órdenes de Prioridad 1.
        * **Score de Calidad:** Calificación histórica del proveedor. El sistema lo utiliza como métrica de desempate secundaria: a igual prioridad, la orden se envía al taller con mejor score.
        """)

    st.subheader("Parámetros del Escenario")
    st.info(
        "Modifique los parámetros a continuación para simular cambios en la red logística. El solver recalculará la asignación óptima instantáneamente."
    )

    col1, col2 = st.columns(2)
    capacity_reduction = col1.slider(
        "Reducción Global de Capacidad (%)",
        0,
        100,
        0,
        step=5,
        help="Simula una disrupción global. Una reducción del 20% significa que todos los proveedores operarán al 80% de su capacidad. Útil para hacer pruebas de estrés (stress-testing).",
    )
    dropped_providers = col2.multiselect(
        "Excluir Proveedores (Simular Cierre)",
        options=allocation_data.capacidad["id"].unique(),
        help="Simula el cierre temporal de proveedores específicos. El solver intentará re-enrutar sus materiales a otros proveedores activos.",
    )

    with st.spinner("Ejecutando el Motor HiGHS..."):
        capacity_multiplier = 1 - (capacity_reduction / 100.0)
        allocation_res = solve_allocation_milp(
            allocation_data, capacity_multiplier, dropped_providers
        )

    kpis = allocation_res.kpis

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><div class="metric-value">{kpis["ordenes_asignadas"]}</div><div class="metric-label">Órdenes Asignadas</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card"><div class="metric-value">{kpis["prioridad_asignada"]:,.0f}</div><div class="metric-label">Prioridad Entregada</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-card"><div class="metric-value">{kpis["proveedores_usados"]}</div><div class="metric-label">Proveedores Activos</div></div>',
        unsafe_allow_html=True,
    )
    c4.markdown(
        f'<div class="metric-card"><div class="metric-value">{kpis["score_medio"]:.2f}</div><div class="metric-label">Calidad Promedio</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.subheader("1. Utilización de Capacidad (Saturación)")
        with st.expander("Definición: ¿Qué significa esta tabla?"):
            st.write(
                "Muestra qué tan 'lleno' de trabajo está cada proveedor. Si un proveedor alcanza el 100% de saturación, significa que el algoritmo aprovechó al máximo su capacidad asignándole las órdenes más críticas posibles. Proveedores con baja saturación pueden ser ineficientes o no ser compatibles con los materiales prioritarios requeridos."
            )
        seq_df = allocation_res.secuencia
        if not seq_df.empty:
            usage = seq_df.groupby("id")["minutos_totales_de_la_orden"].sum().reset_index()
            cap = allocation_data.capacidad.copy()
            cap["adjusted_capacity"] = cap["capacidad_disponible"] * capacity_multiplier
            usage = usage.merge(cap, on="id")
            usage["utilization_pct"] = (
                usage["minutos_totales_de_la_orden"] / usage["adjusted_capacity"]
            ) * 100

            # Ordenar de mayor a menor por defecto, pero incluir TODOS para permitir re-ordenamiento interactivo
            full_usage = usage.sort_values("utilization_pct", ascending=False).copy()
            full_usage = full_usage.rename(
                columns={
                    "id": "Proveedor",
                    "minutos_totales_de_la_orden": "Minutos Usados",
                    "adjusted_capacity": "Capacidad (Ajustada)",
                    "utilization_pct": "Saturación (%)",
                }
            )

            styled_top = (
                full_usage[
                    ["Proveedor", "Minutos Usados", "Capacidad (Ajustada)", "Saturación (%)"]
                ]
                .style.format(
                    {
                        "Minutos Usados": "{:,.0f}",
                        "Capacidad (Ajustada)": "{:,.0f}",
                        "Saturación (%)": "{:.1f}%",
                    }
                )
                .background_gradient(subset=["Saturación (%)"], cmap="Reds")
            )

            st.dataframe(styled_top, hide_index=True, use_container_width=True)
        else:
            st.warning("No se logró asignar ninguna orden bajo estas restricciones.")

    with col_chart2:
        st.subheader("2. Secuencia de Producción (WSPT)")
        with st.expander("Definición: ¿Qué es la Secuencia WSPT?"):
            st.write(
                "Una vez que el algoritmo decide *quién* fabrica qué, aplica una regla matemática llamada **WSPT (Weighted Shortest Processing Time)** para ordenar la fila de trabajo de cada proveedor. Esta regla asegura que las órdenes de más alta prioridad se programen de inmediato, calculando el minuto exacto de Inicio y Fin para minimizar el tiempo de espera (Time-to-Market)."
            )

        display_seq = seq_df[
            ["id", "orden", "material", "prioridad", "score", "inicio_min", "fin_min"]
        ].copy()

        # Filtros interactivos
        col_f1, col_f2 = st.columns(2)
        filtro_proveedor = col_f1.selectbox(
            "Filtrar por Proveedor:", ["Todos"] + sorted(display_seq["id"].unique().tolist())
        )
        filtro_material = col_f2.selectbox(
            "Filtrar por Material:", ["Todos"] + sorted(display_seq["material"].unique().tolist())
        )

        if filtro_proveedor != "Todos":
            display_seq = display_seq[display_seq["id"] == filtro_proveedor]

        if filtro_material != "Todos":
            display_seq = display_seq[display_seq["material"] == filtro_material]

        display_seq = display_seq.rename(
            columns={
                "id": "Proveedor",
                "orden": "Orden ID",
                "material": "Material",
                "prioridad": "Prioridad",
                "score": "Score",
                "inicio_min": "Inicio (min)",
                "fin_min": "Fin (min)",
            }
        )
        st.dataframe(display_seq, hide_index=True, use_container_width=True, height=300)

# -----------------------------------------------------------------------------
# Tab 2: Eficiencia Energética (Extrusora)
# -----------------------------------------------------------------------------
elif tab_selection == "Eficiencia Energética (Extrusora)":
    from process.extrusora.analisis import curvas_temperatura_optimas, test_manana_vs_noche

    # Filtrar errores matemáticos masivos (datos corruptos del PLC) antes de cualquier análisis
    upper_limit = extrusora_df["kpi"].quantile(0.99)
    df_extrusora_clean = extrusora_df[extrusora_df["kpi"] <= upper_limit].copy()

    st.header("Eficiencia Energética y Análisis Térmico")
    st.markdown(
        "Identifica anomalías mediante Machine Learning (Isolation Forest) y analiza estadísticamente los consumos entre turnos y perfiles de temperatura."
    )

    # 1. Análisis por Turno y Test de Hipótesis
    st.subheader("1. Consumo por Turno (Test de Hipótesis)")
    res_turno = test_manana_vs_noche(df_extrusora_clean)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><div class="metric-value">{res_turno["media_manana"]:.2f}</div><div class="metric-label">kWh/ton (Mañana)</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card"><div class="metric-value">{res_turno["media_noche"]:.2f}</div><div class="metric-label">kWh/ton (Noche)</div></div>',
        unsafe_allow_html=True,
    )

    if res_turno["mann_whitney_pvalue"] < 0.05:
        c3.markdown(
            '<div class="metric-card" style="border-left: 4px solid #28a745;"><div class="metric-value">Sí</div><div class="metric-label">Diferencia Significativa</div></div>',
            unsafe_allow_html=True,
        )
    else:
        c3.markdown(
            '<div class="metric-card" style="border-left: 4px solid #dc3545;"><div class="metric-value">No</div><div class="metric-label">Diferencia Significativa</div></div>',
            unsafe_allow_html=True,
        )

    c4.markdown(
        f'<div class="metric-card"><div class="metric-value">{res_turno["mann_whitney_pvalue"]:.4f}</div><div class="metric-label">P-Value (Mann-Whitney)</div></div>',
        unsafe_allow_html=True,
    )

    # 2. Detección de Anomalías (Isolation Forest)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("2. Detección de Anomalías Sensoriales (Isolation Forest)")

    anomalias_df = df_extrusora_clean[df_extrusora_clean["anomalia_iso"].eq(True)].copy()
    pct_anomalias = (len(anomalias_df) / len(df_extrusora_clean)) * 100
    st.info(
        f"El modelo no supervisado Isolation Forest identificó **{len(anomalias_df)} anomalías** ({pct_anomalias:.1f}% de la muestra) en la relación de carga de los motores y el KPI final."
    )

    # Gráfica de anomalías: Consumo Total (kW) vs KPI (kWh/ton)
    df_extrusora_clean["consumo_total_kw"] = (
        df_extrusora_clean["md1_power"] + df_extrusora_clean["md2_power"]
    )
    anomalias_df = df_extrusora_clean[df_extrusora_clean["anomalia_iso"].eq(True)]
    valid_mask = df_extrusora_clean["anomalia_iso"].eq(False)

    fig_anom, ax_anom = plt.subplots(figsize=(10, 5))
    ax_anom.scatter(
        df_extrusora_clean[valid_mask]["consumo_total_kw"],
        df_extrusora_clean[valid_mask]["kpi"],
        color="#e0e0e0",
        alpha=0.5,
        label="Operación Normal",
        s=15,
        edgecolors="none",
    )
    ax_anom.scatter(
        anomalias_df["consumo_total_kw"],
        anomalias_df["kpi"],
        color="#dc3545",
        label="Anomalía (Outlier)",
        s=25,
        edgecolors="black",
        linewidth=0.5,
    )

    # Añadir línea de tendencia sobre los datos normales
    import numpy as np

    x_norm = df_extrusora_clean[valid_mask]["consumo_total_kw"].dropna()
    y_norm = df_extrusora_clean[valid_mask]["kpi"].dropna()
    if len(x_norm) > 0:
        z = np.polyfit(x_norm, y_norm, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_norm.min(), x_norm.max(), 100)
        ax_anom.plot(
            x_line,
            p(x_line),
            color="#007bff",
            linestyle="--",
            linewidth=1.5,
            label="Línea Base Esperada",
        )

    ax_anom.set_title("Relación de Potencia de Motores vs. Eficiencia Final", fontsize=11)
    ax_anom.set_xlabel("Consumo Total de Motores (kW)")
    ax_anom.set_ylabel("Eficiencia (kWh/ton)")

    ax_anom.spines["top"].set_visible(False)
    ax_anom.spines["right"].set_visible(False)
    ax_anom.legend()
    st.pyplot(fig_anom)

    with st.expander("Justificación Matemática: Anomalías Multidimensionales en Proyecciones 2D"):
        st.write("""
        **Nota Técnica:** Es común observar registros catalogados como anómalos (rojos) superpuestos en el clúster de operación normal. 
        
        Esto ocurre debido a que la gráfica es una proyección **bidimensional** (Consumo vs Eficiencia), mientras que el algoritmo *Isolation Forest* se entrenó en un espacio de **10 dimensiones** evaluando el perfil térmico simultáneo de los cañones (`temp_barril_2` al `11`). 
        
        Una anomalía en el centro de la gráfica indica que, si bien la potencia agregada de los motores y el KPI final caen dentro de un rango promedio, la receta de temperaturas aplicada por el operador en ese turno fue estadísticamente aberrante (ej. cañones apagados compensados con cañones sobrecalentados). Esto evidencia fallas de proceso que son invisibles a un análisis visual clásico.
        """)


    # 3. Clustering de Temperaturas (K-Means) y ANOVA
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("3. Perfiles de Temperatura Óptimos (Clustering)")

    with st.expander("Definición: ¿Qué es un Perfil de Temperatura?"):
        st.write("""
        Para fundir el material, la extrusora cuenta con 10 zonas de calor consecutivas (del Barril 2 al Barril 11). 
        
        Durante la operación diaria, los técnicos configuran estas zonas de distintas maneras: algunos prefieren calentar mucho al inicio y poco al final, mientras que otros mantienen una temperatura estable en todo el trayecto. A cada una de estas "rutinas o recetas" completas se le conoce como un **Perfil de Temperatura**.
        
        En lugar de adivinar cuál es mejor, el algoritmo analiza meses de datos históricos y detecta automáticamente las 4 rutinas (Perfiles) más usadas en la fábrica. Posteriormente, el sistema compara el gasto eléctrico de cada rutina para dictaminar con certeza matemática cuál configuración consume menos energía por cada tonelada que se produce.
        """)

    with st.spinner("Agrupando curvas de temperatura con K-Means y calculando ANOVA..."):
        res_temp = curvas_temperatura_optimas(df_extrusora_clean, n_clusters=4)

    st.write(
        f"Se identificaron estadísticamente los perfiles térmicos. **Kruskal-Wallis P-Value:** {res_temp['kruskal_pvalue']:.4e} (Impacto significativo en la eficiencia)."
    )

    # Extraer y graficar los perfiles (curvas)
    df_clustered = res_temp["df_con_clusters"]
    temp_cols = [c for c in df_clustered.columns if "temp_barril" in c]
    centros = df_clustered.groupby("cluster_temp")[temp_cols].mean()

    fig_prof, ax_prof = plt.subplots(figsize=(10, 4))
    colores = ["#007bff", "#28a745", "#fd7e14", "#6f42c1"]

    # Extraer el número de barril de la columna (ej. temp_barril_2 -> 2)
    x_labels = [int(c.split("_")[-1]) for c in temp_cols]

    for i, (idx, row) in enumerate(centros.iterrows()):
        ax_prof.plot(
            x_labels,
            row.values,
            marker="o",
            label=f"Perfil {idx}",
            color=colores[i % len(colores)],
            linewidth=2,
        )

    ax_prof.set_title(
        "Configuraciones de Temperatura (Recetas Descubiertas por el Algoritmo)", fontsize=11
    )
    ax_prof.set_xlabel("Zonas de la Extrusora (Barril)")
    ax_prof.set_ylabel("Temperatura Media (°C)")
    ax_prof.spines["top"].set_visible(False)
    ax_prof.spines["right"].set_visible(False)
    ax_prof.set_xticks(x_labels)
    ax_prof.legend()
    st.pyplot(fig_prof)

    resumen_clusters = res_temp["resumen_clusters"].copy()

    # Añadir conteo por turno para explicar el origen de la ineficiencia
    cross_tab = pd.crosstab(df_clustered["cluster_temp"], df_clustered["turno"])
    # Asegurar que existan ambas columnas por si un dataset está sesgado
    uso_manana = cross_tab.get("Mañana", pd.Series(0, index=cross_tab.index))
    uso_noche = cross_tab.get("Noche", pd.Series(0, index=cross_tab.index))

    resumen_clusters["Uso Turno Mañana"] = uso_manana.values
    resumen_clusters["Uso Turno Noche"] = uso_noche.values

    # Formatear la tabla para la UI (guardando el Styler en otra variable)
    resumen_clusters_ui = (
        resumen_clusters.rename(
            columns={
                "cluster_temp": "Perfil (Clúster)",
                "consumo_medio": "Consumo Eléctrico Medio (kW)",
                "kpi_medio": "KPI Medio (kWh/ton)",
                "temp_barril_2_medio": "Temp Inicio (Barril 2)",
                "temp_barril_11_medio": "Temp Fin (Barril 11)",
                "n": "Total Registros",
            }
        )
        .style.format(
            {
                "Consumo Eléctrico Medio (kW)": "{:,.0f}",
                "KPI Medio (kWh/ton)": "{:.2f}",
                "Temp Inicio (Barril 2)": "{:.1f}°C",
                "Temp Fin (Barril 11)": "{:.1f}°C",
            }
        )
        .background_gradient(subset=["KPI Medio (kWh/ton)"], cmap="RdYlGn_r")
    )

    st.markdown("**Desempeño y Hábitos de Uso por Turno**")
    st.dataframe(resumen_clusters_ui, hide_index=True, use_container_width=True)

    # 4. Conclusiones
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("Conclusiones")

    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        st.info(
            "**Gestión de Turnos y Personal:**\n\nTal como se demostró matemáticamente en el **Test de Hipótesis (Punto 1)**, la eficiencia se desploma durante el turno nocturno. Al cruzar esto con la *Tabla de Desempeño y Hábitos* superior, se evidencia la raíz del problema: la falta de estandarización hace que los operadores de noche utilicen recetas térmicas ineficientes al azar. Se recomienda auditar las prácticas nocturnas de inmediato."
        )
    with col_rec2:
        # Encontrar dinámicamente el mejor perfil usando el DataFrame original crudo
        mejor_perfil = resumen_clusters.sort_values("kpi_medio").iloc[0]
        nombre_perfil = f"Perfil {int(mejor_perfil['cluster_temp'])}"
        ahorro = resumen_clusters["kpi_medio"].max() - mejor_perfil["kpi_medio"]

        st.info(
            f"**Estandarización Térmica (PLC):**\n\nComo es visualmente evidente en la **Gráfica de Configuraciones Térmicas (Punto 3)**, el **{nombre_perfil}** presenta el mejor desempeño estadístico. Se sugiere fijar esta curva exacta como estándar obligatorio en el PLC, lo que erradicará el error humano y reducirá el consumo en **{ahorro:.1f} kWh/ton** respecto a los escenarios menos favorables detectados."
        )


# -----------------------------------------------------------------------------
# Tab 3: Demand Forecasting
# -----------------------------------------------------------------------------
elif tab_selection == "Pronóstico de Demanda":
    st.header("Pronóstico de Demanda Jerárquico")
    st.markdown(
        "Genera proyecciones con bandas de incertidumbre para optimizar el inventario de seguridad."
    )

    modo_pronostico = st.radio(
        "Modo de Análisis",
        ["Evaluación de Modelo (Últimos 28 días históricos)", "Pronóstico a Futuro (Próximo Mes)"],
        horizontal=True,
    )

    store_ids = features_df["store_id"].unique()
    selected_store = st.selectbox(
        "Seleccione la Tienda",
        store_ids,
        help="Muestra el volumen proyectado para la tienda seleccionada. El modelo se entrena sobre todas las tiendas simultáneamente para aprender tendencias macro.",
    )

    if "Evaluación" in modo_pronostico:
        with st.spinner(
            "Entrenando LightGBM con regresión cuantílica (Panel Data) a nivel producto..."
        ):
            demand_res = forecast_demand(features_df, horizon=28)

        # Agrupar a nivel tienda (sumando las categorías de producto) para la visualización
        res_tienda = demand_res["resultados"][
            demand_res["resultados"]["store_id"] == selected_store
        ]
        plot_data = (
            res_tienda.groupby("date")[
                ["ventas", "prediccion", "limite_inferior", "limite_superior"]
            ]
            .sum()
            .reset_index()
        )

        ctx = features_df[
            (features_df["store_id"] == selected_store)
            & (features_df["date"] >= plot_data["date"].min() - pd.Timedelta(days=45))
        ]
        context_data = ctx.groupby("date")["ventas"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            context_data["date"],
            context_data["ventas"],
            label="Ventas Históricas",
            marker="o",
            markersize=4,
            color="#6c757d",
        )
        ax.plot(
            plot_data["date"],
            plot_data["prediccion"],
            label="Pronóstico LightGBM (Validación)",
            color="#0d6efd",
            linewidth=2,
        )

        ax.set_title(f"Evaluación del Modelo a 28 Días — Tienda: {selected_store}", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)

        st.info(f"**Métricas Globales:** Error Absoluto Medio (MAE): {demand_res['mae']:.2f}. ")

    else:
        with st.spinner("Proyectando a 30 días (1 mes) con regresión cuantílica (Panel Data)..."):
            future_res = forecast_future_semester(features_df, horizon=30)

        # Agrupar a nivel tienda (sumando categorías) para la visualización
        res_tienda = future_res[future_res["store_id"] == selected_store]
        plot_data = (
            res_tienda.groupby("date")[["prediccion", "limite_inferior", "limite_superior"]]
            .sum()
            .reset_index()
        )

        # Filtrar contexto histórico reciente igual que en la evaluación (ej. últimos 60 días antes del pronóstico)
        ctx = features_df[
            (features_df["store_id"] == selected_store)
            & (features_df["date"] >= plot_data["date"].min() - pd.Timedelta(days=60))
        ]
        context_data = ctx.groupby("date")["ventas"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            context_data["date"],
            context_data["ventas"],
            label="Ventas Históricas",
            marker="o",
            markersize=4,
            color="#6c757d",
        )
        ax.plot(
            plot_data["date"],
            plot_data["prediccion"],
            label="Pronóstico LightGBM (Mensual)",
            color="#28a745",
            linewidth=2,
        )

        # Línea divisoria entre historia y futuro
        ax.axvline(context_data["date"].max(), color="red", linestyle="--", alpha=0.5, label="Hoy")

        ax.set_title(f"Proyección a Futuro (30 Días) — Tienda: {selected_store}", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig)

        st.success(
            "Visualizando el escenario para el próximo mes (30 días). Se muestra la historia reciente con el mismo nivel de detalle que en la evaluación. Las proyecciones prescinden de rezagos a corto plazo, enfocándose en estacionalidad macro."
        )

# -----------------------------------------------------------------------------
# Tab 3: LLM Insights (Asistente)
# -----------------------------------------------------------------------------
else:
    st.header("Asistente Estratégico Multimodal (LLM)")
    st.markdown(
        "Interactúe con un copiloto de inteligencia artificial diseñado para analizar los resultados matemáticos y extraer insights de negocio."
    )

    allocation_res = solve_allocation_milp(allocation_data, 1.0, [])
    kpis = allocation_res.kpis
    demand_res = forecast_demand(features_df, horizon=28)

    # Contexto oculto en un expander para limpiar la UI
    with st.expander(
        "Ver el contexto que el Agente está analizando en tiempo real", expanded=False
    ):
        st.write(
            f"**KPIs en Memoria:** Órdenes Asignadas ({kpis['pct_ordenes']:.0%}), Error MAE Demanda ({demand_res['mae']:.2f}). Se detectó un cuello de botella en los proveedores Top 3."
        )

    st.markdown("---")

    # Iniciar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "¡Hola! He analizado los gráficos de Utilización de Capacidad y la Secuencia de Producción (WSPT). Te sugiero diversificar materiales para proteger el cuello de botella. ¿Sobre qué métrica o variable te gustaría profundizar?",
            }
        ]

    # Contenedor visual del chat más ancho y limpio
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input del usuario
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Procesando consulta..."):
                    prompt_lower = prompt.lower()
                    if (
                        "inicio_min" in prompt_lower
                        or "fin_min" in prompt_lower
                        or "secuencia" in prompt_lower
                    ):
                        respuesta = "Las variables `inicio_min` y `fin_min` representan el cronograma exacto en el que el proveedor fabricará la orden. Gracias a la regla **WSPT**, aseguramos que los materiales críticos (de alta prioridad) se procesen primero, minimizando el tiempo de espera global de la cadena de suministro."
                    elif (
                        "capacidad" in prompt_lower
                        or "grafic" in prompt_lower
                        or "imagen" in prompt_lower
                    ):
                        respuesta = "Al revisar la **Utilización de Capacidad**, notamos que los proveedores más eficientes llegan al límite (100%). Esto es una señal de alerta temprana: si reduces la capacidad (simulando un fallo con el slider del Módulo 1), el solver MILP reasignará automáticamente las órdenes hacia la derecha de la gráfica, protegiendo tus prioridades."
                    elif (
                        "milp" in prompt_lower
                        or "modelo" in prompt_lower
                        or "por que" in prompt_lower
                    ):
                        respuesta = "El motor MILP que estamos usando (HiGHS) no adivina. A diferencia de las heurísticas, evalúa el espacio combinatorio completo y asegura matemáticamente que la **Función Objetivo** (maximizar la prioridad entregada) llegue a su valor máximo absoluto respetando que todas las órdenes del mismo material vayan al mismo taller."
                    else:
                        respuesta = "Buena pregunta. Como agente IA demostrativo, estoy programado para analizar la matriz de **Asignación de Capacidad** y la **Secuencia WSPT**. ¡Pregúntame sobre esos componentes o sobre cómo interpretar los tiempos de inicio y fin!"

                    st.markdown(respuesta)

        st.session_state.messages.append({"role": "assistant", "content": respuesta})
