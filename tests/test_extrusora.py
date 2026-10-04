import pandas as pd

from process.extrusora import asignar_turno


def test_asignar_turno_cruce_medianoche():
    df = pd.DataFrame(
        {
            "fecha": pd.to_datetime(
                ["2022-11-14 06:00", "2022-11-14 17:59", "2022-11-14 18:00", "2022-11-15 05:59"]
            )
        }
    )
    out = asignar_turno(df)
    assert out["turno"].tolist() == ["Mañana", "Mañana", "Noche", "Noche"]
    # 05:59 del día 15 pertenece al turno Noche del 14
    assert out["turno_id"].tolist() == [
        "2022-11-14_Mañana",
        "2022-11-14_Mañana",
        "2022-11-14_Noche",
        "2022-11-14_Noche",
    ]


def test_limpiar_extrusora():
    from process.extrusora.limpieza import limpiar_extrusora

    df = pd.DataFrame(
        {
            "kpi": [
                100,
                102,
                98,
                101,
                150,
                100000,
                pd.NA,
            ],  # 100000 es un outlier extremo, 150 es parada
            "alimento_de_polvo_rot": [1.5, 1.4, 1.6, 1.5, 0.01, 1.2, 1.0],  # 0.01 es parada
            "md1_power": [50] * 7,
            "md2_power": [50] * 7,
        }
    )
    # Rellenar con dummy vars temporales que el modelo usa si es necesario
    for i in range(2, 12):
        df[f"temp_barril_{i}"] = 100.0

    out = limpiar_extrusora(df)
    # Deberían quedar los primeros 4 registros
    assert len(out) == 4
    assert out["kpi"].max() <= 105


def test_anomalias():
    from process.extrusora.anomalias import detectar_anomalias, ranking_turnos_anomalos

    df = pd.DataFrame(
        {
            "turno_id": ["A", "A", "B", "B"],
            "fecha_turno": [pd.Timestamp("2023-01-01")] * 4,
            "turno": ["Mañana", "Mañana", "Noche", "Noche"],
            "temp_barril_2": [100, 100, 100, 999],  # El último es anomalía
        }
    )
    # Agregamos las 9 columnas restantes para no fallar
    for i in range(3, 12):
        df[f"temp_barril_{i}"] = 100.0

    anomalias = detectar_anomalias(df, contamination=0.25)
    assert "anomalia" in anomalias.columns
    assert anomalias.iloc[3]["anomalia"]

    ranking = ranking_turnos_anomalos(anomalias)
    assert ranking.iloc[0]["turno_id"] == "B"
    assert ranking.iloc[0]["pct_anomalias"] == 0.5
