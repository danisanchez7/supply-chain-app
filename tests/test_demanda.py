"""Tests del módulo de Demanda (Feature Engineering y Estructura Jerárquica)."""

import pandas as pd
import pytest

from process.demanda.prep import crear_features


@pytest.fixture
def mock_sales():
    # Creamos un pequeño panel data de 30 días para 2 tiendas y 2 categorías
    fechas = pd.date_range("2020-01-01", periods=30)
    data = []
    for store in ["TX_1", "CA_1"]:
        for cat in ["FOODS", "HOBBIES"]:
            for fecha in fechas:
                data.append(
                    {
                        "store_id": store,
                        "cat_id": cat,
                        "date": fecha,
                        "ventas": 100,  # Valor constante para probar lags fácilmente
                        "event_name_1": "SuperBowl" if fecha.day == 5 else None,
                    }
                )
    df = pd.DataFrame(data)
    # Agregamos la columna 'd' (días) que viene del M5 original pero no es requerida estricamente por crear_features
    # Sin embargo, crear_features procesa dia_semana y fechas.
    return df


def test_crear_features_estructura(mock_sales):
    """Verifica que el panel data genere las columnas de rezago (lags) y de tiempo correctamente."""
    lags_prueba = [1, 7]
    df_feat = crear_features(mock_sales, lags=lags_prueba)

    # Verificar existencia de columnas
    assert "lag_1" in df_feat.columns
    assert "lag_7" in df_feat.columns
    assert "rolling_mean_7" in df_feat.columns
    assert "dia_semana" in df_feat.columns
    assert "es_fin_semana" in df_feat.columns

    # Como usamos lag_7 y rolling_mean_28, las primeras 28 filas de cada serie (tienda-cat) se descartan por los NaN
    # Originalmente teníamos 30 días por serie. Se pierden 28. Quedan 2 días por serie.
    # 2 días x 2 tiendas x 2 categorías = 8 filas restantes.
    assert len(df_feat) == 8


def test_crear_features_valores(mock_sales):
    """Verifica la lógica matemática de los promedios móviles y variables booleanas."""
    df_feat = crear_features(mock_sales, lags=[1])

    # Dado que ventas = 100 siempre, el rolling_mean_7 debe ser exactamente 100.
    assert (df_feat["rolling_mean_7"] == 100.0).all()
    assert (df_feat["lag_1"] == 100.0).all()

    # Comprobar eventos y fines de semana (0 y 1)
    assert set(df_feat["hay_evento"].unique()).issubset({0, 1})
    assert set(df_feat["es_fin_semana"].unique()).issubset({0, 1})

    # Verificar que el evento se detectó correctamente
    # (event_name_1 desaparece, pero se condensa en 'hay_evento')
    dias_evento = df_feat[df_feat["date"].dt.day == 5]
    if not dias_evento.empty:
        assert (dias_evento["hay_evento"] == 1).all()
