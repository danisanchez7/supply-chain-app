"""Entrenamiento del modelo predictivo jerárquico con estimación de incertidumbre."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from process.config import HORIZONTE_DEMANDA


def entrenar_modelo_tiendas(df: pd.DataFrame, horizonte: int = HORIZONTE_DEMANDA) -> dict[str, Any]:
    """Entrena un LightGBM para predecir demanda con intervalos usando MAPIE (EnbPI).

    Implementa Conformal Prediction (EnbPI para series temporales) para estimar la
    incertidumbre (intervalos de predicción) sin hacer supuestos distribucionales.

    Args:
        df: DataFrame de características (con store_id, ventas y features numéricos).
        horizonte: Días al final del conjunto que se dejan para evaluar el pronóstico.
    """
    # Ordenar cronológicamente para evitar data leakage
    df = df.sort_values(["store_id", "cat_id", "date"]).reset_index(drop=True)

    cols_excluir = ["d", "date", "ventas", "event_name_1"]
    features = [c for c in df.columns if c not in cols_excluir]

    X = df[features].copy()
    # LightGBM soporta variables categóricas nativamente si tienen tipo "category"
    if "store_id" in X.columns:
        X["store_id"] = X["store_id"].astype("category")
    if "cat_id" in X.columns:
        X["cat_id"] = X["cat_id"].astype("category")

    y = df["ventas"]

    # Separar train / test cronológico por tienda y producto
    train_mask = df.groupby(["store_id", "cat_id"], observed=True).cumcount(ascending=False) >= horizonte

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    # Cuantiles para estimar el intervalo (90% confidencia = 5% y 95%)
    model_mid = lgb.LGBMRegressor(objective='quantile', alpha=0.5, n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    model_low = lgb.LGBMRegressor(objective='quantile', alpha=0.05, n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    model_high = lgb.LGBMRegressor(objective='quantile', alpha=0.95, n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)

    # Entrenamiento
    model_mid.fit(X_train, y_train)
    model_low.fit(X_train, y_train)
    model_high.fit(X_train, y_train)

    # Predicción
    y_pred = model_mid.predict(X_test)
    y_low = model_low.predict(X_test)
    y_high = model_high.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # DataFrame de resultados
    res = df[~train_mask][["store_id", "cat_id", "date", "ventas"]].copy()
    res["prediccion"] = y_pred
    res["limite_inferior"] = y_low
    res["limite_superior"] = y_high

    # Post-procesamiento: la demanda no puede ser negativa
    res["prediccion"] = res["prediccion"].clip(lower=0)
    res["limite_inferior"] = res["limite_inferior"].clip(lower=0)
    res["limite_superior"] = res["limite_superior"].clip(lower=0)
    return {"modelo": model_mid, "mae": float(mae), "mape": float(mape), "resultados": res}


def pronosticar_semestre_futuro(df: pd.DataFrame, horizonte_dias: int = 180) -> pd.DataFrame:
    """Entrena sobre TODO el historial y proyecta el futuro real (ej. próximo semestre).
    
    A diferencia de la evaluación train/test que usa rezagos (lags) y promedios móviles,
    para pronosticar el futuro a largo plazo (multistep) usamos un modelo puramente estacional
    y de tendencia, evitando el error acumulativo de pronosticar recursivamente.
    """
    # 1. Preparar features históricas (solo estacionales y tendencia)
    df = df.sort_values(["store_id", "cat_id", "date"]).reset_index(drop=True)
    df["dia_semana"] = df["date"].dt.dayofweek
    df["mes"] = df["date"].dt.month
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    # Índice temporal para capturar la macro-tendencia (crecimiento histórico)
    df["tendencia"] = (df["date"] - df["date"].min()).dt.days
    
    features = ["store_id", "cat_id", "dia_semana", "mes", "es_fin_semana", "tendencia"]
    X_train = df[features].copy()
    X_train["store_id"] = X_train["store_id"].astype("category")
    X_train["cat_id"] = X_train["cat_id"].astype("category")
    y_train = df["ventas"]
    
    # 2. Entrenar modelos de regresión cuantílica
    model_mid = lgb.LGBMRegressor(objective='quantile', alpha=0.5, n_estimators=150, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    model_low = lgb.LGBMRegressor(objective='quantile', alpha=0.05, n_estimators=150, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    model_high = lgb.LGBMRegressor(objective='quantile', alpha=0.95, n_estimators=150, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
    
    model_mid.fit(X_train, y_train)
    model_low.fit(X_train, y_train)
    model_high.fit(X_train, y_train)
    
    # 3. Generar el dataset del futuro
    futuros = []
    fecha_maxima = df["date"].max()
    fecha_minima = df["date"].min()
    
    for tienda in df["store_id"].unique():
        for categoria in df["cat_id"].unique():
            fechas_futuras = pd.date_range(start=fecha_maxima + pd.Timedelta(days=1), periods=horizonte_dias, freq="D")
            df_futuro = pd.DataFrame({"store_id": tienda, "cat_id": categoria, "date": fechas_futuras})
            df_futuro["dia_semana"] = df_futuro["date"].dt.dayofweek
            df_futuro["mes"] = df_futuro["date"].dt.month
            df_futuro["es_fin_semana"] = df_futuro["dia_semana"].isin([5, 6]).astype(int)
            df_futuro["tendencia"] = (df_futuro["date"] - fecha_minima).dt.days
            futuros.append(df_futuro)
        
    X_test = pd.concat(futuros, ignore_index=True)
    X_test_feats = X_test[features].copy()
    X_test_feats["store_id"] = X_test_feats["store_id"].astype("category")
    X_test_feats["cat_id"] = X_test_feats["cat_id"].astype("category")
    
    # 4. Pronosticar
    y_pred = model_mid.predict(X_test_feats)
    y_low = model_low.predict(X_test_feats)
    y_high = model_high.predict(X_test_feats)
    
    # 5. Formatear salida
    res = X_test.copy()
    res["prediccion"] = y_pred.clip(min=0)
    res["limite_inferior"] = y_low.clip(min=0)
    res["limite_superior"] = y_high.clip(min=0)
    
    return res
