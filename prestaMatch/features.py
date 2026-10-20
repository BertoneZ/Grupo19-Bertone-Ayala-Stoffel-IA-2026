import numpy as np
import pandas as pd

MARGEN_PICO_ATRASO = 15


def estimar_factor_escala(train, test):
    percentiles = np.arange(0.30, 0.96, 0.05)
    cocientes = (
        train["Monto_Solicitado"].quantile(percentiles).to_numpy()
        / test["Monto_Solicitado"].quantile(percentiles).to_numpy()
    )
    return float(np.median(cocientes))


def resumir_historial_bancario(bancario):
    h = bancario.copy()
    h["Mes"] = (h["Mes_Referencia"].dt.year - 2024) * 12 + h["Mes_Referencia"].dt.month
    h["Log_Saldo"] = np.log(h["Saldo_Promedio_Mensual"].clip(lower=1))
    atraso = h["Dias_Atraso_Otras_Deudas"]
    por_usuario = h.groupby("ID_Usuario")

    mediana_usuario = por_usuario["Dias_Atraso_Otras_Deudas"].transform("median")
    h["Es_Pico"] = (atraso > mediana_usuario + MARGEN_PICO_ATRASO).astype(int)
    h["Atraso_Sin_Pico"] = atraso.where(h["Es_Pico"] == 0)
    h["Sin_Atraso"] = (atraso == 0).astype(int)

    resumen = por_usuario.agg(
        Saldo_Promedio=("Saldo_Promedio_Mensual", "mean"),
        Log_Saldo_Promedio=("Log_Saldo", "mean"),
        Atraso_Promedio=("Dias_Atraso_Otras_Deudas", "mean"),
        Atraso_Desvio=("Dias_Atraso_Otras_Deudas", "std"),
        Atraso_Maximo=("Dias_Atraso_Otras_Deudas", "max"),
        Atraso_Sin_Picos=("Atraso_Sin_Pico", "mean"),
        Proporcion_Picos=("Es_Pico", "mean"),
        Proporcion_Sin_Atraso=("Sin_Atraso", "mean"),
        Mes_Medio=("Mes", "mean"),
    )

    h["Mes_Centrado"] = h["Mes"] - por_usuario["Mes"].transform("mean")
    h["Log_Saldo_Centrado"] = h["Log_Saldo"] - por_usuario["Log_Saldo"].transform("mean")
    h["Producto"] = h["Mes_Centrado"] * h["Log_Saldo_Centrado"]
    h["Mes_Cuadrado"] = h["Mes_Centrado"] ** 2
    sumas = h.groupby("ID_Usuario")[["Producto", "Mes_Cuadrado"]].sum()
    resumen["Tendencia_Log_Saldo"] = sumas["Producto"] / sumas["Mes_Cuadrado"]

    resumen["Log_Saldo_Inicial"] = (
        resumen["Log_Saldo_Promedio"] + resumen["Tendencia_Log_Saldo"] * (1 - resumen["Mes_Medio"])
    )

    recientes = h[h["Mes"] > h["Mes"].max() - 6]
    resumen["Saldo_Reciente_6m"] = recientes.groupby("ID_Usuario")["Saldo_Promedio_Mensual"].mean()
    return resumen.drop(columns="Mes_Medio")


def construir_features(solicitudes, resumen_bancario, factor_monto=1.0):
    df = solicitudes.merge(
        resumen_bancario, left_on="ID_Usuario", right_index=True, how="left", validate="many_to_one"
    )
    assert len(df) == len(solicitudes)

    monto_equivalente = df["Monto_Solicitado"] * factor_monto
    log_ingreso = np.log(df["Ingreso_Mensual"])
    df["Monto_Sobre_Ingreso"] = monto_equivalente / df["Ingreso_Mensual"]
    df["Log_Monto_Sobre_Ingreso"] = np.log(df["Monto_Sobre_Ingreso"])
    df["Saldo_Sobre_Ingreso"] = df["Saldo_Promedio"] / df["Ingreso_Mensual"]
    df["Log_Saldo_Sobre_Ingreso"] = df["Log_Saldo_Promedio"] - log_ingreso
    df["Log_Saldo_Inicial_Sobre_Ingreso"] = df["Log_Saldo_Inicial"] - log_ingreso
    df["Saldo_Reciente_Sobre_Ingreso"] = df["Saldo_Reciente_6m"] / df["Ingreso_Mensual"]
    return df


FEATURES_MONTO = ["Monto_Sobre_Ingreso", "Log_Monto_Sobre_Ingreso"]

FEATURES_BASE = [
    "Tasa_Interes",
    "Ingreso_Mensual",
    "Saldo_Sobre_Ingreso",
    "Log_Saldo_Sobre_Ingreso",
    "Log_Saldo_Inicial_Sobre_Ingreso",
    "Saldo_Reciente_Sobre_Ingreso",
    "Tendencia_Log_Saldo",
    "Atraso_Promedio",
    "Atraso_Sin_Picos",
    "Atraso_Desvio",
    "Atraso_Maximo",
    "Proporcion_Picos",
    "Proporcion_Sin_Atraso",
]

FEATURE_CATEGORICA = "Motivo_Prestamo"
