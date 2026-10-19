import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PRESUPUESTO = 2_000_000

# Si es False se excluyen las variables que dependen del monto (los montos del test
# son ~6.4 veces mas chicos que los de train, ver seccion 4). Sirve como submission alternativa.
USAR_MONTO = True

# 1. CARGA DE DATASETS
train = pd.read_csv("data/solicitudes_train.csv")
test = pd.read_csv("data/solicitudes_test.csv")
bancario = pd.read_csv("data/comportamiento_bancario.csv")

train["Fecha_Solicitud"] = pd.to_datetime(train["Fecha_Solicitud"])
test["Fecha_Solicitud"] = pd.to_datetime(test["Fecha_Solicitud"])
bancario["Mes_Referencia"] = pd.to_datetime(bancario["Mes_Referencia"])


# 2. FEATURE ENGINEERING BANCARIO (VENTANAS 1M, 3M, 6M, 12M E HISTORIAL COMPLETO)
# Solo se usan meses estrictamente anteriores a la solicitud (sin fuga de informacion).
def construir_features_bancarias(solicitudes, historial):
    assert not solicitudes["ID_Solicitud"].duplicated().any()
    assert not historial.duplicated(subset=["ID_Usuario", "Mes_Referencia"]).any()

    cruces = solicitudes[["ID_Solicitud", "ID_Usuario", "Fecha_Solicitud"]].merge(
        historial, on="ID_Usuario", how="inner"
    )
    # Meses de antiguedad del registro respecto de la solicitud (1 = mes anterior)
    cruces["Meses_Atras"] = (
        (cruces["Fecha_Solicitud"].dt.year - cruces["Mes_Referencia"].dt.year) * 12
        + cruces["Fecha_Solicitud"].dt.month - cruces["Mes_Referencia"].dt.month
    )
    cruces = cruces[cruces["Meses_Atras"] >= 1].copy()
    cruces["Tuvo_Atraso"] = (cruces["Dias_Atraso_Otras_Deudas"] > 0).astype(int)
    cruces["Atraso_Grave"] = (cruces["Dias_Atraso_Otras_Deudas"] > 30).astype(int)

    bloques = []
    for meses in [1, 3, 6, 12]:
        ventana = cruces[cruces["Meses_Atras"] <= meses].groupby("ID_Solicitud")
        resumen = ventana.agg(
            Saldo_Promedio=("Saldo_Promedio_Mensual", "mean"),
            Saldo_Minimo=("Saldo_Promedio_Mensual", "min"),
            Saldo_Desviacion=("Saldo_Promedio_Mensual", "std"),
            Atraso_Maximo=("Dias_Atraso_Otras_Deudas", "max"),
            Atraso_Promedio=("Dias_Atraso_Otras_Deudas", "mean"),
            Proporcion_Meses_Atraso=("Tuvo_Atraso", "mean"),
            Proporcion_Atraso_Grave=("Atraso_Grave", "mean"),
            Meses_Observados=("Meses_Atras", "count"),
        )
        if meses == 1:
            resumen = resumen[["Saldo_Promedio", "Atraso_Maximo"]]
        resumen.columns = [f"{c}_{meses}m" for c in resumen.columns]
        bloques.append(resumen)

    # Historial completo: promedios (no sumas) para que no dependan de cuantos meses hay
    completo = cruces.groupby("ID_Solicitud").agg(
        Atraso_Promedio_Hist=("Dias_Atraso_Otras_Deudas", "mean"),
        Proporcion_Meses_Atraso_Hist=("Tuvo_Atraso", "mean"),
        Saldo_Promedio_Hist=("Saldo_Promedio_Mensual", "mean"),
    )
    bloques.append(completo)

    # Pendiente del saldo en los ultimos 12 meses (regresion lineal saldo ~ tiempo)
    ult12 = cruces[cruces["Meses_Atras"] <= 12].copy()
    ult12["t"] = -ult12["Meses_Atras"]
    ult12["ty"] = ult12["t"] * ult12["Saldo_Promedio_Mensual"]
    ult12["tt"] = ult12["t"] ** 2
    s = ult12.groupby("ID_Solicitud").agg(
        n=("t", "count"), st=("t", "sum"), sy=("Saldo_Promedio_Mensual", "sum"), sty=("ty", "sum"), stt=("tt", "sum")
    )
    pendiente = (s["n"] * s["sty"] - s["st"] * s["sy"]) / (s["n"] * s["stt"] - s["st"] ** 2 + 1e-9)
    bloques.append(pendiente.rename("Pendiente_Saldo_12m"))

    resumen_total = pd.concat(bloques, axis=1)
    resultado = solicitudes.merge(
        resumen_total, left_on="ID_Solicitud", right_index=True, how="left", validate="one_to_one"
    )
    resultado["Tendencia_Saldo_3m_vs_12m"] = resultado["Saldo_Promedio_3m"] - resultado["Saldo_Promedio_12m"]
    resultado["Variabilidad_Saldo_12m"] = resultado["Saldo_Desviacion_12m"] / (resultado["Saldo_Promedio_12m"].abs() + 1)
    resultado["Saldo_Sobre_Ingreso_6m"] = resultado["Saldo_Promedio_6m"] / resultado["Ingreso_Mensual"]
    return resultado


train_features = construir_features_bancarias(train, bancario)
test_features = construir_features_bancarias(test, bancario)

# 3. VARIABLES DERIVADAS DE LA SOLICITUD
for df in [train_features, test_features]:
    df["Relacion_Monto_Ingreso"] = df["Monto_Solicitado"] / df["Ingreso_Mensual"]
    df["Cuota_Sobre_Ingreso"] = df["Relacion_Monto_Ingreso"] * (1 + df["Tasa_Interes"])
    df["Monto_Sobre_Saldo"] = df["Monto_Solicitado"] / (df["Saldo_Promedio_6m"].abs() + 1)

categorias_motivo = sorted(train_features["Motivo_Prestamo"].unique())
for df in [train_features, test_features]:
    df["Motivo_Prestamo"] = pd.Categorical(df["Motivo_Prestamo"], categories=categorias_motivo)

columnas_bancarias = [
    c for c in train_features.columns
    if c.startswith(("Saldo_", "Atraso_", "Proporcion_", "Meses_Observados_", "Pendiente_", "Tendencia_", "Variabilidad_"))
]
columnas_monto = ["Monto_Solicitado", "Relacion_Monto_Ingreso", "Cuota_Sobre_Ingreso", "Monto_Sobre_Saldo"]
columnas_predictoras = ["Tasa_Interes", "Ingreso_Mensual", "Motivo_Prestamo"] + columnas_bancarias
if USAR_MONTO:
    columnas_predictoras += columnas_monto
columnas_numericas = [c for c in columnas_predictoras if c != "Motivo_Prestamo"]

# 4. DIAGNOSTICO DEL CAMBIO DE DISTRIBUCION
# En train se pide en total ~463M y por mes ~38M, pero el test pide solo ~6.8M:
# con 2M de presupuesto se financia ~29% del capital del test, no ~5% como en un mes de train.
# Por eso la validacion simula ese mismo regimen (presupuesto proporcional al capital pedido).
FRACCION_PRESUPUESTO = PRESUPUESTO / test["Monto_Solicitado"].sum()
print(f"Monto medio train: {train['Monto_Solicitado'].mean():,.0f} | test: {test['Monto_Solicitado'].mean():,.0f}")
print(f"Fraccion del capital del test que cubre el presupuesto: {FRACCION_PRESUPUESTO:.1%}")


# 5. MODELO: ENSAMBLE REGRESION LOGISTICA + LIGHTGBM (promedio de probabilidades)
class EnsambleProbabilidades:
    def __init__(self, modelos):
        self.modelos = modelos

    def fit(self, X, y):
        for modelo in self.modelos:
            modelo.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([modelo.predict_proba(X) for modelo in self.modelos], axis=0)


def crear_modelo():
    # Sin class_weight: necesitamos probabilidades calibradas porque entran en la ganancia esperada
    logistica = Pipeline([
        ("preparacion", ColumnTransformer([
            ("numericas", Pipeline([
                ("imputar", SimpleImputer(strategy="median")),
                ("escalar", StandardScaler()),
            ]), columnas_numericas),
            ("motivo", OneHotEncoder(handle_unknown="ignore"), ["Motivo_Prestamo"]),
        ])),
        ("modelo", LogisticRegression(C=0.1, max_iter=2000)),
    ])
    arboles = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5,
        random_state=0, verbose=-1,
    )
    return EnsambleProbabilidades([logistica, arboles])


# 6. ESTRATEGIA DE SELECCION: MAXIMIZAR GANANCIA ESPERADA POR DOLAR PRESTADO
# Ganancia esperada = p * monto * tasa - (1 - p) * monto  ->  por dolar: p * (1 + tasa) - 1
# Como la restriccion es de capital, se ordena por rentabilidad esperada por dolar (mochila greedy).
def seleccionar_prestamos(solicitudes, probabilidades_pago, presupuesto=PRESUPUESTO):
    if len(solicitudes) != len(probabilidades_pago):
        raise ValueError("Debe haber una probabilidad por solicitud")
    if not solicitudes["ID_Solicitud"].is_unique:
        raise ValueError("Los IDs de solicitud deben ser unicos")

    seleccion = solicitudes[["ID_Solicitud", "Monto_Solicitado", "Tasa_Interes"]].copy().reset_index(drop=True)
    p = np.asarray(probabilidades_pago)
    seleccion["Probabilidad_Pago"] = p
    seleccion["Rentabilidad_Esperada"] = p * (1 + seleccion["Tasa_Interes"]) - 1
    seleccion["Ganancia_Esperada"] = seleccion["Rentabilidad_Esperada"] * seleccion["Monto_Solicitado"]
    seleccion["Aprobar"] = 0

    candidatos = seleccion[seleccion["Rentabilidad_Esperada"] > 0].sort_values(
        "Rentabilidad_Esperada", ascending=False, kind="stable"
    )
    capital_utilizado = 0
    for indice, monto in candidatos["Monto_Solicitado"].items():
        if capital_utilizado + monto <= presupuesto:
            seleccion.loc[indice, "Aprobar"] = 1
            capital_utilizado += monto

    if seleccion.loc[seleccion["Aprobar"] == 1, "Monto_Solicitado"].sum() > presupuesto:
        raise ValueError("Presupuesto superado")
    return seleccion


def ganancia_real(solicitudes, aprobar):
    aprobar = np.asarray(aprobar) == 1
    pago = solicitudes["Estado_Final"].to_numpy() == 1
    monto = solicitudes["Monto_Solicitado"].to_numpy()
    tasa = solicitudes["Tasa_Interes"].to_numpy()
    return (monto * tasa)[aprobar & pago].sum() - monto[aprobar & ~pago].sum()


# 7. VALIDACION TEMPORAL WALK-FORWARD (4 MESES) EN REGIMEN SIMILAR AL TEST
# Para cada mes: entrena con todos los meses anteriores, predice ese mes y aprueba con un
# presupuesto = 29% del capital pedido. La ganancia se reescala a un presupuesto de 2M.
print("\n--- VALIDACION WALK-FORWARD ---")
resultados = []
for mes in ["2025-09", "2025-10", "2025-11", "2025-12"]:
    fecha = pd.Timestamp(mes)
    entrenamiento = train_features[train_features["Fecha_Solicitud"] < fecha]
    validacion = train_features[train_features["Fecha_Solicitud"] == fecha].reset_index(drop=True)

    modelo = crear_modelo().fit(entrenamiento[columnas_predictoras], entrenamiento["Estado_Final"])
    probs = modelo.predict_proba(validacion[columnas_predictoras])[:, 1]

    presupuesto_mes = FRACCION_PRESUPUESTO * validacion["Monto_Solicitado"].sum()
    aprobados = seleccionar_prestamos(validacion, probs, presupuesto_mes)["Aprobar"]
    oraculo = seleccionar_prestamos(validacion, validacion["Estado_Final"] * 0.999, presupuesto_mes)["Aprobar"]
    escala = PRESUPUESTO / presupuesto_mes
    resultados.append({
        "mes": mes,
        "AUC": roc_auc_score(validacion["Estado_Final"], probs),
        "ganancia": ganancia_real(validacion, aprobados) * escala,
        "oraculo": ganancia_real(validacion, oraculo) * escala,
    })
    r = resultados[-1]
    print(f"{mes}: AUC {r['AUC']:.4f} | ganancia USD {r['ganancia']:,.0f} | maximo posible USD {r['oraculo']:,.0f}")

resultados = pd.DataFrame(resultados)
print(f"PROMEDIO: AUC {resultados['AUC'].mean():.4f} | ganancia USD {resultados['ganancia'].mean():,.0f} "
      f"({resultados['ganancia'].mean() / resultados['oraculo'].mean():.1%} del maximo posible)")

# 8. ENTRENAR MODELO FINAL CON TODO TRAIN Y GENERAR SUBMISSION
print("\n--- ENTRENANDO MODELO FINAL PARA SUBMISSION ---")
modelo_final = crear_modelo().fit(train_features[columnas_predictoras], train_features["Estado_Final"])
probabilidades_pago_test = modelo_final.predict_proba(test_features[columnas_predictoras])[:, 1]
seleccion_test = seleccionar_prestamos(test_features, probabilidades_pago_test)
decisiones_test = seleccion_test.set_index("ID_Solicitud")["Aprobar"]
aprob_test = seleccion_test["Aprobar"] == 1
print(f"Ganancia esperada segun el modelo: USD {seleccion_test.loc[aprob_test, 'Ganancia_Esperada'].sum():,.0f}")


# 9. GUARDAR SUBMISSION.CSV
def guardar_submission(solicitudes_test, decisiones):
    assert len(solicitudes_test) == 3061 and len(decisiones) == 3061
    assert solicitudes_test["ID_Solicitud"].is_unique
    assert set(decisiones.index) == set(solicitudes_test["ID_Solicitud"])
    assert decisiones.isin([0, 1]).all()

    ordenadas = decisiones.reindex(solicitudes_test["ID_Solicitud"]).astype(int)
    montos = solicitudes_test["Monto_Solicitado"].reset_index(drop=True)
    capital_aprobado = montos.loc[ordenadas.to_numpy() == 1].sum()

    if capital_aprobado > PRESUPUESTO:
        raise ValueError("Presupuesto superado: no se guarda submission.csv")

    submission = pd.DataFrame({
        "ID_Solicitud": solicitudes_test["ID_Solicitud"].to_numpy(),
        "Decision_aprobacion": ordenadas.to_numpy(),
    })
    nombre = "submission.csv" if USAR_MONTO else "submission_sin_monto.csv"
    submission.to_csv(nombre, index=False)
    print(f"\n--- {nombre.upper()} GENERADA EXITOSAMENTE ---")
    print("Cantidad de aprobados:", int(ordenadas.sum()))
    print(f"Capital utilizado (USD): {capital_aprobado:,.2f}")
    print(f"Capital restante (USD): {PRESUPUESTO - capital_aprobado:,.2f}")
    return submission


submission = guardar_submission(test, decisiones_test)
