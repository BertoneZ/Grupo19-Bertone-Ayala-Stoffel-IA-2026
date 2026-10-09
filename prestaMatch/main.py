import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Cargar datasets
train = pd.read_csv("data/solicitudes_train.csv")
test = pd.read_csv("data/solicitudes_test.csv")
bancario = pd.read_csv("data/comportamiento_bancario.csv")

# Mostrar primeras filas
# print("TRAIN")
# print(train.head())
# print("\nTEST")
# print(test.head())
# print("\nCOMPORTAMIENTO BANCARIO")
# print(bancario.head())

# print("\nDimensiones:")
# print("Train:", train.shape)
# print("Test:", test.shape)
# print("Bancario:", bancario.shape)


# print("\n--- COLUMNAS TRAIN ---")
# print(train.columns)

# print("\n--- TIPOS DE DATOS TRAIN ---")
# print(train.dtypes)

# print("\n--- VALORES NULOS TRAIN ---")
# print(train.isnull().sum())

# print("\n--- VALORES NULOS TEST ---")
# print(test.isnull().sum())

# print("\n--- VALORES NULOS BANCARIO ---")
# print(bancario.isnull().sum())

# print("\n--- DISTRIBUCIÓN ESTADO FINAL ---")
# print(train["Estado_Final"].value_counts())

# print("\n--- DISTRIBUCIÓN PORCENTUAL ---")
# print(train["Estado_Final"].value_counts(normalize=True) * 100)

# print("\n--- DUPLICADOS ---")
# print("Train:", train.duplicated().sum())
# print("Test:", test.duplicated().sum())
# print("Bancario:", bancario.duplicated().sum())

# print("\n--- MOTIVOS DE PRÉSTAMO ---")
# print(train["Motivo_Prestamo"].value_counts())

# print("\n--- RANGOS NUMÉRICOS ---")

# print("Monto:")
# print(train["Monto_Solicitado"].min(),
#       train["Monto_Solicitado"].max())

# print("Ingreso:")
# print(train["Ingreso_Mensual"].min(),
#       train["Ingreso_Mensual"].max())

# print("Tasa:")
# print(train["Tasa_Interes"].min(),
#       train["Tasa_Interes"].max())

# print("Atrasos:")
# print(bancario["Dias_Atraso_Otras_Deudas"].min(),
#       bancario["Dias_Atraso_Otras_Deudas"].max())

train["Fecha_Solicitud"] = pd.to_datetime(train["Fecha_Solicitud"])
test["Fecha_Solicitud"] = pd.to_datetime(test["Fecha_Solicitud"])
bancario["Mes_Referencia"] = pd.to_datetime(bancario["Mes_Referencia"])
# ==============================
# 5. REVISAR RANGOS DE FECHAS
# ==============================
# print("\n--- RANGO TRAIN ---")
# print(train["Fecha_Solicitud"].min())
# print(train["Fecha_Solicitud"].max())

# print("\n--- RANGO TEST ---")
# print(test["Fecha_Solicitud"].min())
# print(test["Fecha_Solicitud"].max())

# print("\n--- RANGO BANCARIO ---")
# print(bancario["Mes_Referencia"].min())
# print(bancario["Mes_Referencia"].max())

# FEATURES BANCARIAS: SOLO LOS SEIS MESES ANTERIORES
def construir_features_bancarias(solicitudes, historial):
    assert solicitudes["ID_Solicitud"].is_unique
    assert not historial.duplicated(
        ["ID_Usuario", "Mes_Referencia"]
    ).any()
    cruces = solicitudes[
        ["ID_Solicitud", "ID_Usuario", "Fecha_Solicitud"]
    ].merge(historial, on="ID_Usuario", how="left")

    # Incluye seis meses anteriores; excluye el mes de solicitud.
    inicio = cruces["Fecha_Solicitud"] - pd.DateOffset(months=6)
    anteriores = cruces.loc[
        (cruces["Mes_Referencia"] >= inicio)
        & (cruces["Mes_Referencia"] < cruces["Fecha_Solicitud"])
    ].copy()

    assert (
        anteriores["Mes_Referencia"] < anteriores["Fecha_Solicitud"]
    ).all()

    anteriores["Tuvo_Atraso"] = (
        anteriores["Dias_Atraso_Otras_Deudas"] > 0
    )

    features = anteriores.groupby("ID_Solicitud").agg(
        Saldo_Promedio_6m=("Saldo_Promedio_Mensual", "mean"),
        Saldo_Desviacion_6m=("Saldo_Promedio_Mensual", "std"),
        Atraso_Maximo_6m=("Dias_Atraso_Otras_Deudas", "max"),
        Proporcion_Meses_Atraso_6m=("Tuvo_Atraso", "mean"),
        Meses_Observados_6m=("Mes_Referencia", "count"),
    )

    ultimos = anteriores.sort_values("Mes_Referencia").drop_duplicates(
        "ID_Solicitud", keep="last"
    )
    features["Saldo_Ultimo_Disponible_6m"] = ultimos.set_index(
        "ID_Solicitud"
    )["Saldo_Promedio_Mensual"]

    resultado = solicitudes.merge(
        features, on="ID_Solicitud", how="left", validate="one_to_one"
    )
    resultado["Meses_Observados_6m"] = (
        resultado["Meses_Observados_6m"].fillna(0).astype(int)
    )
    assert len(resultado) == len(solicitudes)
    assert resultado["ID_Solicitud"].equals(solicitudes["ID_Solicitud"])
    assert resultado["Meses_Observados_6m"].between(0, 6).all()
    return resultado

train_features = construir_features_bancarias(train, bancario)
test_features = construir_features_bancarias(test, bancario)
train_features["Relacion_Monto_Ingreso"] = (
    train_features["Monto_Solicitado"] / train_features["Ingreso_Mensual"]
)
test_features["Relacion_Monto_Ingreso"] = (
    test_features["Monto_Solicitado"] / test_features["Ingreso_Mensual"]
)

columnas_features = [
    columna for columna in train_features.columns
    if columna not in train.columns and columna != "Relacion_Monto_Ingreso"
]
for nombre, datos in [("TRAIN", train_features), ("TEST", test_features)]:
    print(f"\n--- FEATURES BANCARIAS {nombre} ---")
    print("Dimensiones:", datos.shape)
    print(datos[["ID_Solicitud"] + columnas_features].head())
    print("Nulos:")
    print(datos[columnas_features].isna().sum())
    print("Meses observados:")
    print(datos["Meses_Observados_6m"].value_counts().sort_index())

    
# SEPARACION TEMPORAL: APRENDER DEL PASADO Y EVALUAR EN MESES POSTERIORES
fecha_corte = pd.Timestamp("2025-10-01")
entrenamiento = train_features.loc[
    train_features["Fecha_Solicitud"] < fecha_corte
].copy()
validacion = train_features.loc[
    train_features["Fecha_Solicitud"] >= fecha_corte
].copy()
# Comprobar que ninguna solicitud se pierde o aparece en ambos conjuntos.
assert not entrenamiento.empty and not validacion.empty
assert len(entrenamiento) + len(validacion) == len(train_features)
assert set(entrenamiento["ID_Solicitud"]).isdisjoint(validacion["ID_Solicitud"])
assert entrenamiento["Fecha_Solicitud"].max() < validacion["Fecha_Solicitud"].min()
assert validacion["Fecha_Solicitud"].max() < test_features["Fecha_Solicitud"].min()


# X contiene los predictores; y contiene la respuesta que queremos aprender.
# Los identificadores y la fecha no se incluyen como variables predictoras.
columnas_predictoras = [
    "Monto_Solicitado",
    "Tasa_Interes",
    "Ingreso_Mensual",
    "Relacion_Monto_Ingreso",
    "Motivo_Prestamo",
] + columnas_features
X_train = entrenamiento[columnas_predictoras].copy()
y_train = entrenamiento["Estado_Final"].copy()
X_validacion = validacion[columnas_predictoras].copy()
y_validacion = validacion["Estado_Final"].copy()
X_test = test_features[columnas_predictoras].copy()

# Motivo_Prestamo sigue siendo texto: se codificara en la etapa del modelo.
# No ajustamos escaladores ni otras transformaciones en esta etapa.
for nombre, datos in [("ENTRENAMIENTO", entrenamiento), ("VALIDACION", validacion)]:
    print(f"\n--- SEPARACION TEMPORAL: {nombre} ---")
    print("Solicitudes:", len(datos))
    print("Usuarios distintos:", datos["ID_Usuario"].nunique())
    print("Fechas:", datos["Fecha_Solicitud"].min(), "a", datos["Fecha_Solicitud"].max())
    distribucion = datos["Estado_Final"].value_counts().reindex([0, 1], fill_value=0)
    print(pd.DataFrame({
        "Cantidad": distribucion,
        "Porcentaje": (distribucion / len(datos) * 100).round(2),
    }))

usuarios_train = set(entrenamiento["ID_Usuario"])
usuarios_validacion = set(validacion["ID_Usuario"])
usuarios_compartidos = usuarios_train & usuarios_validacion
solicitudes_usuarios_nuevos = (~validacion["ID_Usuario"].isin(usuarios_train)).sum()
print("\nUsuarios compartidos entre entrenamiento y validacion:", len(usuarios_compartidos))
print("Solicitudes de validacion de usuarios nuevos:", solicitudes_usuarios_nuevos)
print("El test final tiene usuarios nuevos; tendremos en cuenta esta diferencia al evaluar.")
print("\nDimensiones X_train:", X_train.shape)
print("Dimensiones X_validacion:", X_validacion.shape)
print("Dimensiones X_test:", X_test.shape)
print("Variables predictoras:", columnas_predictoras)

# REGRESION LOGISTICA: PREPARAR, APRENDER Y EVALUAR
columnas_numericas = [c for c in columnas_predictoras if c != "Motivo_Prestamo"]
# Estandarizar coloca las variables numericas en escalas comparables.
# One-hot crea una columna 0/1 por motivo, sin inventar un orden entre motivos.
preparacion = ColumnTransformer([
    ("numericas", StandardScaler(), columnas_numericas),
    ("motivo", OneHotEncoder(handle_unknown="ignore"), ["Motivo_Prestamo"]),
])

# El pipeline aprende el escalado, las categorias y el modelo solo con train.
modelo_logistico = Pipeline([
    ("preparacion", preparacion),
    ("modelo", LogisticRegression(max_iter=1000)),
])
modelo_logistico.fit(X_train, y_train)

# Clase 1 = pago. La clasificacion usa un umbral de 0.5 para esta evaluacion.
# Clasificar como pago no es todavia una decision de aprobar el prestamo.
predicciones_validacion = modelo_logistico.predict(X_validacion)
indice_pago = list(modelo_logistico.classes_).index(1)
probabilidades_pago_validacion = modelo_logistico.predict_proba(X_validacion)[:, indice_pago]

# Referencia sencilla: predecir siempre la clase mas frecuente en train.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
predicciones_baseline = baseline.predict(X_validacion)
probabilidades_baseline = baseline.predict_proba(X_validacion)[:, list(baseline.classes_).index(1)]


def mostrar_evaluacion(nombre, reales, predicciones, probabilidades):
    print(f"\n--- EVALUACION: {nombre} ---")
    print(f"Accuracy: {accuracy_score(reales, predicciones):.4f}")
    # ROC-AUC necesita ejemplos de ambas clases.
    if reales.nunique() == 2:
        print(f"ROC-AUC (probabilidad de pago): {roc_auc_score(reales, probabilidades):.4f}")
    print("Matriz: filas = resultado real; columnas = prediccion")
    print(pd.DataFrame(
        confusion_matrix(reales, predicciones, labels=[0, 1]),
        index=["Real default", "Real pago"],
        columns=["Predice default", "Predice pago"],
    ))
    print(classification_report(
        reales, predicciones, labels=[0, 1],
        target_names=["Default (0)", "Pago (1)"], zero_division=0,
    ))


mostrar_evaluacion("BASELINE", y_validacion, predicciones_baseline, probabilidades_baseline)
mostrar_evaluacion(
    "REGRESION LOGISTICA", y_validacion,
    predicciones_validacion, probabilidades_pago_validacion,
)

# Evaluar tambien solicitudes de personas ausentes del entrenamiento.
mascara_nuevos = ~validacion["ID_Usuario"].isin(usuarios_train)
if mascara_nuevos.any():
    mostrar_evaluacion(
        "LOGISTICA - USUARIOS NUEVOS", y_validacion.loc[mascara_nuevos],
        predicciones_validacion[mascara_nuevos.to_numpy()],
        probabilidades_pago_validacion[mascara_nuevos.to_numpy()],
    )

ejemplos_validacion = validacion[["ID_Solicitud", "Estado_Final"]].copy()
ejemplos_validacion["Prediccion"] = predicciones_validacion
ejemplos_validacion["Probabilidad_Pago"] = probabilidades_pago_validacion
print("\n--- EJEMPLOS DE PREDICCIONES EN VALIDACION ---")
print(ejemplos_validacion.head().to_string(index=False))


# SELECCION FINANCIERA: LAS RESPUESTAS REALES NO INTERVIENEN EN LA DECISION.
PRESUPUESTO = 2_000_000
def seleccionar_prestamos(solicitudes, probabilidades_pago):
    """Seleccion voraz por rentabilidad esperada; no garantiza el optimo global.

    Las probabilidades deben venir en el mismo orden que las solicitudes.
    Devuelve una fila por solicitud, conservando el orden original.
    """
    if len(solicitudes) != len(probabilidades_pago):
        raise ValueError("Debe haber una probabilidad por solicitud")
    if not solicitudes["ID_Solicitud"].is_unique:
        raise ValueError("Los IDs de solicitud deben ser unicos")
    # Seleccionar solo estos datos impide utilizar Estado_Final por accidente.
    seleccion = solicitudes[
        ["ID_Solicitud", "Monto_Solicitado", "Tasa_Interes"]
    ].copy().set_index("ID_Solicitud")
    seleccion["Probabilidad_Pago"] = list(probabilidades_pago)
    p = seleccion["Probabilidad_Pago"]
    monto = seleccion["Monto_Solicitado"]
    tasa = seleccion["Tasa_Interes"]
    if not p.between(0, 1).all():
        raise ValueError("Las probabilidades deben estar entre 0 y 1, sin nulos")
    if not monto.between(1, float("inf"), inclusive="left").all():
        raise ValueError("Los montos deben ser positivos y finitos")
    if not tasa.between(0, float("inf"), inclusive="left").all():
        raise ValueError("Las tasas deben ser no negativas y finitas")

    seleccion["Ganancia_Esperada"] = p * monto * tasa - (1 - p) * monto
    seleccion["Rentabilidad_Esperada"] = seleccion["Ganancia_Esperada"] / monto
    seleccion["Aprobar"] = 0
    candidatos = seleccion.loc[seleccion["Ganancia_Esperada"] > 0].sort_values(
        "Rentabilidad_Esperada", ascending=False, kind="stable"
    )
    capital_utilizado = 0
    for id_solicitud, prestamo in candidatos.iterrows():
        monto_solicitado = prestamo["Monto_Solicitado"]
        if capital_utilizado + monto_solicitado <= PRESUPUESTO:
            seleccion.loc[id_solicitud, "Aprobar"] = 1
            capital_utilizado += monto_solicitado
        # Si no cabe, seguimos buscando otros prestamos que si puedan caber.

    capital_aprobado = seleccion.loc[
        seleccion["Aprobar"] == 1, "Monto_Solicitado"
    ].sum()
    if capital_aprobado > PRESUPUESTO:
        raise ValueError("La seleccion supera el presupuesto")
    assert capital_aprobado <= 2_000_000
    return seleccion.reset_index()


seleccion_validacion = seleccionar_prestamos(validacion, probabilidades_pago_validacion)
# Solo despues de decidir incorporamos la respuesta real, por ID.
evaluacion_financiera = seleccion_validacion.merge(
    validacion[["ID_Solicitud", "Estado_Final"]],
    on="ID_Solicitud", how="left", validate="one_to_one",
)
evaluacion_financiera["Ganancia_Real"] = 0.0
aprobados = evaluacion_financiera["Aprobar"] == 1
pagaron = evaluacion_financiera["Estado_Final"] == 1
evaluacion_financiera.loc[aprobados & pagaron, "Ganancia_Real"] = (
    evaluacion_financiera.loc[aprobados & pagaron, "Monto_Solicitado"]
    * evaluacion_financiera.loc[aprobados & pagaron, "Tasa_Interes"]
)
evaluacion_financiera.loc[aprobados & ~pagaron, "Ganancia_Real"] = (
    -evaluacion_financiera.loc[aprobados & ~pagaron, "Monto_Solicitado"]
)
capital_validacion = evaluacion_financiera.loc[aprobados, "Monto_Solicitado"].sum()
assert capital_validacion <= 2_000_000
ganancia_esperada_validacion = evaluacion_financiera.loc[aprobados, "Ganancia_Esperada"].sum()
ganancia_real_validacion = evaluacion_financiera["Ganancia_Real"].sum()

print("\n--- EVALUACION FINANCIERA EN VALIDACION ---")
print("Cantidad de aprobados:", int(aprobados.sum()))
print(f"Capital utilizado (USD): {capital_validacion:,.2f}")
print(f"Capital restante (USD): {PRESUPUESTO - capital_validacion:,.2f}")
print(f"Ganancia esperada (USD): {ganancia_esperada_validacion:,.2f}")
print(f"Ganancia real (USD): {ganancia_real_validacion:,.2f}")
print("La ganancia de estos tres meses no es una prediccion del score del test.")

# CONTROL DE ENTREGA FINAL CON EL FORMATO CONFIRMADO.
def guardar_submission(solicitudes_test, decisiones):
    """Decisiones: Series indexada por ID_Solicitud, con 1=aprobar y 0=rechazar.
    Exporta ID_Solicitud y Decision_aprobacion en el orden del test.
    """
    presupuesto = 2_000_000
    if len(solicitudes_test) != 3061 or len(decisiones) != 3061:
        raise ValueError("Se requieren exactamente 3061 solicitudes y decisiones")
    if not solicitudes_test["ID_Solicitud"].is_unique or not decisiones.index.is_unique:
        raise ValueError("Los identificadores de las solicitudes no deben repetirse")
    if set(decisiones.index) != set(solicitudes_test["ID_Solicitud"]):
        raise ValueError("Las decisiones deben corresponder exactamente a los IDs del test")
    if not decisiones.isin([0, 1]).all():
        raise ValueError("Cada decision debe ser 0 o 1, sin valores faltantes")

    # Reordenar por ID, aunque la seleccion haya ordenado por rentabilidad.
    ordenadas = decisiones.reindex(solicitudes_test["ID_Solicitud"]).astype(int)
    montos = solicitudes_test["Monto_Solicitado"].reset_index(drop=True)
    if montos.isna().any() or not montos.gt(0).all():
        raise ValueError("Los montos del test deben ser positivos y no tener nulos")
    capital_aprobado = montos.loc[ordenadas.to_numpy() == 1].sum()

    # La excepcion protege incluso si Python se ejecuta con asserts desactivados.
    if capital_aprobado > presupuesto:
        raise ValueError("Presupuesto superado: no se guarda submission.csv")
    assert capital_aprobado <= 2_000_000

    submission = pd.DataFrame({
        "ID_Solicitud": solicitudes_test["ID_Solicitud"].to_numpy(),
        "Decision_aprobacion": ordenadas.to_numpy(),
    })
    assert len(submission) == 3061
    assert submission["Decision_aprobacion"].isin([0, 1]).all()
    assert submission["ID_Solicitud"].tolist() == solicitudes_test["ID_Solicitud"].tolist()

    print("\n--- CONTROL DE SUBMISSION ---")
    print("Cantidad de aprobados:", int(ordenadas.sum()))
    print("Capital utilizado (USD):", capital_aprobado)
    print("Capital restante (USD):", presupuesto - capital_aprobado)
    submission.to_csv("submission.csv", index=False)
    return submission

# Mantener el modelo de validacion y entrenar una copia con todo el historico.
# Se conserva la misma configuracion; la evaluacion anterior no se recalcula.
modelo_final = clone(modelo_logistico)
modelo_final.fit(train_features[columnas_predictoras], train_features["Estado_Final"])
indice_pago_final = list(modelo_final.classes_).index(1)
probabilidades_pago_test = modelo_final.predict_proba(X_test)[:, indice_pago_final]
seleccion_test = seleccionar_prestamos(test, probabilidades_pago_test)
decisiones_test = seleccion_test.set_index("ID_Solicitud")["Aprobar"]
submission = guardar_submission(test, decisiones_test)
