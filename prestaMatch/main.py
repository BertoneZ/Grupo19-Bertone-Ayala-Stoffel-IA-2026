import pandas as pd

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
