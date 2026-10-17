import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. CARGA DE DATASETS
train = pd.read_csv("data/solicitudes_train.csv")
test = pd.read_csv("data/solicitudes_test.csv")
bancario = pd.read_csv("data/comportamiento_bancario.csv")

train["Fecha_Solicitud"] = pd.to_datetime(train["Fecha_Solicitud"])
test["Fecha_Solicitud"] = pd.to_datetime(test["Fecha_Solicitud"])
bancario["Mes_Referencia"] = pd.to_datetime(bancario["Mes_Referencia"])

# 2. FEATURE ENGINEERING BANCARIO (VENTANAS 6M Y 3M)
def construir_features_bancarias(solicitudes, historial):
    assert "ID_Usuario" in solicitudes.columns and "ID_Solicitud" in solicitudes.columns
    assert "ID_Usuario" in historial.columns and "Mes_Referencia" in historial.columns
    assert not solicitudes["ID_Solicitud"].duplicated().any()
    assert not historial.duplicated(subset=["ID_Usuario", "Mes_Referencia"]).any()

    cruces = solicitudes[["ID_Solicitud", "ID_Usuario", "Fecha_Solicitud"]].merge(
        historial, on="ID_Usuario", how="left"
    )

    # Ventana 6 meses
    inicio_6m = cruces["Fecha_Solicitud"] - pd.DateOffset(months=6)
    mask_6m = (cruces["Mes_Referencia"] >= inicio_6m) & (cruces["Mes_Referencia"] < cruces["Fecha_Solicitud"])
    anteriores_6m = cruces[mask_6m].copy()
    anteriores_6m["Tuvo_Atraso"] = (anteriores_6m["Dias_Atraso_Otras_Deudas"] > 0).astype(int)

    # Ventana 3 meses
    inicio_3m = cruces["Fecha_Solicitud"] - pd.DateOffset(months=3)
    mask_3m = (cruces["Mes_Referencia"] >= inicio_3m) & (cruces["Mes_Referencia"] < cruces["Fecha_Solicitud"])
    anteriores_3m = cruces[mask_3m].copy()
    anteriores_3m["Tuvo_Atraso_3m"] = (anteriores_3m["Dias_Atraso_Otras_Deudas"] > 0).astype(int)

    resumen_6m = anteriores_6m.groupby("ID_Solicitud").agg(
        Saldo_Promedio_6m=("Saldo_Promedio_Mensual", "mean"),
        Saldo_Desviacion_6m=("Saldo_Promedio_Mensual", "std"),
        Atraso_Maximo_6m=("Dias_Atraso_Otras_Deudas", "max"),
        Proporcion_Meses_Atraso_6m=("Tuvo_Atraso", "mean"),
        Meses_Observados_6m=("Mes_Referencia", "count"),
        Saldo_Ultimo_Disponible_6m=("Saldo_Promedio_Mensual", "last"),
    )

    resumen_3m = anteriores_3m.groupby("ID_Solicitud").agg(
        Saldo_Promedio_3m=("Saldo_Promedio_Mensual", "mean"),
        Atraso_Maximo_3m=("Dias_Atraso_Otras_Deudas", "max"),
        Proporcion_Meses_Atraso_3m=("Tuvo_Atraso_3m", "mean"),
    )

    resumen_total = resumen_6m.join(resumen_3m, how="left")

    resumen_total["Saldo_Desviacion_6m"] = resumen_total["Saldo_Desviacion_6m"].fillna(0)
    resumen_total["Saldo_Promedio_3m"] = resumen_total["Saldo_Promedio_3m"].fillna(resumen_total["Saldo_Promedio_6m"])
    resumen_total["Atraso_Maximo_3m"] = resumen_total["Atraso_Maximo_3m"].fillna(0)
    resumen_total["Proporcion_Meses_Atraso_3m"] = resumen_total["Proporcion_Meses_Atraso_3m"].fillna(0)
    resumen_total["Tendencia_Saldo_Reciente"] = (
        resumen_total["Saldo_Ultimo_Disponible_6m"] - resumen_total["Saldo_Promedio_6m"]
    )

    resultado = solicitudes.merge(
        resumen_total, on="ID_Solicitud", how="left", validate="one_to_one"
    )

    cols_a_cero = [
        "Atraso_Maximo_6m", "Proporcion_Meses_Atraso_6m", "Meses_Observados_6m",
        "Atraso_Maximo_3m", "Proporcion_Meses_Atraso_3m", "Tendencia_Saldo_Reciente"
    ]
    for c in cols_a_cero:
        resultado[c] = resultado[c].fillna(0)

    for c in ["Saldo_Promedio_6m", "Saldo_Desviacion_6m", "Saldo_Ultimo_Disponible_6m", "Saldo_Promedio_3m"]:
        resultado[c] = resultado[c].fillna(resultado[c].median())

    return resultado

train_features = construir_features_bancarias(train, bancario)
test_features = construir_features_bancarias(test, bancario)

# 3. VARIABLES DERIVADAS SELECCIONADAS (SIN RUIDO)
for df in [train_features, test_features]:
    df["Relacion_Monto_Ingreso"] = df["Monto_Solicitado"] / (df["Ingreso_Mensual"] + 1e-5)
    df["Interes_Total_Pagar"] = df["Monto_Solicitado"] * df["Tasa_Interes"]

# 4. PARTICION TEMPORAL (ENTRENAMIENTO VS VALIDACION)
fecha_corte = pd.Timestamp("2025-10-01")
entrenamiento = train_features.loc[train_features["Fecha_Solicitud"] < fecha_corte].copy()
validacion = train_features.loc[train_features["Fecha_Solicitud"] >= fecha_corte].copy()

columnas_predictoras = [
    "Monto_Solicitado",
    "Tasa_Interes",
    "Ingreso_Mensual",
    "Relacion_Monto_Ingreso",
    "Interes_Total_Pagar",
    "Motivo_Prestamo",
    "Saldo_Promedio_6m",
    "Saldo_Desviacion_6m",
    "Atraso_Maximo_6m",
    "Proporcion_Meses_Atraso_6m",
    "Meses_Observados_6m",
    "Saldo_Ultimo_Disponible_6m",
    "Saldo_Promedio_3m",
    "Atraso_Maximo_3m",
    "Proporcion_Meses_Atraso_3m",
    "Tendencia_Saldo_Reciente",
]

X_train = entrenamiento[columnas_predictoras].copy()
y_train = entrenamiento["Estado_Final"].copy()
X_validacion = validacion[columnas_predictoras].copy()
y_validacion = validacion["Estado_Final"].copy()
X_test = test_features[columnas_predictoras].copy()

columnas_numericas = [c for c in columnas_predictoras if c != "Motivo_Prestamo"]

preparacion = ColumnTransformer([
    ("numericas", StandardScaler(), columnas_numericas),
    ("motivo", OneHotEncoder(handle_unknown="ignore"), ["Motivo_Prestamo"]),
])

# 5. ESTRATEGIA DE SELECCION FINANCIERA CON FILTRO BLINDADO
PRESUPUESTO = 2_000_000

def seleccionar_prestamos(solicitudes, probabilidades_pago, margen_seguridad=0.03, piso_prob=0.74):
    if len(solicitudes) != len(probabilidades_pago):
        raise ValueError("Debe haber una probabilidad por solicitud")
    if not solicitudes["ID_Solicitud"].is_unique:
        raise ValueError("Los IDs de solicitud deben ser unicos")

    seleccion = solicitudes[
        ["ID_Solicitud", "Monto_Solicitado", "Tasa_Interes"]
    ].copy().set_index("ID_Solicitud")
    seleccion["Probabilidad_Pago"] = list(probabilidades_pago)
    p = seleccion["Probabilidad_Pago"]
    monto = seleccion["Monto_Solicitado"]
    tasa = seleccion["Tasa_Interes"]

    seleccion["Ganancia_Esperada"] = p * monto * tasa - (1 - p) * monto
    seleccion["Rentabilidad_Esperada"] = seleccion["Ganancia_Esperada"] / monto
    seleccion["Aprobar"] = 0

    # Doble candado: margen sobre break-even Y piso mínimo de probabilidad absoluta
    umbral_minimo = 1.0 / (1.0 + tasa) + margen_seguridad
    filtro_seguro = (
        (seleccion["Ganancia_Esperada"] > 0) & 
        (seleccion["Probabilidad_Pago"] >= umbral_minimo) &
        (seleccion["Probabilidad_Pago"] >= piso_prob)
    )

    candidatos = seleccion.loc[filtro_seguro].sort_values(
        "Rentabilidad_Esperada", ascending=False, kind="stable"
    )

    # Si sobra dinero, completar con candidatos con ganancia positiva
    if candidatos["Monto_Solicitado"].sum() < PRESUPUESTO:
        resto = seleccion.loc[
            (seleccion["Ganancia_Esperada"] > 0) & (~filtro_seguro)
        ].sort_values("Rentabilidad_Esperada", ascending=False, kind="stable")
        candidatos = pd.concat([candidatos, resto])

    capital_utilizado = 0
    for id_solicitud, prestamo in candidatos.iterrows():
        monto_solicitado = prestamo["Monto_Solicitado"]
        if capital_utilizado + monto_solicitado <= PRESUPUESTO:
            seleccion.loc[id_solicitud, "Aprobar"] = 1
            capital_utilizado += monto_solicitado

    capital_aprobado = seleccion.loc[seleccion["Aprobar"] == 1, "Monto_Solicitado"].sum()
    if capital_aprobado > PRESUPUESTO:
        raise ValueError("Presupuesto superado")
    return seleccion.reset_index()

# 6. MICRO-GRID SEARCH INTELIGENTE EN VALIDACION LOCAL
print("\n--- INICIANDO BUSQUEDA DEL SUPER-RECORD LOCAL ---")
mejor_ganancia_real = -float("inf")
mejor_config = {}

# Exploramos alrededor de la zona ganadora (peso 2.0 a 2.8, C 0.03 a 0.08, margenes 0.02 a 0.045)
pesos_default = [2.0, 2.2, 2.4, 2.6]
valores_c = [0.03, 0.05, 0.08]
margenes = [0.02, 0.03, 0.035, 0.04]
pisos = [0.72, 0.75, 0.78]

total_combinaciones = len(pesos_default) * len(valores_c) * len(margenes) * len(pisos)
print(f"Probando {total_combinaciones} combinaciones de alta rentabilidad...")

for w0 in pesos_default:
    for c_val in valores_c:
        pipe = Pipeline([
            ("preparacion", preparacion),
            ("modelo", LogisticRegression(
                C=c_val,
                class_weight={0: w0, 1: 1.0},
                max_iter=1000,
                random_state=42
            )),
        ])
        pipe.fit(X_train, y_train)
        probs_val = pipe.predict_proba(X_validacion)[:, list(pipe.classes_).index(1)]

        for m_seg in margenes:
            for p_min in pisos:
                sel_val = seleccionar_prestamos(
                    validacion, probs_val, margen_seguridad=m_seg, piso_prob=p_min
                )
                eval_fin = sel_val.merge(
                    validacion[["ID_Solicitud", "Estado_Final"]],
                    on="ID_Solicitud", how="left"
                )
                aprob = eval_fin["Aprobar"] == 1
                pag = eval_fin["Estado_Final"] == 1

                ganancia_real = (
                    eval_fin.loc[aprob & pag, "Monto_Solicitado"] * eval_fin.loc[aprob & pag, "Tasa_Interes"]
                ).sum() - eval_fin.loc[aprob & ~pag, "Monto_Solicitado"].sum()

                if ganancia_real > mejor_ganancia_real:
                    mejor_ganancia_real = ganancia_real
                    mejor_config = {
                        "peso_default": w0,
                        "C": c_val,
                        "margen": m_seg,
                        "piso_prob": p_min
                    }

print("\n" + "="*50)
print(f"NUEVO RECORD ENCONTRADO!")
print(f"Configuracion: {mejor_config}")
print(f"Ganancia Real Maxima en Validacion: USD {mejor_ganancia_real:,.2f}")
print("="*50)

# 7. ENTRENAR MODELO FINAL CON EL RECORD Y GENERAR SUBMISSION
print("\n--- ENTRENANDO MODELO FINAL PARA SUBMISSION ---")
modelo_optimo = Pipeline([
    ("preparacion", preparacion),
    ("modelo", LogisticRegression(
        C=mejor_config["C"],
        class_weight={0: mejor_config["peso_default"], 1: 1.0},
        max_iter=1000,
        random_state=42
    )),
])
modelo_optimo.fit(train_features[columnas_predictoras], train_features["Estado_Final"])

probabilidades_pago_test = modelo_optimo.predict_proba(X_test)[:, list(modelo_optimo.classes_).index(1)]
seleccion_test = seleccionar_prestamos(
    test,
    probabilidades_pago_test,
    margen_seguridad=mejor_config["margen"],
    piso_prob=mejor_config["piso_prob"]
)
decisiones_test = seleccion_test.set_index("ID_Solicitud")["Aprobar"]

# 8. GUARDAR SUBMISSION.CSV
def guardar_submission(solicitudes_test, decisiones):
    presupuesto = 2_000_000
    assert len(solicitudes_test) == 3061 and len(decisiones) == 3061
    assert solicitudes_test["ID_Solicitud"].is_unique
    assert set(decisiones.index) == set(solicitudes_test["ID_Solicitud"])
    assert decisiones.isin([0, 1]).all()

    ordenadas = decisiones.reindex(solicitudes_test["ID_Solicitud"]).astype(int)
    montos = solicitudes_test["Monto_Solicitado"].reset_index(drop=True)
    capital_aprobado = montos.loc[ordenadas.to_numpy() == 1].sum()

    if capital_aprobado > presupuesto:
        raise ValueError("Presupuesto superado: no se guarda submission.csv")
    assert capital_aprobado <= 2_000_000

    submission = pd.DataFrame({
        "ID_Solicitud": solicitudes_test["ID_Solicitud"].to_numpy(),
        "Decision_aprobacion": ordenadas.to_numpy(),
    })
    submission.to_csv("submission.csv", index=False)
    print("\n--- SUBMISSION GENERADA EXITOSAMENTE ---")
    print("Cantidad de aprobados:", int(ordenadas.sum()))
    print(f"Capital utilizado (USD): {capital_aprobado:,.2f}")
    print(f"Capital restante (USD): {presupuesto - capital_aprobado:,.2f}")
    return submission

submission = guardar_submission(test, decisiones_test)