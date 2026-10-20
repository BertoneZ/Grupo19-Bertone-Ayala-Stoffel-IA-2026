from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from datos import cargar_datos
from features import (
    FEATURE_CATEGORICA, FEATURES_BASE, FEATURES_MONTO,
    construir_features, estimar_factor_escala, resumir_historial_bancario,
)
from modelos import SEMILLA, LightGBMVariasSemillas, crear_ensamble, crear_logistica
from negocio import PRESUPUESTO, guardar_submission, seleccionar_prestamos, simular_ganancia_como_test

CARPETA_SALIDA = Path(__file__).parent / "submissions"


def predecir_fuera_de_muestra(crear_modelo, datos, features):
    y = datos["Estado_Final"].to_numpy()
    prob = np.zeros(len(datos))
    particion = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    for idx_train, idx_valid in particion.split(datos, y, groups=datos["ID_Usuario"]):
        modelo = crear_modelo().fit(datos.iloc[idx_train][features], y[idx_train])
        prob[idx_valid] = modelo.predict_proba(datos.iloc[idx_valid][features])[:, 1]
    return prob


def main():
    train, test, bancario = cargar_datos()

    factor_monto = estimar_factor_escala(train, test)
    fraccion_presupuesto = PRESUPUESTO / test["Monto_Solicitado"].sum()
    print(f"Los montos del test son {factor_monto:.2f} veces mas chicos que los de train.")
    print(f"El presupuesto cubre el {fraccion_presupuesto:.1%} del capital pedido en el test.")

    resumen_bancario = resumir_historial_bancario(bancario)
    train_f = construir_features(train, resumen_bancario, factor_monto=1.0)
    test_f = construir_features(test, resumen_bancario, factor_monto=factor_monto)

    features_A = FEATURES_MONTO + FEATURES_BASE
    features_B = FEATURES_BASE
    columnas_A = features_A + [FEATURE_CATEGORICA]
    columnas_B = features_B + [FEATURE_CATEGORICA]

    print("\n--- VALIDACION CRUZADA (ganancia simulada como si fuera el test, escala 2M USD) ---")
    y = train_f["Estado_Final"].to_numpy()
    prob_logistica = predecir_fuera_de_muestra(lambda: crear_logistica(features_A, FEATURE_CATEGORICA), train_f, columnas_A)
    prob_lgbm = predecir_fuera_de_muestra(lambda: LightGBMVariasSemillas(features_A, FEATURE_CATEGORICA), train_f, columnas_A)
    prob_logistica_B = predecir_fuera_de_muestra(lambda: crear_logistica(features_B, FEATURE_CATEGORICA), train_f, columnas_B)
    prob_lgbm_B = predecir_fuera_de_muestra(lambda: LightGBMVariasSemillas(features_B, FEATURE_CATEGORICA), train_f, columnas_B)

    resultados = {
        "Regresion logistica (A)": prob_logistica,
        "LightGBM (A)": prob_lgbm,
        "Ensamble (A)  <- submission A": (prob_logistica + prob_lgbm) / 2,
        "Ensamble sin monto (B)  <- submission B": (prob_logistica_B + prob_lgbm_B) / 2,
        "Maximo teorico (adivinando todo)": y * 0.999,
    }
    for nombre, prob in resultados.items():
        auc = roc_auc_score(y, prob)
        ganancia = simular_ganancia_como_test(train_f, prob, fraccion_presupuesto)
        print(f"  {nombre:42s} AUC {auc:.4f} | ganancia USD {ganancia:>9,.0f}")
    print("  (Kaggle publico mide solo una parte del test, por eso alli el numero es menor.)")

    print("\n--- SUBMISSIONS ---")
    for nombre_archivo, features, columnas in [
        ("submission_A_principal.csv", features_A, columnas_A),
        ("submission_B_sin_monto.csv", features_B, columnas_B),
    ]:
        modelo = crear_ensamble(features, FEATURE_CATEGORICA).fit(train_f[columnas], y)
        prob_test = modelo.predict_proba(test_f[columnas])[:, 1]
        aprobar = seleccionar_prestamos(test_f, prob_test)
        guardar_submission(test_f, aprobar, CARPETA_SALIDA / nombre_archivo)


if __name__ == "__main__":
    main()
