from pathlib import Path

import pandas as pd

CARPETA_DATOS = Path(__file__).parent / "data"


def cargar_datos():
    train = pd.read_csv(CARPETA_DATOS / "solicitudes_train.csv")
    test = pd.read_csv(CARPETA_DATOS / "solicitudes_test.csv")
    bancario = pd.read_csv(CARPETA_DATOS / "comportamiento_bancario.csv")

    train["Fecha_Solicitud"] = pd.to_datetime(train["Fecha_Solicitud"])
    test["Fecha_Solicitud"] = pd.to_datetime(test["Fecha_Solicitud"])
    bancario["Mes_Referencia"] = pd.to_datetime(bancario["Mes_Referencia"])

    assert train["ID_Solicitud"].is_unique and test["ID_Solicitud"].is_unique
    assert not bancario.duplicated(["ID_Usuario", "Mes_Referencia"]).any()
    assert test["ID_Usuario"].isin(bancario["ID_Usuario"]).all()
    return train, test, bancario
