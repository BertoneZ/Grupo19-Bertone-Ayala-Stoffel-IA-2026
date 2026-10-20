import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEMILLA = 42
MOTIVOS = ["Consumo", "Educacion", "Negocio", "Otros", "Salud", "Vehiculo", "Vivienda"]


def crear_logistica(features_numericas, categorica):
    preparacion = ColumnTransformer([
        ("numericas", StandardScaler(), features_numericas),
        ("motivo", OneHotEncoder(handle_unknown="ignore"), [categorica]),
    ])
    return Pipeline([
        ("preparacion", preparacion),
        ("modelo", LogisticRegression(C=0.1, max_iter=5000)),
    ])


class LightGBMVariasSemillas:
    def __init__(self, features_numericas, categorica, semillas=(SEMILLA, SEMILLA + 1, SEMILLA + 2)):
        self.features = features_numericas + [categorica]
        self.categorica = categorica
        self.semillas = semillas

    def _preparar(self, X):
        X = X[self.features].copy()
        X[self.categorica] = pd.Categorical(X[self.categorica], categories=MOTIVOS)
        return X

    def fit(self, X, y):
        self.modelos = [
            lgb.LGBMClassifier(
                n_estimators=1500, learning_rate=0.01, num_leaves=4, min_child_samples=200,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
                random_state=semilla, verbose=-1,
            ).fit(self._preparar(X), y)
            for semilla in self.semillas
        ]
        return self

    def predict_proba(self, X):
        X = self._preparar(X)
        return np.mean([m.predict_proba(X) for m in self.modelos], axis=0)


class Ensamble:
    def __init__(self, modelos):
        self.modelos = modelos

    def fit(self, X, y):
        for modelo in self.modelos:
            modelo.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.modelos], axis=0)


def crear_ensamble(features_numericas, categorica):
    return Ensamble([
        crear_logistica(features_numericas, categorica),
        LightGBMVariasSemillas(features_numericas, categorica),
    ])
