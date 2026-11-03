"""Outils partagés entre le notebook d'entraînement et l'application Streamlit.

IMPORTANT : la classe IQRClipper est sauvegardée DANS les pipelines (.pkl).
Ce fichier doit donc être présent à côté de app.py pour que joblib puisse
recharger les modèles.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class IQRClipper(BaseEstimator, TransformerMixin):
    """Ramène les valeurs aberrantes (méthode IQR) dans les bornes [Q1-1.5*IQR ; Q3+1.5*IQR].

    Les bornes sont apprises UNIQUEMENT sur les données d'entraînement (pas de fuite de données)
    et réutilisées telles quelles sur les nouvelles données.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        # IQR nul (colonne quasi constante) : on ne coupe rien
        flat = iqr == 0
        self.lower_[flat] = np.nanmin(X, axis=0)[flat]
        self.upper_[flat] = np.nanmax(X, axis=0)[flat]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)  # les NaN restent NaN (imputés ensuite)

    def get_feature_names_out(self, input_features=None):
        return input_features


def build_preprocessor(num_cols, cat_cols):
    """Prétraitement complet : outliers -> valeurs manquantes -> normalisation / encodage."""
    num_pipe = Pipeline([
        ("outliers", IQRClipper()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
