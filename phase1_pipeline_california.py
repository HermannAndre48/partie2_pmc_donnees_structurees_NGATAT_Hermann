import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Charger le dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
# StandardScaler (https://scikitlearn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
# normalise chaque feature : (X - mean) / std.
# Résultat : mean = 0, std = 1 sur le train set.
# Pourquoi fitter sur X_train uniquement : si on fitte sur X entier,
# les stats du scaler "voient" le test set avant l'évaluation (data leakage).
# TODO : faire un premier split train/test avec test_size=0.2 et random_state=42
# TODO : faire un second split train/val sur le résultat précédent (val_size=0.2 du train)
# TODO : instancier un StandardScaler et le fitter sur X_train UNIQUEMENT
# TODO : transformer X_train, X_val, X_test avec le scaler fitté
# TODO : afficher les shapes de X_train, X_val, X_test
# TODO : afficher les stats descriptives de X_train_norm (mean et std par feature)
# TODO : afficher les feature_names du dataset ET vérifier qu'il y en a bien 8