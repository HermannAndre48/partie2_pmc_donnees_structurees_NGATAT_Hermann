import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Chargement Pima (URL directe, pas de compte Kaggle requis)
pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indiansdiabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(pima_url, names=cols)
# TODO : afficher df['Outcome'].value_counts() pour voir la distribution de classes
# noter les proportions exactes avant d'entraîner quoi que ce soit
# TODO : afficher (df == 0).sum() pour toutes les colonnes
# Glucose=0, BMI=0, Insulin=0, SkinThickness=0 sont physiologiquement impossibles
# ce sont des NaN déguisés en zéros, encodage courant dans les datasets médicaux
anciens
# on les laisse pour l'instant, mais les noter : c'est un point de fragilité réel
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
# TODO : split train/test 80/20, random_state=42
# TODO : instancier StandardScaler, le fitter sur X_train UNIQUEMENT, transformer train et
test
# (fitter sur X_test = data leakage : le modèle verrait les stats du test avant
évaluation)
# TODO : construire un modèle Sequential binaire
# architecture : Dense(64, relu, input_shape=(8,)) -> Dense(32, relu) -> Dense(1,
sigmoid)
# compiler avec optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
# TODO : entraîner 100 epochs, validation_split=0.2, batch_size=32
# stocker le résultat dans une variable `history
# TODO : afficher la val_accuracy finale (max sur toutes les epochs)
# et vérifier que model.predict(X_val).mean() est proche de 0.35 (pas 0.05)