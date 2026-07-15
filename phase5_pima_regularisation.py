import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
# X_train_norm, X_val, y_train, y_val issus de la Phase 4
# (ou reconstruire le preprocessing en tête de fichier)
def build_pima_regularized(l2_lambda=0.01, use_dropout=False):
"""
Modèle Pima avec régularisation L2 optionnelle et Dropout optionnel.
Si use_dropout=True, insère un Dropout(0.3) après chaque couche cachée.
"""
# TODO : créer un keras.Sequential vide
# TODO : ajouter Dense(64, relu, input_shape=(8,)) avec
kernel_regularizer=regularizers.l2(l2_lambda)
# TODO : si use_dropout est True, ajouter layers.Dropout(0.3) immédiatement après
# TODO : ajouter Dense(32, relu) avec kernel_regularizer=regularizers.l2(l2_lambda)
# TODO : si use_dropout est True, ajouter layers.Dropout(0.3) immédiatement après
# TODO : ajouter Dense(1, activation='sigmoid') en couche de sortie (pas de
régularisation ici)
# TODO : compiler avec optimizer='adam', loss='binary_crossentropy', metrics=
['accuracy']
# TODO : retourner le modèle
pass
# Callback Early Stopping : surveille val_loss, arrêt si pas d'amélioration pendant 15
epochs.
# restore_best_weights=True récupère les poids du meilleur epoch, pas du dernier.
early_stopping = keras.callbacks.EarlyStopping(
monitor='val_loss', patience=15, restore_best_weights=True
)
# --- Configuration 1 : baseline non régularisé (architecture identique à Phase 4) ---
# TODO : construire model_baseline avec build_pima_regularized(l2_lambda=0.0,
use_dropout=False)
# TODO : appeler model_baseline.fit(X_train_norm, y_train, epochs=300,
validation_split=0.2,
# callbacks=[early_stopping], verbose=0)
# TODO : stocker le retour dans history_baseline
# TODO : afficher l'epoch d'arrêt (len(history_baseline.history['val_loss'])) et
max(val_accuracy)
# --- Configuration 2 : L2 seul (l2_lambda=0.01, use_dropout=False) ---
# TODO : construire model_l2 avec build_pima_regularized(l2_lambda=0.01,
use_dropout=False)
# TODO : entraîner avec les mêmes paramètres et le même callback
# TODO : stocker dans history_l2, afficher l'epoch d'arrêt et le max val_accuracy
# --- Configuration 3 : L2 + Dropout (l2_lambda=0.01, use_dropout=True) ---
# TODO : construire model_l2_drop avec build_pima_regularized(l2_lambda=0.01,
use_dropout=True)
# TODO : entraîner et stocker dans history_l2_drop
# TODO : afficher l'epoch d'arrêt et le max val_accuracy
# TODO : tracer les trois courbes val_loss côte à côte avec matplotlib
# titres : "Baseline", "L2 seul", "L2 + Dropout"
# ajouter une ligne verticale à l'epoch d'arrêt pour chaque config
# sauvegarder en phase5_pima_3configs.png