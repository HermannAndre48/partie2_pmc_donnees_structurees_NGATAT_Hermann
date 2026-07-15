"""
===============================================================================
PHASE 3 : DIAGNOSTIC TENSORBOARD SUR CALIFORNIA HOUSING
===============================================================================

Objectif : brancher TensorBoard sur deux runs de California Housing
(un avec normalisation, un sans) et comparer les scalars dans l'UI.

TensorBoard est votre outil de diagnostic PRINCIPAL : à partir d'ici,
plus de diagnostics à l'aveugle.

TROIS SITUATIONS CLASSIQUES OBSERVÉES EN TENSORBOARD :
=========================================================

(a) TRAIN et VAL DESCENDENT ENSEMBLE ✓ SAIN
    - Les deux courbes baissent régulièrement
    - Petit écart entre train et val (normal : val toujours un peu plus haut)
    - Interprétation : apprentissage équilibré, pas d'overfitting

(b) TRAIN DESCEND, VAL STAGNE ou REMONTE ⚠️ OVERFITTING
    - Train loss baisse, val loss plafonne ou augmente
    - Le modèle overfitte : trop de capacité, pas assez de régularisation
    - Interprétation : généralisation mauvaise, trop adapter au train set

(c) VAL PASSE SOUS TRAIN ❌ SUSPECT - PROBLÈME EN AMONT
    - Val loss < train loss sur toute la courbe (très inhabituel)
    - Presque toujours un problème : scaler appris sur tout, fuite de données,
      val trop facile, split défaillant
    - Interprétation : revérifier split et scaler.fit() sur X_train uniquement

HORODATAGE DES DOSSIERS :
Pourquoi timestamp (HHMMSS) sur chaque run ?
- Si on relance le même run sans timestamp, TensorBoard écrit dans le MÊME
  dossier et MÉLANGE les courbes
- L'horodatage garantit que chaque run a son propre espace isolé
- Format : logs/fit/california_norm_143022 vs logs/fit/california_raw_143045

OUTIL : tensorboard --logdir=logs/fit
Ouvrir http://localhost:6006 dans le navigateur
===============================================================================
"""

import numpy as np
import datetime
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============ PHASE 1 : LOAD & NORMALIZE ============
print("=" * 60)
print("Phase 1 : Chargement et normalisation California Housing")
print("=" * 60)

housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split train/test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)

# Scaler fitté UNIQUEMENT sur train
scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

print(f"X_train_norm shape : {X_train_norm.shape}")
print(f"X_val_norm shape : {X_val_norm.shape}")
print()


# ============ PHASE 2 : BUILD MODEL ============
def build_regression_model(input_dim):
    """Construire un PMC de régression simple."""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_dim=input_dim),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)  # Pas d'activation (régression)
    ])
    return model


# ============ PHASE 3 : TENSORBOARD TRAINING ============
def train_with_tensorboard(X_train, y_train, X_val, y_val, run_name, epochs=50):
    """
    Entraîne un modèle de régression avec callback TensorBoard horodaté.
    
    Args:
        X_train, y_train : données d'entraînement
        X_val, y_val : données de validation
        run_name : nom du run (ex: "california_norm")
        epochs : nombre d'epochs
    
    Returns:
        (model, history)
    """
    # Construire log_dir avec timestamp HHMMSS
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = f"logs/fit/{run_name}_{timestamp}"
    
    # Instancier TensorBoard callback
    # histogram_freq=1 : enregistrer les distributions des poids à chaque epoch
    tb_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Instancier et compiler le modèle
    model = build_regression_model(input_dim=X_train.shape[1])
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Entraîner avec callback TensorBoard
    print(f"\n📊 Entraînement '{run_name}' ({epochs} epochs, batch_size=32)")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[tb_callback],
        verbose=0
    )
    
    print(f"✓ Run '{run_name}' terminé.")
    print(f"  Logs TensorBoard : {log_dir}")
    print(f"  Final val_loss : {history.history['val_loss'][-1]:.4f}")
    
    return model, history


print("\n" + "=" * 60)
print("PHASE 3 : TensorBoard - Deux runs avec callback")
print("=" * 60)

# Run 1 : données NORMALISÉES (bon comportement attendu = situation a)
print("\n🔵 Run 1 : California NORMALISÉ (bon comportement attendu)")
model_norm, history_norm = train_with_tensorboard(
    X_train_norm, y_train,
    X_val_norm, y_val,
    run_name="california_norm",
    epochs=50
)

# Run 2 : données BRUTES (comportement dégradé = situation b/c)
print("\n🔴 Run 2 : California BRUT (comportement dégradé à observer)")
model_raw, history_raw = train_with_tensorboard(
    X_train, y_train,
    X_val, y_val,
    run_name="california_raw",
    epochs=50
)

# ============ DIAGNOSTIC : INTERPRÉTATION DES COURBES ============
print("\n" + "=" * 60)
print("DIAGNOSTIC : Analyse des deux runs")
print("=" * 60)

# Comparaison des loss finales
norm_final_loss = history_norm.history['val_loss'][-1]
norm_final_mae = history_norm.history['val_mae'][-1]
raw_final_loss = history_raw.history['val_loss'][-1]
raw_final_mae = history_raw.history['val_mae'][-1]

print(f"\n🔵 California NORMALISÉ :")
print(f"  Val Loss final : {norm_final_loss:.6f}")
print(f"  Val MAE final : {norm_final_mae:.4f}")
print(f"  Train Loss final : {history_norm.history['loss'][-1]:.6f}")

print(f"\n🔴 California BRUT :")
print(f"  Val Loss final : {raw_final_loss:.6f}")
print(f"  Val MAE final : {raw_final_mae:.4f}")
print(f"  Train Loss final : {history_raw.history['loss'][-1]:.6f}")

# Analyser la situation a/b/c pour chaque run
print("\n" + "=" * 60)
print("CLASSIFICATION SITUATION (a/b/c)")
print("=" * 60)

def classify_situation(history_dict, run_name):
    """Classifier la situation observée en (a), (b), ou (c)."""
    train_loss = np.array(history_dict['loss'])
    val_loss = np.array(history_dict['val_loss'])
    
    # Vérifier si val < train (situation c)
    val_under_train = np.mean(val_loss) < np.mean(train_loss) * 0.95
    if val_under_train:
        situation = "(c) VAL PASSE SOUS TRAIN ❌ SUSPECT"
        explanation = "Problème en amont : scaler appris sur tout? Fuite de données? Split défaillant?"
        return situation, explanation
    
    # Vérifier l'écart train-val au final
    final_gap = val_loss[-1] - train_loss[-1]
    val_increasing = val_loss[-10:].mean() > val_loss[-20:-10].mean()
    
    if val_increasing and final_gap > train_loss[-1] * 0.1:
        situation = "(b) TRAIN DESCEND, VAL STAGNE/REMONTE ⚠️ OVERFITTING"
        explanation = "Le modèle overfitte : trop de capacité relative aux données."
        return situation, explanation
    
    situation = "(a) TRAIN ET VAL DESCENDENT ENSEMBLE ✓ SAIN"
    explanation = "Apprentissage équilibré. Le modèle généralise correctement."
    return situation, explanation

sit_norm, exp_norm = classify_situation(history_norm.history, "california_norm")
sit_raw, exp_raw = classify_situation(history_raw.history, "california_raw")

print(f"\n🔵 California NORMALISÉ :")
print(f"  {sit_norm}")
print(f"  → {exp_norm}")

print(f"\n🔴 California BRUT :")
print(f"  {sit_raw}")
print(f"  → {exp_raw}")

print("\n" + "=" * 60)
print("INSTRUCTIONS TENSORBOARD")
print("=" * 60)
print("\n📊 Ouvrir TensorBoard dans un terminal :")
print("  tensorboard --logdir=logs/fit")
print("\nPuis accéder à : http://localhost:6006")
print("\nDans l'onglet 'SCALARS' :")
print("  ✓ Cochez les deux runs (california_norm_HHMMSS et california_raw_HHMMSS)")
print("  ✓ Vous devez voir deux courbes avec des couleurs différentes")
print("  ✓ Val_loss normalisé descend proprement")
print("  ✓ Val_loss brut explose ou plafonne très haut")
print("  → Même architecture, même optimizer, même durée : seule normalisation change")
print()

# ============ ADVERSARIAL : DEUX INSTANCES TENSORBOARD ============
print("=" * 60)
print("ADVERSARIAL : Deux instances TensorBoard sur même port")
print("=" * 60)

print("\n⚠️  Test : lancer deux instances de TensorBoard sur port 6006")
print("   Commande 1 (terminal 1) : tensorboard --logdir=logs/fit --port=6006")
print("   Commande 2 (terminal 2) : tensorboard --logdir=logs/fit --port=6006")
print()
print("Résultat attendu :")
print("  ✗ L'une des deux refuse de démarrer : 'Address already in use'")
print("  → En production, ce genre de conflit silencieux bloque les pipelines")
print()
print("Réflexe de libération du port :")
print("  pkill -f tensorboard")
print("  Puis relancer une seule instance.")
print()

print("=" * 60)
print("✓ Phase 3 terminée : TensorBoard configuré et runs lancés")
print("=" * 60)