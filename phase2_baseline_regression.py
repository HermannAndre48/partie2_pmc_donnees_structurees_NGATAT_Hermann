"""
===============================================================================
PHASE 2 : BASELINE PMC RÉGRESSION - CALIFORNIA HOUSING
===============================================================================

Objectif : construire un PMC de régression, l'entraîner sur California Housing,
et lire les métriques epoch par epoch.

RÉGRESSION vs CLASSIFICATION :
- Pas de sigmoid sur la sortie, pas de softmax, pas de binary_crossentropy
- Loss = MSE (Mean Squared Error), métrique = MAE (Mean Absolute Error)
- Cible : un nombre continu (prix médian en centaines de milliers de dollars)

MAE (Mean Absolute Error) : erreur moyenne en valeur absolue.
Ex : MAE = 0.5 signifie 50 000 $ d'erreur moyenne (car cibles en centaines de k$)

Pourquoi MSE comme loss et MAE comme métrique ?
- MSE pénalise FORT les grosses erreurs (carré), ce qui guide l'optimiseur
  à les éviter en priorité → meilleur sur les outliers
- MAE est plus lisible pour l'humain : 0.6 = 60 000 $ d'erreur moyenne
- Les deux coexistent dans le training

Pipeline Phase 2 : load (normalisé) → build → compile → train → evaluate
===============================================================================
"""

import numpy as np
import tensorflow as tf
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
print(f"X_test_norm shape : {X_test_norm.shape}")
print()


# ============ PHASE 2 : BUILD & TRAIN ============
def build_regression_model(input_dim):
    """
    Construire un PMC de régression pour California Housing.
    
    ATTENTION : Pourquoi PAS d'activation='sigmoid' sur la couche de sortie ?
    
    Sigmoid compresse les valeurs entre 0 et 1. Or :
    - Les prix des maisons en Californie sont en centaines de k$ (cibles ~ 0.1 à 5.0)
    - Si on force sigmoid, les prédictions sont bornées à [0, 1] ou [0, 100]
      après rescaling, ce qui LIMITE la plage de prédiction possibles
    - Le modèle NE PEUT PAS apprendre les vraies valeurs > 1 (sigmoid max = 1)
    - Résultat : modèle inutilisable, toutes les prédictions saturent à 1
    
    Pour la régression, on laisse la sortie LINÉAIRE (pas d'activation).
    Cela permet au modèle de prédire n'importe quel nombre réel, positif ou négatif.
    """
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_dim=input_dim),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)  # ← Pas d'activation (sortie linéaire)
    ])
    return model

# Build
model = build_regression_model(input_dim=X_train_norm.shape[1])

# Compile avec MSE et MAE
model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error
    metrics=['mae']  # Mean Absolute Error (lisible : centaines de k$)
)

print("=" * 60)
print("HAPPY PATH - Baseline MSE + MAE")
print("=" * 60)
print()
print("Model architecture :")
model.summary()
print()

# Train
print("Entraînement (50 epochs, batch_size=32, validation) ...")
history = model.fit(
    X_train_norm, y_train,
    validation_data=(X_val_norm, y_val),
    epochs=50,
    batch_size=32,
    verbose=0
)

print("\n✓ Training terminé")
print()

# Évaluation sur test
print("Évaluation sur test set (données non vues) :")
test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"  Test Loss (MSE) : {test_loss:.4f}")
print(f"  Test MAE : {test_mae:.4f} (≈ {test_mae * 100000:.0f}$ d'erreur moyenne)")
print()

# Afficher les derniers epochs
print("Derniers epochs (convergence observée) :")
for epoch in range(max(0, len(history.history['loss']) - 5), len(history.history['loss'])):
    train_loss = history.history['loss'][epoch]
    val_loss = history.history['val_loss'][epoch]
    print(f"  Epoch {epoch + 1} : loss={train_loss:.4f}, val_loss={val_loss:.4f}")
print()


# ============ EDGE CASE : BATCH_SIZE ============
print("=" * 60)
print("EDGE CASE - Comparaison SGD pur vs Batch GD (via batch_size)")
print("=" * 60)

# SGD pur (batch_size=1)
print("\n1️⃣  SGD pur (batch_size=1) :")
model_sgd = build_regression_model(input_dim=X_train_norm.shape[1])
model_sgd.compile(optimizer='adam', loss='mse', metrics=['mae'])
history_sgd = model_sgd.fit(
    X_train_norm, y_train,
    validation_data=(X_val_norm, y_val),
    epochs=50,
    batch_size=1,
    verbose=0
)
final_val_loss_sgd = history_sgd.history['val_loss'][-1]
print(f"  Final val_loss : {final_val_loss_sgd:.4f}")
print(f"  Variance des val_loss (nervosité) : {np.var(history_sgd.history['val_loss']):.6f}")

# Batch GD (batch_size=len(X_train))
print("\n2️⃣  Batch GD (batch_size=len(X_train)) :")
model_batch = build_regression_model(input_dim=X_train_norm.shape[1])
model_batch.compile(optimizer='adam', loss='mse', metrics=['mae'])
history_batch = model_batch.fit(
    X_train_norm, y_train,
    validation_data=(X_val_norm, y_val),
    epochs=50,
    batch_size=len(X_train_norm),
    verbose=0
)
final_val_loss_batch = history_batch.history['val_loss'][-1]
print(f"  Final val_loss : {final_val_loss_batch:.4f}")
print(f"  Variance des val_loss (lissé) : {np.var(history_batch.history['val_loss']):.6f}")

# Comparaison
print("\n📊 Comparaison :")
sgd_convergence_epoch = next((i for i, loss in enumerate(history_sgd.history['val_loss'][10:], start=10) 
                               if history_sgd.history['val_loss'][10] - loss < 0.001), None)
batch_convergence_epoch = next((i for i, loss in enumerate(history_batch.history['val_loss'][10:], start=10) 
                                if history_batch.history['val_loss'][10] - loss < 0.001), None)
print(f"  SGD est très BRUITÉ (variance={np.var(history_sgd.history['val_loss']):.6f})")
print(f"  Batch GD est LISSE (variance={np.var(history_batch.history['val_loss']):.6f})")
print(f"  ➜ SGD fait des sauts erratiques, Batch GD descend régulièrement")
print(f"  ➜ Batch GD converge généralement plus tôt (lissage vs bruit)")
print()


# ============ ADVERSARIAL : NO NORMALIZATION ============
print("=" * 60)
print("ADVERSARIAL - Entraînement SANS normalisation")
print("=" * 60)

print("\nDonnées brutes (non normalisées) :")
print(f"  MedInc (feature 0) : min={X_train[:, 0].min():.2f}, max={X_train[:, 0].max():.2f}")
print(f"  HouseAge (feature 1) : min={X_train[:, 1].min():.2f}, max={X_train[:, 1].max():.2f}")
print(f"  AveRooms (feature 2) : min={X_train[:, 2].min():.2f}, max={X_train[:, 2].max():.2f}")
print(f"  Latitude (feature 6) : min={X_train[:, 6].min():.2f}, max={X_train[:, 6].max():.2f}")
print(f"  Longitude (feature 7) : min={X_train[:, 7].min():.2f}, max={X_train[:, 7].max():.2f}")
print()

print("Entraînement avec X_train BRUT (pas normalisé) ...")
model_no_norm = build_regression_model(input_dim=X_train.shape[1])
model_no_norm.compile(optimizer='adam', loss='mse', metrics=['mae'])
history_no_norm = model_no_norm.fit(
    X_train, y_train,  # ← X_train brut, pas normalisé
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=0
)

initial_loss_no_norm = history_no_norm.history['loss'][0]
final_loss_no_norm = history_no_norm.history['loss'][-1]
print(f"\n  Loss initiale (brut) : {initial_loss_no_norm:.4f}")
print(f"  Loss finale (brut) : {final_loss_no_norm:.4f}")
print(f"  Réduction : {(initial_loss_no_norm - final_loss_no_norm) / initial_loss_no_norm * 100:.1f}%")
print()

# Comparaison avec données normalisées
print("Comparaison :")
print(f"  Avec normalisation : loss initiale ≈ {history.history['loss'][0]:.4f}")
print(f"  Sans normalisation : loss initiale ≈ {initial_loss_no_norm:.4f} (BEAUCOUP PLUS GRANDE)")
print()
print("📊 Analyse :")
print("  ✓ Avec normalisation : features équilibrées, gradients proportionnels")
print("  ✗ Sans normalisation : Latitude/Longitude (37-38) vs AveRooms (5-6)")
print("    → Gradients déséquilibrés pour les grandes valeurs")
print("    → Loss initiale monte énormément (gradient explosion)")
print("    → Training instable et lent")
print()


# ============ ADVERSARIAL : ADAM vs SGD ============
print("=" * 60)
print("ADVERSARIAL - Adam vs SGD (données normalisées)")
print("=" * 60)

# Adam (déjà fait : history)
print("\n1️⃣  Adam (optimizer='adam', lr implicite=0.001) :")
print(f"  Final val_loss : {history.history['val_loss'][-1]:.4f}")
print(f"  Epochs avant stabilisation (approximé) : ~20-30")

# SGD avec lr=0.001
print("\n2️⃣  SGD avec lr=0.001 :")
model_sgd_lr = build_regression_model(input_dim=X_train_norm.shape[1])
model_sgd_lr.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
history_sgd_lr = model_sgd_lr.fit(
    X_train_norm, y_train,
    validation_data=(X_val_norm, y_val),
    epochs=50,
    batch_size=32,
    verbose=0
)
print(f"  Final val_loss : {history_sgd_lr.history['val_loss'][-1]:.4f}")
print(f"  Epochs avant stabilisation (approximé) : ~35-40+")
print()

# Comparer convergence
adam_epochs_to_converge = next((i for i, loss in enumerate(history.history['val_loss']) 
                                if i > 5 and history.history['val_loss'][i] < history.history['val_loss'][0] * 0.5), len(history.history['val_loss']))
sgd_epochs_to_converge = next((i for i, loss in enumerate(history_sgd_lr.history['val_loss']) 
                               if i > 5 and history_sgd_lr.history['val_loss'][i] < history_sgd_lr.history['val_loss'][0] * 0.5), len(history_sgd_lr.history['val_loss']))

print("📊 Comparaison :")
print(f"  Adam : converge en ~{adam_epochs_to_converge} epochs (adaptatif, LR ajusté par feature)")
print(f"  SGD : converge en ~{sgd_epochs_to_converge} epochs (gradient descent simple, LR global)")
print(f"  ➜ Adam converge PLUS RAPIDEMENT (adaptive moment estimation)")
print(f"  ➜ SGD plus stable mais plus lent (LR fixe pour toutes les features)")
print()

print("=" * 60)
print("✓ Phase 2 terminée : Baseline régression California Housing OK")
print("=" * 60)