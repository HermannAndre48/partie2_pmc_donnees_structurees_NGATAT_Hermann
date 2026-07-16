# -*- coding: utf-8 -*-
"""
PHASE 6 : Recherche d'hyperparamètres avec keras-tuner RandomSearch
=======================================================================

Objectif :
---------
Automatiser la recherche d'hyperparamètres sur le dataset Pima en utilisant
keras-tuner avec stratégie RandomSearch.

Espace de recherche (5 dimensions, 15 trials) :
- learning_rate : [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
- units_1 : [32, 64, 96, 128]
- units_2 : [16, 32, 48, 64]
- activation : ['relu', 'tanh']
- dropout_rate : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

PROTECTION CONTRE L'UNDERFITTING :
- max_epochs=100 pour chaque trial (suffisant pour Early Stopping)
- patience=10 pour EarlyStopping
- Variance testée avec seed=42 vs seed=43 pour valider stabilité

COMMENT SAVOIR SI ON A ASSEZ CHERCHÉ ?
- Si max_trials=1 donne très différent de max_trials=15 (delta > 2%), la recherche ajoute du signal
- Comparer val_accuracy pour short max_epochs (100) vs long max_epochs (200)
  Si la différence de meilleur model > 1%, alors 100 epochs tue les bons modèles prématurément
"""

import sys
import os

# Set environment for better output
os.environ['PYTHONIOENCODING'] = 'utf-8'

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

print("=" * 70)
print("PHASE 6 : Keras-tuner RandomSearch Pima Diabetes")
print("=" * 70)

# ========== ÉTAPE 1 : CHARGEMENT ET PRÉPARATION DES DONNÉES ==========
print("\n" + "=" * 70)
print("ÉTAPE 1 : Chargement Pima depuis OpenML")
print("=" * 70)

try:
    data = fetch_openml(name='diabetes', version=1, as_frame=True, parser='auto')
    X_raw = data.data
    y_raw = data.target
    target_col = data.target_names[0] if hasattr(data, 'target_names') else 'class'
    
    print(f"[OK] Dataset chargé : shape = {X_raw.shape}")
    print(f"  Target colonne : {target_col}")
    
    # Identifier la colonne cible et la supprimer si elle est en X
    if target_col in X_raw.columns:
        X_raw = X_raw.drop(columns=[target_col])
    
    # Encoder target catégorique en numérique (0/1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    print(f"[OK] Target convertie : {np.unique(y_raw)} → {np.unique(y_encoded)}")
    print(f"  Distribution : {np.bincount(y_encoded)}")
    
except Exception as e:
    print(f"[FAIL] Erreur chargement OpenML : {e}")
    raise

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw.values, y_encoded, test_size=0.2, random_state=42
)

# Normalisation : fit scaler SEULEMENT sur train, puis apply on test
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

print(f"[OK] Split & Scaling done")
print(f"  X_train_norm : shape={X_train_norm.shape}, mean≈{X_train_norm.mean():.4f}, std≈{X_train_norm.std():.4f}")
print(f"  X_test_norm : shape={X_test_norm.shape}, mean≈{X_test_norm.mean():.4f}, std≈{X_test_norm.std():.4f}")

# ========== ÉTAPE 2 : DÉFINIR LE MODÈLE AVEC ESPACE D'HYPERPARAMÈTRES ==========
print("\n" + "=" * 70)
print("ÉTAPE 2 : HyperModel Pima (5 dimensions)")
print("=" * 70)

def build_pima_model(hp):
    """
    Construit un modèle Sequential avec hyperparamètres échantillonnés.
    
    Espace d'exploration :
    - learning_rate : 5 choix discrets [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    - units_1 : 4 choix [32, 64, 96, 128]
    - units_2 : 4 choix [16, 32, 48, 64]
    - activation : 2 choix ['relu', 'tanh']
    - dropout_rate : 6 choix [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    Total combinaisons possibles : 5*4*4*2*6 = 960
    RandomSearch avec 15 trials échantillonne 15/960 ≈ 1.6% de l'espace
    """
    
    model = keras.Sequential()
    
    # Couche 1 avec dropout optionnel
    units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    activation = hp.Choice('activation', values=['relu', 'tanh'])
    model.add(layers.Dense(units_1, activation=activation, input_shape=(8,)))
    
    # Dropout après couche 1 (si dropout_rate > 0)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    # Couche 2
    units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
    model.add(layers.Dense(units_2, activation=activation))
    
    # Dropout après couche 2
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    # Couche sortie
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compiler avec learning_rate variable
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

print("[OK] HyperModel défini avec 5 dimensions")

# ========== ÉTAPE 3 : HAPPY PATH - MAIN TUNING (seed=42) ==========
print("\n" + "=" * 70)
print("HAPPY PATH : RandomSearch avec 15 trials (seed=42)")
print("=" * 70)

# Supprimer répertoire précédent si existe (pour clean rebuild)
if os.path.exists('tuning_pima'):
    shutil.rmtree('tuning_pima')

tuner = kt.RandomSearch(
    build_pima_model,
    objective='val_accuracy',
    max_trials=15,
    seed=42,
    directory='tuning_pima',
    project_name='pima_random',
    executions_per_trial=1,
    overwrite=False
)

print("[OK] Tuner RandomSearch initialisé")
print("\n--- Résumé de l'espace de recherche ---")
tuner.search_space_summary()

print("\n--- Lancement de la recherche (15 trials) ---")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_tuning = tuner.search(
    X_train_norm, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

print("[OK] Recherche terminée")

# ========== ÉTAPE 4 : ANALYSE DES RÉSULTATS ==========
print("\n" + "=" * 70)
print("ÉTAPE 4 : Analyse des résultats (15 trials)")
print("=" * 70)

best_hp = tuner.get_best_hyperparameters()[0]
print("\n--- Meilleurs hyperparamètres (trial 1) ---")
print(f"  learning_rate : {best_hp.get('learning_rate')}")
print(f"  units_1 : {best_hp.get('units_1')}")
print(f"  units_2 : {best_hp.get('units_2')}")
print(f"  activation : {best_hp.get('activation')}")
print(f"  dropout_rate : {best_hp.get('dropout_rate')}")

# Afficher top 5 trials
print("\n--- Top 5 meilleurs trials ---")
tuner.results_summary(num_trials=5)

# Analyser invariants dans top 5
print("\n--- Invariants dans top 5 (signal concentré) ---")
top_hps = tuner.get_best_hyperparameters(num_trials=5)
top_configs = {}
for i, hp in enumerate(top_hps):
    hp_values = hp.values
    lr_val = hp_values.get('learning_rate', 'N/A')
    print(f"\nTrial {i+1} : learning_rate = {lr_val}")
    for key, val in hp_values.items():
        if key not in top_configs:
            top_configs[key] = []
        top_configs[key].append(val)
        print(f"  {key} = {val}")

print("\n--- Analyse des plages récurrentes (top 5) ---")
for param, values in top_configs.items():
    unique_vals = list(set(values))
    print(f"  {param} : {unique_vals} (appears {len(values)} times in top 5)")

# ========== ÉTAPE 5 : ENTRAÎNER LE MEILLEUR MODÈLE JUSQU'À CONVERGENCE ==========
print("\n" + "=" * 70)
print("ÉTAPE 5 : Entraîner meilleur modèle à convergence")
print("=" * 70)

best_model = tuner.hypermodel.build(best_hp)
print("[OK] Meilleur modèle construit")

history_best = best_model.fit(
    X_train_norm, y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

best_val_acc = max(history_best.history['val_accuracy'])
best_test_pred = best_model.predict(X_test_norm, verbose=0)
best_test_pred_binary = (best_test_pred > 0.5).astype(int).flatten()
best_test_acc = (best_test_pred_binary == y_test).mean()

print(f"[OK] Meilleur modèle entraîné : {len(history_best.history['loss'])} epochs")
print(f"  Best val_accuracy : {best_val_acc:.4f}")
print(f"  Test accuracy : {best_test_acc:.4f}")

# ========== VÉRIFICATIONS HAPPY PATH ==========
print("\n" + "=" * 70)
print("VÉRIFICATIONS HAPPY PATH")
print("=" * 70)

check1 = len(top_hps) >= 5
print(f"[OK] CHECK 1 : Au moins 5 trials disponibles ? {check1}")

check2 = best_val_acc > 0.76
print(f"{'[OK]' if check2 else '[WARN]'} CHECK 2 : Best val_accuracy > 0.76 ? {best_val_acc:.4f} {'[OK]' if check2 else '[FAIL]'}")

# ========== EDGE CASE : UNE SEULE ITÉRATION (max_trials=1) ==========
print("\n" + "=" * 70)
print("EDGE CASE : RandomSearch avec max_trials=1 (single random config)")
print("=" * 70)

if os.path.exists('tuning_pima_edge'):
    shutil.rmtree('tuning_pima_edge')

tuner_edge = kt.RandomSearch(
    build_pima_model,
    objective='val_accuracy',
    max_trials=1,
    seed=42,
    directory='tuning_pima_edge',
    project_name='pima_edge',
    executions_per_trial=1,
    overwrite=False
)

print("[WARN] Tuner RandomSearch (max_trials=1) lancé...")
early_stop_edge = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
tuner_edge.search(
    X_train_norm, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop_edge],
    verbose=0
)

best_hp_edge = tuner_edge.get_best_hyperparameters()[0]
best_model_edge = tuner_edge.hypermodel.build(best_hp_edge)
history_edge = best_model_edge.fit(
    X_train_norm, y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stop_edge],
    verbose=0
)

edge_val_acc = max(history_edge.history['val_accuracy'])
print(f"[WARN] Edge case val_accuracy (single trial) : {edge_val_acc:.4f}")
print(f"  Gain de recherche 15→1 trials : {best_val_acc - edge_val_acc:.4f} ({(best_val_acc - edge_val_acc)*100:.2f}%)")

check_edge = (best_val_acc - edge_val_acc) > 0.01
print(f"{'[OK]' if check_edge else '[WARN]'} CHECK EDGE : Gain > 1% ? {check_edge}")

# ========== ADVERSARIAL : MAUVAIS ESPACE DE RECHERCHE ==========
print("\n" + "=" * 70)
print("ADVERSARIAL : Learning rate catastrophique [10.0-100.0]")
print("=" * 70)

def build_pima_model_adversarial(hp):
    """Même modèle, mais learning_rate dans plage catastrophique."""
    model = keras.Sequential()
    units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    activation = hp.Choice('activation', values=['relu', 'tanh'])
    model.add(layers.Dense(units_1, activation=activation, input_shape=(8,)))
    
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
    model.add(layers.Dense(units_2, activation=activation))
    
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # [WARN] MAUVAIS : learning_rate > 1.0 diverge
    learning_rate = hp.Float('learning_rate', min_value=10.0, max_value=100.0)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

if os.path.exists('tuning_pima_adv'):
    shutil.rmtree('tuning_pima_adv')

tuner_adv = kt.RandomSearch(
    build_pima_model_adversarial,
    objective='val_accuracy',
    max_trials=5,  # Moins d'essais pour sauver du temps
    seed=42,
    directory='tuning_pima_adv',
    project_name='pima_adv',
    executions_per_trial=1,
    overwrite=False
)

print("[WARN] Tuner avec mauvais espace de recherche lancé...")
tuner_adv.search(
    X_train_norm, y_train,
    epochs=50,  # Moins d'epochs
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
    verbose=0
)

best_hp_adv = tuner_adv.get_best_hyperparameters()[0]
best_model_adv = tuner_adv.hypermodel.build(best_hp_adv)
history_adv = best_model_adv.fit(
    X_train_norm, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
    verbose=0
)

adv_val_acc = max(history_adv.history['val_accuracy'])
print(f"[FAIL] Adversarial val_accuracy : {adv_val_acc:.4f}")
print(f"  Comparé à naive baseline (classe majorité) ≈ 0.651 : {adv_val_acc:.4f}")

check_adv = adv_val_acc < 0.70
print(f"[OK] CHECK ADV : Val_acc < 0.70 (collapse) ? {check_adv}")

# ========== STABILITÉ : seed=42 vs seed=43 ==========
print("\n" + "=" * 70)
print("STABILITÉ : Variance avec seed=43 vs seed=42")
print("=" * 70)

if os.path.exists('tuning_pima_seed43'):
    shutil.rmtree('tuning_pima_seed43')

tuner_s43 = kt.RandomSearch(
    build_pima_model,
    objective='val_accuracy',
    max_trials=15,
    seed=43,  # AUTRE SEED
    directory='tuning_pima_seed43',
    project_name='pima_seed43',
    executions_per_trial=1,
    overwrite=False
)

print("[WARN] Tuner RandomSearch (seed=43) lancé...")
tuner_s43.search(
    X_train_norm, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
    verbose=0
)

best_hp_s43 = tuner_s43.get_best_hyperparameters()[0]
best_model_s43 = tuner_s43.hypermodel.build(best_hp_s43)
history_s43 = best_model_s43.fit(
    X_train_norm, y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

s43_val_acc = max(history_s43.history['val_accuracy'])

print(f"[OK] Seed 42 best val_accuracy : {best_val_acc:.4f}")
print(f"[OK] Seed 43 best val_accuracy : {s43_val_acc:.4f}")

seed_variance = abs(best_val_acc - s43_val_acc)
print(f"  Variance absolue : {seed_variance:.4f} ({seed_variance*100:.2f}%)")

check_stability = seed_variance < 0.02
print(f"{'[OK]' if check_stability else '[WARN]'} CHECK STABILITY : Variance < 2% ? {check_stability}")

# ========== VISUALISATION : Distribution val_accuracy (seed 42 vs 43) ==========
print("\n" + "=" * 70)
print("VISUALISATION : Distribution val_accuracy (seed 42 vs seed 43)")
print("=" * 70)

# Récupérer val_accuracy de tous les trials
def get_trial_accuracies(tuner):
    """Extrait les val_accuracy max de chaque trial du tuner."""
    accuracies = []
    for trial in tuner.oracle.trials.values():
        if hasattr(trial, 'best_value') and trial.best_value is not None:
            accuracies.append(trial.best_value)
    return sorted(accuracies, reverse=True)

acc_s42 = get_trial_accuracies(tuner)
acc_s43 = get_trial_accuracies(tuner_s43)

print(f"Seed 42 - 15 trials val_accuracy : {[f'{a:.4f}' for a in acc_s42[:5]]} ... (top 5)")
print(f"Seed 43 - 15 trials val_accuracy : {[f'{a:.4f}' for a in acc_s43[:5]]} ... (top 5)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1 : Distribution seed 42
axes[0].bar(range(len(acc_s42)), acc_s42, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axhline(y=best_val_acc, color='red', linestyle='--', linewidth=2, label=f'Best: {best_val_acc:.4f}')
axes[0].axhline(y=0.76, color='green', linestyle='--', linewidth=2, label='Target: 0.76')
axes[0].set_xlabel('Trial rank (sorted desc)')
axes[0].set_ylabel('Val Accuracy')
axes[0].set_title('Seed 42 : Distribution of 15 trials')
axes[0].set_ylim([0.65, 0.8])
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Subplot 2 : Distribution seed 43
axes[1].bar(range(len(acc_s43)), acc_s43, color='coral', alpha=0.7, edgecolor='black')
axes[1].axhline(y=s43_val_acc, color='red', linestyle='--', linewidth=2, label=f'Best: {s43_val_acc:.4f}')
axes[1].axhline(y=0.76, color='green', linestyle='--', linewidth=2, label='Target: 0.76')
axes[1].set_xlabel('Trial rank (sorted desc)')
axes[1].set_ylabel('Val Accuracy')
axes[1].set_title('Seed 43 : Distribution of 15 trials')
axes[1].set_ylim([0.65, 0.8])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('phase6_stability_comparison.png', dpi=150, bbox_inches='tight')
print("[OK] Figure sauvegardée : phase6_stability_comparison.png")
plt.close()

# ========== RÉSUMÉ FINAL ==========
print("\n" + "=" * 70)
print("RÉSUMÉ FINAL : PHASE 6 TERMINÉE")
print("=" * 70)

summary_df = pd.DataFrame({
    'Configuration': [
        'RandomSearch 15 trials (s42)',
        'RandomSearch 1 trial (s42)',
        'Adversarial bad LR (5 trials)',
        'RandomSearch 15 trials (s43)'
    ],
    'Val Accuracy': [
        f'{best_val_acc:.4f}',
        f'{edge_val_acc:.4f}',
        f'{adv_val_acc:.4f}',
        f'{s43_val_acc:.4f}'
    ],
    'Status': [
        '[OK] Main result' if best_val_acc > 0.76 else '[WARN] Target missed',
        '[WARN] Single random config',
        '[FAIL] Collapsed (bad space)',
        f'{"[OK]" if abs(s43_val_acc - best_val_acc) < 0.02 else "[WARN]"} Stability check'
    ]
})

print("\n" + summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("OBJECTIF PHASE 6 :")
print(f"  Main best val_accuracy : {best_val_acc:.4f}")
print(f"  Target >= 0.76 : {'[OK] ATTEINT' if best_val_acc >= 0.76 else '[WARN] MANQUÉ'}")
print(f"  Gain recherche (15→1 trials) : {(best_val_acc - edge_val_acc)*100:.2f}%")
print(f"  Stabilité (variance seed) : {seed_variance*100:.2f}%")
print("=" * 70)
