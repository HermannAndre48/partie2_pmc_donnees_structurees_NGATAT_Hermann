"""
===============================================================================
PHASE 4 : DONNÉES ET BASELINE PIMA DIABETES (CLASSIFICATION BINAIRE)
===============================================================================

Objectif : poser un baseline solide et mesurable sur Pima Indians Diabetes.
Ce chiffre de référence est essentiel pour évaluer la régularisation (Phase 5)
et le tuning automatique (Phase 6). Sans baseline, on ne sait pas si L2 améliore.

DATASET PIMA INDIANS DIABETES :
- 768 femmes d'origine Pima
- 8 features médicales : Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age, etc.
- 1 target binaire : Outcome (0 = non-diabétique, 1 = diabétique)
- Distribution : ~65% classe 0, ~35% classe 1 → DÉSÉQUILIBRÉ

⚠️  PIÈGE : Un modèle stupide qui prédit toujours 0 obtient 65% d'accuracy !
    C'est la fréquence de base. Notre baseline doit faire mieux.

ARCHITECTURE CLASSIFICATION BINAIRE :
- Couche d'entrée : 8 features
- Couche cachée : Dense(64, relu)
- Couche cachée : Dense(32, relu)
- Couche de sortie : Dense(1, sigmoid) ← Sigmoid pour prédire P(y=1) ∈ [0, 1]
- Loss : binary_crossentropy ← Pas MSE ! Cross-entropy pour classification
- Métrique : accuracy ← Pourcentage de prédictions correctes

⚠️  Zéros suspects : Glucose=0, BMI=0, Insulin=0, SkinThickness=0 sont
    physiologiquement impossibles. C'est un encodage de NaN : à noter comme
    point de fragilité.

Pipeline : load → distribution → split → scale → build → compile → train → evaluate
===============================================================================
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============ PHASE 1 : LOAD & DISTRIBUTION ============
print("=" * 70)
print("PHASE 4 : Baseline Classification Binaire - Pima Diabetes")
print("=" * 70)

# Chargement Pima via UCI ML Repository (source fiable)
pima_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(pima_url, names=cols, header=None)

print("\n" + "=" * 70)
print("DISTRIBUTION DES CLASSES")
print("=" * 70)

# Convertir Outcome en entier (bincount nécessite des int)
outcome_int = df['Outcome'].astype(int).values
class_counts = np.bincount(outcome_int)
total = len(df)
print(f"\nDistribution classes :")
for class_idx, count in enumerate(class_counts):
    percentage = (count / total) * 100
    print(f"  {class_idx} : {count:4d} ({percentage:.1f}%)")

# Question piège
print(f"\n❓ QUESTION : Accuracy d'un modèle stupide qui prédit toujours 0 ?")
print(f"   RÉPONSE : {(class_counts[0] / total) * 100:.1f}%")
print(f"   → Notre baseline DOIT dépasser ce seuil pour avoir appris quelque chose!")

# Afficher les zéros suspects
print("\n" + "=" * 70)
print("COLONNES AVEC DES ZÉROS SUSPECTS (NaN encodés en 0)")
print("=" * 70)

zero_counts = (df == 0).sum()
suspicious_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nZéros suspects (physiologiquement impossibles) :")
for col in suspicious_cols:
    if col in zero_counts.index:
        count = zero_counts[col]
        if count > 0:
            percentage = (count / len(df)) * 100
            print(f"  {col:15s} : {count:3d} zéros ({percentage:.1f}%)")

print("\n⚠️  Ces zéros représentent probablement des valeurs manquantes.")
print("   C'est un point de fragilité du dataset.")
print()

# ============ PHASE 2 : EDGE CASE - IMPUTATION MÉDIANE ============
print("=" * 70)
print("EDGE CASE : Impact de l'imputation des zéros par la médiane")
print("=" * 70)

# Copie pour imputation
df_imputed = df.copy()
print("\nAvant imputation :")
print(f"  Zéros totaux dans colonnes suspects : {(df[suspicious_cols] == 0).sum().sum()}")

# Imputer les zéros par la médiane
for col in suspicious_cols:
    mask = df_imputed[col] == 0
    if mask.sum() > 0:
        median_val = df_imputed.loc[df_imputed[col] != 0, col].median()
        df_imputed.loc[mask, col] = median_val

print(f"\nAprès imputation :")
print(f"  Zéros totals dans colonnes suspects : {(df_imputed[suspicious_cols] == 0).sum().sum()}")
print("  (Les zéros ont été remplacés par la médiane)")
print()

# ============ PHASE 3 : TRAIN/TEST SPLIT & SCALING ============
print("=" * 70)
print("TRAIN/TEST SPLIT & NORMALISATION")
print("=" * 70)

# Utiliser le dataset ORIGINAL (avec zéros)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDataset original (avec zéros suspects) :")
print(f"  X_train shape : {X_train.shape}")
print(f"  X_test shape : {X_test.shape}")
print(f"  y_train distribution : {np.bincount(y_train)}")
print(f"  y_test distribution : {np.bincount(y_test)}")

# StandardScaler fitté UNIQUEMENT sur train
scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)

print(f"\n✓ Scaler fitté sur X_train uniquement (pas de data leakage)")
print()

# ============ PHASE 4 : BUILD & COMPILE ============
print("=" * 70)
print("MODÈLE : Classification Binaire PMC")
print("=" * 70)

def build_binary_classifier(input_dim):
    """
    Construire un PMC de classification binaire pour Pima.
    
    Architecture :
      - Input : input_dim features
      - Dense(64, relu)
      - Dense(32, relu)
      - Output : Dense(1, sigmoid) ← Sigmoid pour P(y=1) ∈ [0, 1]
    
    Compilation :
      - optimizer : adam
      - loss : binary_crossentropy ← Pour classification binaire
      - metrics : accuracy ← Pourcentage de prédictions correctes
    """
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_dim=input_dim),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # ← Sigmoid pour binaire
    ])
    return model

# Build
model = build_binary_classifier(input_dim=X_train_norm.shape[1])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # ← Loss pour classification binaire
    metrics=['accuracy']  # ← Métrique lisible : % correct
)

print("\nArchitecture :")
model.summary()
print()

# ============ PHASE 5 : TRAINING ============
print("=" * 70)
print("HAPPY PATH : Training Baseline (100 epochs)")
print("=" * 70)

print(f"\nEntraînement sur X_train_norm (614 samples)...")
print(f"Validation sur 20% du train (123 samples) via validation_split=0.2...")
history = model.fit(
    X_train_norm, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Résultats
val_acc_final = history.history['val_accuracy'][-1]
train_acc_final = history.history['accuracy'][-1]
val_acc_max = np.max(history.history['val_accuracy'])

print(f"\n✓ Training terminé")
print(f"  Train accuracy final : {train_acc_final:.4f}")
print(f"  Val accuracy final : {val_acc_final:.4f}")
print(f"  Val accuracy max : {val_acc_max:.4f} (epoch {np.argmax(history.history['val_accuracy']) + 1})")

# Vérifier qu'on dépasse le baseline stupide
baseline_stupide = (class_counts[0] / total) * 100 / 100
print(f"\n  Baseline stupide (toujours prédire 0) : {baseline_stupide:.4f}")
if val_acc_max > baseline_stupide:
    print(f"  ✓ Notre modèle ({val_acc_max:.4f}) DÉPASSE le baseline ({baseline_stupide:.4f})")
    print(f"    → Le modèle a appris quelque chose de réel !")
else:
    print(f"  ✗ Notre modèle ({val_acc_max:.4f}) N'AMÉLIORE PAS le baseline ({baseline_stupide:.4f})")
    print(f"    → RED FLAG : Le modèle préfère la classe majoritaire")

print()

# ============ PHASE 6 : TEST SET ============
print("=" * 70)
print("ÉVALUATION SUR TEST SET (données jamais vues)")
print("=" * 70)

test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"\nTest Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")

# Vérifier les prédictions
y_pred_proba = model.predict(X_test_norm, verbose=0)  # Probabilités
y_pred = (y_pred_proba > 0.5).astype(int).flatten()   # Binaire 0/1

pred_mean = y_pred.mean()
pred_proba_mean = y_pred_proba.mean()

print(f"\nAnalyse des prédictions sur test :")
print(f"  Moyenne des probabilités brutes : {pred_proba_mean:.4f}")
print(f"  Moyenne des prédictions binaires : {pred_mean:.4f}")
print(f"  (Doit être proche de {class_counts[1] / total:.4f}, la fréquence de classe 1)")

if pred_mean < 0.10:
    print(f"\n  ⚠️  RED FLAG : model.predict().mean() = {pred_mean:.4f}")
    print(f"     Le modèle préfère la classe 0 (non-diabétique)")
    print(f"     À relancer avec class_weight={{0: 1.0, 1: 1.9}}")
else:
    print(f"\n  ✓ OK : Le modèle voit bien la classe 1 (diabétique)")

print()

# ============ ADVERSARIAL : IMPUTATION IMPACT ============
print("=" * 70)
print("ADVERSARIAL : Imputation par médiane → Impact sur accuracy")
print("=" * 70)

# Utiliser dataset imputé
X_imp = df_imputed.drop('Outcome', axis=1).values
y_imp = df_imputed['Outcome'].values

X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(
    X_imp, y_imp, test_size=0.2, random_state=42
)

# Scaler sur données imputées
scaler_imp = StandardScaler()
scaler_imp.fit(X_train_imp)
X_train_imp_norm = scaler_imp.transform(X_train_imp)
X_test_imp_norm = scaler_imp.transform(X_test_imp)

# Entraîner un modèle sur données imputées
model_imp = build_binary_classifier(input_dim=X_train_imp_norm.shape[1])
model_imp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(f"\nEntraînement sur données IMPUTÉES (zéros → médiane)...")
history_imp = model_imp.fit(
    X_train_imp_norm, y_train_imp,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

val_acc_imp = np.max(history_imp.history['val_accuracy'])
test_loss_imp, test_acc_imp = model_imp.evaluate(X_test_imp_norm, y_test_imp, verbose=0)

print(f"\n✓ Model sur données imputées :")
print(f"  Val accuracy max : {val_acc_imp:.4f}")
print(f"  Test accuracy : {test_acc_imp:.4f}")

print(f"\nComparaison :")
print(f"  Baseline (avec zéros) : val_acc_max = {val_acc_max:.4f}, test_acc = {test_acc:.4f}")
print(f"  Imputé (zéros → médiane) : val_acc_max = {val_acc_imp:.4f}, test_acc = {test_acc_imp:.4f}")

diff = test_acc_imp - test_acc
print(f"\n  Δ Test Accuracy : {diff:+.4f} ({diff * 100:+.2f}%)")

if abs(diff) < 0.02:
    print(f"  → Impact FAIBLE : imputation ne change pas grand-chose (~2%)")
elif diff > 0.05:
    print(f"  → Impact POSITIF : imputation améliore de {diff * 100:.2f}%")
else:
    print(f"  → Impact NÉGATIF : imputation dégrade de {abs(diff) * 100:.2f}%")

print()

print("=" * 70)
print("✓ Phase 4 terminée : Baseline établie et tests complets")
print("=" * 70)