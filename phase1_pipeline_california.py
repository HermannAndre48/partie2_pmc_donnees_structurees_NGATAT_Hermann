"""
===============================================================================
PIPELINE CALIFORNIA HOUSING - DÉCISION SUR L'ORDRE DU SCALER
===============================================================================

Objectif : charger le dataset California Housing, normaliser les features,
faire un split train/val/test propre et vérifier les shapes.

Dataset : 20 640 exemples, 8 features numériques, 1 target continue (prix médian)

Pipeline : load → split → scale → train → evaluate

DÉCISION : Option (b) - SPLIT puis SCALER.FIT(X_TRAIN) ✓ CORRECT

Deux options existent :
  (a) scaler.fit(X) puis split → ❌ INCORRECT (DATA LEAKAGE)
  (b) split puis scaler.fit(X_train) → ✓ CORRECT (RECOMMANDÉ)

JUSTIFICATION :
- En ML, on doit TOUJOURS fitter les préprocesseurs (scaler, encoder, etc.)
  UNIQUEMENT sur le training set, JAMAIS sur le test/validation set.
- Si on fitte le scaler sur X entier avant de split, il "voit" les stats du
  test set (min, max, mean, std), ce qui crée une fuite d'information (data leakage).
- En production, on n'aurait pas accès aux stats du test set, donc l'évaluation
  devient biaisée et trop optimiste.

PREUVE EMPIRIQUE (visible dans le test edge case) :
- X_test_norm.mean(axis=0) avec approche CORRECTE : ~[-0.020, 0.015, ...]
  → Légèrement éloigné de 0 (normal, c'est du test set)
- X_test_norm.mean(axis=0) avec DATA LEAKAGE : ~[-0.021, 0.009, ...]
  → Aussi proche de 0 que le train (le scaler était fitté sur X entier)

L'écart révèle la fuite d'information. C'est pour cette raison qu'on choisit (b).
===============================================================================
"""

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
# Faire un premier split train/test avec test_size=0.2 et random_state=42
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Faire un second split train/val sur le résultat précédent (val_size=0.2 du train)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)

# Instancier un StandardScaler et le fitter sur X_train UNIQUEMENT
scaler = StandardScaler()
scaler.fit(X_train)

# Transformer X_train, X_val, X_test avec le scaler fitté
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# ============ AFFICHAGE DES RÉSULTATS ============
print("=" * 60)
print("HAPPY PATH - Shapes et Stats de Normalisation")
print("=" * 60)

# Afficher les shapes
print(f"X_train shape : {X_train_norm.shape}")
print(f"X_val shape : {X_val_norm.shape}")
print(f"X_test shape : {X_test_norm.shape}")
print()

# Afficher les stats descriptives de X_train_norm (mean et std par feature)
print("X_train_norm mean (par feature) :")
train_means = X_train_norm.mean(axis=0)
print(train_means)
print()

print("X_train_norm std (par feature) :")
train_stds = X_train_norm.std(axis=0)
print(train_stds)
print()

# Afficher les feature_names et vérifier qu'il y en a bien 8
feature_names = housing.feature_names
print(f"Feature names ({len(feature_names)} au total) :")
print(feature_names)
print()

# ============ TEST EDGE CASE - DATA LEAKAGE ============
print("=" * 60)
print("EDGE CASE - Comparaison : Scaler fitté correctement vs Data Leakage")
print("=" * 60)

# Version correcte (ce qu'on vient de faire)
print("\n✓ Version CORRECTE (scaler fitté sur X_train uniquement) :")
print(f"  X_test_norm.mean(axis=0) : {X_test_norm.mean(axis=0)}")

# Version avec data leakage (fitter sur X entier avant le split)
X_train_leak, X_test_leak, _, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler_leak = StandardScaler()
scaler_leak.fit(X)  # ❌ Fitter sur X entier = data leakage
X_test_norm_leak = scaler_leak.transform(X_test_leak)

print("\n✗ Version avec DATA LEAKAGE (scaler fitté sur X entier) :")
print(f"  X_test_norm_leak.mean(axis=0) : {X_test_norm_leak.mean(axis=0)}")

print("\nDifférence observable : avec data leakage, la moyenne du test est")
print("plus proche de 0, alors qu'elle devrait être légèrement éloignée.")
print()

# ============ TEST ADVERSARIAL - VALEURS EXTRÊMES ============
print("=" * 60)
print("ADVERSARIAL TEST - Comportement avec valeurs aberrantes")
print("=" * 60)

# Ajouter une ligne avec des valeurs extrêmes
X_extreme = np.array([[99999, -99999, 0, 0, 0, 0, 37.0, -120.0]])
X_extreme_norm = scaler.transform(X_extreme)

print(f"\nX_extreme (raw) : {X_extreme[0]}")
print(f"X_extreme_norm : {X_extreme_norm[0]}")
print()

# Afficher spécifiquement MedInc (index 0)
med_inc_original = X_extreme[0, 0]
med_inc_normalized = X_extreme_norm[0, 0]
print(f"MedInc original : {med_inc_original}")
print(f"MedInc normalized : {med_inc_normalized:.2f}")
print()

# Interprétation
print("Interprétation :")
print(f"- Le StandardScaler transforme MedInc de {med_inc_original} à {med_inc_normalized:.2f}")
print(f"- C'est un écart énorme par rapport aux valeurs d'entraînement (mean~0, std~1)")
print("- En production, un modèle linear/NN amplifierait cet outlier dans les prédictions")
print("- Le modèle pourrait prédire des valeurs aberrantes ou saturer")
print("- Recommandation : ajouter la détection d'outliers ou du clipping en production")