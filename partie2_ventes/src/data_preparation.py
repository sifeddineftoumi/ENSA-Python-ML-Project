"""
Projet ENSA Berrechid 2025-2026
Partie 2 : Prédiction des ventes de vêtements
Script 3 : Préparation des données pour le modeling

Ce script prépare les données pour les modèles de prédiction :
- Split train/test (80/20)
- Normalisation si nécessaire
- Sauvegarde des ensembles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

sns.set_style('whitegrid')

print("="*70)
print("PRÉPARATION DES DONNÉES POUR LE MODELING")
print("="*70)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1] Chargement des données...")

df = pd.read_csv('../data/sales_data.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

print(f"✓ Données chargées : {df.shape[0]} lignes")
print(f"  - Date de début : {df.index.min().date()}")
print(f"  - Date de fin : {df.index.max().date()}")

# ============================================================================
# 2. DÉCOUPAGE TRAIN/TEST
# ============================================================================
print("\n[2] Découpage Train/Test (80/20)...")

# Calculer le point de séparation (80% pour train)
split_point = int(len(df) * 0.8)
split_date = df.index[split_point]

# Séparation
train_data = df.iloc[:split_point].copy()
test_data = df.iloc[split_point:].copy()

print(f"\n📊 Résultats du découpage :")
print(f"  TRAIN SET :")
print(f"    - Taille : {len(train_data)} jours ({len(train_data)/len(df)*100:.1f}%)")
print(f"    - Période : {train_data.index.min().date()} → {train_data.index.max().date()}")
print(f"    - Ventes moyennes : {train_data['sales'].mean():.0f}")
print(f"\n  TEST SET :")
print(f"    - Taille : {len(test_data)} jours ({len(test_data)/len(df)*100:.1f}%)")
print(f"    - Période : {test_data.index.min().date()} → {test_data.index.max().date()}")
print(f"    - Ventes moyennes : {test_data['sales'].mean():.0f}")

# ============================================================================
# 3. VÉRIFICATION DE LA REPRÉSENTATIVITÉ
# ============================================================================
print("\n[3] Vérification de la représentativité...")

# Comparer les distributions
from scipy import stats

# Test de Kolmogorov-Smirnov
ks_statistic, ks_pvalue = stats.ks_2samp(train_data['sales'], test_data['sales'])

print(f"\n📊 Test de Kolmogorov-Smirnov :")
print(f"  - Statistique : {ks_statistic:.4f}")
print(f"  - p-value : {ks_pvalue:.4f}")

if ks_pvalue > 0.05:
    print("  ✓ Les distributions sont similaires (p > 0.05)")
else:
    print("  ⚠️ Les distributions diffèrent significativement (p < 0.05)")

# Statistiques comparatives
print(f"\n📈 Statistiques comparatives :")
print(f"  Métrique       | Train   | Test    | Différence")
print(f"  " + "-"*55)
print(f"  Moyenne        | {train_data['sales'].mean():7.1f} | {test_data['sales'].mean():7.1f} | {abs(train_data['sales'].mean() - test_data['sales'].mean()):7.1f}")
print(f"  Écart-type     | {train_data['sales'].std():7.1f} | {test_data['sales'].std():7.1f} | {abs(train_data['sales'].std() - test_data['sales'].std()):7.1f}")
print(f"  Min            | {train_data['sales'].min():7.0f} | {test_data['sales'].min():7.0f} | {abs(train_data['sales'].min() - test_data['sales'].min()):7.0f}")
print(f"  Max            | {train_data['sales'].max():7.0f} | {test_data['sales'].max():7.0f} | {abs(train_data['sales'].max() - test_data['sales'].max()):7.0f}")

# ============================================================================
# 4. NORMALISATION (Pour LSTM)
# ============================================================================
print("\n[4] Normalisation des données (pour LSTM)...")

# Créer le scaler sur les données d'entraînement uniquement
scaler = MinMaxScaler(feature_range=(0, 1))
train_sales_scaled = scaler.fit_transform(train_data[['sales']])
test_sales_scaled = scaler.transform(test_data[['sales']])

print(f"✓ Normalisation effectuée [0, 1]")
print(f"  - Min original : {train_data['sales'].min()}")
print(f"  - Max original : {train_data['sales'].max()}")

# Sauvegarder le scaler
os.makedirs('../outputs/models', exist_ok=True)
joblib.dump(scaler, '../outputs/models/scaler.pkl')
print(f"✓ Scaler sauvegardé : ../outputs/models/scaler.pkl")

# ============================================================================
# 5. SAUVEGARDE DES ENSEMBLES
# ============================================================================
print("\n[5] Sauvegarde des ensembles...")

# Sauvegarder les datasets
train_data.to_csv('../data/train_data.csv')
test_data.to_csv('../data/test_data.csv')

print(f"✓ Train set sauvegardé : ../data/train_data.csv")
print(f"✓ Test set sauvegardé : ../data/test_data.csv")

# Sauvegarder les données normalisées
np.save('../data/train_scaled.npy', train_sales_scaled)
np.save('../data/test_scaled.npy', test_sales_scaled)

print(f"✓ Données normalisées sauvegardées")

# ============================================================================
# 6. VISUALISATION DU DÉCOUPAGE
# ============================================================================
print("\n[6] Visualisation du découpage...")

fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Découpage Train/Test des Données', fontsize=18, fontweight='bold')

# Vue d'ensemble
axes[0].plot(train_data.index, train_data['sales'], 
             color='steelblue', linewidth=1.5, label='Train Set')
axes[0].plot(test_data.index, test_data['sales'], 
             color='coral', linewidth=1.5, label='Test Set')
axes[0].axvline(x=split_date, color='red', linestyle='--', linewidth=2, 
                label=f'Split: {split_date.date()}')
axes[0].set_title('Série Temporelle Complète', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Ventes', fontsize=12)
axes[0].legend(fontsize=11, loc='upper left')
axes[0].grid(True, alpha=0.3)

# Distributions comparées
axes[1].hist(train_data['sales'], bins=40, alpha=0.6, color='steelblue', 
             label='Train', density=True, edgecolor='black')
axes[1].hist(test_data['sales'], bins=40, alpha=0.6, color='coral', 
             label='Test', density=True, edgecolor='black')
axes[1].axvline(train_data['sales'].mean(), color='blue', linestyle='--', 
                linewidth=2, label=f'Moy. Train: {train_data["sales"].mean():.0f}')
axes[1].axvline(test_data['sales'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Moy. Test: {test_data["sales"].mean():.0f}')
axes[1].set_title('Distributions Train vs Test', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Ventes', fontsize=12)
axes[1].set_ylabel('Densité', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../outputs/visualizations/7_train_test_split.png', dpi=150, bbox_inches='tight')
print("✓ Graphique sauvegardé : 7_train_test_split.png")

# ============================================================================
# 7. CRÉATION D'UN RÉSUMÉ
# ============================================================================
print("\n[7] Création du résumé de préparation...")

summary = {
    'total_samples': len(df),
    'train_size': len(train_data),
    'test_size': len(test_data),
    'split_date': str(split_date.date()),
    'train_mean': float(train_data['sales'].mean()),
    'test_mean': float(test_data['sales'].mean()),
    'train_std': float(train_data['sales'].std()),
    'test_std': float(test_data['sales'].std()),
    'ks_pvalue': float(ks_pvalue),
    'scaler_min': float(train_data['sales'].min()),
    'scaler_max': float(train_data['sales'].max())
}

import json
with open('../outputs/data_preparation_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("✓ Résumé sauvegardé : ../outputs/data_preparation_summary.json")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "="*70)
print("✅ PRÉPARATION DES DONNÉES TERMINÉE !")
print("="*70)

print("\n📊 Fichiers créés :")
print("  1. train_data.csv - Ensemble d'entraînement")
print("  2. test_data.csv - Ensemble de test")
print("  3. train_scaled.npy - Données normalisées (train)")
print("  4. test_scaled.npy - Données normalisées (test)")
print("  5. scaler.pkl - Objet scaler pour dénormalisation")
print("  6. data_preparation_summary.json - Résumé")

print("\n📈 Statistiques :")
print(f"  - Train : {len(train_data)} jours (80%)")
print(f"  - Test : {len(test_data)} jours (20%)")
print(f"  - Split date : {split_date.date()}")

print("\n🎯 Prochaine étape : Modèle ARIMA (4_model_arima.py)")
print("="*70)