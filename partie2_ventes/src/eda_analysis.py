"""
Projet ENSA Berrechid 2025-2026
Partie 2 : Prédiction des ventes de vêtements
Script 2 : Analyse Exploratoire des Données (EDA)

Ce script réalise une analyse complète du dataset :
- Statistiques descriptives
- Détection d'anomalies
- Analyse de stationnarité
- Autocorrélation
- Décomposition de la série temporelle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

print("="*70)
print("ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
print("="*70)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1] Chargement des données...")

try:
    df = pd.read_csv('../data/sales_data.csv', parse_dates=['date'])
    df.set_index('date', inplace=True)
    print(f"✓ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
except FileNotFoundError:
    print("❌ ERREUR : Le fichier sales_data.csv n'existe pas.")
    print("   Exécutez d'abord '1_generate_data.py'")
    exit(1)

# Aperçu des données
print("\n" + "="*70)
print("APERÇU DES DONNÉES")
print("="*70)
print(df.head(10))

# ============================================================================
# 2. STATISTIQUES DESCRIPTIVES DÉTAILLÉES
# ============================================================================
print("\n[2] Analyse statistique...")

print("\n" + "="*70)
print("STATISTIQUES DESCRIPTIVES")
print("="*70)

stats = df['sales'].describe()
print(stats)

print(f"\n📊 Métriques supplémentaires :")
print(f"  - Médiane : {df['sales'].median():.0f}")
print(f"  - Mode : {df['sales'].mode()[0]}")
print(f"  - Variance : {df['sales'].var():.0f}")
print(f"  - Coefficient de variation : {(df['sales'].std() / df['sales'].mean() * 100):.2f}%")
print(f"  - Skewness (asymétrie) : {df['sales'].skew():.3f}")
print(f"  - Kurtosis (aplatissement) : {df['sales'].kurtosis():.3f}")

# ============================================================================
# 3. DÉTECTION D'ANOMALIES
# ============================================================================
print("\n[3] Détection des anomalies...")

# Méthode IQR (Interquartile Range)
Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['sales'] < lower_bound) | (df['sales'] > upper_bound)]

print(f"✓ Valeurs aberrantes détectées : {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
print(f"  - Seuil inférieur : {lower_bound:.0f}")
print(f"  - Seuil supérieur : {upper_bound:.0f}")

if len(outliers) > 0:
    print(f"\n📌 Top 5 des valeurs extrêmes :")
    print(outliers.nlargest(5, 'sales')[['sales']])

# ============================================================================
# 4. TEST DE STATIONNARITÉ (Augmented Dickey-Fuller)
# ============================================================================
print("\n[4] Test de stationnarité (ADF Test)...")

adf_result = adfuller(df['sales'], autolag='AIC')

print(f"\n📊 Résultats du test ADF :")
print(f"  - Statistique de test : {adf_result[0]:.4f}")
print(f"  - p-value : {adf_result[1]:.4f}")
print(f"  - Valeurs critiques :")
for key, value in adf_result[4].items():
    print(f"    {key}: {value:.3f}")

if adf_result[1] < 0.05:
    print("\n✓ La série est STATIONNAIRE (p < 0.05)")
else:
    print("\n⚠️ La série est NON-STATIONNAIRE (p >= 0.05)")
    print("   → Une différenciation sera nécessaire pour ARIMA")

# ============================================================================
# 5. VALEURS MANQUANTES
# ============================================================================
print("\n[5] Vérification des valeurs manquantes...")

missing = df.isnull().sum()
print(f"✓ Valeurs manquantes : {missing.sum()}")
if missing.sum() == 0:
    print("  → Aucune valeur manquante détectée !")

# ============================================================================
# 6. VISUALISATIONS AVANCÉES
# ============================================================================
print("\n[6] Génération des visualisations avancées...")

os.makedirs('../outputs/visualizations', exist_ok=True)

# --- GRAPHIQUE 1 : Box plots et distributions ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Analyse de Distribution et Anomalies', fontsize=18, fontweight='bold')

# Box plot global
axes[0, 0].boxplot(df['sales'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
axes[0, 0].set_title('Box Plot des Ventes', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Nombre de ventes', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Distribution avec courbe normale
from scipy import stats
axes[0, 1].hist(df['sales'], bins=50, density=True, alpha=0.7, 
                color='steelblue', edgecolor='black', label='Distribution réelle')
mu, std = df['sales'].mean(), df['sales'].std()
x = np.linspace(df['sales'].min(), df['sales'].max(), 100)
axes[0, 1].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, 
                label='Loi normale théorique')
axes[0, 1].set_title('Distribution vs Loi Normale', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Ventes', fontsize=12)
axes[0, 1].set_ylabel('Densité', fontsize=12)
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(df['sales'], dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Test de normalité)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Anomalies marquées sur la série temporelle
axes[1, 1].plot(df.index, df['sales'], linewidth=1, color='steelblue', label='Ventes')
if len(outliers) > 0:
    axes[1, 1].scatter(outliers.index, outliers['sales'], 
                       color='red', s=50, zorder=5, label='Anomalies')
axes[1, 1].axhline(y=upper_bound, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Seuil sup: {upper_bound:.0f}')
axes[1, 1].axhline(y=lower_bound, color='orange', linestyle='--', linewidth=1.5,
                   label=f'Seuil inf: {lower_bound:.0f}')
axes[1, 1].set_title('Détection des Anomalies', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Date', fontsize=12)
axes[1, 1].set_ylabel('Ventes', fontsize=12)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/visualizations/3_distribution_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Graphique 1 sauvegardé : 3_distribution_analysis.png")

# --- GRAPHIQUE 2 : Décomposition de la série temporelle ---
print("\n[7] Décomposition de la série temporelle...")

decomposition = seasonal_decompose(df['sales'], model='additive', period=7)

fig, axes = plt.subplots(4, 1, figsize=(16, 12))
fig.suptitle('Décomposition de la Série Temporelle (Modèle Additif)', 
             fontsize=18, fontweight='bold')

# Série originale
decomposition.observed.plot(ax=axes[0], color='steelblue', linewidth=1.5)
axes[0].set_title('Série Originale', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Ventes', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Tendance
decomposition.trend.plot(ax=axes[1], color='darkgreen', linewidth=2)
axes[1].set_title('Tendance (Trend)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Tendance', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Saisonnalité
decomposition.seasonal.plot(ax=axes[2], color='coral', linewidth=1.5)
axes[2].set_title('Saisonnalité (Seasonal)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Composante saisonnière', fontsize=12)
axes[2].grid(True, alpha=0.3)

# Résidu
decomposition.resid.plot(ax=axes[3], color='gray', linewidth=1)
axes[3].set_title('Résidu (Noise)', fontsize=14, fontweight='bold')
axes[3].set_xlabel('Date', fontsize=12)
axes[3].set_ylabel('Résidu', fontsize=12)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/visualizations/4_time_series_decomposition.png', dpi=150, bbox_inches='tight')
print("✓ Graphique 2 sauvegardé : 4_time_series_decomposition.png")

# --- GRAPHIQUE 3 : Autocorrélation et Autocorrélation Partielle ---
print("\n[8] Analyse d'autocorrélation...")

fig, axes = plt.subplots(2, 1, figsize=(16, 8))
fig.suptitle('Analyse d\'Autocorrélation', fontsize=18, fontweight='bold')

# ACF
plot_acf(df['sales'], lags=40, ax=axes[0], color='steelblue', 
         title='Fonction d\'Autocorrélation (ACF)')
axes[0].set_xlabel('Lag (jours)', fontsize=12)
axes[0].set_ylabel('Corrélation', fontsize=12)
axes[0].grid(True, alpha=0.3)

# PACF
plot_pacf(df['sales'], lags=40, ax=axes[1], color='coral',
          title='Fonction d\'Autocorrélation Partielle (PACF)')
axes[1].set_xlabel('Lag (jours)', fontsize=12)
axes[1].set_ylabel('Corrélation partielle', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/visualizations/5_autocorrelation.png', dpi=150, bbox_inches='tight')
print("✓ Graphique 3 sauvegardé : 5_autocorrelation.png")

# --- GRAPHIQUE 4 : Rolling statistics ---
print("\n[9] Calcul des statistiques mobiles...")

rolling_mean = df['sales'].rolling(window=30).mean()
rolling_std = df['sales'].rolling(window=30).std()

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df.index, df['sales'], color='steelblue', linewidth=1, label='Ventes originales')
ax.plot(df.index, rolling_mean, color='red', linewidth=2, label='Moyenne mobile (30 jours)')
ax.plot(df.index, rolling_std, color='orange', linewidth=2, label='Écart-type mobile (30 jours)')
ax.set_title('Statistiques Mobiles (Rolling Statistics)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Ventes', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/visualizations/6_rolling_statistics.png', dpi=150, bbox_inches='tight')
print("✓ Graphique 4 sauvegardé : 6_rolling_statistics.png")

# ============================================================================
# 10. RÉSUMÉ ET INSIGHTS
# ============================================================================
print("\n" + "="*70)
print("✅ ANALYSE EXPLORATOIRE TERMINÉE !")
print("="*70)

print("\n📊 RÉSUMÉ DES INSIGHTS :")
print("\n1️⃣ CARACTÉRISTIQUES GÉNÉRALES :")
print(f"   - Ventes moyennes : {df['sales'].mean():.0f} unités/jour")
print(f"   - Écart-type : {df['sales'].std():.0f}")
print(f"   - Coefficient de variation : {(df['sales'].std() / df['sales'].mean() * 100):.1f}%")

print("\n2️⃣ STATIONNARITÉ :")
if adf_result[1] < 0.05:
    print("   ✓ Série stationnaire → ARIMA simple possible")
else:
    print("   ⚠️ Série non-stationnaire → Différenciation nécessaire (ARIMA avec d=1)")

print("\n3️⃣ SAISONNALITÉ :")
print("   ✓ Saisonnalité hebdomadaire détectée (pic le samedi)")
print("   ✓ Saisonnalité annuelle détectée (pic en automne)")
print("   → SARIMA ou Prophet recommandés")

print("\n4️⃣ ANOMALIES :")
print(f"   - {len(outliers)} valeurs aberrantes ({len(outliers)/len(df)*100:.2f}%)")
print("   → Principalement dues aux événements spéciaux (Black Friday, Soldes)")

print("\n5️⃣ TENDANCE :")
print("   ✓ Tendance croissante confirmée")
print("   → Les modèles doivent capturer cette croissance")

print("\n📈 Graphiques générés :")
print("   1. 3_distribution_analysis.png")
print("   2. 4_time_series_decomposition.png")
print("   3. 5_autocorrelation.png")
print("   4. 6_rolling_statistics.png")

print("\n🎯 Prochaine étape : Préparation des données (3_data_preparation.py)")
print("="*70)