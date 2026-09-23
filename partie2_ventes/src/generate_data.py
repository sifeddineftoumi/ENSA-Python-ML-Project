"""
Projet ENSA Berrechid 2025-2026
Partie 2 : Prédiction des ventes de vêtements
Script 1 : Génération de données synthétiques réalistes

Ce script génère un dataset de ventes de vêtements avec :
- Tendance croissante (croissance du business)
- Saisonnalité hebdomadaire (weekend vs semaine)
- Saisonnalité annuelle (saisons, fêtes)
- Bruit aléatoire réaliste
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Configuration
np.random.seed(42)  # Pour la reproductibilité
sns.set_style('whitegrid')

print("="*70)
print("GÉNÉRATION DE DONNÉES DE VENTES DE VÊTEMENTS")
print("="*70)

# ============================================================================
# 1. PARAMÈTRES DU DATASET
# ============================================================================
print("\n[1] Configuration des paramètres...")

# Période : 2 ans de données quotidiennes
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
n_days = len(date_range)

print(f"✓ Période : {start_date.date()} à {end_date.date()}")
print(f"✓ Nombre de jours : {n_days}")

# ============================================================================
# 2. COMPOSANTES DE LA SÉRIE TEMPORELLE
# ============================================================================
print("\n[2] Génération des composantes...")

# --- TENDANCE : Croissance linéaire du business ---
# Les ventes augmentent progressivement (expansion du magasin)
base_sales = 100  # Ventes de base par jour
growth_rate = 0.15  # Croissance de 15% sur 2 ans
trend = np.linspace(base_sales, base_sales * (1 + growth_rate), n_days)

# --- SAISONNALITÉ ANNUELLE : Impact des saisons ---
# Hiver (Déc-Fév) : +30% (manteaux, pulls)
# Printemps (Mar-Mai) : normal
# Été (Jui-Aoû) : -20% (vêtements légers, moins de ventes)
# Automne (Sep-Nov) : +40% (rentrée, nouvelle collection)

def seasonal_factor(date):
    """Facteur saisonnier basé sur le mois"""
    month = date.month
    
    # Hiver (Décembre, Janvier, Février)
    if month in [12, 1, 2]:
        return 1.30
    # Printemps (Mars, Avril, Mai)
    elif month in [3, 4, 5]:
        return 1.00
    # Été (Juin, Juillet, Août)
    elif month in [6, 7, 8]:
        return 0.80
    # Automne (Septembre, Octobre, Novembre)
    else:
        return 1.40

seasonal_component = np.array([seasonal_factor(date) for date in date_range])

# --- SAISONNALITÉ HEBDOMADAIRE : Weekend vs Semaine ---
# Lundi-Jeudi : normal
# Vendredi : +20% (début du weekend)
# Samedi : +50% (pic de ventes)
# Dimanche : +30% (shopping familial)

def weekly_factor(date):
    """Facteur hebdomadaire basé sur le jour de la semaine"""
    day = date.weekday()  # 0=Lundi, 6=Dimanche
    
    if day in [0, 1, 2, 3]:  # Lun-Jeu
        return 1.00
    elif day == 4:  # Vendredi
        return 1.20
    elif day == 5:  # Samedi
        return 1.50
    else:  # Dimanche
        return 1.30

weekly_component = np.array([weekly_factor(date) for date in date_range])

# --- ÉVÉNEMENTS SPÉCIAUX : Promotions et fêtes ---
# Black Friday, Soldes d'hiver/été, Rentrée scolaire
events = np.ones(n_days)

for i, date in enumerate(date_range):
    # Black Friday (dernier vendredi de novembre)
    if date.month == 11 and date.day >= 23 and date.day <= 29 and date.weekday() == 4:
        events[i] = 2.50  # +150% de ventes
    
    # Soldes d'hiver (janvier)
    elif date.month == 1 and date.day <= 31:
        events[i] = 1.80
    
    # Soldes d'été (juillet)
    elif date.month == 7 and date.day >= 1 and date.day <= 31:
        events[i] = 1.70
    
    # Rentrée scolaire (septembre)
    elif date.month == 9 and date.day >= 1 and date.day <= 15:
        events[i] = 1.60

# --- BRUIT ALÉATOIRE : Variabilité naturelle ---
# Fluctuations quotidiennes dues à la météo, événements locaux, etc.
noise = np.random.normal(0, 8, n_days)  # Écart-type de 8 unités

# ============================================================================
# 3. COMBINAISON DES COMPOSANTES
# ============================================================================
print("\n[3] Combinaison des composantes...")

# Formule : Ventes = Tendance × Saisonnalité × Événements + Bruit
sales = trend * seasonal_component * weekly_component * events + noise

# S'assurer que les ventes sont positives
sales = np.maximum(sales, 10)

# Arrondir à l'unité (nombre entier de vêtements vendus)
sales = np.round(sales).astype(int)

print(f"✓ Ventes moyennes : {sales.mean():.0f} unités/jour")
print(f"✓ Ventes min : {sales.min()} unités")
print(f"✓ Ventes max : {sales.max()} unités")

# ============================================================================
# 4. CRÉATION DU DATAFRAME
# ============================================================================
print("\n[4] Création du DataFrame...")

df = pd.DataFrame({
    'date': date_range,
    'sales': sales,
    'day_of_week': [date.strftime('%A') for date in date_range],
    'month': [date.month for date in date_range],
    'year': [date.year for date in date_range],
    'is_weekend': [1 if date.weekday() >= 5 else 0 for date in date_range],
    'season': [
        'Winter' if m in [12, 1, 2]
        else 'Spring' if m in [3, 4, 5]
        else 'Summer' if m in [6, 7, 8]
        else 'Autumn'
        for m in [date.month for date in date_range]
    ]
})

print(f"✓ DataFrame créé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Aperçu du dataset
print("\n" + "="*70)
print("APERÇU DES DONNÉES")
print("="*70)
print(df.head(10))

print("\n" + "="*70)
print("STATISTIQUES DESCRIPTIVES")
print("="*70)
print(df['sales'].describe())

# ============================================================================
# 5. SAUVEGARDE DU DATASET
# ============================================================================
print("\n[5] Sauvegarde du dataset...")

os.makedirs('../data', exist_ok=True)
df.to_csv('../data/sales_data.csv', index=False)

print("✓ Dataset sauvegardé : ../data/sales_data.csv")

# ============================================================================
# 6. VISUALISATIONS
# ============================================================================
print("\n[6] Génération des visualisations...")

os.makedirs('../outputs/visualizations', exist_ok=True)

# --- Graphique 1 : Série temporelle complète ---
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle('Analyse des Ventes de Vêtements (2022-2023)', 
             fontsize=18, fontweight='bold', y=0.995)

# Série complète
axes[0].plot(df['date'], df['sales'], linewidth=1.5, color='steelblue')
axes[0].set_title('Ventes Quotidiennes - Vue d\'ensemble', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Nombre de vêtements vendus', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Zoom sur 3 mois
zoom_start = df[df['date'] == '2023-09-01'].index[0]
zoom_end = df[df['date'] == '2023-11-30'].index[0]
axes[1].plot(df['date'][zoom_start:zoom_end], 
             df['sales'][zoom_start:zoom_end], 
             linewidth=2, color='darkgreen', marker='o', markersize=3)
axes[1].set_title('Zoom : Septembre - Novembre 2023 (Période de rentrée)', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Ventes', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Distribution des ventes
axes[2].hist(df['sales'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[2].axvline(df['sales'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Moyenne: {df["sales"].mean():.0f}')
axes[2].set_title('Distribution des Ventes', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Nombre de ventes', fontsize=12)
axes[2].set_ylabel('Fréquence', fontsize=12)
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../outputs/visualizations/1_sales_overview.png', dpi=150, bbox_inches='tight')
print("✓ Graphique 1 sauvegardé : 1_sales_overview.png")

# --- Graphique 2 : Patterns hebdomadaires et mensuels ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Patterns de Ventes', fontsize=18, fontweight='bold')

# Ventes par jour de la semaine
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_sales = df.groupby('day_of_week')['sales'].mean().reindex(day_order)
axes[0, 0].bar(range(7), weekly_sales.values, color='skyblue', edgecolor='black')
axes[0, 0].set_xticks(range(7))
axes[0, 0].set_xticklabels(['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'], fontsize=11)
axes[0, 0].set_title('Ventes Moyennes par Jour de la Semaine', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Ventes moyennes', fontsize=11)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Ventes par mois
monthly_sales = df.groupby('month')['sales'].mean()
month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jui', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
axes[0, 1].bar(range(1, 13), monthly_sales.values, color='lightcoral', edgecolor='black')
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].set_xticklabels(month_names, fontsize=10)
axes[0, 1].set_title('Ventes Moyennes par Mois', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('Ventes moyennes', fontsize=11)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Ventes par saison
season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
season_sales = df.groupby('season')['sales'].mean().reindex(season_order)
colors = ['#ADD8E6', '#90EE90', '#FFD700', '#FF8C00']
axes[1, 0].bar(range(4), season_sales.values, color=colors, edgecolor='black')
axes[1, 0].set_xticks(range(4))
axes[1, 0].set_xticklabels(['Hiver', 'Printemps', 'Été', 'Automne'], fontsize=11)
axes[1, 0].set_title('Ventes Moyennes par Saison', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Ventes moyennes', fontsize=11)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Weekend vs Semaine
weekend_comparison = df.groupby('is_weekend')['sales'].mean()
axes[1, 1].bar(['Semaine', 'Weekend'], weekend_comparison.values, 
               color=['#4682B4', '#FF6347'], edgecolor='black')
axes[1, 1].set_title('Ventes : Semaine vs Weekend', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Ventes moyennes', fontsize=11)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../outputs/visualizations/2_sales_patterns.png', dpi=150, bbox_inches='tight')
print("✓ Graphique 2 sauvegardé : 2_sales_patterns.png")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "="*70)
print("✅ GÉNÉRATION DE DONNÉES TERMINÉE AVEC SUCCÈS !")
print("="*70)
print("\n📊 Dataset créé :")
print(f"  - Fichier : sales_data.csv")
print(f"  - Période : 2 ans (730 jours)")
print(f"  - Ventes moyennes : {df['sales'].mean():.0f} unités/jour")
print(f"  - Écart-type : {df['sales'].std():.0f}")
print("\n📈 Caractéristiques :")
print("  ✓ Tendance croissante (+15% sur 2 ans)")
print("  ✓ Saisonnalité annuelle (automne = pic)")
print("  ✓ Saisonnalité hebdomadaire (samedi = pic)")
print("  ✓ Événements spéciaux (Black Friday, Soldes)")
print("\n🎯 Prochaine étape : Analyse exploratoire (2_eda_analysis.py)")
print("="*70)