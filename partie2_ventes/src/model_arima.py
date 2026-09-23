"""
Projet ENSA Berrechid 2025-2026
Partie 2 : Prédiction des ventes de vêtements
Script 4 : Modèle ARIMA/SARIMA

Ce script implémente et évalue un modèle SARIMA
(Seasonal AutoRegressive Integrated Moving Average)
sur un ensemble d'entraînement et un ensemble de test chronologiques.
"""

import os
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# ============================================================================
# CONFIGURATION DES CHEMINS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


print("=" * 70)
print("MODÈLE SARIMA - PRÉDICTION DES VENTES")
print("=" * 70)


# ============================================================================
# FONCTION MAPE
# ============================================================================

def calculate_mape(y_true, y_pred):
    """
    Calcule le MAPE en excluant les observations dont la valeur réelle
    est égale à zéro afin d'éviter une division par zéro.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0

    if not np.any(mask):
        return np.nan

    return np.mean(
        np.abs(
            (y_true[mask] - y_pred[mask]) / y_true[mask]
        )
    ) * 100


# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================

print("\n[1] Chargement des données...")

train_path = os.path.join(DATA_DIR, "train_data.csv")
test_path = os.path.join(DATA_DIR, "test_data.csv")

train_data = pd.read_csv(
    train_path,
    parse_dates=["date"],
    index_col="date"
)

test_data = pd.read_csv(
    test_path,
    parse_dates=["date"],
    index_col="date"
)

train_data = train_data.sort_index()
test_data = test_data.sort_index()

# Vérification des colonnes
if "sales" not in train_data.columns:
    raise ValueError("La colonne 'sales' est absente de train_data.csv.")

if "sales" not in test_data.columns:
    raise ValueError("La colonne 'sales' est absente de test_data.csv.")

# Vérification des valeurs manquantes
if train_data["sales"].isnull().any():
    raise ValueError("Des valeurs manquantes sont présentes dans le train set.")

if test_data["sales"].isnull().any():
    raise ValueError("Des valeurs manquantes sont présentes dans le test set.")

print(f"✓ Train : {len(train_data)} jours")
print(f"✓ Test  : {len(test_data)} jours")

print(
    f"✓ Période train : "
    f"{train_data.index.min().date()} → {train_data.index.max().date()}"
)

print(
    f"✓ Période test  : "
    f"{test_data.index.min().date()} → {test_data.index.max().date()}"
)


# ============================================================================
# 2. CONFIGURATION DU MODÈLE SARIMA
# ============================================================================

print("\n[2] Configuration du modèle SARIMA...")

# Paramètres SARIMA :
# (p, d, q) x (P, D, Q, s)
#
# p, d, q : composantes non saisonnières
# P, D, Q : composantes saisonnières
# s       : période saisonnière
#
# Ici, s=7 correspond à une saisonnalité hebdomadaire.

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 7)

print("\n📊 Paramètres du modèle :")
print(f"  - Order (p,d,q) : {order}")
print(f"  - Seasonal Order (P,D,Q,s) : {seasonal_order}")

print("\n  Interprétation :")
print("    • p=1 : une composante autorégressive")
print("    • d=1 : une différenciation non saisonnière")
print("    • q=1 : une composante de moyenne mobile")
print("    • P=1 : une composante autorégressive saisonnière")
print("    • D=1 : une différenciation saisonnière")
print("    • Q=1 : une composante de moyenne mobile saisonnière")
print("    • s=7 : périodicité hebdomadaire")

print(
    "\n  Remarque : cette configuration est utilisée pour "
    "l'évaluation du modèle SARIMA."
)


# ============================================================================
# 3. ENTRAÎNEMENT DU MODÈLE
# ============================================================================

print("\n[3] Entraînement du modèle SARIMA...")
print("⏳ Cela peut prendre quelques instants...")

start_time = time.time()

model = SARIMAX(
    train_data["sales"],
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)

training_time = time.time() - start_time

print(
    f"✓ Modèle entraîné en {training_time:.2f} secondes"
)


# ============================================================================
# 4. PRÉDICTIONS SUR L'ENSEMBLE DE TEST
# ============================================================================

print("\n[4] Génération des prédictions sur l'ensemble de test...")

# Utilisation de get_forecast() afin d'obtenir à la fois :
# - les prédictions ponctuelles
# - les intervalles de prévision natifs du modèle SARIMA

forecast_result = model_fit.get_forecast(
    steps=len(test_data)
)

forecast_mean = forecast_result.predicted_mean
confidence_intervals = forecast_result.conf_int(alpha=0.05)

# Réindexation explicite sur les dates du test
predictions = pd.Series(
    forecast_mean.values,
    index=test_data.index,
    name="predicted_sales"
)

lower_bound = pd.Series(
    confidence_intervals.iloc[:, 0].values,
    index=test_data.index,
    name="lower_bound"
)

upper_bound = pd.Series(
    confidence_intervals.iloc[:, 1].values,
    index=test_data.index,
    name="upper_bound"
)

print(f"✓ {len(predictions)} prédictions générées")
print("✓ Intervalles de prévision à 95 % calculés")


# ============================================================================
# 5. ÉVALUATION DES PERFORMANCES
# ============================================================================

print("\n[5] Évaluation des performances...")

actual_values = test_data["sales"]

mse = mean_squared_error(
    actual_values,
    predictions
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    actual_values,
    predictions
)

mape = calculate_mape(
    actual_values,
    predictions
)

r2 = r2_score(
    actual_values,
    predictions
)

print("\n" + "=" * 70)
print("MÉTRIQUES DE PERFORMANCE - SARIMA")
print("=" * 70)

print(f"  MSE  (Mean Squared Error)      : {mse:.2f}")
print(f"  RMSE (Root Mean Squared Error) : {rmse:.2f}")
print(f"  MAE  (Mean Absolute Error)     : {mae:.2f}")
print(f"  MAPE (Mean Absolute Percentage) : {mape:.2f}%")
print(f"  R²   (Coefficient de détermination) : {r2:.4f}")


# ============================================================================
# 6. INTERPRÉTATION DES ERREURS
# ============================================================================

print("\n📊 Interprétation des performances :")

print(
    f"  • L'erreur absolue moyenne est de "
    f"{mae:.0f} unités de vente."
)

print(
    f"  • L'erreur absolue en pourcentage moyenne "
    f"(MAPE) est de {mape:.2f}%."
)

print(
    f"  • Le coefficient R² obtenu sur l'ensemble de test "
    f"est de {r2:.4f}."
)

if r2 >= 0:
    print(
        "  • Le modèle explique une partie de la variabilité "
        "observée sur l'ensemble de test."
    )
else:
    print(
        "  • Le R² négatif indique que les prédictions du modèle "
        "sont moins proches des observations que la référence "
        "constituée par la moyenne des valeurs du test."
    )


# ============================================================================
# 7. ANALYSE DES ERREURS DE PRÉVISION
# ============================================================================

print("\n[6] Analyse des erreurs de prévision...")

# Important :
# real_sales - predicted_sales correspond aux erreurs de prévision
# sur l'ensemble de test.
forecast_errors = actual_values - predictions

print("\n📊 Statistiques des erreurs de prévision :")

print(
    f"  - Moyenne      : {forecast_errors.mean():.2f}"
)

print(
    f"  - Écart-type   : {forecast_errors.std():.2f}"
)

print(
    f"  - Minimum      : {forecast_errors.min():.2f}"
)

print(
    f"  - Maximum      : {forecast_errors.max():.2f}"
)

print(
    "\n  Remarque : ces valeurs correspondent aux erreurs "
    "de prévision sur le test set et non aux résidus internes "
    "utilisés lors de l'estimation du modèle."
)


# ============================================================================
# 8. VISUALISATIONS
# ============================================================================

print("\n[7] Génération des visualisations...")


# ----------------------------------------------------------------------------
# GRAPHIQUE 1 : PRÉDICTIONS VS RÉALITÉ
# ----------------------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(16, 10)
)

fig.suptitle(
    "Modèle SARIMA - Résultats",
    fontsize=18,
    fontweight="bold"
)


# Vue d'ensemble
axes[0].plot(
    train_data.index,
    train_data["sales"],
    color="steelblue",
    linewidth=1,
    label="Train",
    alpha=0.7
)

axes[0].plot(
    test_data.index,
    test_data["sales"],
    color="black",
    linewidth=2,
    label="Valeurs réelles",
    marker="o",
    markersize=3
)

axes[0].plot(
    test_data.index,
    predictions,
    color="red",
    linewidth=2,
    label="Prédictions SARIMA",
    linestyle="--"
)

# Vrai intervalle de prévision SARIMA à 95 %
axes[0].fill_between(
    test_data.index,
    lower_bound,
    upper_bound,
    color="red",
    alpha=0.2,
    label="Intervalle de prévision 95 %"
)

axes[0].axvline(
    x=test_data.index[0],
    color="green",
    linestyle="--",
    linewidth=2,
    label="Début du test"
)

axes[0].set_title(
    f"Prédictions vs réalité - RMSE : {rmse:.2f}, MAPE : {mape:.2f}%",
    fontsize=14,
    fontweight="bold"
)

axes[0].set_xlabel("Date", fontsize=12)
axes[0].set_ylabel("Ventes", fontsize=12)
axes[0].legend(
    fontsize=10,
    loc="upper left"
)
axes[0].grid(
    True,
    alpha=0.3
)


# Zoom sur le test set
axes[1].plot(
    test_data.index,
    test_data["sales"],
    color="black",
    linewidth=2,
    label="Valeurs réelles",
    marker="o",
    markersize=4
)

axes[1].plot(
    test_data.index,
    predictions,
    color="red",
    linewidth=2,
    label="Prédictions SARIMA",
    linestyle="--",
    marker="s",
    markersize=3
)

axes[1].fill_between(
    test_data.index,
    lower_bound,
    upper_bound,
    color="red",
    alpha=0.2,
    label="Intervalle de prévision 95 %"
)

axes[1].set_title(
    "Zoom sur la période de test",
    fontsize=14,
    fontweight="bold"
)

axes[1].set_xlabel(
    "Date",
    fontsize=12
)

axes[1].set_ylabel(
    "Ventes",
    fontsize=12
)

axes[1].legend(
    fontsize=11
)

axes[1].grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATIONS_DIR,
        "8_sarima_predictions.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "✓ Graphique 1 sauvegardé : "
    "8_sarima_predictions.png"
)


# ----------------------------------------------------------------------------
# GRAPHIQUE 2 : ANALYSE DES ERREURS DE PRÉVISION
# ----------------------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 10)
)

fig.suptitle(
    "Analyse des erreurs de prévision - SARIMA",
    fontsize=18,
    fontweight="bold"
)


# Erreurs dans le temps
axes[0, 0].plot(
    test_data.index,
    forecast_errors,
    color="purple",
    linewidth=1.5
)

axes[0, 0].axhline(
    y=0,
    color="red",
    linestyle="--",
    linewidth=2
)

axes[0, 0].fill_between(
    test_data.index,
    0,
    forecast_errors,
    alpha=0.3,
    color="purple"
)

axes[0, 0].set_title(
    "Erreurs de prévision au fil du temps",
    fontsize=13,
    fontweight="bold"
)

axes[0, 0].set_xlabel(
    "Date",
    fontsize=11
)

axes[0, 0].set_ylabel(
    "Erreur",
    fontsize=11
)

axes[0, 0].grid(
    True,
    alpha=0.3
)


# Distribution des erreurs
axes[0, 1].hist(
    forecast_errors,
    bins=30,
    color="purple",
    edgecolor="black",
    alpha=0.7,
    density=True
)

mu = forecast_errors.mean()
std = forecast_errors.std()

if std > 0:
    x = np.linspace(
        forecast_errors.min(),
        forecast_errors.max(),
        100
    )

    axes[0, 1].plot(
        x,
        stats.norm.pdf(x, mu, std),
        "r-",
        linewidth=2,
        label="Loi normale théorique"
    )

axes[0, 1].axvline(
    mu,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Moyenne : {mu:.2f}"
)

axes[0, 1].set_title(
    "Distribution des erreurs de prévision",
    fontsize=13,
    fontweight="bold"
)

axes[0, 1].set_xlabel(
    "Erreur",
    fontsize=11
)

axes[0, 1].set_ylabel(
    "Densité",
    fontsize=11
)

axes[0, 1].legend(
    fontsize=10
)

axes[0, 1].grid(
    True,
    alpha=0.3
)


# Q-Q plot
stats.probplot(
    forecast_errors,
    dist="norm",
    plot=axes[1, 0]
)

axes[1, 0].set_title(
    "Q-Q Plot des erreurs de prévision",
    fontsize=13,
    fontweight="bold"
)

axes[1, 0].grid(
    True,
    alpha=0.3
)


# Prédictions vs réalité
axes[1, 1].scatter(
    actual_values,
    predictions,
    alpha=0.6,
    color="steelblue",
    s=50
)

min_val = min(
    actual_values.min(),
    predictions.min()
)

max_val = max(
    actual_values.max(),
    predictions.max()
)

axes[1, 1].plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    linewidth=2,
    label="Référence y = x"
)

axes[1, 1].set_title(
    f"Prédictions vs réalité (R² = {r2:.3f})",
    fontsize=13,
    fontweight="bold"
)

axes[1, 1].set_xlabel(
    "Ventes réelles",
    fontsize=11
)

axes[1, 1].set_ylabel(
    "Ventes prédites",
    fontsize=11
)

axes[1, 1].legend(
    fontsize=10
)

axes[1, 1].grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATIONS_DIR,
        "9_sarima_residuals.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "✓ Graphique 2 sauvegardé : "
    "9_sarima_residuals.png"
)


# ============================================================================
# 9. SAUVEGARDE DU MODÈLE ET DES RÉSULTATS
# ============================================================================

print("\n[8] Sauvegarde du modèle et des résultats...")


# Sauvegarde du modèle entraîné sur le train set
model_path = os.path.join(
    MODELS_DIR,
    "sarima_model.pkl"
)

with open(model_path, "wb") as f:
    pickle.dump(model_fit, f)

print(
    "✓ Modèle sauvegardé : "
    "sarima_model.pkl"
)


# Sauvegarde des prédictions
results_df = pd.DataFrame({
    "date": test_data.index,
    "real_sales": actual_values.values,
    "predicted_sales": predictions.values,
    "lower_bound": lower_bound.values,
    "upper_bound": upper_bound.values,
    "residuals": forecast_errors.values
})

predictions_path = os.path.join(
    OUTPUT_DIR,
    "sarima_predictions.csv"
)

results_df.to_csv(
    predictions_path,
    index=False
)

print(
    "✓ Prédictions sauvegardées : "
    "sarima_predictions.csv"
)


# ============================================================================
# 10. SAUVEGARDE DES MÉTRIQUES
# ============================================================================

metrics = {
    "model": "SARIMA",
    "order": str(order),
    "seasonal_order": str(seasonal_order),
    "training_time": float(training_time),
    "test_observations": int(len(test_data)),
    "mse": float(mse),
    "rmse": float(rmse),
    "mae": float(mae),
    "mape": float(mape),
    "r2": float(r2),
    "prediction_interval": "95%",
    "mape_zero_values_excluded": True
}

metrics_path = os.path.join(
    OUTPUT_DIR,
    "sarima_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print(
    "✓ Métriques sauvegardées : "
    "sarima_metrics.json"
)


# ============================================================================
# 11. RÉSUMÉ FINAL
# ============================================================================

print("\n" + "=" * 70)
print("MODÈLE SARIMA TERMINÉ")
print("=" * 70)

print("\n📊 Résumé des performances :")

print(
    f"  • RMSE : {rmse:.2f}"
)

print(
    f"  • MAE  : {mae:.2f}"
)

print(
    f"  • MAPE : {mape:.2f}%"
)

print(
    f"  • R²   : {r2:.4f}"
)

print(
    f"  • Observations de test : {len(test_data)}"
)

print(
    "\n📈 Fichiers générés :"
)

print(
    "  1. sarima_model.pkl - Modèle entraîné"
)

print(
    "  2. sarima_predictions.csv - Prédictions et intervalles"
)

print(
    "  3. sarima_metrics.json - Métriques"
)

print(
    "  4. 8_sarima_predictions.png - Prédictions et intervalle"
)

print(
    "  5. 9_sarima_residuals.png - Analyse des erreurs"
)

print(
    "\n🎯 Prochaine étape : Modèle Prophet (model_prophet.py)"
)

print("=" * 70)
