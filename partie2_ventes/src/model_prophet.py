"""
Projet ENSA Berrechid 2025-2026
Partie 2 : Prédiction des ventes de vêtements
Script : Modèle Prophet

Évaluation de deux configurations Prophet :
1. Saisonnalité annuelle activée
2. Saisonnalité annuelle désactivée

Les deux configurations sont évaluées sur le même jeu de test
(146 observations) et comparées selon le RMSE.

La configuration retenue est ensuite utilisée pour produire :
- les prédictions du jeu de test ;
- les intervalles de prédiction à 95 % ;
- les métriques ;
- les visualisations ;
- le modèle Prophet d'évaluation.
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

from prophet import Prophet
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from scipy import stats

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# ============================================================================
# 0. CONFIGURATION DES CHEMINS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


print("=" * 70)
print("MODÈLE PROPHET - PRÉDICTION DES VENTES")
print("=" * 70)


# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================

print("\n[1] Chargement des données...")

train_file = os.path.join(DATA_DIR, "train_data.csv")
test_file = os.path.join(DATA_DIR, "test_data.csv")

train_data = pd.read_csv(
    train_file,
    parse_dates=["date"]
)

test_data = pd.read_csv(
    test_file,
    parse_dates=["date"]
)

train_data = train_data.sort_values("date").reset_index(drop=True)
test_data = test_data.sort_values("date").reset_index(drop=True)

required_columns = {"date", "sales"}

if not required_columns.issubset(train_data.columns):
    raise ValueError(
        "Le fichier train_data.csv doit contenir les colonnes 'date' et 'sales'."
    )

if not required_columns.issubset(test_data.columns):
    raise ValueError(
        "Le fichier test_data.csv doit contenir les colonnes 'date' et 'sales'."
    )

if train_data["sales"].isna().any() or test_data["sales"].isna().any():
    raise ValueError("Des valeurs manquantes sont présentes dans la colonne sales.")

print(f"✓ Train : {len(train_data)} jours")
print(f"✓ Test  : {len(test_data)} jours")

print(
    f"✓ Période train : "
    f"{train_data['date'].min().date()} → {train_data['date'].max().date()}"
)

print(
    f"✓ Période test  : "
    f"{test_data['date'].min().date()} → {test_data['date'].max().date()}"
)


# ============================================================================
# 2. PRÉPARATION DES DONNÉES POUR PROPHET
# ============================================================================

print("\n[2] Préparation des données pour Prophet...")

df_train = train_data[["date", "sales"]].copy()
df_train.columns = ["ds", "y"]

df_test = test_data[["date", "sales"]].copy()
df_test.columns = ["ds", "y"]

print("✓ Format Prophet appliqué : colonnes 'ds' et 'y'")


# ============================================================================
# 3. FONCTION DE CALCUL DU MAPE
# ============================================================================

def calculate_mape(y_true, y_pred):
    """
    Calcule le MAPE en ignorant les observations dont la valeur réelle est 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0

    if not np.any(mask):
        return np.nan

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100


# ============================================================================
# 4. FONCTION D'ÉVALUATION D'UNE CONFIGURATION PROPHET
# ============================================================================

def evaluate_prophet_configuration(yearly_seasonality):
    """
    Entraîne et évalue une configuration Prophet sur le même jeu de test.
    """

    label = (
        "Prophet_Annuel"
        if yearly_seasonality
        else "Prophet_Sans_Annuel"
    )

    print("\n" + "-" * 70)
    print(f"CONFIGURATION : {label}")
    print("-" * 70)

    print(f"  Saisonnalité annuelle     : {'Activée' if yearly_seasonality else 'Désactivée'}")
    print("  Saisonnalité hebdomadaire : Activée")
    print("  Saisonnalité quotidienne  : Désactivée")
    print("  Mode                      : Multiplicatif")
    print("  Changepoint prior scale   : 0.05")
    print("  Seasonality prior scale   : 10")
    print("  Intervalle de prédiction  : 95 %")

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        interval_width=0.95
    )

    print("\n  Entraînement en cours...")

    start_time = time.time()

    model.fit(df_train)

    training_time = time.time() - start_time

    print(f"  ✓ Entraînement terminé en {training_time:.2f} secondes")

    # ------------------------------------------------------------------------
    # Prédictions
    # ------------------------------------------------------------------------

    future = model.make_future_dataframe(
        periods=len(test_data),
        freq="D"
    )

    forecast = model.predict(future)

    test_forecast = forecast[
        forecast["ds"].isin(df_test["ds"])
    ].copy()

    test_forecast = test_forecast.sort_values("ds").reset_index(drop=True)

    if len(test_forecast) != len(df_test):
        raise ValueError(
            f"Nombre de prédictions incorrect : "
            f"{len(test_forecast)} au lieu de {len(df_test)}."
        )

    predictions = test_forecast["yhat"].to_numpy()
    actual = df_test["y"].to_numpy()

    # ------------------------------------------------------------------------
    # Métriques
    # ------------------------------------------------------------------------

    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    mape = calculate_mape(actual, predictions)
    r2 = r2_score(actual, predictions)

    residuals = actual - predictions

    print("\n  MÉTRIQUES")
    print(f"  MSE  : {mse:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}")

    return {
        "label": label,
        "yearly_seasonality": yearly_seasonality,
        "model": model,
        "forecast": forecast,
        "test_forecast": test_forecast,
        "predictions": predictions,
        "actual": actual,
        "residuals": residuals,
        "training_time": training_time,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2
    }


# ============================================================================
# 5. COMPARAISON DES DEUX CONFIGURATIONS
# ============================================================================

print("\n[3] Comparaison des configurations Prophet...")

results_annual = evaluate_prophet_configuration(
    yearly_seasonality=True
)

results_without_annual = evaluate_prophet_configuration(
    yearly_seasonality=False
)


# ============================================================================
# 6. SÉLECTION SELON LE RMSE
# ============================================================================

print("\n" + "=" * 70)
print("COMPARAISON DES CONFIGURATIONS PROPHET")
print("=" * 70)

comparison = pd.DataFrame([
    {
        "configuration": results_annual["label"],
        "yearly_seasonality": True,
        "MSE": results_annual["mse"],
        "RMSE": results_annual["rmse"],
        "MAE": results_annual["mae"],
        "MAPE (%)": results_annual["mape"],
        "R²": results_annual["r2"],
        "Temps (s)": results_annual["training_time"]
    },
    {
        "configuration": results_without_annual["label"],
        "yearly_seasonality": False,
        "MSE": results_without_annual["mse"],
        "RMSE": results_without_annual["rmse"],
        "MAE": results_without_annual["mae"],
        "MAPE (%)": results_without_annual["mape"],
        "R²": results_without_annual["r2"],
        "Temps (s)": results_without_annual["training_time"]
    }
])

print("\n")
print(
    comparison[
        [
            "configuration",
            "MSE",
            "RMSE",
            "MAE",
            "MAPE (%)",
            "R²",
            "Temps (s)"
        ]
    ].to_string(index=False)
)

# Sélection de la configuration selon le RMSE
best_idx = comparison["RMSE"].idxmin()

selected_row = comparison.loc[best_idx]

if selected_row["yearly_seasonality"]:
    selected_results = results_annual
else:
    selected_results = results_without_annual

selected_configuration = selected_results["label"]

print("\n" + "-" * 70)
print("CONFIGURATION RETENUE SELON LE RMSE")
print("-" * 70)

print(f"  Configuration : {selected_configuration}")
print(f"  RMSE          : {selected_results['rmse']:.2f}")
print(f"  MAE           : {selected_results['mae']:.2f}")
print(f"  MAPE          : {selected_results['mape']:.2f}%")
print(f"  R²            : {selected_results['r2']:.4f}")


# ============================================================================
# 7. SAUVEGARDE DE LA COMPARAISON DES CONFIGURATIONS
# ============================================================================

prophet_config_comparison = {
    "selection_criterion": "RMSE",
    "test_observations": int(len(test_data)),
    "configurations": [
        {
            "configuration": results_annual["label"],
            "yearly_seasonality": True,
            "mse": float(results_annual["mse"]),
            "rmse": float(results_annual["rmse"]),
            "mae": float(results_annual["mae"]),
            "mape": float(results_annual["mape"]),
            "r2": float(results_annual["r2"]),
            "training_time": float(results_annual["training_time"])
        },
        {
            "configuration": results_without_annual["label"],
            "yearly_seasonality": False,
            "mse": float(results_without_annual["mse"]),
            "rmse": float(results_without_annual["rmse"]),
            "mae": float(results_without_annual["mae"]),
            "mape": float(results_without_annual["mape"]),
            "r2": float(results_without_annual["r2"]),
            "training_time": float(results_without_annual["training_time"])
        }
    ],
    "selected_configuration": selected_configuration
}

config_comparison_file = os.path.join(
    OUTPUT_DIR,
    "prophet_configuration_comparison.json"
)

with open(config_comparison_file, "w", encoding="utf-8") as f:
    json.dump(
        prophet_config_comparison,
        f,
        indent=4,
        ensure_ascii=False
    )

comparison_csv_file = os.path.join(
    OUTPUT_DIR,
    "prophet_configuration_comparison.csv"
)

comparison.to_csv(
    comparison_csv_file,
    index=False
)

print(
    f"\n✓ Comparaison sauvegardée : "
    f"{config_comparison_file}"
)

print(
    f"✓ Tableau sauvegardé : "
    f"{comparison_csv_file}"
)


# ============================================================================
# 8. RÉCUPÉRATION DES RÉSULTATS DE LA CONFIGURATION RETENUE
# ============================================================================

model = selected_results["model"]
forecast = selected_results["forecast"]
test_forecast = selected_results["test_forecast"]

predictions = selected_results["predictions"]
actual = selected_results["actual"]
residuals = selected_results["residuals"]

training_time = selected_results["training_time"]

mse = selected_results["mse"]
rmse = selected_results["rmse"]
mae = selected_results["mae"]
mape = selected_results["mape"]
r2 = selected_results["r2"]


# ============================================================================
# 9. VISUALISATIONS
# ============================================================================

print("\n[4] Génération des visualisations...")


# ---------------------------------------------------------------------------
# GRAPHIQUE 10 : PRÉDICTIONS VS RÉALITÉ
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

fig.suptitle(
    f"Modèle Prophet - Résultats ({selected_configuration})",
    fontsize=18,
    fontweight="bold"
)

# Vue d'ensemble
axes[0].plot(
    df_train["ds"],
    df_train["y"],
    color="steelblue",
    linewidth=1,
    label="Train",
    alpha=0.7
)

axes[0].plot(
    df_test["ds"],
    df_test["y"],
    color="black",
    linewidth=2,
    label="Valeurs réelles",
    marker="o",
    markersize=3
)

axes[0].plot(
    test_forecast["ds"],
    test_forecast["yhat"],
    color="red",
    linewidth=2,
    label="Prédictions Prophet",
    linestyle="--"
)

axes[0].fill_between(
    test_forecast["ds"],
    test_forecast["yhat_lower"],
    test_forecast["yhat_upper"],
    color="red",
    alpha=0.2,
    label="Intervalle de prédiction 95 %"
)

axes[0].axvline(
    x=df_test["ds"].iloc[0],
    color="green",
    linestyle="--",
    linewidth=2,
    label="Début du test"
)

axes[0].set_title(
    f"Prédictions vs Réalité - RMSE : {rmse:.2f}, MAPE : {mape:.2f}%",
    fontsize=14,
    fontweight="bold"
)

axes[0].set_xlabel("Date", fontsize=12)
axes[0].set_ylabel("Ventes", fontsize=12)
axes[0].legend(fontsize=10, loc="upper left")
axes[0].grid(True, alpha=0.3)


# Zoom test
axes[1].plot(
    df_test["ds"],
    df_test["y"],
    color="black",
    linewidth=2,
    label="Valeurs réelles",
    marker="o",
    markersize=4
)

axes[1].plot(
    test_forecast["ds"],
    test_forecast["yhat"],
    color="red",
    linewidth=2,
    label="Prédictions",
    marker="s",
    markersize=4
)

axes[1].fill_between(
    test_forecast["ds"],
    test_forecast["yhat_lower"],
    test_forecast["yhat_upper"],
    color="red",
    alpha=0.2,
    label="Intervalle 95 %"
)

axes[1].set_title(
    "Zoom sur la période de test",
    fontsize=14,
    fontweight="bold"
)

axes[1].set_xlabel("Date", fontsize=12)
axes[1].set_ylabel("Ventes", fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plot_10 = os.path.join(
    VISUALIZATIONS_DIR,
    "10_prophet_predictions.png"
)

plt.savefig(
    plot_10,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("✓ Graphique 10 sauvegardé : 10_prophet_predictions.png")


# ---------------------------------------------------------------------------
# GRAPHIQUE 11 : COMPOSANTES PROPHET
# ---------------------------------------------------------------------------

fig = model.plot_components(
    forecast,
    figsize=(16, 12)
)

fig.suptitle(
    "Décomposition Prophet : Tendance et Saisonnalités",
    fontsize=18,
    fontweight="bold",
    y=1.00
)

plt.tight_layout()

plot_11 = os.path.join(
    VISUALIZATIONS_DIR,
    "11_prophet_components.png"
)

plt.savefig(
    plot_11,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("✓ Graphique 11 sauvegardé : 11_prophet_components.png")


# ---------------------------------------------------------------------------
# GRAPHIQUE 12 : ANALYSE DES ERREURS DE PRÉVISION
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

fig.suptitle(
    "Analyse des erreurs de prévision - Prophet",
    fontsize=18,
    fontweight="bold"
)


# Erreurs temporelles
axes[0, 0].plot(
    df_test["ds"],
    residuals,
    color="green",
    linewidth=1.5
)

axes[0, 0].axhline(
    y=0,
    color="red",
    linestyle="--",
    linewidth=2
)

axes[0, 0].fill_between(
    df_test["ds"],
    0,
    residuals,
    alpha=0.3,
    color="green"
)

axes[0, 0].set_title(
    "Erreurs de prévision au fil du temps",
    fontsize=13,
    fontweight="bold"
)

axes[0, 0].set_xlabel("Date", fontsize=11)
axes[0, 0].set_ylabel("Erreur (réel - prédit)", fontsize=11)
axes[0, 0].grid(True, alpha=0.3)


# Distribution
axes[0, 1].hist(
    residuals,
    bins=30,
    color="green",
    edgecolor="black",
    alpha=0.7,
    density=True
)

mu = residuals.mean()
std = residuals.std()

if std > 0:
    x = np.linspace(
        residuals.min(),
        residuals.max(),
        100
    )

    axes[0, 1].plot(
        x,
        stats.norm.pdf(x, mu, std),
        "r-",
        linewidth=2,
        label="Loi normale"
    )

axes[0, 1].axvline(
    mu,
    color="darkgreen",
    linestyle="--",
    linewidth=2,
    label=f"Moyenne : {mu:.2f}"
)

axes[0, 1].set_title(
    "Distribution des erreurs",
    fontsize=13,
    fontweight="bold"
)

axes[0, 1].set_xlabel(
    "Erreur (réel - prédit)",
    fontsize=11
)

axes[0, 1].set_ylabel(
    "Densité",
    fontsize=11
)

axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)


# Q-Q plot
stats.probplot(
    residuals,
    dist="norm",
    plot=axes[1, 0]
)

axes[1, 0].set_title(
    "Q-Q Plot des erreurs",
    fontsize=13,
    fontweight="bold"
)

axes[1, 0].grid(True, alpha=0.3)


# Prédictions vs réalité
axes[1, 1].scatter(
    df_test["y"],
    predictions,
    alpha=0.6,
    color="green",
    s=50
)

min_val = min(
    df_test["y"].min(),
    predictions.min()
)

max_val = max(
    df_test["y"].max(),
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
    f"Prédictions vs Réalité (R² = {r2:.3f})",
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

axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

plot_12 = os.path.join(
    VISUALIZATIONS_DIR,
    "12_prophet_residuals.png"
)

plt.savefig(
    plot_12,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("✓ Graphique 12 sauvegardé : 12_prophet_residuals.png")


# ============================================================================
# 10. SAUVEGARDE DU MODÈLE D'ÉVALUATION
# ============================================================================

print("\n[5] Sauvegarde du modèle et des résultats...")


model_file = os.path.join(
    MODELS_DIR,
    "prophet_model.pkl"
)

with open(model_file, "wb") as f:
    pickle.dump(model, f)

print("✓ Modèle sauvegardé : prophet_model.pkl")


# ============================================================================
# 11. SAUVEGARDE DES PRÉDICTIONS
# ============================================================================

results_df = pd.DataFrame({
    "date": df_test["ds"],
    "real_sales": actual,
    "predicted_sales": predictions,
    "residuals": residuals,
    "lower_bound": test_forecast["yhat_lower"].values,
    "upper_bound": test_forecast["yhat_upper"].values
})

predictions_file = os.path.join(
    OUTPUT_DIR,
    "prophet_predictions.csv"
)

results_df.to_csv(
    predictions_file,
    index=False
)

print("✓ Prédictions sauvegardées : prophet_predictions.csv")


# ============================================================================
# 12. SAUVEGARDE DES MÉTRIQUES
# ============================================================================

metrics = {
    "model": "Prophet",
    "selected_configuration": selected_configuration,
    "yearly_seasonality": bool(
        selected_results["yearly_seasonality"]
    ),
    "training_time": float(training_time),
    "mse": float(mse),
    "rmse": float(rmse),
    "mae": float(mae),
    "mape": float(mape),
    "r2": float(r2),
    "test_observations": int(len(test_data)),
    "selection_criterion": "RMSE",
    "prediction_interval": 0.95
}

metrics_file = os.path.join(
    OUTPUT_DIR,
    "prophet_metrics.json"
)

with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=4,
        ensure_ascii=False
    )

print("✓ Métriques sauvegardées : prophet_metrics.json")


# ============================================================================
# 13. RÉSUMÉ FINAL
# ============================================================================

print("\n" + "=" * 70)
print("MODÈLE PROPHET TERMINÉ")
print("=" * 70)

print("\nConfiguration retenue selon le RMSE :")
print(f"  • {selected_configuration}")

print("\nPerformances sur le jeu de test :")
print(f"  • Observations : {len(test_data)}")
print(f"  • RMSE         : {rmse:.2f}")
print(f"  • MAE          : {mae:.2f}")
print(f"  • MAPE         : {mape:.2f}%")
print(f"  • R²           : {r2:.4f}")

print("\nFichiers générés :")
print("  1. prophet_model.pkl")
print("  2. prophet_predictions.csv")
print("  3. prophet_metrics.json")
print("  4. prophet_configuration_comparison.json")
print("  5. prophet_configuration_comparison.csv")
print("  6. 10_prophet_predictions.png")
print("  7. 11_prophet_components.png")
print("  8. 12_prophet_residuals.png")

print("\n✓ Évaluation Prophet terminée.")
print("=" * 70)
