"""
Projet ENSA Berrechid 2025-2026
Partie 2 : Prédiction des ventes de vêtements
Script 7 : Comparaison des modèles SARIMA, Prophet et LSTM

Ce script compare les performances des trois modèles sur le même
ensemble de test et identifie le modèle retenu selon le critère RMSE.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

print("=" * 70)
print("COMPARAISON DES MODÈLES DE PRÉDICTION")
print("=" * 70)

# ============================================================================
# 1. CHARGEMENT DES MÉTRIQUES
# ============================================================================
print("\n[1] Chargement des métriques de chaque modèle...")

with open(os.path.join(OUTPUT_DIR, "sarima_metrics.json"), "r") as f:
    sarima_metrics = json.load(f)

with open(os.path.join(OUTPUT_DIR, "prophet_metrics.json"), "r") as f:
    prophet_metrics = json.load(f)

with open(os.path.join(OUTPUT_DIR, "lstm_metrics.json"), "r") as f:
    lstm_metrics = json.load(f)

print("✓ Métriques chargées pour SARIMA, Prophet et LSTM")


# ============================================================================
# 2. CRÉATION DU TABLEAU COMPARATIF
# ============================================================================
print("\n[2] Création du tableau comparatif...")

comparison_df = pd.DataFrame({
    "Modèle": ["SARIMA", "Prophet", "LSTM"],
    "MSE": [
        sarima_metrics["mse"],
        prophet_metrics["mse"],
        lstm_metrics["mse"]
    ],
    "RMSE": [
        sarima_metrics["rmse"],
        prophet_metrics["rmse"],
        lstm_metrics["rmse"]
    ],
    "MAE": [
        sarima_metrics["mae"],
        prophet_metrics["mae"],
        lstm_metrics["mae"]
    ],
    "MAPE (%)": [
        sarima_metrics["mape"],
        prophet_metrics["mape"],
        lstm_metrics["mape"]
    ],
    "R²": [
        sarima_metrics["r2"],
        prophet_metrics["r2"],
        lstm_metrics["r2"]
    ],
    "Temps (s)": [
        sarima_metrics["training_time"],
        prophet_metrics["training_time"],
        lstm_metrics["training_time"]
    ]
})

print("\n" + "=" * 70)
print("TABLEAU COMPARATIF DES MODÈLES")
print("=" * 70)
print(comparison_df.to_string(index=False))


# ============================================================================
# 3. VÉRIFICATION DES PRÉDICTIONS
# ============================================================================
print("\n[3] Vérification des ensembles de prédictions...")

prediction_files = {
    "SARIMA": "sarima_predictions.csv",
    "Prophet": "prophet_predictions.csv",
    "LSTM": "lstm_predictions.csv"
}

predictions = {}

for model_name, filename in prediction_files.items():
    path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier de prédictions introuvable pour {model_name}: {filename}"
        )

    df_pred = pd.read_csv(path, parse_dates=["date"])

    required_columns = {"date", "real_sales", "predicted_sales"}

    if not required_columns.issubset(df_pred.columns):
        raise ValueError(
            f"Colonnes manquantes dans {filename}. "
            f"Colonnes attendues : {required_columns}"
        )

    if df_pred["date"].duplicated().any():
        raise ValueError(
            f"Des dates dupliquées ont été détectées dans {filename}."
        )

    if df_pred[["real_sales", "predicted_sales"]].isnull().any().any():
        raise ValueError(
            f"Des valeurs manquantes ont été détectées dans {filename}."
        )

    predictions[model_name] = df_pred.sort_values("date").reset_index(drop=True)

    print(
        f"✓ {model_name:<8} : "
        f"{len(df_pred)} observations | "
        f"{df_pred['date'].min().date()} → {df_pred['date'].max().date()}"
    )


# Vérification du nombre d'observations
n_predictions = {
    model: len(df)
    for model, df in predictions.items()
}

if len(set(n_predictions.values())) != 1:
    raise ValueError(
        "Les modèles ne contiennent pas le même nombre d'observations de test."
    )

test_observations = next(iter(n_predictions.values()))

# Vérification des dates
reference_dates = predictions["SARIMA"]["date"]

for model_name, df_pred in predictions.items():

    if not df_pred["date"].equals(reference_dates):
        raise ValueError(
            f"Les dates de test de {model_name} ne correspondent pas "
            "à celles des autres modèles."
        )

print(
    f"✓ Les trois modèles sont comparés sur le même ensemble de test "
    f"({test_observations} observations)."
)


# ============================================================================
# 4. IDENTIFICATION DU MODÈLE RETENU SELON LE RMSE
# ============================================================================
print("\n[4] Identification du modèle retenu selon le RMSE...")

# Critère principal :
# le modèle retenu est celui présentant le RMSE le plus faible.
selection_criterion = "RMSE"

best_model_idx = comparison_df["RMSE"].idxmin()
best_model = comparison_df.loc[best_model_idx, "Modèle"]

selected_metrics = comparison_df.loc[best_model_idx]

print("\n" + "=" * 70)
print("MODÈLE RETENU SELON LE CRITÈRE RMSE")
print("=" * 70)

print(f"\n  Modèle retenu : {best_model}")
print(f"\n  Performances sur l'ensemble de test :")
print(f"    • RMSE : {selected_metrics['RMSE']:.2f}")
print(f"    • MAE  : {selected_metrics['MAE']:.2f}")
print(f"    • MAPE : {selected_metrics['MAPE (%)']:.2f}%")
print(f"    • R²   : {selected_metrics['R²']:.4f}")

print(
    f"\n  Nombre d'observations de test : {test_observations}"
)


# ============================================================================
# 5. ORDRE DES MODÈLES SELON LE RMSE
# ============================================================================
print("\n📊 Ordre des modèles selon le RMSE (croissant) :")

ranking = comparison_df.sort_values(
    "RMSE",
    ascending=True
)[["Modèle", "RMSE", "MAE", "MAPE (%)", "R²"]].reset_index(drop=True)

for position, row in ranking.iterrows():
    print(
        f"  {position + 1}. {row['Modèle']:8s} - "
        f"RMSE : {row['RMSE']:6.2f} | "
        f"MAE : {row['MAE']:6.2f} | "
        f"MAPE : {row['MAPE (%)']:5.2f}% | "
        f"R² : {row['R²']:.4f}"
    )


# ============================================================================
# 6. ANALYSE DES MÉTRIQUES
# ============================================================================
print("\n[5] Analyse détaillée des performances...")

print("\n📈 Valeurs extrêmes observées par métrique :")

min_rmse_model = comparison_df.loc[
    comparison_df["RMSE"].idxmin(), "Modèle"
]

min_mae_model = comparison_df.loc[
    comparison_df["MAE"].idxmin(), "Modèle"
]

min_mape_model = comparison_df.loc[
    comparison_df["MAPE (%)"].idxmin(), "Modèle"
]

max_r2_model = comparison_df.loc[
    comparison_df["R²"].idxmax(), "Modèle"
]

fastest_model = comparison_df.loc[
    comparison_df["Temps (s)"].idxmin(), "Modèle"
]

print(
    f"\n  • RMSE minimale : {min_rmse_model} "
    f"({comparison_df['RMSE'].min():.2f})"
)

print(
    f"  • MAE minimale : {min_mae_model} "
    f"({comparison_df['MAE'].min():.2f})"
)

print(
    f"  • MAPE minimale : {min_mape_model} "
    f"({comparison_df['MAPE (%)'].min():.2f}%)"
)

print(
    f"  • R² maximal : {max_r2_model} "
    f"({comparison_df['R²'].max():.4f})"
)

print(
    f"  • Temps d'entraînement minimal : {fastest_model} "
    f"({comparison_df['Temps (s)'].min():.2f}s)"
)


# ============================================================================
# 7. VISUALISATIONS COMPARATIVES
# ============================================================================
print("\n[6] Génération des visualisations comparatives...")

# --- GRAPHIQUE 1 : Barres comparatives des métriques ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

fig.suptitle(
    "Comparaison des modèles - Métriques d'évaluation",
    fontsize=18,
    fontweight="bold"
)

colors = ["#3498db", "#2ecc71", "#e74c3c"]

# RMSE
axes[0, 0].bar(
    comparison_df["Modèle"],
    comparison_df["RMSE"],
    color=colors,
    edgecolor="black"
)

axes[0, 0].set_title(
    "RMSE - valeur plus faible",
    fontsize=14,
    fontweight="bold"
)

axes[0, 0].set_ylabel("RMSE", fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis="y")

for i, value in enumerate(comparison_df["RMSE"]):
    axes[0, 0].text(
        i,
        value + 1,
        f"{value:.2f}",
        ha="center",
        fontweight="bold"
    )


# MAE
axes[0, 1].bar(
    comparison_df["Modèle"],
    comparison_df["MAE"],
    color=colors,
    edgecolor="black"
)

axes[0, 1].set_title(
    "MAE - valeur plus faible",
    fontsize=14,
    fontweight="bold"
)

axes[0, 1].set_ylabel("MAE", fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis="y")

for i, value in enumerate(comparison_df["MAE"]):
    axes[0, 1].text(
        i,
        value + 1,
        f"{value:.2f}",
        ha="center",
        fontweight="bold"
    )


# MAPE
axes[1, 0].bar(
    comparison_df["Modèle"],
    comparison_df["MAPE (%)"],
    color=colors,
    edgecolor="black"
)

axes[1, 0].set_title(
    "MAPE - valeur plus faible",
    fontsize=14,
    fontweight="bold"
)

axes[1, 0].set_ylabel("MAPE (%)", fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis="y")

for i, value in enumerate(comparison_df["MAPE (%)"]):
    axes[1, 0].text(
        i,
        value + 0.2,
        f"{value:.2f}%",
        ha="center",
        fontweight="bold"
    )


# R²
axes[1, 1].bar(
    comparison_df["Modèle"],
    comparison_df["R²"],
    color=colors,
    edgecolor="black"
)

axes[1, 1].set_title(
    "R² - valeur plus élevée",
    fontsize=14,
    fontweight="bold"
)

axes[1, 1].set_ylabel("R²", fontsize=12)

# Le R² de SARIMA est négatif.
# On ne force donc pas l'axe à partir de zéro.
r2_min = comparison_df["R²"].min()
r2_max = comparison_df["R²"].max()

margin = max((r2_max - r2_min) * 0.15, 0.05)

axes[1, 1].set_ylim(
    r2_min - margin,
    r2_max + margin
)

axes[1, 1].grid(True, alpha=0.3, axis="y")

for i, value in enumerate(comparison_df["R²"]):
    offset = 0.02 if value >= 0 else -0.08

    axes[1, 1].text(
        i,
        value + offset,
        f"{value:.4f}",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATIONS_DIR,
        "16_models_comparison_metrics.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "✓ Graphique 1 sauvegardé : "
    "16_models_comparison_metrics.png"
)


# ============================================================================
# 8. COMPARAISON DES PRÉDICTIONS
# ============================================================================
print("\n[7] Génération de la comparaison visuelle des prédictions...")

sarima_pred = predictions["SARIMA"]
prophet_pred = predictions["Prophet"]
lstm_pred = predictions["LSTM"]

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

fig.suptitle(
    "Comparaison visuelle des prédictions",
    fontsize=18,
    fontweight="bold"
)


# SARIMA
axes[0].plot(
    sarima_pred["date"],
    sarima_pred["real_sales"],
    color="black",
    linewidth=2,
    label="Valeurs réelles",
    marker="o",
    markersize=3
)

axes[0].plot(
    sarima_pred["date"],
    sarima_pred["predicted_sales"],
    color="#3498db",
    linewidth=2,
    label="SARIMA",
    linestyle="--",
    marker="s",
    markersize=3
)

axes[0].set_title(
    f"SARIMA - RMSE : {sarima_metrics['rmse']:.2f}",
    fontsize=14,
    fontweight="bold"
)

axes[0].set_ylabel("Ventes", fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)


# Prophet
axes[1].plot(
    prophet_pred["date"],
    prophet_pred["real_sales"],
    color="black",
    linewidth=2,
    label="Valeurs réelles",
    marker="o",
    markersize=3
)

axes[1].plot(
    prophet_pred["date"],
    prophet_pred["predicted_sales"],
    color="#2ecc71",
    linewidth=2,
    label="Prophet",
    linestyle="--",
    marker="s",
    markersize=3
)

axes[1].set_title(
    f"Prophet - RMSE : {prophet_metrics['rmse']:.2f}",
    fontsize=14,
    fontweight="bold"
)

axes[1].set_ylabel("Ventes", fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)


# LSTM
axes[2].plot(
    lstm_pred["date"],
    lstm_pred["real_sales"],
    color="black",
    linewidth=2,
    label="Valeurs réelles",
    marker="o",
    markersize=3
)

axes[2].plot(
    lstm_pred["date"],
    lstm_pred["predicted_sales"],
    color="#e74c3c",
    linewidth=2,
    label="LSTM",
    linestyle="--",
    marker="s",
    markersize=3
)

axes[2].set_title(
    f"LSTM - RMSE : {lstm_metrics['rmse']:.2f}",
    fontsize=14,
    fontweight="bold"
)

axes[2].set_xlabel("Date", fontsize=12)
axes[2].set_ylabel("Ventes", fontsize=12)
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATIONS_DIR,
        "17_models_comparison_predictions.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "✓ Graphique 2 sauvegardé : "
    "17_models_comparison_predictions.png"
)


# ============================================================================
# 9. COMPARAISON SUR LE MÊME GRAPHIQUE
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(
    sarima_pred["date"],
    sarima_pred["real_sales"],
    color="black",
    linewidth=3,
    label="Valeurs réelles",
    marker="o",
    markersize=4
)

ax.plot(
    sarima_pred["date"],
    sarima_pred["predicted_sales"],
    color="#3498db",
    linewidth=2,
    label="SARIMA",
    linestyle="--",
    alpha=0.8
)

ax.plot(
    prophet_pred["date"],
    prophet_pred["predicted_sales"],
    color="#2ecc71",
    linewidth=2,
    label="Prophet",
    linestyle="--",
    alpha=0.8
)

ax.plot(
    lstm_pred["date"],
    lstm_pred["predicted_sales"],
    color="#e74c3c",
    linewidth=2,
    label="LSTM",
    linestyle="--",
    alpha=0.8
)

ax.set_title(
    "Comparaison des prédictions sur l'ensemble de test",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Ventes", fontsize=12)
ax.legend(fontsize=12, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUALIZATIONS_DIR,
        "18_models_comparison_all.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    "✓ Graphique 3 sauvegardé : "
    "18_models_comparison_all.png"
)


# ============================================================================
# 10. SAUVEGARDE DU RAPPORT
# ============================================================================
print("\n[8] Génération du rapport de comparaison...")

report = {
    "selection_criterion": selection_criterion,
    "test_observations": int(test_observations),

    "comparison_table": comparison_df.to_dict("records"),

    # Cette clé est conservée car forecasting.py l'utilise.
    "best_model": {
        "name": best_model,
        "rmse": float(selected_metrics["RMSE"]),
        "mae": float(selected_metrics["MAE"]),
        "mape": float(selected_metrics["MAPE (%)"]),
        "r2": float(selected_metrics["R²"])
    },

    "ranking": ranking.to_dict("records")
}

with open(
    os.path.join(OUTPUT_DIR, "comparison_report.json"),
    "w"
) as f:
    json.dump(report, f, indent=4)

print("✓ Rapport sauvegardé : comparison_report.json")


# Tableau CSV
comparison_df.to_csv(
    os.path.join(OUTPUT_DIR, "comparison_table.csv"),
    index=False
)

print("✓ Tableau sauvegardé : comparison_table.csv")


# ============================================================================
# 11. RÉSUMÉ FINAL
# ============================================================================
print("\n" + "=" * 70)
print("COMPARAISON DES MODÈLES TERMINÉE")
print("=" * 70)

print(
    f"\nModèle retenu selon le critère {selection_criterion} : "
    f"{best_model}"
)

print("\nPerformances du modèle retenu sur l'ensemble de test :")
print(
    f"  • RMSE : {selected_metrics['RMSE']:.2f}"
)
print(
    f"  • MAE  : {selected_metrics['MAE']:.2f}"
)
print(
    f"  • MAPE : {selected_metrics['MAPE (%)']:.2f}%"
)
print(
    f"  • R²   : {selected_metrics['R²']:.4f}"
)

print(
    f"\nEnsemble de test : {test_observations} observations"
)

print("\nFichiers générés :")
print("  1. comparison_report.json")
print("  2. comparison_table.csv")
print("  3. 16_models_comparison_metrics.png")
print("  4. 17_models_comparison_predictions.png")
print("  5. 18_models_comparison_all.png")

print(
    "\nProchaine étape : prédictions futures "
    "(forecasting.py)"
)

print("=" * 70)
