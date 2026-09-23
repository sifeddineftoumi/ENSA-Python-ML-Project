"""
Projet ENSA Berrechid 2025-2026
Partie 2 : Prédiction des ventes de vêtements
Script 8 : Prédictions Futures

Ce script utilise le modèle retenu lors de la comparaison
pour effectuer une prévision sur les 30 jours suivant
la dernière observation historique.

Le modèle retenu est réentraîné sur l'ensemble des données
historiques disponibles avant la prévision finale.
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from prophet import Prophet

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


COMPARISON_REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "comparison_report.json"
)

DATA_FILE = os.path.join(
    DATA_DIR,
    "sales_data.csv"
)

FINAL_MODEL_FILE = os.path.join(
    MODELS_DIR,
    "prophet_final_model.pkl"
)

FORECAST_FILE = os.path.join(
    OUTPUT_DIR,
    "future_predictions.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "future_forecast_summary.json"
)


print("=" * 70)
print("PRÉDICTIONS FUTURES - 30 PROCHAINS JOURS")
print("=" * 70)


# ============================================================================
# 1. IDENTIFICATION DU MODÈLE RETENU
# ============================================================================

print("\n[1] Identification du modèle retenu...")

if not os.path.exists(COMPARISON_REPORT_FILE):
    raise FileNotFoundError(
        f"Rapport de comparaison introuvable : {COMPARISON_REPORT_FILE}"
    )

with open(COMPARISON_REPORT_FILE, "r", encoding="utf-8") as f:
    report = json.load(f)

best_model = report["best_model"]["name"]

print(f"✓ Modèle retenu selon le RMSE : {best_model}")
print(f"  • RMSE : {report['best_model']['rmse']:.2f}")
print(f"  • MAE : {report['best_model']['mae']:.2f}")
print(f"  • MAPE : {report['best_model']['mape']:.2f}%")
print(f"  • R² : {report['best_model']['r2']:.4f}")


# ============================================================================
# 2. CHARGEMENT DE L'ENSEMBLE COMPLET DES DONNÉES
# ============================================================================

print("\n[2] Chargement de l'ensemble complet des données...")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset introuvable : {DATA_FILE}"
    )

df_complete = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

df_complete = df_complete.sort_values("date").reset_index(drop=True)

if "sales" not in df_complete.columns:
    raise ValueError(
        "La colonne 'sales' est absente du dataset."
    )

if df_complete["sales"].isna().any():
    raise ValueError(
        "Le dataset contient des valeurs manquantes dans la colonne 'sales'."
    )

print(f"✓ Dataset complet : {len(df_complete)} jours")
print(f"  • Première date : {df_complete['date'].min().date()}")
print(f"  • Dernière date : {df_complete['date'].max().date()}")


# ============================================================================
# 3. PRÉPARATION DU MODÈLE FINAL
# ============================================================================

print("\n[3] Préparation du modèle final...")

forecast_days = 30

if best_model != "Prophet":
    raise ValueError(
        f"Le modèle retenu est '{best_model}'. "
        "Cette version finale de forecasting.py est configurée "
        "pour le modèle Prophet."
    )

print("✓ Le modèle final sera réentraîné sur les 730 observations historiques.")


# ============================================================================
# 4. RÉENTRAÎNEMENT DE PROPHET SUR TOUTES LES DONNÉES
# ============================================================================

print("\n[4] Réentraînement de Prophet sur toutes les données...")

prophet_data = df_complete[["date", "sales"]].copy()

prophet_data = prophet_data.rename(
    columns={
        "date": "ds",
        "sales": "y"
    }
)

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    interval_width=0.95
)

model.fit(prophet_data)

print("✓ Prophet réentraîné sur l'ensemble complet des données.")


# ============================================================================
# 5. SAUVEGARDE DU MODÈLE FINAL
# ============================================================================

print("\n[5] Sauvegarde du modèle final...")

with open(FINAL_MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"✓ Modèle final sauvegardé : {FINAL_MODEL_FILE}")


# ============================================================================
# 6. GÉNÉRATION DES 30 PRÉVISIONS FUTURES
# ============================================================================

print("\n[6] Génération des prévisions futures...")

last_historical_date = df_complete["date"].max()

future_dates = pd.date_range(
    start=last_historical_date + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D"
)

future_df = pd.DataFrame({
    "ds": future_dates
})

forecast = model.predict(future_df)

predictions = forecast["yhat"].values
lower_bound = forecast["yhat_lower"].values
upper_bound = forecast["yhat_upper"].values

print(f"✓ {forecast_days} prévisions générées")
print(
    f"  • Période prévue : "
    f"{future_dates.min().date()} → {future_dates.max().date()}"
)


# ============================================================================
# 7. CRÉATION DU DATAFRAME DES PRÉVISIONS
# ============================================================================

print("\n[7] Création du DataFrame des prévisions...")

forecast_df = pd.DataFrame({
    "date": future_dates,
    "predicted_sales": predictions,
    "lower_bound": lower_bound,
    "upper_bound": upper_bound
})

forecast_df["day_of_week"] = forecast_df["date"].dt.day_name()
forecast_df["month"] = forecast_df["date"].dt.month
forecast_df["is_weekend"] = (
    forecast_df["date"].dt.weekday >= 5
).astype(int)

print("✓ DataFrame créé")

print("\n📊 Aperçu des prévisions :")
print(forecast_df.head(10).to_string(index=False))


# ============================================================================
# 8. STATISTIQUES DES PRÉVISIONS
# ============================================================================

print("\n📈 Statistiques des prévisions :")

print(
    f"  • Ventes moyennes prédites : "
    f"{predictions.mean():.0f} unités/jour"
)

print(
    f"  • Minimum : "
    f"{predictions.min():.0f} unités"
)

print(
    f"  • Maximum : "
    f"{predictions.max():.0f} unités"
)

print(
    f"  • Écart-type : "
    f"{predictions.std():.0f}"
)


# ============================================================================
# 9. VISUALISATIONS
# ============================================================================

print("\n[8] Génération des visualisations...")


# --------------------------------------------------------------------------
# GRAPHIQUE 1 : VUE D'ENSEMBLE
# --------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(18, 8))

ax.plot(
    df_complete["date"],
    df_complete["sales"],
    linewidth=1.5,
    label="Données historiques"
)

ax.plot(
    forecast_df["date"],
    forecast_df["predicted_sales"],
    linewidth=2.5,
    label="Prévisions futures",
    marker="o",
    markersize=5,
    linestyle="--"
)

ax.fill_between(
    forecast_df["date"],
    forecast_df["lower_bound"],
    forecast_df["upper_bound"],
    alpha=0.2,
    label="Intervalle de prévision à 95 %"
)

ax.axvline(
    x=last_historical_date,
    linestyle="--",
    linewidth=2.5,
    label="Fin des données historiques"
)

ax.set_title(
    f"Prévisions futures - 30 prochains jours ({best_model})",
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Ventes", fontsize=14)
ax.legend(fontsize=12, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()

graph1_file = os.path.join(
    VISUALIZATIONS_DIR,
    "19_future_forecast_overview.png"
)

plt.savefig(
    graph1_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("✓ Graphique 1 sauvegardé : 19_future_forecast_overview.png")


# --------------------------------------------------------------------------
# GRAPHIQUE 2 : PRÉVISIONS DÉTAILLÉES
# --------------------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(16, 10)
)

fig.suptitle(
    f"Prévisions détaillées - {best_model}",
    fontsize=18,
    fontweight="bold"
)


lookback_days = 60

historical_recent = df_complete.iloc[-lookback_days:]


axes[0].plot(
    historical_recent["date"],
    historical_recent["sales"],
    linewidth=2,
    label="Historique récent",
    marker="o",
    markersize=3
)

axes[0].plot(
    forecast_df["date"],
    forecast_df["predicted_sales"],
    linewidth=2.5,
    label="Prévisions",
    marker="s",
    markersize=5
)

axes[0].fill_between(
    forecast_df["date"],
    forecast_df["lower_bound"],
    forecast_df["upper_bound"],
    alpha=0.2,
    label="Intervalle de prévision à 95 %"
)

axes[0].axvline(
    x=last_historical_date,
    linestyle="--",
    linewidth=2,
    label="Fin des données historiques"
)

axes[0].set_title(
    "Transition : historique → prévisions",
    fontsize=14,
    fontweight="bold"
)

axes[0].set_xlabel("Date", fontsize=12)
axes[0].set_ylabel("Ventes", fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)


axes[1].plot(
    forecast_df["date"],
    forecast_df["predicted_sales"],
    linewidth=3,
    label="Prévisions",
    marker="o",
    markersize=6
)

axes[1].fill_between(
    forecast_df["date"],
    forecast_df["lower_bound"],
    forecast_df["upper_bound"],
    alpha=0.3,
    label="Intervalle de prévision à 95 %"
)

mean_prediction = forecast_df["predicted_sales"].mean()

axes[1].axhline(
    y=mean_prediction,
    linestyle="--",
    linewidth=2,
    label=f"Moyenne : {mean_prediction:.0f}"
)

axes[1].set_title(
    "30 prochains jours - prévisions détaillées",
    fontsize=14,
    fontweight="bold"
)

axes[1].set_xlabel("Date", fontsize=12)
axes[1].set_ylabel("Ventes", fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)


min_idx = forecast_df["predicted_sales"].idxmin()
max_idx = forecast_df["predicted_sales"].idxmax()


axes[1].annotate(
    f"Min : {forecast_df.loc[min_idx, 'predicted_sales']:.0f}",
    xy=(
        forecast_df.loc[min_idx, "date"],
        forecast_df.loc[min_idx, "predicted_sales"]
    ),
    xytext=(10, -30),
    textcoords="offset points",
    bbox=dict(
        boxstyle="round,pad=0.5",
        alpha=0.7
    ),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=0"
    )
)

axes[1].annotate(
    f"Max : {forecast_df.loc[max_idx, 'predicted_sales']:.0f}",
    xy=(
        forecast_df.loc[max_idx, "date"],
        forecast_df.loc[max_idx, "predicted_sales"]
    ),
    xytext=(10, 30),
    textcoords="offset points",
    bbox=dict(
        boxstyle="round,pad=0.5",
        alpha=0.7
    ),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=0"
    )
)

plt.tight_layout()

graph2_file = os.path.join(
    VISUALIZATIONS_DIR,
    "20_future_forecast_detailed.png"
)

plt.savefig(
    graph2_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("✓ Graphique 2 sauvegardé : 20_future_forecast_detailed.png")


# --------------------------------------------------------------------------
# GRAPHIQUE 3 : ANALYSE PAR JOUR DE LA SEMAINE
# --------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 6)
)

fig.suptitle(
    "Analyse des prévisions futures",
    fontsize=18,
    fontweight="bold"
)


day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

daily_avg = (
    forecast_df
    .groupby("day_of_week")["predicted_sales"]
    .mean()
    .reindex(day_order)
)

axes[0].bar(
    range(7),
    daily_avg.values,
    edgecolor="black",
    alpha=0.7
)

axes[0].set_xticks(range(7))

axes[0].set_xticklabels(
    ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"],
    fontsize=11
)

axes[0].set_title(
    "Ventes prédites par jour de la semaine",
    fontsize=13,
    fontweight="bold"
)

axes[0].set_ylabel(
    "Ventes moyennes",
    fontsize=11
)

axes[0].grid(
    True,
    alpha=0.3,
    axis="y"
)

for i, value in enumerate(daily_avg.values):
    if not np.isnan(value):
        axes[0].text(
            i,
            value + 2,
            f"{value:.0f}",
            ha="center",
            fontweight="bold"
        )


weekend_comparison = (
    forecast_df
    .groupby("is_weekend")["predicted_sales"]
    .mean()
)

week_labels = []

week_values = []

if 0 in weekend_comparison.index:
    week_labels.append("Semaine")
    week_values.append(weekend_comparison.loc[0])

if 1 in weekend_comparison.index:
    week_labels.append("Weekend")
    week_values.append(weekend_comparison.loc[1])

axes[1].bar(
    week_labels,
    week_values,
    edgecolor="black",
    alpha=0.7
)

axes[1].set_title(
    "Ventes prédites : semaine vs weekend",
    fontsize=13,
    fontweight="bold"
)

axes[1].set_ylabel(
    "Ventes moyennes",
    fontsize=11
)

axes[1].grid(
    True,
    alpha=0.3,
    axis="y"
)

for i, value in enumerate(week_values):
    axes[1].text(
        i,
        value + 2,
        f"{value:.0f}",
        ha="center",
        fontweight="bold",
        fontsize=12
    )

plt.tight_layout()

graph3_file = os.path.join(
    VISUALIZATIONS_DIR,
    "21_future_forecast_analysis.png"
)

plt.savefig(
    graph3_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("✓ Graphique 3 sauvegardé : 21_future_forecast_analysis.png")


# ============================================================================
# 10. SAUVEGARDE DES PRÉVISIONS
# ============================================================================

print("\n[9] Sauvegarde des prévisions futures...")

forecast_df.to_csv(
    FORECAST_FILE,
    index=False
)

print(
    f"✓ Prévisions sauvegardées : {FORECAST_FILE}"
)


# ============================================================================
# 11. CRÉATION DU RÉSUMÉ
# ============================================================================

summary = {
    "model_used": best_model,
    "model_training_observations": int(len(df_complete)),
    "historical_period": (
        f"{df_complete['date'].min().date()} "
        f"to {df_complete['date'].max().date()}"
    ),
    "forecast_period": (
        f"{forecast_df['date'].min().date()} "
        f"to {forecast_df['date'].max().date()}"
    ),
    "days_predicted": int(forecast_days),
    "average_sales": float(predictions.mean()),
    "min_sales": float(predictions.min()),
    "max_sales": float(predictions.max()),
    "std_sales": float(predictions.std()),
    "best_day": forecast_df.loc[
        forecast_df["predicted_sales"].idxmax(),
        "date"
    ].strftime("%Y-%m-%d"),
    "worst_day": forecast_df.loc[
        forecast_df["predicted_sales"].idxmin(),
        "date"
    ].strftime("%Y-%m-%d"),
    "selection_criterion": "RMSE on test set",
    "interval_width": 0.95
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=4
    )

print(
    f"✓ Résumé sauvegardé : {SUMMARY_FILE}"
)


# ============================================================================
# 12. RÉSUMÉ FINAL
# ============================================================================

print("\n" + "=" * 70)
print("PRÉVISIONS FUTURES TERMINÉES !")
print("=" * 70)

print("\n📊 Résumé des prévisions :")

print(
    f"  • Modèle utilisé : {best_model}"
)

print(
    f"  • Données d'entraînement final : "
    f"{len(df_complete)} jours"
)

print(
    f"  • Période historique : "
    f"{df_complete['date'].min().date()} → "
    f"{df_complete['date'].max().date()}"
)

print(
    f"  • Période prévue : "
    f"{forecast_df['date'].min().date()} → "
    f"{forecast_df['date'].max().date()}"
)

print(
    f"  • Ventes moyennes prédites : "
    f"{predictions.mean():.0f} unités/jour"
)

print(
    f"  • Jour avec la prévision maximale : "
    f"{summary['best_day']} "
    f"({predictions.max():.0f} unités)"
)

print(
    f"  • Jour avec la prévision minimale : "
    f"{summary['worst_day']} "
    f"({predictions.min():.0f} unités)"
)

print("\n📈 Fichiers générés :")

print("  1. future_predictions.csv")
print("  2. future_forecast_summary.json")
print("  3. prophet_final_model.pkl")
print("  4. 19_future_forecast_overview.png")
print("  5. 20_future_forecast_detailed.png")
print("  6. 21_future_forecast_analysis.png")

print("\n" + "=" * 70)
