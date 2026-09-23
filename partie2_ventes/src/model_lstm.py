"""
Projet ENSA Berrechid 2025-2026
Partie 2 : Prédiction des ventes de vêtements
Script : Modèle LSTM

Ce script entraîne un réseau LSTM pour prédire les ventes
et évalue ses performances sur le jeu de test.

Méthodologie :
- 584 observations pour l'entraînement
- 146 observations pour le test
- 30 jours de contexte pour les séquences
- validation chronologique sur une partie du train
- scaler ajusté uniquement sur les données d'entraînement
"""

import os
import json
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# 1. REPRODUCTIBILITÉ
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. CHEMINS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

TRAIN_FILE = os.path.join(DATA_DIR, "train_data.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_data.csv")

TRAIN_SCALED_FILE = os.path.join(
    DATA_DIR,
    "train_scaled.npy"
)

TEST_SCALED_FILE = os.path.join(
    DATA_DIR,
    "test_scaled.npy"
)

SCALER_FILE = os.path.join(
    MODELS_DIR,
    "scaler.pkl"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================

print("=" * 70)
print("MODÈLE LSTM - PRÉDICTION DES VENTES")
print("=" * 70)

train_data = pd.read_csv(
    TRAIN_FILE,
    parse_dates=["date"]
)

test_data = pd.read_csv(
    TEST_FILE,
    parse_dates=["date"]
)

train_scaled = np.load(
    TRAIN_SCALED_FILE
)

test_scaled = np.load(
    TEST_SCALED_FILE
)

scaler = joblib.load(
    SCALER_FILE
)


# Vérifications
required_columns = {"date", "sales"}

if not required_columns.issubset(train_data.columns):
    raise ValueError(
        "Le fichier train_data.csv doit contenir les colonnes "
        "'date' et 'sales'."
    )

if not required_columns.issubset(test_data.columns):
    raise ValueError(
        "Le fichier test_data.csv doit contenir les colonnes "
        "'date' et 'sales'."
    )

if train_data["sales"].isna().any():
    raise ValueError(
        "Des valeurs manquantes sont présentes dans train_data.csv."
    )

if test_data["sales"].isna().any():
    raise ValueError(
        "Des valeurs manquantes sont présentes dans test_data.csv."
    )

train_data = train_data.sort_values("date").reset_index(drop=True)
test_data = test_data.sort_values("date").reset_index(drop=True)

print(f"\nNombre de données train : {len(train_data)}")
print(f"Nombre de données test  : {len(test_data)}")

print(
    f"Période train : "
    f"{train_data['date'].min().date()} → "
    f"{train_data['date'].max().date()}"
)

print(
    f"Période test  : "
    f"{test_data['date'].min().date()} → "
    f"{test_data['date'].max().date()}"
)


# ============================================================
# 4. PRÉPARATION DES SÉQUENCES
# ============================================================

seq_length = 30


def create_sequences(data, sequence_length):
    """
    Crée les séquences temporelles utilisées par le LSTM.

    Chaque séquence contient les 'sequence_length' observations
    précédentes et la valeur suivante constitue la cible.
    """

    X = []
    y = []

    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# ------------------------------------------------------------
# Séquences d'entraînement
# ------------------------------------------------------------

X_train, y_train = create_sequences(
    train_scaled,
    seq_length
)

print(f"\nSéquences train : {len(X_train)}")


# ============================================================
# 5. PRÉPARATION CORRECTE DU TEST
# ============================================================

"""
Pour prédire le premier jour du test, le LSTM a besoin
des 30 jours précédents.

Ces 30 jours correspondent aux 30 dernières observations
du jeu d'entraînement.

On concatène donc :

    30 derniers jours du train
    +
    146 jours du test

Ce procédé permet d'obtenir exactement 146 séquences
correspondant aux 146 observations du jeu de test.
"""

test_context = np.concatenate(
    [
        train_scaled[-seq_length:],
        test_scaled
    ]
)

X_test, y_test = create_sequences(
    test_context,
    seq_length
)

print(f"Séquences test : {len(X_test)}")
print(f"Observations test attendues : {len(test_data)}")


if len(X_test) != len(test_data):
    raise ValueError(
        f"Erreur : {len(X_test)} prédictions test obtenues "
        f"au lieu de {len(test_data)}."
    )


# ============================================================
# 6. VALIDATION CHRONOLOGIQUE
# ============================================================

"""
Les séquences d'entraînement sont séparées chronologiquement
en deux parties :

- 90 % : entraînement
- 10 % : validation

Aucune donnée du jeu de test n'est utilisée pour la validation.
"""

validation_ratio = 0.10

validation_size = int(
    len(X_train) * validation_ratio
)

train_size = len(X_train) - validation_size

X_train_fit = X_train[:train_size]
y_train_fit = y_train[:train_size]

X_val = X_train[train_size:]
y_val = y_train[train_size:]

print("\nSéparation chronologique train/validation :")
print(f"  • Séquences d'entraînement : {len(X_train_fit)}")
print(f"  • Séquences de validation   : {len(X_val)}")

if len(X_val) == 0:
    raise ValueError(
        "Le nombre de séquences de validation est nul."
    )


# ============================================================
# 7. CONSTRUCTION DU MODÈLE
# ============================================================

print("\nConstruction du réseau LSTM...")

model = Sequential([
    Input(shape=(seq_length, 1)),

    LSTM(
        50,
        activation="relu",
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(
        50,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        25,
        activation="relu"
    ),

    Dense(1)
])


model.compile(
    optimizer=Adam(),
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ============================================================
# 8. ENTRAÎNEMENT
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

print("\nDébut de l'entraînement...")

start_time = time.time()

history = model.fit(
    X_train_fit,
    y_train_fit,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
    shuffle=False
)

training_time = time.time() - start_time

epochs_completed = len(
    history.history["loss"]
)

print(
    f"\nTemps d'entraînement : "
    f"{training_time:.2f} secondes"
)

print(
    f"Nombre d'epochs réalisés : "
    f"{epochs_completed}"
)


# ============================================================
# 9. PRÉDICTION SUR LE TEST
# ============================================================

print("\nGénération des prédictions sur le test...")

predictions_scaled = model.predict(
    X_test,
    verbose=0
).flatten()


# Retour aux valeurs originales
predictions = scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).flatten()

actual = test_data["sales"].values


# ============================================================
# 10. CALCUL DES MÉTRIQUES
# ============================================================

mse = mean_squared_error(
    actual,
    predictions
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    actual,
    predictions
)


# MAPE robuste aux valeurs nulles
non_zero = actual != 0

if np.any(non_zero):

    mape = np.mean(
        np.abs(
            (
                actual[non_zero]
                - predictions[non_zero]
            )
            / actual[non_zero]
        )
    ) * 100

else:

    mape = np.nan


r2 = r2_score(
    actual,
    predictions
)


# ============================================================
# 11. AFFICHAGE DES RÉSULTATS
# ============================================================

print("\n" + "=" * 70)
print("RÉSULTATS LSTM")
print("=" * 70)

print(
    f"Nombre de prédictions test : "
    f"{len(predictions)}"
)

print(
    f"MSE  : {mse:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"MAPE : {mape:.2f}%"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 12. SAUVEGARDE DES PRÉDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "date": test_data["date"],
    "real_sales": actual,
    "predicted_sales": predictions,
    "residuals": actual - predictions
})

predictions_file = os.path.join(
    OUTPUT_DIR,
    "lstm_predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)

print(
    f"\nPrédictions sauvegardées : "
    f"{predictions_file}"
)


# ============================================================
# 13. SAUVEGARDE DES MÉTRIQUES
# ============================================================

metrics = {
    "model": "LSTM",
    "sequence_length": seq_length,
    "validation_method": "chronological",
    "validation_ratio": validation_ratio,
    "train_sequences": len(X_train_fit),
    "validation_sequences": len(X_val),
    "training_time": training_time,
    "epochs": epochs_completed,
    "batch_size": 32,
    "mse": float(mse),
    "rmse": float(rmse),
    "mae": float(mae),
    "mape": float(mape),
    "r2": float(r2),
    "test_observations": len(test_data),
    "test_predictions": len(predictions),
    "random_seed": SEED
}

metrics_file = os.path.join(
    OUTPUT_DIR,
    "lstm_metrics.json"
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

print(
    f"Métriques sauvegardées : "
    f"{metrics_file}"
)


# ============================================================
# 14. COURBE D'APPRENTISSAGE
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    history.history["loss"],
    label="Loss entraînement"
)

plt.plot(
    history.history["val_loss"],
    label="Loss validation"
)

plt.title(
    "LSTM - Courbe d'apprentissage"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

learning_curve_file = os.path.join(
    OUTPUT_DIR,
    "13_lstm_learning_curve.png"
)

plt.savefig(
    learning_curve_file,
    dpi=300
)

plt.close()


# ============================================================
# 15. PRÉDICTIONS VS VALEURS RÉELLES
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    test_data["date"],
    actual,
    label="Ventes réelles"
)

plt.plot(
    test_data["date"],
    predictions,
    label="Prédictions LSTM"
)

plt.title(
    "LSTM - Prédictions vs ventes réelles"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Ventes"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

prediction_plot_file = os.path.join(
    OUTPUT_DIR,
    "14_lstm_predictions.png"
)

plt.savefig(
    prediction_plot_file,
    dpi=300
)

plt.close()


# ============================================================
# 16. ERREURS DE PRÉDICTION
# ============================================================

errors = actual - predictions

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    test_data["date"],
    errors
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title(
    "LSTM - Erreurs de prédiction"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Erreur"
)

plt.grid(True)

plt.tight_layout()

error_plot_file = os.path.join(
    OUTPUT_DIR,
    "15_lstm_errors.png"
)

plt.savefig(
    error_plot_file,
    dpi=300
)

plt.close()


# ============================================================
# 17. SAUVEGARDE DU MODÈLE
# ============================================================

model_file = os.path.join(
    MODELS_DIR,
    "lstm_model.keras"
)

model.save(
    model_file
)

print(
    f"Modèle sauvegardé : "
    f"{model_file}"
)


# ============================================================
# 18. FIN
# ============================================================

print("\n" + "=" * 70)
print("ENTRAÎNEMENT LSTM TERMINÉ")
print("=" * 70)

print(
    f"\nRésumé :"
)

print(
    f"  • Séquences train initiales : "
    f"{len(X_train)}"
)

print(
    f"  • Séquences utilisées pour entraînement : "
    f"{len(X_train_fit)}"
)

print(
    f"  • Séquences de validation : "
    f"{len(X_val)}"
)

print(
    f"  • Prédictions test : "
    f"{len(predictions)}"
)

print(
    f"  • RMSE : "
    f"{rmse:.2f}"
)

print(
    f"  • MAE : "
    f"{mae:.2f}"
)

print(
    f"  • MAPE : "
    f"{mape:.2f}%"
)

print(
    f"  • R² : "
    f"{r2:.4f}"
)
