"""
Projet ENSA Berrechid 2025-2026
Partie 1 : Réseau de Neurones Convolutionnel (CNN) sur MNIST
Auteur : [Ton Nom]
Date : Décembre 2024

Description :
Ce script implémente un CNN pour la classification des chiffres manuscrits
de la base de données MNIST (28x28 pixels, 10 classes : 0-9)
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Créer le dossier outputs s'il n'existe pas
os.makedirs('outputs', exist_ok=True)

print("="*60)
print("PARTIE 1 : CNN SUR MNIST")
print("="*60)

# ============================================================================
# 1. CHARGEMENT ET EXPLORATION DES DONNÉES
# ============================================================================
print("\n[1] Chargement des données MNIST...")

# Charger MNIST (téléchargement automatique si nécessaire)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"✓ Données chargées avec succès !")
print(f"  - Ensemble d'entraînement : {X_train.shape[0]} images")
print(f"  - Ensemble de test : {X_test.shape[0]} images")
print(f"  - Dimensions des images : {X_train.shape[1]}x{X_train.shape[2]} pixels")
print(f"  - Nombre de classes : {len(np.unique(y_train))}")

# Afficher quelques exemples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Exemples d\'images MNIST', fontsize=16, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/mnist_examples.png', dpi=150, bbox_inches='tight')
print("\n✓ Exemples d'images sauvegardés : outputs/mnist_examples.png")

# ============================================================================
# 2. PRÉPARATION DES DONNÉES
# ============================================================================
print("\n[2] Préparation des données...")

# Normalisation : convertir les valeurs de [0, 255] vers [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Redimensionner pour le CNN : ajouter une dimension de canal
# Shape passe de (60000, 28, 28) à (60000, 28, 28, 1)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Conversion des labels en one-hot encoding
# Exemple : 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

print(f"✓ Normalisation effectuée (valeurs entre 0 et 1)")
print(f"✓ Shape après preprocessing :")
print(f"  - X_train : {X_train.shape}")
print(f"  - X_test : {X_test.shape}")
print(f"  - y_train : {y_train_cat.shape}")

# ============================================================================
# 3. CONSTRUCTION DU MODÈLE CNN
# ============================================================================
print("\n[3] Construction du réseau de neurones convolutionnel...")

model = keras.Sequential([
    # ---- Bloc Convolutionnel 1 ----
    # 32 filtres 3x3, activation ReLU
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                  input_shape=(28, 28, 1), name='conv1'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
    
    # ---- Bloc Convolutionnel 2 ----
    # 64 filtres 3x3, extraction de features plus complexes
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
    
    # ---- Bloc Convolutionnel 3 ----
    # 128 filtres pour capturer des patterns avancés
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv3'),
    
    # ---- Aplatissement (Flatten) ----
    # Convertir les feature maps 3D en vecteur 1D
    layers.Flatten(name='flatten'),
    
    # ---- Couches Fully Connected ----
    layers.Dense(128, activation='relu', name='dense1'),
    layers.Dropout(0.5, name='dropout'),  # Régularisation pour éviter l'overfitting
    
    # ---- Couche de Sortie ----
    # 10 neurones (1 par classe), activation softmax pour probabilités
    layers.Dense(10, activation='softmax', name='output')
], name='MNIST_CNN')

# Afficher l'architecture du modèle
print("\n" + "="*60)
print("ARCHITECTURE DU CNN")
print("="*60)
model.summary()

# ============================================================================
# 4. COMPILATION DU MODÈLE
# ============================================================================
print("\n[4] Compilation du modèle...")

model.compile(
    optimizer='adam',  # Optimiseur Adam (adaptive learning rate)
    loss='categorical_crossentropy',  # Loss pour classification multi-classes
    metrics=['accuracy']  # Métrique : précision
)

print("✓ Modèle compilé avec succès !")
print("  - Optimizer : Adam")
print("  - Loss : Categorical Crossentropy")
print("  - Metrics : Accuracy")

# ============================================================================
# 5. ENTRAÎNEMENT DU MODÈLE
# ============================================================================
print("\n[5] Entraînement du modèle...")
print("⏳ Cela peut prendre quelques minutes...")

history = model.fit(
    X_train, y_train_cat,
    batch_size=128,  # Traiter 128 images à la fois
    epochs=10,  # 10 passages complets sur les données
    validation_split=0.1,  # 10% des données pour validation
    verbose=1
)

print("\n✓ Entraînement terminé !")

# ============================================================================
# 6. ÉVALUATION SUR L'ENSEMBLE DE TEST
# ============================================================================
print("\n[6] Évaluation sur l'ensemble de test...")

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

print(f"\n{'='*60}")
print(f"RÉSULTATS FINAUX")
print(f"{'='*60}")
print(f"✓ Loss sur test : {test_loss:.4f}")
print(f"✓ Accuracy sur test : {test_accuracy*100:.2f}%")

# ============================================================================
# 7. VISUALISATION DE L'ENTRAÎNEMENT
# ============================================================================
print("\n[7] Génération des graphiques...")

# Graphique 1 : Accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Évolution de la Précision (Accuracy)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Graphique 2 : Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Évolution de la Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/training_history.png', dpi=150, bbox_inches='tight')
print("✓ Graphiques sauvegardés : outputs/training_history.png")

# ============================================================================
# 8. MATRICE DE CONFUSION
# ============================================================================
print("\n[8] Génération de la matrice de confusion...")

# Prédictions sur le test set
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Matrice de Confusion', fontsize=16, fontweight='bold')
plt.xlabel('Classe Prédite')
plt.ylabel('Classe Réelle')
plt.savefig('outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Matrice de confusion sauvegardée : outputs/confusion_matrix.png")

# Rapport de classification détaillé
print("\n" + "="*60)
print("RAPPORT DE CLASSIFICATION DÉTAILLÉ")
print("="*60)
print(classification_report(y_test, y_pred_classes))

# ============================================================================
# 9. SAUVEGARDE DU MODÈLE
# ============================================================================
print("\n[9] Sauvegarde du modèle...")

model.save('outputs/mnist_cnn_model.h5')
print("✓ Modèle sauvegardé : outputs/mnist_cnn_model.h5")

# ============================================================================
# 10. TEST SUR QUELQUES EXEMPLES
# ============================================================================
print("\n[10] Test de prédiction sur des exemples...")

# Sélectionner 10 images aléatoires du test set
indices = np.random.choice(len(X_test), 10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Prédictions du Modèle CNN', fontsize=16, fontweight='bold')

for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
    img = X_test[idx]
    true_label = y_test[idx]
    
    # Prédiction
    pred = model.predict(img.reshape(1, 28, 28, 1), verbose=0)
    pred_label = np.argmax(pred)
    confidence = np.max(pred) * 100
    
    # Affichage
    ax.imshow(img.squeeze(), cmap='gray')
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f'Vrai: {true_label} | Prédit: {pred_label}\nConfiance: {confidence:.1f}%',
                 color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/predictions_examples.png', dpi=150, bbox_inches='tight')
print("✓ Exemples de prédictions sauvegardés : outputs/predictions_examples.png")

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("\n" + "="*60)
print("✅ PARTIE 1 TERMINÉE AVEC SUCCÈS !")
print("="*60)
print("\n📊 Fichiers générés dans le dossier 'outputs/' :")
print("  1. mnist_examples.png          - Exemples d'images MNIST")
print("  2. training_history.png        - Courbes d'entraînement")
print("  3. confusion_matrix.png        - Matrice de confusion")
print("  4. predictions_examples.png    - Exemples de prédictions")
print("  5. mnist_cnn_model.h5          - Modèle sauvegardé")
print("\n📈 Performance finale :")
print(f"  - Accuracy : {test_accuracy*100:.2f}%")
print(f"  - Loss : {test_loss:.4f}")
print("\n🎓 Prochaine étape : Test sur image externe (test_image.py)")
print("="*60)