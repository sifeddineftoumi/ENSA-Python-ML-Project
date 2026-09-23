"""
Projet ENSA Berrechid 2025-2026
Test du modèle CNN sur une image externe

Ce script permet de tester le modèle entraîné sur n'importe quelle
image de chiffre manuscrit (format PNG, JPG, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import sys

print("="*60)
print("TEST DU MODÈLE CNN SUR IMAGE EXTERNE")
print("="*60)

# ============================================================================
# 1. CHARGEMENT DU MODÈLE
# ============================================================================
print("\n[1] Chargement du modèle entraîné...")

try:
    model = keras.models.load_model('outputs/mnist_cnn_model.h5')
    print("✓ Modèle chargé avec succès !")
except:
    print("❌ ERREUR : Le modèle n'a pas été trouvé.")
    print("   Exécutez d'abord 'mnist_cnn.py' pour entraîner le modèle.")
    sys.exit(1)

# ============================================================================
# 2. FONCTION DE PRÉTRAITEMENT D'IMAGE
# ============================================================================
def preprocess_image(image_path):
    """
    Prépare une image pour la prédiction :
    - Conversion en niveaux de gris
    - Redimensionnement 28x28
    - Inversion si nécessaire (fond blanc → fond noir)
    - Normalisation
    """
    try:
        # Charger l'image
        img = Image.open(image_path)
        
        # Convertir en niveaux de gris
        img = img.convert('L')
        
        # Redimensionner à 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convertir en array numpy
        img_array = np.array(img)
        
        # MNIST a un fond noir et chiffre blanc
        # Si l'image a fond blanc, inverser
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normaliser [0, 255] → [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape pour le modèle : (28, 28) → (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array, img
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'image : {e}")
        return None, None

# ============================================================================
# 3. FONCTION DE PRÉDICTION
# ============================================================================
def predict_digit(image_path):
    """
    Prédit le chiffre présent dans l'image
    """
    # Prétraiter l'image
    img_processed, img_original = preprocess_image(image_path)
    
    if img_processed is None:
        return
    
    # Prédiction
    predictions = model.predict(img_processed, verbose=0)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    # Affichage des résultats
    print("\n" + "="*60)
    print("RÉSULTAT DE LA PRÉDICTION")
    print("="*60)
    print(f"✓ Chiffre prédit : {predicted_class}")
    print(f"✓ Confiance : {confidence:.2f}%")
    print("\nProbabilités pour chaque classe :")
    for i, prob in enumerate(predictions[0]):
        bar = "█" * int(prob * 50)
        print(f"  {i} : {prob*100:5.2f}% {bar}")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Image originale redimensionnée
    axes[0].imshow(img_original, cmap='gray')
    axes[0].set_title('Image preprocessée (28x28)', fontsize=12)
    axes[0].axis('off')
    
    # Graphique des probabilités
    axes[1].bar(range(10), predictions[0], color='steelblue')
    axes[1].set_xlabel('Chiffre', fontsize=12)
    axes[1].set_ylabel('Probabilité', fontsize=12)
    axes[1].set_title('Distribution des probabilités', fontsize=12)
    axes[1].set_xticks(range(10))
    axes[1].grid(axis='y', alpha=0.3)
    
    # Mettre en évidence la prédiction
    axes[1].bar(predicted_class, predictions[0][predicted_class], 
                color='red', label=f'Prédit: {predicted_class}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/prediction_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Résultat sauvegardé : outputs/prediction_result.png")

# ============================================================================
# 4. MODE INTERACTIF
# ============================================================================
def interactive_mode():
    """
    Mode interactif pour tester plusieurs images
    """
    print("\n" + "="*60)
    print("MODE INTERACTIF")
    print("="*60)
    print("\nVous pouvez tester plusieurs images.")
    print("Tapez 'q' pour quitter.\n")
    
    while True:
        image_path = input("📁 Chemin de l'image à tester : ").strip()
        
        if image_path.lower() == 'q':
            print("\n👋 Au revoir !")
            break
        
        predict_digit(image_path)
        print("\n" + "-"*60 + "\n")

# ============================================================================
# 5. EXÉCUTION PRINCIPALE
# ============================================================================
if __name__ == "__main__":
    print("\n📌 Options :")
    print("  1. Tester une image spécifique")
    print("  2. Mode interactif (tester plusieurs images)")
    
    choice = input("\nVotre choix (1 ou 2) : ").strip()
    
    if choice == '1':
        image_path = input("\n📁 Chemin de l'image : ").strip()
        predict_digit(image_path)
    elif choice == '2':
        interactive_mode()
    else:
        print("❌ Choix invalide. Relancez le script.")

print("\n" + "="*60)
print("✅ PROGRAMME TERMINÉ")
print("="*60)