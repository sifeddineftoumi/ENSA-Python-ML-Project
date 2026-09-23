"""
Script pour créer une image de test simple
Génère une image avec un chiffre manuscrit simulé
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def create_test_digit(digit, filename='test_digit.png'):
    """
    Crée une image simple d'un chiffre pour tester le modèle
    
    Args:
        digit (int): Le chiffre à dessiner (0-9)
        filename (str): Nom du fichier de sortie
    """
    # Créer une image blanche 200x200
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Dessiner le chiffre en noir au centre
    try:
        # Essayer d'utiliser une police système
        font = ImageFont.truetype("arial.ttf", 150)
    except:
        # Utiliser la police par défaut si arial n'est pas disponible
        font = ImageFont.load_default()
    
    # Centrer le texte
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((200 - text_width) // 2, (200 - text_height) // 2)
    draw.text(position, text, fill='black', font=font)
    
    # Sauvegarder
    img.save(filename)
    print(f"✓ Image créée : {filename}")
    
    # Afficher l'image
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title(f'Chiffre généré : {digit}')
    plt.axis('off')
    plt.show()
    
    return filename

# Exemple d'utilisation
if __name__ == "__main__":
    print("="*60)
    print("CRÉATION D'IMAGE DE TEST")
    print("="*60)
    
    digit = input("\n📝 Quel chiffre voulez-vous générer (0-9) ? : ")
    
    try:
        digit = int(digit)
        if 0 <= digit <= 9:
            create_test_digit(digit)
        else:
            print("❌ Erreur : Entrez un chiffre entre 0 et 9")
    except ValueError:
        print("❌ Erreur : Valeur invalide")