\# ENSA Python ML Project



Projet réalisé dans le cadre du module de programmation Python à l'\*\*ENSA Berrechid — 2025-2026\*\*.



Ce projet regroupe deux applications principales en intelligence artificielle et analyse de données :



1\. \*\*Classification d'images manuscrites MNIST à l'aide d'un réseau de neurones convolutif (CNN)\*\*.

2\. \*\*Prévision des ventes quotidiennes de vêtements\*\* à l'aide de plusieurs modèles de séries temporelles et d'apprentissage profond.



Une application permet également de présenter les résultats du projet.



\---



\## 1. Objectifs



Les objectifs du projet sont de mettre en pratique différentes étapes d'un processus de Machine Learning :



\- préparation et exploration des données ;

\- entraînement de modèles ;

\- évaluation sur un jeu de test ;

\- comparaison de plusieurs modèles ;

\- sélection d'un modèle selon un critère d'évaluation défini ;

\- génération de prévisions futures ;

\- visualisation et présentation des résultats.



\---



\# 2. Partie 1 — Classification MNIST avec CNN



La première partie porte sur la reconnaissance de chiffres manuscrits à partir du jeu de données \*\*MNIST\*\*.



Le modèle utilisé est un \*\*réseau de neurones convolutif (CNN)\*\*.



\### Architecture générale



Le réseau utilise plusieurs couches convolutives et des couches de classification afin d'identifier les chiffres de 0 à 9.



\### Résultat obtenu



Le modèle a obtenu une accuracy d'environ :



\*\*99,24 %\*\*



Le projet contient également des scripts permettant de tester le modèle sur une image.



\### Fichiers principaux



```text

partie1\_cnn\_mnist/

├── mnist\_cnn.py

├── create\_test\_image.py

├── test\_image.py

├── test\_digit.png

└── requirements.txt

```



Les résultats générés pendant l'entraînement sont stockés localement dans `outputs/` mais ne sont pas versionnés sur GitHub.



\---



\# 3. Partie 2 — Prévision des ventes



La deuxième partie consiste à prévoir les ventes quotidiennes de vêtements à partir d'une série temporelle.



Les données utilisées sont \*\*synthétiques\*\* et couvrent 730 jours :



```text

01/01/2022 → 31/12/2023

```



Le jeu de données est organisé chronologiquement selon la méthodologie suivante :



```text

730 observations

&#x20;       │

&#x20;       ├── 584 observations → entraînement

&#x20;       │

&#x20;       └── 146 observations → test

```



Le jeu de test est conservé séparément afin de comparer les modèles sur exactement les mêmes observations.



\---



\## 3.1 Préparation des données



La préparation comprend notamment :



\- vérification des données ;

\- traitement de la structure temporelle ;

\- séparation chronologique entraînement/test ;

\- normalisation pour le modèle LSTM ;

\- sauvegarde des données préparées.



Le scaler est ajusté uniquement sur les données d'entraînement afin d'éviter une fuite d'information entre les ensembles.



\---



\## 3.2 Analyse exploratoire



Le script `eda\_analysis.py` permet d'explorer la série temporelle et d'étudier notamment :



\- l'évolution des ventes ;

\- les tendances ;

\- les variations temporelles ;

\- les caractéristiques de la série.



\---



\# 4. Modèles de prévision



Trois approches ont été évaluées.



\## SARIMA



Configuration utilisée pour l'évaluation :



```text

SARIMA(1,1,1)(1,1,1,7)

```



Le modèle prend en compte une composante saisonnière hebdomadaire.



\## Prophet



Deux configurations ont été comparées :



\- avec saisonnalité annuelle ;

\- sans saisonnalité annuelle.



La configuration avec saisonnalité annuelle a été retenue pour la suite selon le critère RMSE.



\## LSTM



Le modèle LSTM utilise notamment :



\- deux couches LSTM de 50 unités ;

\- Dropout de 0,2 ;

\- une fenêtre temporelle de 30 jours ;

\- Early Stopping ;

\- validation chronologique.



\---



\# 5. Comparaison des modèles



Les trois modèles ont été évalués sur les \*\*146 mêmes observations du jeu de test\*\*.



Le critère principal utilisé pour la sélection est le \*\*RMSE (Root Mean Squared Error)\*\*.



| Modèle | RMSE | MAE | MAPE | R² |

|---|---:|---:|---:|---:|

| SARIMA | 107,32 | 92,93 | 48,23 % | -1,97 |

| Prophet | 36,97 | 23,79 | 13,90 % | 0,65 |

| LSTM | 53,44 | 36,10 | 19,07 % | 0,26 |



Selon le critère RMSE, la configuration Prophet évaluée avec saisonnalité annuelle présente le RMSE le plus faible parmi les trois modèles testés.



Le script `comparison.py` réalise automatiquement cette comparaison.



\---



\# 6. Prévision future



Après la comparaison, le modèle retenu selon le RMSE est réentraîné sur l'ensemble des \*\*730 observations historiques\*\*.



Une prévision de \*\*30 jours\*\* est ensuite générée :



```text

01/01/2024 → 30/01/2024

```



La prévision comprend également des intervalles de prédiction à 95 %.



Les résultats obtenus sur cette période sont notamment :



\- moyenne : environ \*\*287 unités/jour\*\* ;

\- minimum prévu : \*\*217 unités/jour\*\* ;

\- maximum prévu : \*\*367 unités/jour\*\*.



Ces valeurs constituent des prévisions produites à partir des données synthétiques du projet et ne représentent pas des ventes réelles.



\---



\# 7. Application de présentation



Le dossier `app\_presentation/` contient une application permettant de présenter les résultats du projet.



```text

app\_presentation/

├── app.py

└── requirements\_app.txt

```



\---



\# 8. Structure du projet



```text

ENSA-Python-ML-Project/

│

├── app\_presentation/

│   ├── app.py

│   └── requirements\_app.txt

│

├── partie1\_cnn\_mnist/

│   ├── create\_test\_image.py

│   ├── mnist\_cnn.py

│   ├── test\_image.py

│   ├── test\_digit.png

│   ├── requirements.txt

│   └── outputs/

│

├── partie2\_ventes/

│   ├── data/

│   │   ├── sales\_data.csv

│   │   ├── train\_data.csv

│   │   └── test\_data.csv

│   │

│   ├── src/

│   │   ├── generate\_data.py

│   │   ├── data\_preparation.py

│   │   ├── eda\_analysis.py

│   │   ├── model\_arima.py

│   │   ├── model\_prophet.py

│   │   ├── model\_lstm.py

│   │   ├── comparison.py

│   │   └── forecasting.py

│   │

│   └── outputs/

│

├── .gitignore

└── README.md

```



Les dossiers `outputs/` sont générés localement pendant l'exécution et sont exclus du dépôt Git.



\---



\# 9. Technologies utilisées



\### Programmation



\- Python

\- NumPy

\- Pandas



\### Machine Learning / Deep Learning



\- TensorFlow / Keras

\- Scikit-learn



\### Séries temporelles



\- Statsmodels

\- Prophet



\### Visualisation



\- Matplotlib



\### Application



\- Streamlit



\### Gestion de versions



\- Git

\- GitHub



\---



\# 10. Installation



Cloner le repository :



```bash

git clone https://github.com/sifeddineftoumi/ENSA-Python-ML-Project.git

cd ENSA-Python-ML-Project

```



Créer un environnement virtuel :



```bash

python -m venv .venv

```



Activer l'environnement sous Windows PowerShell :



```powershell

.venv\\Scripts\\Activate.ps1

```



Installer les dépendances de la partie MNIST :



```powershell

pip install -r partie1\_cnn\_mnist\\requirements.txt

```



Installer les dépendances de la partie prévision :



```powershell

pip install -r partie2\_ventes\\requirements.txt

```



Pour l'application :



```powershell

pip install -r app\_presentation\\requirements\_app.txt

```



\---



\# 11. Exécution



\## MNIST



Depuis la racine du projet :



```powershell

python partie1\_cnn\_mnist\\mnist\_cnn.py

```



Les scripts de test peuvent ensuite être utilisés pour effectuer des prédictions sur des images.



\---



\## Prévision des ventes



\### Génération des données



```powershell

python partie2\_ventes\\src\\generate\_data.py

```



\### Préparation



```powershell

python partie2\_ventes\\src\\data\_preparation.py

```



\### Analyse exploratoire



```powershell

python partie2\_ventes\\src\\eda\_analysis.py

```



\### SARIMA



```powershell

python partie2\_ventes\\src\\model\_arima.py

```



\### Prophet



```powershell

python partie2\_ventes\\src\\model\_prophet.py

```



\### LSTM



```powershell

python partie2\_ventes\\src\\model\_lstm.py

```



\### Comparaison



```powershell

python partie2\_ventes\\src\\comparison.py

```



\### Prévision des 30 prochains jours



```powershell

python partie2\_ventes\\src\\forecasting.py

```



\---



\# 12. Reproductibilité et gestion des fichiers générés



Les modèles entraînés et certains fichiers intermédiaires ne sont pas inclus dans le repository.



Cela concerne notamment :



```text

\*.h5

\*.keras

\*.pkl

\*.npy

```



ainsi que les résultats générés dans :



```text

partie1\_cnn\_mnist/outputs/

partie2\_ventes/outputs/

```



Ces fichiers peuvent être régénérés en exécutant les scripts correspondants.



Les environnements Python locaux (`.venv/` et `.venv-1/`) sont également exclus du dépôt.



\---



\# 13. Limites



La partie prévision des ventes utilise un jeu de données synthétique créé pour les besoins du projet.



Les performances obtenues ne peuvent donc pas être interprétées comme une mesure de performance sur des données commerciales réelles.



Par ailleurs, les modèles ont été évalués avec des configurations définies dans le cadre du projet. La comparaison présentée ne constitue pas une recherche exhaustive d'hyperparamètres.



\---



\# 14. Perspectives



Plusieurs améliorations peuvent être envisagées :



\- utiliser des données réelles lorsque celles-ci sont disponibles ;

\- effectuer une recherche systématique d'hyperparamètres ;

\- tester d'autres modèles de séries temporelles ;

\- améliorer l'application de visualisation ;

\- mettre en place une automatisation complète du pipeline ;

\- déployer l'application ;

\- ajouter des mécanismes de suivi des performances des modèles.



\---



\## Auteur



\*\*Sif Eddine Toumi\*\*



ENSA Berrechid — Génie Informatique



Projet Python / Machine Learning — 2025-2026

