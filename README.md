# Télécom - Prédiction du Churn Client avec Deep Learning

## Description du Projet

Ce projet vise à prédire le churn (départ) des clients dans une entreprise de télécommunications en utilisant les techniques du Deep Learning. Le modèle est construit avec TensorFlow/Keras et entraîné sur un ensemble de données contenant les informations des clients et leur statut de départ.

L'objectif principal est de développer un modèle capable d'identifier les clients susceptibles de partir, permettant ainsi à l'entreprise de mettre en place des stratégies de rétention proactives.

---

## Objectifs

- Analyser les données clients pour identifier les facteurs influençant le churn
- Préparer et normaliser les données pour le machine learning
- Construire un réseau de neurones profond pour la classification binaire
- Évaluer les performances du modèle avec différentes métriques
- Fournir des prédictions exploitables pour la rétention client

---

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les éléments suivants :

### Bibliothèques Python requises

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Dépendances système

- Python 3.7 ou supérieur
- Jupyter Notebook ou Google Colab (recommandé)
- 4 GB de RAM minimum
- Connexion internet (pour télécharger les bibliothèques)

### Fichiers de données requis

- `Telco-Customer-Churn.csv` : Ensemble de données contenant les informations des clients

---

## Structure du Projet

Le notebook est organisé en plusieurs sections principales :

### 1. Importation des Bibliothèques
- **pandas** : Manipulation et analyse des données tabulaires
- **numpy** : Opérations numériques et calculs sur les tableaux
- **matplotlib & seaborn** : Visualisation des données
- **scikit-learn** : Preprocessing, évaluation des modèles
- **tensorflow.keras** : Construction et entraînement du réseau de neurones

### 2. Analyse Exploratoire des Données (EDA)

Cette section comprend :
- Affichage des premières lignes du dataset
- Inspection des colonnes et types de données
- Vérification des valeurs manquantes
- Statistiques descriptives
- Analyse de la distribution du churn

### 3. Nettoyage et Prétraitement des Données

Les étapes incluent :
- Suppression des colonnes non pertinentes
- Conversion des types de données
- Gestion des valeurs manquantes
- Encodage des variables catégorielles (One-Hot Encoding)
- Normalisation des caractéristiques avec StandardScaler

### 4. Division des Données

- Séparation en ensemble d'entraînement (80%) et test (20%)
- Normalisation indépendante des deux ensembles
- Équilibre des classes avec class_weight

### 5. Construction du Modèle

Le réseau de neurones comprend :
- **Couche d'entrée** : Accepte 31 caractéristiques
- **Couche cachée 1** : 128 neurones + activation ReLU + Dropout (40%)
- **Couche cachée 2** : 64 neurones + activation ReLU + Dropout (30%)
- **Couche cachée 3** : 32 neurones + activation ReLU
- **Couche de sortie** : 1 neurone + activation Sigmoid (classification binaire)

### 6. Compilation et Entraînement

Configuration :
- **Optimiseur** : Adam
- **Fonction de perte** : Binary Crossentropy
- **Métrique** : Accuracy
- **Callbacks** : Early Stopping pour éviter le surapprentissage

### 7. Évaluation du Modèle

Métriques utilisées :
- Classification Report (Précision, Recall, F1-Score)
- Matrice de Confusion
- Courbe ROC et AUC
- Historique d'entraînement (perte et exactitude)

---

## Utilisation

### Installation et Configuration

1. **Cloner le repository** :
```bash
git clone https://github.com/hervoz/Telecom-churm-Traitement-DL.git
cd Telecom-churm-Traitement-DL
```

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

3. **Préparer les données** :
   - Téléchargez le fichier `Telco-Customer-Churn.csv`
   - Placez-le dans le répertoire du projet

4. **Exécuter le notebook** :
   - Ouvrez `ChurnTraitement.ipynb` dans Jupyter Notebook ou Google Colab
   - Exécutez les cellules dans l'ordre

### Exécution Pas à Pas

1. **Importer les bibliothèques** : Exécute l'installation de toutes les dépendances
2. **Charger et explorer les données** : Analyse exploratoire et statistiques
3. **Nettoyer les données** : Préparation pour le modèle
4. **Normaliser les données** : Mise à l'échelle des caractéristiques
5. **Entraîner le modèle** : Construction et entraînement du réseau de neurones
6. **Visualiser les résultats** : Graphiques de l'historique d'entraînement
7. **Évaluer le modèle** : Calcul des performances

---

## Résultats

### Caractéristiques du Dataset

- **Nombre d'observations** : 7 043 clients
- **Nombre de caractéristiques** : 31 après encodage
- **Distribution du churn** :
  - Pas de churn (Classe 0) : 73.46%
  - Churn (Classe 1) : 26.54%

### Performance du Modèle

Le modèle atteint les performances suivantes :
- **Précision** : Faible taux de faux positifs
- **Recall** : Identification efficace des clients à risque
- **F1-Score** : Équilibre entre précision et recall
- **AUC-ROC** : Mesure globale de discrimination

### Visualisations

Le notebook génère plusieurs visualisations :
- Courbes de perte et d'exactitude pendant l'entraînement
- Distribution du churn
- Matrice de confusion
- Courbe ROC

---

## Fonctionnalités Principales

### Gestion du Déséquilibre des Données

Utilisation de `class_weight="balanced"` pour gérer le déséquilibre entre les classes.

### Régularisation

- **Dropout layers** : Prévention du surapprentissage
- **Early Stopping** : Arrêt de l'entraînement si la validation n'améliore plus

### Normalisation Appropriée

- Utilisation de `StandardScaler` séparément sur l'ensemble d'entraînement et test
- Évite la fuite d'informations entre les ensembles

---

## Améliorations Futures

- Implémenter d'autres architectures (LSTM, GRU)
- Effectuer une recherche en grille (GridSearch) pour optimiser les hyperparamètres
- Ajouter de nouvelles caractéristiques (feature engineering)
- Tester d'autres algorithmes (Random Forest, XGBoost)
- Déployer le modèle dans un environnement de production
- Mettre en place un système de monitoring

---

## Fichiers du Projet

```
Telecom-churm-Traitement-DL/
├── README.md                    # Ce fichier
├── ChurnTraitement.ipynb        # Notebook principal
├── Telco-Customer-Churn.csv     # Dataset
└── requirements.txt             # Dépendances Python
```

---

## Auteur

- **Auteur** : Akounda Abner
- **Date** : 2024
- **Repository** : [GitHub - Telecom-churm-Traitement-DL](https://github.com/hervoz/Telecom-churm-Traitement-DL)

---

## Licence

Ce projet est fourni à titre éducatif. Veuillez consulter les termes de licence appropriés.

---

## Notes Importantes

### Pour Google Colab

Le notebook est conçu pour être exécuté sur Google Colab. Assurez-vous de :
- Disposer d'une connexion Google Drive fonctionnelle
- Charger le fichier CSV dans Google Colab
- Disposer de quotas GPU suffisants pour l'entraînement

### Optimisation du Modèle

Pour améliorer les performances :
1. Ajustez les hyperparamètres (nombre de neurones, taux Dropout)
2. Augmentez le nombre d'epochs
3. Modifiez la taille des batches
4. Testez différents taux d'apprentissage

### Ressources Recommandées

- Documentation TensorFlow : https://www.tensorflow.org/
- Scikit-learn : https://scikit-learn.org/
- Machine Learning Basics : https://en.wikipedia.org/wiki/Machine_learning
