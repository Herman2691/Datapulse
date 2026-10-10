# 🚀 DATAPULSE Pro

<div align="center">

![Version](https://img.shields.io/badge/version-3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.49+-red.svg)

**Plateforme Professionnelle d'Analyse ML & Data Science**

Une application web complète pour l'analyse de données et le Machine Learning, développée avec Streamlit et Scikit-learn.

[Fonctionnalités](#-fonctionnalités) • [Installation](#-installation) • [Utilisation](#-utilisation) • [Documentation](#-documentation)

</div>

---

## 📋 Table des Matières

- [À Propos](#-à-propos)
- [Fonctionnalités](#-fonctionnalités)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du Projet](#-structure-du-projet)
- [Algorithmes Disponibles](#-algorithmes-disponibles)
- [Documentation](#-documentation)
- [Contribution](#-contribution)
- [Auteur](#-auteur)
- [Licence](#-licence)

---

## 🎯 À Propos

**DATAPULSE Pro** est une plateforme web professionnelle conçue pour faciliter l'analyse de données et l'application d'algorithmes de Machine Learning. Elle offre une interface intuitive et moderne permettant aux data scientists, analystes et chercheurs de réaliser des analyses complètes sans écrire de code.

### Points Forts

- ✅ **Interface Intuitive** : Design moderne et professionnel avec navigation fluide
- ✅ **Analyses Complètes** : EDA, ML supervisé, clustering, analyses factorielles
- ✅ **Support Format** : CSV et Excel (.xlsx, .xls)
- ✅ **Reproductible** : découpage réglable, paramètres exportés, code scikit-learn équivalent
- ✅ **Visualisations Avancées** : Graphiques interactifs et informatifs
- ✅ **Export Flexible** : CSV, Excel, JSON
- ✅ **Historique Intégré** : Suivi de toutes vos analyses

---

## ✨ Fonctionnalités

### 📊 Analyse Exploratoire des Données (EDA)

- **Statistiques Descriptives Complètes**
  - Moyennes, médianes, écarts-types
  - Valeurs manquantes et uniques
  - Distribution des types de données
  
- **Matrices de Corrélation**
  - Heatmaps interactives
  - Identification des corrélations fortes
  - Test de Pearson
  
- **Analyses Bivariées**
  - Scatter plots avec régression
  - Violin plots
  - Heatmaps de contingence
  - Tests statistiques (Chi², corrélation)

### 🎯 Machine Learning Supervisé

#### Classification
- 🌳 **Random Forest** : Ensemble d'arbres de décision
- 🎯 **KNN** : K-Nearest Neighbors
- 🧠 **RNA** : Réseaux de neurones (MLPClassifier)
- ⚡ **SVM** : Support Vector Machine
- 📈 **Régression Logistique**
- 🌲 **Arbre de Décision**

#### Régression
- 🌳 **Random Forest Regressor**
- 🎯 **KNN Regressor**
- 📉 **Régression Linéaire**
- 🌲 **Arbre de Décision Regressor**

#### Évaluation des Modèles
- Matrices de confusion (brutes et normalisées)
- Courbes d'apprentissage automatiques
- Validation croisée (5-fold)
- Métriques complètes (Accuracy, R², MSE, MAE, etc.)
- Importance des features
- Visualisation des arbres de décision

### 🔍 Machine Learning Non-Supervisé

#### Clustering
- **K-Means**
  - Méthode du coude automatique
  - Score Silhouette
  - Projection ACP 2D
  
- **CAH** (Classification Hiérarchique Ascendante)
  - Dendrogramme interactif
  - Méthode de Ward
  - Seuils de coupure suggérés

#### Réduction de Dimension
- **ACP** (Analyse en Composantes Principales)
  - Scree plot
  - Variance expliquée
  - Projections 2D/3D
  - Contribution des variables

### 🎓 Reproductibilité et Apprentissage

- **Réglages du découpage** : proportion du jeu de test (10 à 40 %) et graine aléatoire, sur les pages Modélisation et Comparaison
- **Prétraitement sans fuite de données** : imputation, normalisation et encodage dans un `Pipeline` scikit-learn, appris sur le jeu d'entraînement uniquement (et réappris dans chaque pli de validation croisée)
- **💻 Montre-moi le code** : après chaque entraînement, le code scikit-learn commenté qui reproduit exactement l'analyse, avec les résultats attendus
- **Export des paramètres (JSON)** : versions, empreinte des données, découpage, prétraitement, hyperparamètres
- **Référence pour comparaison (JSON)** : paramètres, métriques et prédictions ligne par ligne du jeu de test, pour comparer le résultat d'un code écrit à la main
- **🎓 Comparer mon code** : déposez le `mes_resultats.json` produit par votre propre code ; DATAPULSE le compare au dernier entraînement (ou à une référence importée) et affiche le verdict, chaque contrôle et des pistes en cas d'écart

### 📊 Analyses Factorielles

- **ACM** (Analyse des Correspondances Multiples)
  - Pour variables catégorielles multiples
  - Plan factoriel des individus
  - Inertie expliquée par dimension
  
- **AFC** (Analyse Factorielle des Correspondances)
  - Pour 2 variables catégorielles
  - Table de contingence interactive
  - Test du Chi²
  - Projection lignes/colonnes

### 🧪 Test et Prédiction

- Interface de test interactive
- Saisie manuelle des features
- Prédiction en temps réel
- Probabilités de classification
- Résultats visuels élégants

### 📈 Fonctionnalités Avancées

- **Comparaison de Modèles**
  - Évaluation parallèle de plusieurs algorithmes
  - Classement automatique
  - Graphiques comparatifs
  
- **Historique des Analyses**
  - Suivi de toutes les analyses effectuées
  - Statistiques d'utilisation
  - Visualisations de l'historique
  
- **Export Professionnel**
  - Export CSV
  - Export Excel avec feuilles multiples
  - Export JSON de l'historique
  - Paramètres, code Python et référence de comparaison du dernier modèle
  - Sauvegarde de modèles (en développement)

---

## 🛠 Technologies

### Backend
- **Python 3.8+**
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques
- **Scikit-learn** : Algorithmes ML
- **SciPy** : Analyses statistiques avancées

### Frontend
- **Streamlit** : Framework web
- **Plotly** : Graphiques interactifs
- **Matplotlib & Seaborn** : Visualisations statiques

### Traitement de Données
- **OpenPyXL** : Support Excel
- **StandardScaler** : Normalisation
- **OneHotEncoder** : Encodage catégoriel
- **SimpleImputer** : Gestion des valeurs manquantes

---

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de packages Python)

### Étapes d'Installation

1. **Cloner le repository**
```bash
git clone https://github.com/Herman2691/Datapulse.git
cd Datapulse
```

2. **Créer un environnement virtuel (recommandé)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

### Fichier requirements.txt

```txt
streamlit>=1.49.0
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
Pillow>=10.0.0
```

---

## 💻 Utilisation

### Démarrage de l'Application

```bash
streamlit run Datapulse.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse : `http://localhost:8501`

### Workflow Typique

#### 1️⃣ Chargement des Données
- Accédez à **"📂 Chargement & EDA"**
- Uploadez votre fichier CSV ou Excel
- Visualisez l'aperçu des données

#### 2️⃣ Exploration des Données
- Consultez les statistiques descriptives
- Générez la matrice de corrélation
- Effectuez des analyses bivariées

#### 3️⃣ Définir la Variable Cible
- Sélectionnez la colonne à prédire
- Vérifiez la distribution de la cible

#### 4️⃣ Modélisation Supervisée
- Allez dans **"🎯 Modélisation Supervisée"**
- Choisissez le type d'analyse (Classification/Régression)
- Sélectionnez un algorithme
- Réglez si besoin la proportion de test et la graine aléatoire
- Lancez l'entraînement

#### 5️⃣ Évaluation
- Consultez les métriques de performance
- Analysez la courbe d'apprentissage
- Vérifiez la matrice de confusion (classification)
- Ouvrez **« 💻 Montre-moi le code »** pour voir le code scikit-learn équivalent

#### 6️⃣ Test du Modèle
- Accédez à **"🧪 Test du Modèle"**
- Saisissez les valeurs des features
- Obtenez une prédiction en temps réel

#### 🎓 Comparer mon code (Optionnel)
- Refaites l'analyse avec votre propre code scikit-learn (même proportion de test, même graine)
- Téléchargez `comparateur.py` depuis la page **"🎓 Comparer mon code"** et ajoutez `sauvegarder_resultats(X_test, y_pred, modele)` à la fin de votre code
- Déposez `mes_resultats.json` : DATAPULSE indique si votre résultat est identique et, sinon, quoi vérifier

#### 7️⃣ Analyses Non-Supervisées (Optionnel)
- Clustering pour découvrir des groupes
- ACP pour réduire la dimensionnalité
- ACM/AFC pour variables catégorielles

#### 8️⃣ Export
- Allez dans **"📤 Export"**
- Téléchargez vos résultats
- Sauvegardez l'historique
- Récupérez le code, les paramètres et la référence du dernier modèle (onglet **💾 Modèle**)

---

## 📁 Structure du Projet

```
Datapulse/
│
├── Datapulse.py              # Application Streamlit principale
├── data_processor.py         # Module de traitement ML
├── code_generator.py         # Génération du code scikit-learn équivalent
├── comparateur.py            # Comparaison d'un résultat étudiant avec la référence
├── plateforme/               # Prototype de la plateforme d'apprentissage (module 1, JupyterLite)
├── requirements.txt          # Dépendances Python
└── README.md                 # Documentation
```

### Description des Fichiers Principaux

#### `Datapulse.py`
Interface utilisateur Streamlit avec :
- Navigation par pages
- Design professionnel moderne
- Gestion de l'état de session
- Routage des fonctionnalités

#### `data_processor.py`
Classe `DataProcessor` gérant :
- Chargement de données (CSV/Excel)
- Prétraitement automatique dans un `Pipeline` (sans fuite de données)
- Export des paramètres exacts et de la référence de comparaison
- Entraînement de modèles
- Analyses statistiques
- Visualisations avancées
- Clustering et réduction de dimension
- ACM et AFC

#### `code_generator.py`
Génère, à partir des paramètres exportés en JSON, le code Python commenté qui reproduit exactement un entraînement. Utilisable aussi en ligne de commande :
```bash
python code_generator.py parametres.json
```

#### `comparateur.py`
Compare le résultat d'un code écrit à la main avec la référence exportée par DATAPULSE : même jeu de test, mêmes prédictions, mêmes probabilités (tolérances du plan). En cas d'écart, il ne dit pas seulement « faux » : il propose des pistes (graine, `test_size`, `stratify`, normalisation oubliée ou ajustée avant le découpage, hyperparamètres).

À la fin du code de l'étudiant :
```python
from comparateur import sauvegarder_resultats
sauvegarder_resultats(X_test, y_pred, modele)   # crée mes_resultats.json
```
Puis, dans DATAPULSE (page **🎓 Comparer mon code**) ou en ligne de commande :
```bash
python comparateur.py reference.json mes_resultats.json
```

#### `plateforme/`
Prototype du module 1 de la plateforme d'apprentissage : deux notebooks (leçon, quiz, code à trous avec indices, défi, comparaison avec DATAPULSE, bilan) qui s'exécutent dans le navigateur avec JupyterLite. Voir [plateforme/README.md](plateforme/README.md).

---

## 🤖 Algorithmes Disponibles

### Classification (6 algorithmes)

| Algorithme | Description | Cas d'Usage |
|-----------|-------------|-------------|
| **Random Forest** | Ensemble d'arbres de décision | Excellence générale, robuste |
| **KNN** | K plus proches voisins | Données non-linéaires |
| **RNA (MLP)** | Réseau de neurones | Patterns complexes |
| **SVM** | Machine à vecteurs de support | Séparation non-linéaire |
| **Régression Logistique** | Modèle linéaire probabiliste | Baseline rapide |
| **Arbre de Décision** | Arbre de règles | Interprétabilité |

### Régression (4 algorithmes)

| Algorithme | Description | Cas d'Usage |
|-----------|-------------|-------------|
| **Random Forest Reg** | Ensemble d'arbres | Relations non-linéaires |
| **KNN Reg** | K plus proches voisins | Patterns locaux |
| **Régression Linéaire** | Modèle linéaire | Baseline, relations simples |
| **Arbre de Décision Reg** | Arbre de règles | Seuils de décision |

### Clustering (2 méthodes)

| Méthode | Description | Caractéristiques |
|---------|-------------|------------------|
| **K-Means** | Partitionnement centroïde | Rapide, sphérique |
| **CAH** | Hiérarchique Ward | Dendrogramme, flexible |

### Analyses Factorielles (3 méthodes)

| Méthode | Type | Variables |
|---------|------|-----------|
| **ACP** | Réduction dimension | Quantitatives |
| **ACM** | Correspondances multiples | Catégorielles multiples |
| **AFC** | Correspondances simples | 2 catégorielles |

---

## 📚 Documentation

### Configuration des Modèles

Les hyperparamètres par défaut sont optimisés pour la plupart des cas d'usage :

```python
# Random Forest (Classification)
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# KNN
KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='euclidean'
)
```

### Prétraitement Automatique

Le système applique automatiquement, dans un `Pipeline` appris sur le jeu d'entraînement uniquement :
- **Imputation** : Médiane (numériques et booléennes), mode (catégorielles)
- **Normalisation** : StandardScaler pour features numériques (les booléens deviennent 0/1)
- **Encodage** : OneHotEncoder pour features catégorielles
- **Split** : 80 % train / 20 % test par défaut (réglable de 10 à 40 %), stratifié en classification, graine 42 par défaut (réglable)

Les colonnes d'un autre type (dates lues depuis Excel, par exemple) sont ignorées et signalées dans les paramètres exportés. Une date lue depuis un CSV est du texte : elle est traitée comme une catégorie.

### Format des Données

#### CSV
```csv
feature1,feature2,feature3,target
1.5,catA,10,classe1
2.3,catB,15,classe2
```

#### Excel
- Feuille unique
- Première ligne = en-têtes
- Pas de cellules fusionnées

---

## 🤝 Contribution

Les contributions sont les bienvenues ! 

### Comment Contribuer

1. **Fork** le projet
2. Créez une **branche** (`git checkout -b feature/AmazingFeature`)
3. **Committez** vos changements (`git commit -m 'Add AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une **Pull Request**

### Guidelines

- Code Python conforme à PEP 8
- Commentaires en français
- Tests unitaires pour nouvelles fonctionnalités
- Documentation mise à jour

---

## 👨‍💻 Auteur

**Herman Kandolo**
- Chercheur en Intelligence Artificielle et Data Science
- 📧 Email : hermankandolo2022@gmail.com
- 🔗 LinkedIn : [Herman Kandolo](https://linkedin.com/in/herman-kandolo-209b73364)
- 🐙 GitHub : [@hermankandolo](https://github.com/Herman2691)

---

## 📄 Licence

Ce projet est sous licence **MIT** - voir le fichier [LICENSE](LICENSE) pour plus de détails.

```
MIT License

Copyright (c) 2025 Herman Kandolo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🙏 Remerciements

- **Streamlit** pour le framework web
- **Scikit-learn** pour les algorithmes ML
- **Plotly** pour les visualisations interactives
- La communauté **Python** pour les outils open-source

---

## 📞 Support

Pour toute question ou problème :
- 🐛 Ouvrir une [Issue](https://github.com/Herman2691/Datapulse/issues)
- 💬 Démarrer une [Discussion](https://github.com/Herman2691/Datapulse/discussions)
- 📧 Contact direct : hermanKandolo2022@gmail.com

---
- tester l'application en ligne : https://datapulse-t69vfezujmrcobcdsgrpt3.streamlit.app/
## 🗺 Roadmap

### Version 3.1 (Q2 2025)
- [ ] Sauvegarde et chargement de modèles (.pkl)
- [ ] Support de nouveaux formats (Parquet, JSON)
- [ ] Optimisation automatique des hyperparamètres
- [ ] Dashboard de monitoring temps réel

### Version 4.0 (Q3 2025)
- [ ] Deep Learning (TensorFlow/PyTorch)
- [ ] Séries temporelles (ARIMA, Prophet)
- [ ] NLP et Text Mining
- [ ] API REST pour intégration

---

<div align="center">

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐**

Made with ❤️ by Herman Kandolo

</div>
