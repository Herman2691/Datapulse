# 🎓 Plateforme d'apprentissage du ML : prototype du module 1

Prototype du module 1 « Ma première classification » du plan `plan-plateforme-apprentissage-ml.md` (section 4). L'étudiant manipule dans DATAPULSE, lit le code, complète des exercices à trous, écrit seul l'analyse, puis compare son résultat avec DATAPULSE.

Tout s'exécute **dans le navigateur** (JupyterLite / Pyodide) : rien à installer pour l'étudiant, aucun serveur de calcul.

**▶️ Ouvrir le module 1 en ligne** : https://herman2691.github.io/Datapulse/lab/index.html?path=module1_premiere_classification.ipynb

## Contenu

```
plateforme/
├── construire_site.py                        # Construit le site JupyterLite (dossier _site)
├── requirements-site.txt                     # Versions testées pour construire le site
├── overrides.json                            # Interface JupyterLite en français par défaut
└── module1/
    ├── module1_premiere_classification.ipynb # Étapes A à D : leçon, quiz, DATAPULSE, lecture du code, code à trous
    ├── module1_defi.ipynb                    # Étapes E à G : défi, comparaison avec DATAPULSE, bilan
    ├── module1_outils.py                     # Quiz, vérifications, indices, progression, bilan
    ├── cancer_du_sein.csv                    # Données du module (à charger dans DATAPULSE)
    ├── reference_module1.json                # Référence DATAPULSE pour la comparaison
    └── generer_reference.py                  # Régénère le CSV et la référence avec DATAPULSE
```

`comparateur.py`, à la racine du dépôt, est ajouté au site lors de la construction.

## Déroulé pour l'étudiant

| Étape | Où | Ce qui est vérifié automatiquement |
|---|---|---|
| A. Leçon et quiz | notebook 1 | 3 questions, explication pour chaque réponse ; seule la première réponse compte |
| B. Manipulation | DATAPULSE en ligne | réponses libres enregistrées (non notées) |
| C. Lire le code | notebook 1 | — (réponses aux questions dépliables) |
| D1. Découpage | notebook 1 | taille du test, `stratify`, graine, ordre des sorties, `test_size=20` au lieu de `0.2` |
| D2. Modèle | notebook 1 | Pipeline, `StandardScaler`, entraîné, sur le bon jeu (pas sur tout X ni sur le test), hyperparamètres |
| D3. Évaluation | notebook 1 | prédiction sur le test, accuracy (pas sur le train), matrice de confusion, rappel des tumeurs malignes |
| E. Défi | notebook 2 (vide) | consigne respectée, sur le cancer du sein ou le jeu de données du projet fil rouge |
| F. Comparaison | notebook 2 | `comparateur.py` : même jeu de test, mêmes prédictions, probabilités à 1e-4 près, pistes en cas d'écart |
| G. Bilan | notebook 2 | critères de la section 4.3 du plan, tentatives et indices par exercice, journal de bord |

Chaque exercice à trous propose 3 niveaux d'indices ; la solution se débloque après 3 tentatives. Les erreurs Python fréquentes (`...` oubliés, cellule non exécutée, modèle non entraîné) sont suivies d'un conseil en français.

La progression est enregistrée dans `progression_module1.json`, le journal dans `journal_de_bord.md`. Dans JupyterLite, ces fichiers restent **dans le navigateur de l'étudiant**.

## Construire et tester le site

Dans un environnement virtuel séparé de celui de DATAPULSE :

```bash
pip install -r plateforme/requirements-site.txt
python plateforme/construire_site.py
python -m http.server -d plateforme/_site 8000
```

Puis ouvrir http://localhost:8000. Le dossier `plateforme/_site` est un site statique : il se publie tel quel sur GitHub Pages, Netlify, etc.

### Publication automatique

Le workflow `.github/workflows/plateforme-pages.yml` construit le site et le publie sur GitHub Pages à chaque modification de `plateforme/` ou de `comparateur.py` dans `master`. On peut aussi le lancer à la main depuis l'onglet *Actions* (« Run workflow »).

Prérequis, une seule fois : *Settings → Pages → Build and deployment → Source : GitHub Actions*.

Sans JupyterLite, les notebooks s'ouvrent aussi dans Jupyter classique, depuis le dossier `plateforme/module1` (choisir le noyau Python 3).

## Vérifications effectuées

- **Dans le navigateur** (Chrome, JupyterLite, Pyodide 314.0.6 : Python 3.14, scikit-learn 1.8.0, pandas 3.0) : parcours complet jusqu'à « Module 1 validé », interface en français, fichiers de progression enregistrés.
- **Écart navigateur / DATAPULSE** : la référence a été produite avec scikit-learn 1.7.2, le navigateur utilise la 1.8.0 ; les probabilités diffèrent de 2,7·10⁻¹⁵, très en dessous de la tolérance (10⁻⁴).
- **Étudiants simulés** (Jupyter classique) : notebook non rempli, `test_size=20`, `stratify` oublié, mauvaise graine, sorties dans le désordre, modèle sans Pipeline, non entraîné, entraîné sur toutes les données, `C=0.1`, prédiction sur le train, accuracy sur le train, défi avec une autre graine, défi vide, réponses libres laissées vides.

## Limites du prototype

- **Pas de comptes** : la progression est propre à un navigateur. Vider les données du site ou changer d'ordinateur la fait perdre. Les comptes (plan, section 9) sont l'étape suivante.
- **Anti-triche minimal** : le défi démarre dans un notebook vide, mais rien n'empêche de recopier le notebook 1 ou les solutions.
- **Versions** : si DATAPULSE ou le navigateur changent de version de scikit-learn, relancer `generer_reference.py` et la vérification (`demarrer()` affiche un avertissement si les versions diffèrent).
- **Premier chargement** : la première cellule télécharge Python et scikit-learn dans le navigateur (quelques dizaines de secondes).
- Avec ce jeu de données, la matrice de confusion est symétrique (`[[41, 1], [1, 71]]`) : une matrice écrite avec `y_test` et `y_pred` inversés ne peut pas être détectée.
