"""
cours.py - Contenu théorique des modules de la plateforme d'apprentissage

Affiché dans la page « 📚 Cours » de DATAPULSE. Les exercices de code de chaque module
sont dans les notebooks de plateforme/ (JupyterLite, publiés sur GitHub Pages).

Auteur: Herman Kandolo
"""

import os

URL_PLATEFORME = "https://herman2691.github.io/Datapulse/lab/index.html?path="

# Parcours complet (plan, section 2) : seuls les modules avec un contenu sont ouverts
PARCOURS = [
    {"numero": 0, "titre": "Mise à niveau Python", "contenu": "Variables, listes, fonctions, pandas de base"},
    {"numero": 1, "titre": "Première classification", "contenu": "Régression logistique, découpage, métriques de base"},
    {"numero": 2, "titre": "Arbres, forêts et KNN", "contenu": "Comparer plusieurs algorithmes, hyperparamètres"},
    {"numero": 3, "titre": "Régression", "contenu": "Régression linéaire, MSE, MAE, R²"},
    {"numero": 4, "titre": "Évaluer correctement", "contenu": "Validation croisée, surapprentissage, fuite de données"},
    {"numero": 5, "titre": "Pipelines propres", "contenu": "Imputation, normalisation, encodage dans un Pipeline"},
    {"numero": 6, "titre": "Non supervisé", "contenu": "K-means, ACP, CAH"},
    {"numero": 7, "titre": "Sauvegarder et servir un modèle", "contenu": "joblib, API REST"},
    {"numero": 8, "titre": "Déployer et surveiller", "contenu": "Conteneurisation, hébergement, suivi"},
]

MODULE_1 = {
    "titre": "Module 1 : Ma première classification",
    "duree": "environ 2 h 30 (cours, manipulation, exercices et défi)",
    "donnees": os.path.join(os.path.dirname(os.path.abspath(__file__)), "plateforme", "module1", "cancer_du_sein.csv"),
    "cible": "target",
    "notebook_exercices": URL_PLATEFORME + "module1_premiere_classification.ipynb",
    "notebook_defi": URL_PLATEFORME + "module1_defi.ipynb",
    "objectifs": [
        "Expliquer ce qu'est un problème de classification (variables explicatives, cible).",
        "Expliquer pourquoi on sépare les données en jeu d'entraînement et jeu de test.",
        "Entraîner une régression logistique et obtenir une prédiction.",
        "Lire une matrice de confusion et calculer l'accuracy, la précision et le rappel.",
        "Reproduire en code un résultat obtenu dans DATAPULSE.",
    ],
}

LECON_MODELE = """
Un **modèle** est une fonction apprise à partir d'exemples. On lui montre des centaines de cas
dont on connaît la réponse ; il en tire une règle qu'il applique ensuite à des cas nouveaux.

Notre jeu de données décrit **569 tumeurs du sein**. Pour chacune :

- **30 mesures** prises sur une image au microscope (rayon, texture, périmètre…) :
  ce sont les **variables explicatives**, notées `X` ;
- le **diagnostic** : maligne (`0`) ou bénigne (`1`). C'est la **cible**, notée `y` :
  ce que le modèle doit prédire.

Prédire une **catégorie** (maligne ou bénigne) s'appelle une **classification**.
Prédire un **nombre** (un prix, une température) s'appelle une **régression** (module 3).
"""

LECON_TEST = """
Un modèle qui récite ses exemples ne prouve rien, comme un élève qui apprend par cœur les corrigés.
On met donc de côté une partie des données, **le jeu de test**, que le modèle ne voit jamais pendant
l'apprentissage. Son score sur ce jeu dit s'il sait **généraliser** à des cas nouveaux.

Deux réglages comptent :

- **la proportion de test** (souvent 20 %) : plus de test donne une évaluation plus fiable,
  mais laisse moins d'exemples pour apprendre ;
- **la graine aléatoire** (`random_state`) : le découpage est tiré au hasard ; fixer la graine
  rend ce hasard **reproductible** (même graine, mêmes lignes de test, mêmes résultats).
"""

LECON_LOGISTIQUE = """
La régression logistique calcule un **score** à partir des variables (une somme pondérée), puis le
transforme en **probabilité** entre 0 et 1 grâce à la fonction sigmoïde. Elle applique enfin un
**seuil** : au-dessus, elle prédit la classe 1 (bénigne) ; en dessous, la classe 0 (maligne).

Déplace le seuil pour voir comment la décision change.
"""

LECON_MATRICE = """
La **matrice de confusion** compte les bonnes et les mauvaises réponses du modèle sur le jeu de test.
Les **lignes** sont les vraies classes, les **colonnes** les prédictions.

- **accuracy** = part de prédictions correctes ;
- **rappel** (des tumeurs malignes) = sur toutes les tumeurs malignes, combien le modèle en a trouvé ;
- **précision** (des tumeurs malignes) = quand le modèle dit « maligne », à quel point il a raison.

Pour un cancer, un **faux négatif** (une tumeur maligne prédite bénigne) est l'erreur la plus grave :
on regarde donc de près le **rappel** des tumeurs malignes.
"""

# Matrice obtenue par DATAPULSE sur le module (test 20 %, graine 42) : [[VP, FN], [FP, VN]] pour la classe maligne
MATRICE_MODULE = [[41, 1], [1, 71]]

CODE_MODULE = '''# 1. Charger les données
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# 2. Séparer entraînement / test (même réglage que DATAPULSE)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Pipeline : normalisation + modèle (la normalisation est
#    ajustée sur le train uniquement, pour éviter toute fuite)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

modele = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

# 4. Entraîner
modele.fit(X_train, y_train)

# 5. Évaluer sur le jeu de test
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = modele.predict(X_test)
print("Accuracy :", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))'''

QUESTIONS_CODE = [
    ("Pourquoi `stratify=y` ?",
     "Il y a 212 tumeurs malignes pour 357 bénignes (37 % / 63 %). Sans stratification, le hasard peut "
     "mettre trop de tumeurs malignes d'un côté. `stratify=y` garde les mêmes proportions dans le jeu "
     "d'entraînement et dans le jeu de test."),
    ("Pourquoi `random_state=42` ?",
     "Le découpage est tiré au hasard. Fixer la graine rend ce hasard reproductible : même graine, mêmes "
     "lignes de test, mêmes résultats. C'est ce qui permet de comparer ton code avec DATAPULSE. "
     "Le nombre 42 n'a rien de spécial."),
    ("Pourquoi la normalisation est-elle dans le `Pipeline` ?",
     "Les mesures n'ont pas la même échelle (une aire vaut des centaines, une texture quelques unités). "
     "`StandardScaler` les ramène à une échelle comparable. Placé dans le `Pipeline`, il calcule ses moyennes "
     "sur le jeu d'entraînement seulement. Sinon, des informations du jeu de test « fuiraient » dans "
     "l'apprentissage : c'est une fuite de données, et le score serait trop optimiste."),
]

QUIZ_MODULE_1 = [
    {
        "question": "Pourquoi ne pas évaluer le modèle sur les données d'entraînement ?",
        "choix": [
            "Parce que ce serait trop lent.",
            "Parce que le modèle a déjà vu ces exemples : un bon score ne prouverait pas qu'il sait prédire de nouveaux cas.",
            "Parce que scikit-learn l'interdit.",
        ],
        "bonne": 1,
        "explications": [
            "Non, la vitesse n'est pas en cause. Un élève qui récite les réponses d'un exercice déjà corrigé prouve-t-il qu'il a compris ?",
            "Exactement. Un modèle peut « réciter » ses exemples. Seules des données jamais vues montrent s'il généralise.",
            "Non, scikit-learn te laisse faire, et c'est justement un piège classique.",
        ],
    },
    {
        "question": "Dans le jeu de données du cancer du sein, quelle est la cible ?",
        "choix": [
            "Le rayon moyen de la tumeur (mean radius).",
            "Le diagnostic : tumeur maligne (0) ou bénigne (1).",
            "Les 30 mesures de la tumeur.",
        ],
        "bonne": 1,
        "explications": [
            "Non, c'est une variable explicative : une mesure qui aide à prédire.",
            "Oui. La cible est ce qu'on veut prédire ; les 30 mesures sont les variables explicatives.",
            "Non, ce sont les variables explicatives (X). La cible (y) est ce qu'on cherche à prédire.",
        ],
    },
    {
        "question": "Le modèle prédit une probabilité de 0,8 pour une tumeur. Qu'est-ce que cela signifie ?",
        "choix": [
            "Le modèle a raison 80 % du temps.",
            "Le modèle estime à 80 % la probabilité que la tumeur soit bénigne (classe 1) ; avec le seuil de 0,5, il prédit « bénigne ».",
            "80 % des variables ont servi à la prédiction.",
        ],
        "bonne": 1,
        "explications": [
            "Non, ça, c'est l'accuracy, mesurée sur tout le jeu de test. La probabilité concerne une seule tumeur.",
            "Oui. La régression logistique estime une probabilité, puis applique un seuil (0,5 par défaut).",
            "Non, toutes les variables servent. La probabilité porte sur la classe de la tumeur.",
        ],
    },
]
