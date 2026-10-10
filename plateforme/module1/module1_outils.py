"""
module1_outils.py - Outils du module 1 « Ma première classification »

Quiz, vérification des exercices à trous, indices progressifs, comparaison avec
DATAPULSE et bilan. La progression est enregistrée dans progression_module1.json,
à côté des notebooks.

Dans un notebook :
    from module1_outils import *
    demarrer()

Auteur: Herman Kandolo
"""

import json
import os
import sys
from datetime import datetime

import numpy as np

FICHIER_PROGRESSION = "progression_module1.json"
FICHIER_JOURNAL = "journal_de_bord.md"
REFERENCE_PAR_DEFAUT = "reference_module1.json"
EXERCICES = ("D1", "D2", "D3")
TENTATIVES_AVANT_SOLUTION = 3

__all__ = ["demarrer", "quiz", "repondre", "reflexion", "verifier", "indice", "solution",
           "verifier_defi", "comparer_avec_datapulse", "expliquer_ecart", "journal", "bilan"]


# ---------------------------------------------------------------------------
# Affichage et progression
# ---------------------------------------------------------------------------

def _afficher(texte):
    """Affiche du Markdown dans un notebook, du texte brut ailleurs"""
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display
        if get_ipython() is not None:
            display(Markdown(texte))
            return
    except ImportError:
        pass
    print(texte)


def _charger():
    if os.path.exists(FICHIER_PROGRESSION):
        with open(FICHIER_PROGRESSION, encoding="utf-8") as f:
            return json.load(f)
    return {"debut": datetime.now().isoformat(timespec="seconds"), "quiz": {}, "reflexions": {},
            "exercices": {}, "defi": {}, "comparaison": {}}


def _sauver(progression):
    progression["mise_a_jour"] = datetime.now().isoformat(timespec="seconds")
    with open(FICHIER_PROGRESSION, "w", encoding="utf-8") as f:
        json.dump(progression, f, ensure_ascii=False, indent=2)


def _exercice(progression, nom):
    return progression["exercices"].setdefault(
        nom, {"tentatives": 0, "indices": 0, "valide": False, "solution_vue": False})


def _espace():
    """Variables de l'étudiant (celles du notebook)"""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            return shell.user_ns
    except ImportError:
        pass
    return sys._getframe(2).f_globals


# ---------------------------------------------------------------------------
# Messages d'erreur reformulés
# ---------------------------------------------------------------------------

def _conseil_erreur(etype, evalue):
    """Traduit les erreurs fréquentes des débutants en conseil lisible"""
    message = str(evalue)
    if "ellipsis" in message.lower():
        return "Il reste des `...` dans ton code : remplace-les par ta réponse."
    if etype is NameError:
        nom = message.split("'")[1] if "'" in message else "cette variable"
        return (f"Python ne connaît pas `{nom}`. As-tu exécuté les cellules précédentes, dans l'ordre ? "
                "Menu *Exécuter* → *Exécuter toutes les cellules au-dessus*. Vérifie aussi l'orthographe.")
    if etype.__name__ == "NotFittedError":
        return "Le modèle n'est pas encore entraîné : appelle `modele.fit(X_train, y_train)` avant `predict`."
    if etype is ModuleNotFoundError and ("module1_outils" in message or "comparateur" in message):
        return "Le fichier `module1_outils.py` (ou `comparateur.py`) doit être dans le même dossier que ce notebook."
    if etype is ModuleNotFoundError:
        return "Cette bibliothèque n'est pas installée, ou son nom est mal orthographié."
    if "parameter of" in message or "must be" in message:
        return "Un paramètre a une valeur non acceptée : la dernière ligne de l'erreur dit lequel et ce qui est attendu."
    if etype is AttributeError:
        return ("Une variable n'a pas la forme attendue, souvent parce qu'une cellule précédente a échoué : "
                "corrige-la et réexécute-la avant celle-ci.")
    if etype is SyntaxError:
        return "Erreur d'écriture : vérifie les parenthèses, les virgules et les guillemets autour de la flèche ^."
    return None


def _activer_messages_clairs():
    try:
        from IPython import get_ipython
    except ImportError:
        return
    shell = get_ipython()
    if shell is None:
        return

    def gestionnaire(shell, etype, evalue, tb, tb_offset=None):
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        conseil = _conseil_erreur(etype, evalue)
        if conseil:
            _afficher(f"💡 **Conseil** : {conseil}")

    shell.set_custom_exc((Exception,), gestionnaire)


# ---------------------------------------------------------------------------
# Démarrage
# ---------------------------------------------------------------------------

def demarrer():
    """À exécuter au début de chaque notebook du module"""
    import sklearn
    _activer_messages_clairs()
    progression = _charger()
    _sauver(progression)
    lignes = [f"✅ Outils du module 1 chargés (scikit-learn {sklearn.__version__})."]
    if os.path.exists(REFERENCE_PAR_DEFAUT):
        with open(REFERENCE_PAR_DEFAUT, encoding="utf-8") as f:
            version_ref = json.load(f)["parametres"]["versions"]["scikit-learn"]
        if version_ref.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            lignes.append(f"⚠️ La référence DATAPULSE a été produite avec scikit-learn {version_ref} : "
                          "de très légers écarts de probabilités sont possibles.")
    valides = [e for e in EXERCICES if progression["exercices"].get(e, {}).get("valide")]
    if valides:
        lignes.append(f"Déjà validés : {', '.join(valides)}.")
    _afficher("\n\n".join(lignes))


# ---------------------------------------------------------------------------
# Étape A : quiz
# ---------------------------------------------------------------------------

QUIZ = {
    1: {
        "question": "Pourquoi ne pas évaluer le modèle sur les données d'entraînement ?",
        "choix": {
            "a": "Parce que ce serait trop lent.",
            "b": "Parce que le modèle a déjà vu ces exemples : un bon score ne prouverait pas qu'il sait "
                 "prédire de nouveaux cas.",
            "c": "Parce que scikit-learn l'interdit.",
        },
        "bonne": "b",
        "explications": {
            "a": "Non, la vitesse n'est pas en cause. Pense à un élève qui récite par cœur les réponses d'un "
                 "exercice déjà corrigé : est-ce que cela prouve qu'il a compris ?",
            "b": "Exactement. Un modèle peut « réciter » ses exemples. Seules des données jamais vues "
                 "montrent s'il généralise.",
            "c": "Non, scikit-learn te laisse faire, et c'est justement un piège classique.",
        },
    },
    2: {
        "question": "Dans le jeu de données du cancer du sein, quelle est la cible ?",
        "choix": {
            "a": "Le rayon moyen de la tumeur (`mean radius`).",
            "b": "Le diagnostic : tumeur maligne (0) ou bénigne (1).",
            "c": "Les 30 mesures de la tumeur.",
        },
        "bonne": "b",
        "explications": {
            "a": "Non, c'est une variable explicative : une mesure qui aide à prédire.",
            "b": "Oui. La cible est ce qu'on veut prédire ; les 30 mesures sont les variables explicatives.",
            "c": "Non, ce sont les variables explicatives (X). La cible (y) est ce qu'on cherche à prédire.",
        },
    },
    3: {
        "question": "Le modèle prédit une probabilité de 0,8 pour une tumeur. Qu'est-ce que cela signifie ?",
        "choix": {
            "a": "Le modèle a raison 80 % du temps.",
            "b": "Le modèle estime à 80 % la probabilité que cette tumeur soit de la classe 1 (bénigne) ; "
                 "comme 0,8 dépasse le seuil de 0,5, il prédit « bénigne ».",
            "c": "80 % des variables ont servi à la prédiction.",
        },
        "bonne": "b",
        "explications": {
            "a": "Non, ça, c'est l'accuracy, mesurée sur tout le jeu de test. La probabilité concerne une seule tumeur.",
            "b": "Oui. La régression logistique estime une probabilité, puis applique un seuil (0,5 par défaut).",
            "c": "Non, toutes les variables servent. La probabilité porte sur la classe de la tumeur.",
        },
    },
}


def quiz(numero=None):
    """Affiche une question du quiz (ou toutes)"""
    numeros = [numero] if numero else list(QUIZ)
    for n in numeros:
        q = QUIZ[n]
        choix = "\n".join(f"- **{lettre})** {texte}" for lettre, texte in q["choix"].items())
        _afficher(f"**Question {n}.** {q['question']}\n\n{choix}\n\nRéponds avec `repondre({n}, \"a\")`.")


def repondre(numero, lettre):
    """Enregistre la réponse à une question du quiz et explique"""
    if numero not in QUIZ:
        _afficher(f"❓ Il n'y a pas de question {numero} : les questions vont de 1 à {len(QUIZ)}.")
        return
    q = QUIZ[numero]
    lettre = str(lettre).strip().lower()
    if lettre not in q["choix"]:
        _afficher(f"❓ Choisis parmi {', '.join(q['choix'])}.")
        return
    correct = lettre == q["bonne"]
    progression = _charger()
    reponse = progression["quiz"].setdefault(str(numero), {"premiere_reponse_correcte": correct, "tentatives": 0})
    reponse["tentatives"] += 1
    reponse["trouvee"] = reponse.get("trouvee", False) or correct
    _sauver(progression)
    _afficher(("✅ " if correct else "❌ ") + q["explications"][lettre])


def reflexion(cle, texte):
    """Enregistre une réponse libre (non notée)"""
    if _est_vide(texte):
        _afficher("✏️ Écris ta réponse après « Ta réponse : », puis exécute à nouveau la cellule.")
        return
    progression = _charger()
    progression["reflexions"][cle] = texte
    _sauver(progression)
    _afficher("📝 Réponse enregistrée. Il n'y a pas de mauvaise réponse ici : l'important est d'y avoir réfléchi.")


# ---------------------------------------------------------------------------
# Étape D : code à trous
# ---------------------------------------------------------------------------

def _donnees_attendues():
    """Découpage attendu, recalculé à partir des données d'origine"""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X, y, X_train, X_test, y_train, y_test


def _modele_attendu(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000, random_state=42))]).fit(X_train, y_train)


def _est_vide(texte):
    """Vrai si l'étudiant n'a rien écrit après les intitulés du modèle (lignes finissant par « : » ou « ? »)"""
    lignes = [l.strip() for l in str(texte).strip().splitlines() if l.strip()]
    return all(l.endswith(":") or l.endswith("?") for l in lignes)


def _manquantes(espace, noms):
    return [n for n in noms if n not in espace]


def _verifier_d1(espace):
    manquantes = _manquantes(espace, ["X_train", "X_test", "y_train", "y_test"])
    if manquantes:
        return False, f"Je ne trouve pas {', '.join(manquantes)} : as-tu exécuté ta cellule ?"
    X_train, X_test, y_train, y_test = (espace[n] for n in ("X_train", "X_test", "y_train", "y_test"))
    if getattr(X_test, "ndim", 2) != 2 or getattr(y_train, "ndim", 1) != 1:
        return False, ("Les résultats ne sont pas dans le bon ordre. `train_test_split` renvoie "
                       "`X_train, X_test, y_train, y_test`, dans cet ordre.")
    X, y, _, X_test_ok, _, _ = _donnees_attendues()
    if len(X_test) + len(X_train) != len(X):
        return False, f"Ton découpage contient {len(X_train) + len(X_test)} lignes au lieu de {len(X)} : as-tu bien découpé X et y ?"
    if len(X_test) != len(X_test_ok):
        part = len(X_test) / len(X)
        conseil = " Attention : `test_size=20` veut dire 20 lignes, `test_size=0.2` veut dire 20 %." if len(X_test) == 20 else ""
        return False, (f"Ton jeu de test contient {len(X_test)} lignes ({part:.0%}). "
                       f"La consigne demande 20 % des données.{conseil}")
    if set(X_test.index) == set(X_test_ok.index):
        if not (X_train.index.equals(y_train.index) and X_test.index.equals(y_test.index)):
            return False, "X et y ne correspondent plus ligne à ligne : découpe-les ensemble, dans le même appel."
        return True, (f"Découpage correct : {len(X_train)} lignes pour l'entraînement, {len(X_test)} pour le test. "
                      "La stratification garde la même proportion de tumeurs malignes dans les deux.")
    from sklearn.model_selection import train_test_split
    _, X_test_sans_strat = train_test_split(X, test_size=0.2, random_state=42)
    if set(X_test.index) == set(X_test_sans_strat.index):
        return False, ("Bonne taille et bonne graine, mais il manque la stratification : ajoute `stratify=y` "
                       "pour garder la même proportion de chaque classe dans le train et le test.")
    return False, ("Bonne taille, mais pas les mêmes lignes de test que DATAPULSE. Vérifie `random_state=42` "
                   "(la graine) et `stratify=y`.")


def _verifier_d2(espace):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    if "modele" not in espace:
        return False, "Je ne trouve pas la variable `modele` : as-tu exécuté ta cellule ?"
    modele = espace["modele"]
    if isinstance(modele, LogisticRegression):
        return False, ("Ton modèle est une régression logistique seule. Mets-la dans un `Pipeline`, "
                       "après un `StandardScaler`, pour normaliser les variables.")
    if not isinstance(modele, Pipeline):
        return False, f"`modele` devrait être un `Pipeline`, mais c'est un {type(modele).__name__}."
    etapes = [etape for _, etape in modele.steps]
    if not any(isinstance(e, StandardScaler) for e in etapes):
        return False, "Ton Pipeline ne contient pas de `StandardScaler()` : la normalisation manque."
    if not isinstance(etapes[-1], LogisticRegression):
        return False, "La dernière étape du Pipeline doit être la `LogisticRegression(...)`."
    try:
        check_is_fitted(modele)
    except NotFittedError:
        return False, "Ton modèle n'est pas entraîné : appelle `modele.fit(X_train, y_train)`."
    _, _, X_train, X_test, y_train, _ = _donnees_attendues()
    scaler = next(e for e in etapes if isinstance(e, StandardScaler))
    if scaler.n_samples_seen_ == 569:
        return False, ("Ton modèle a été entraîné sur toutes les données (569 lignes). Entraîne-le sur "
                       "`X_train, y_train` uniquement : le jeu de test doit rester inconnu du modèle.")
    if scaler.n_samples_seen_ == len(X_test):
        return False, "Ton modèle a été entraîné sur le jeu de test ! Utilise `modele.fit(X_train, y_train)`."
    if not np.allclose(scaler.mean_, X_train.mean().to_numpy()):
        return False, "Ton modèle n'a pas été entraîné sur le X_train attendu : valide d'abord l'exercice D1."
    clf = etapes[-1]
    differences = [f"`{p}={clf.get_params()[p]!r}` (attendu `{v!r}`)"
                   for p, v in {"max_iter": 1000, "random_state": 42, "C": 1.0}.items()
                   if clf.get_params()[p] != v]
    attendu = _modele_attendu(X_train, y_train)
    if np.abs(modele.predict_proba(X_test) - attendu.predict_proba(X_test)).max() > 1e-4:
        if differences:
            return False, "Ta régression logistique n'a pas les réglages demandés : " + ", ".join(differences) + "."
        return False, "Ton modèle ne donne pas les mêmes probabilités que le modèle attendu. Relis la consigne."
    return True, ("Modèle correct : la normalisation a été apprise sur le jeu d'entraînement seulement, "
                  "puis la régression logistique a été entraînée.")


def _verifier_d3(espace):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
    manquantes = _manquantes(espace, ["y_pred", "accuracy", "matrice", "rappel"])
    if manquantes:
        return False, f"Je ne trouve pas {', '.join(manquantes)} : as-tu exécuté ta cellule ?"
    _, _, X_train, X_test, y_train, y_test = _donnees_attendues()
    attendu = _modele_attendu(X_train, y_train)
    y_pred_ok = attendu.predict(X_test)
    y_pred = np.asarray(espace["y_pred"])
    if len(y_pred) == len(X_train):
        return False, "`y_pred` contient les prédictions du jeu d'entraînement. Prédis sur `X_test`."
    if len(y_pred) != len(y_pred_ok) or not np.array_equal(y_pred, y_pred_ok):
        return False, "`y_pred` n'est pas `modele.predict(X_test)` avec le modèle de D2 : vérifie D2, puis réexécute."
    accuracy = espace["accuracy"]
    acc_ok = accuracy_score(y_test, y_pred_ok)
    if np.isscalar(accuracy) and abs(accuracy - accuracy_score(y_train, attendu.predict(X_train))) < 1e-9:
        return False, "Ton accuracy est calculée sur le jeu d'entraînement. Calcule-la avec `y_test` et `y_pred`."
    if not np.isscalar(accuracy) or abs(accuracy - acc_ok) > 1e-6:
        return False, f"Ton accuracy vaut {accuracy}, alors qu'on attend {acc_ok:.4f}. Utilise `accuracy_score(y_test, y_pred)`."
    matrice = np.asarray(espace["matrice"])
    cm_ok = confusion_matrix(y_test, y_pred_ok)
    if matrice.shape == cm_ok.shape and np.array_equal(matrice, cm_ok.T) and not np.array_equal(cm_ok, cm_ok.T):
        return False, ("Ta matrice est inversée (transposée) : l'ordre compte. Écris `confusion_matrix(y_test, y_pred)`, "
                       "les vraies valeurs d'abord. Les lignes sont alors les vraies classes, les colonnes les prédictions.")
    if matrice.shape != cm_ok.shape or not np.array_equal(matrice, cm_ok):
        return False, "Ta matrice de confusion ne correspond pas : utilise `confusion_matrix(y_test, y_pred)`."
    rappel = espace["rappel"]
    rappel_ok = recall_score(y_test, y_pred_ok, pos_label=0)
    if np.isscalar(rappel) and abs(rappel - precision_score(y_test, y_pred_ok, pos_label=0)) < 1e-9 \
            and abs(rappel - rappel_ok) > 1e-9:
        return False, "Ton rappel vaut en fait la précision : as-tu inversé `y_test` et `y_pred` ?"
    if not np.isscalar(rappel) or abs(rappel - rappel_ok) > 1e-6:
        return False, ("Ton rappel ne correspond pas. On cherche la part des tumeurs malignes (classe 0) "
                       "que le modèle a bien détectées : `recall_score(y_test, y_pred, pos_label=0)`.")
    return True, (f"Bravo ! Accuracy {acc_ok:.4f}, et le modèle détecte {rappel_ok:.0%} des tumeurs malignes "
                  f"({cm_ok[0, 0]} sur {cm_ok[0].sum()}).")


VERIFICATIONS = {"D1": _verifier_d1, "D2": _verifier_d2, "D3": _verifier_d3}

INDICES = {
    "D1": [
        "La consigne : 20 % de données de test, découpage stratifié, graine 42.",
        "`test_size` est une proportion (un nombre entre 0 et 1). `stratify` reçoit la colonne dont on "
        "veut garder les proportions : la cible. `random_state` reçoit la graine.",
        "Il faut écrire `test_size=0.2`, `stratify=y` et `random_state=42`.",
    ],
    "D2": [
        "Le Pipeline enchaîne deux étapes : d'abord la normalisation, puis le modèle.",
        "La normalisation s'écrit `StandardScaler()`, le modèle `LogisticRegression(max_iter=1000, random_state=42)`. "
        "Pour entraîner, on appelle `fit` avec les données d'entraînement.",
        "`(\"scaler\", StandardScaler())`, `(\"clf\", LogisticRegression(max_iter=1000, random_state=42))`, "
        "puis `modele.fit(X_train, y_train)`.",
    ],
    "D3": [
        "On prédit sur le jeu de test, puis on compare les prédictions aux vraies valeurs du test.",
        "Les fonctions de `sklearn.metrics` prennent toujours les vraies valeurs d'abord, puis les prédictions. "
        "Pour le rappel des tumeurs malignes, la classe qui nous intéresse est 0.",
        "`modele.predict(X_test)`, `accuracy_score(y_test, y_pred)`, `confusion_matrix(y_test, y_pred)`, "
        "`recall_score(y_test, y_pred, pos_label=0)`.",
    ],
}

SOLUTIONS = {
    "D1": '''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)''',
    "D2": '''from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

modele = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
modele.fit(X_train, y_train)''',
    "D3": '''from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

y_pred = modele.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
matrice = confusion_matrix(y_test, y_pred)
rappel = recall_score(y_test, y_pred, pos_label=0)

print("Accuracy :", accuracy)
print(matrice)
print("Rappel (tumeurs malignes) :", rappel)''',
}


def verifier(exercice):
    """Vérifie un exercice à trous (« D1 », « D2 » ou « D3 »)"""
    exercice = exercice.upper()
    if exercice not in VERIFICATIONS:
        _afficher(f"❓ Exercice inconnu : choisis parmi {', '.join(EXERCICES)}.")
        return
    try:
        ok, message = VERIFICATIONS[exercice](_espace())
    except Exception as e:  # une variable de l'étudiant n'a pas la forme attendue
        ok, message = False, f"Je n'arrive pas à vérifier tes variables ({type(e).__name__} : {e}). Réexécute tes cellules dans l'ordre."
    progression = _charger()
    etat = _exercice(progression, exercice)
    if not etat["valide"]:
        etat["tentatives"] += 1
        etat["valide"] = ok
    _sauver(progression)
    if ok:
        _afficher(f"✅ **{exercice} validé.** {message}")
    else:
        restantes = TENTATIVES_AVANT_SOLUTION - etat["tentatives"]
        suite = (f"Besoin d'aide ? `indice(\"{exercice}\")`." if restantes > 0 else
                 f"Tu peux maintenant voir la solution : `solution(\"{exercice}\")`.")
        _afficher(f"❌ {message}\n\n{suite}")


def indice(exercice):
    """Donne l'indice suivant (3 niveaux, du plus vague au plus précis)"""
    exercice = exercice.upper()
    if exercice not in INDICES:
        _afficher(f"❓ Exercice inconnu : choisis parmi {', '.join(EXERCICES)}.")
        return
    progression = _charger()
    etat = _exercice(progression, exercice)
    niveau = min(etat["indices"], len(INDICES[exercice]) - 1)
    etat["indices"] = niveau + 1
    _sauver(progression)
    _afficher(f"💡 **Indice {niveau + 1}/{len(INDICES[exercice])}** : {INDICES[exercice][niveau]}")


def solution(exercice):
    """Montre la solution, après 3 tentatives (ou une fois l'exercice validé)"""
    exercice = exercice.upper()
    if exercice not in SOLUTIONS:
        _afficher(f"❓ Exercice inconnu : choisis parmi {', '.join(EXERCICES)}.")
        return
    progression = _charger()
    etat = _exercice(progression, exercice)
    if not etat["valide"] and etat["tentatives"] < TENTATIVES_AVANT_SOLUTION:
        restantes = TENTATIVES_AVANT_SOLUTION - etat["tentatives"]
        _afficher(f"🔒 Encore {restantes} tentative(s) avant la solution. Essaie un `indice(\"{exercice}\")` !")
        return
    etat["solution_vue"] = True
    _sauver(progression)
    _afficher(f"```python\n{SOLUTIONS[exercice]}\n```\n\nRecopie-la dans ta cellule, comprends chaque ligne, puis `verifier(\"{exercice}\")`.")


# ---------------------------------------------------------------------------
# Étapes E et F : défi et comparaison avec DATAPULSE
# ---------------------------------------------------------------------------

def verifier_defi():
    """Vérifie que le défi respecte la consigne (sur n'importe quel jeu de données)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.validation import check_is_fitted
    espace = _espace()
    problemes = []
    manquantes = _manquantes(espace, ["X_train", "X_test", "y_test", "modele", "y_pred", "accuracy", "matrice"])
    if manquantes:
        problemes.append(f"Variables introuvables : {', '.join(manquantes)}.")
    else:
        X_train, X_test, y_test, modele, y_pred = (espace[n] for n in ("X_train", "X_test", "y_test", "modele", "y_pred"))
        part = len(X_test) / (len(X_train) + len(X_test))
        if abs(part - 0.2) > 0.01:
            problemes.append(f"Ton jeu de test représente {part:.0%} des données au lieu de 20 %.")
        if not isinstance(modele, Pipeline) or not any(isinstance(e, StandardScaler) for _, e in modele.steps):
            problemes.append("`modele` doit être un Pipeline avec un `StandardScaler` (la normalisation).")
        elif not isinstance(modele.steps[-1][1], LogisticRegression):
            problemes.append("La dernière étape du Pipeline doit être une `LogisticRegression`.")
        else:
            try:
                check_is_fitted(modele)
                if not np.array_equal(np.asarray(y_pred), modele.predict(X_test)):
                    problemes.append("`y_pred` doit être `modele.predict(X_test)`.")
                elif abs(espace["accuracy"] - accuracy_score(y_test, y_pred)) > 1e-6:
                    problemes.append("`accuracy` doit être `accuracy_score(y_test, y_pred)`.")
                elif not np.array_equal(np.asarray(espace["matrice"]), confusion_matrix(y_test, y_pred)):
                    problemes.append("`matrice` doit être `confusion_matrix(y_test, y_pred)`.")
            except Exception as e:
                problemes.append(f"Le modèle n'est pas utilisable ({type(e).__name__}) : l'as-tu entraîné ?")
    progression = _charger()
    progression["defi"] = {"valide": not problemes, "problemes": problemes,
                           "date": datetime.now().isoformat(timespec="seconds")}
    _sauver(progression)
    if problemes:
        _afficher("❌ Pas encore :\n\n" + "\n".join(f"- {p}" for p in problemes))
        return
    _afficher("✅ **Défi validé** : ton code respecte la consigne. Passe à la comparaison avec DATAPULSE !")


def _comparateur():
    try:
        import comparateur
    except ModuleNotFoundError:
        # Dans le dépôt DATAPULSE, comparateur.py est deux dossiers plus haut
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))
        import comparateur
    return comparateur


def comparer_avec_datapulse(reference=REFERENCE_PAR_DEFAUT):
    """Compare le résultat du défi avec la référence DATAPULSE

    reference : fichier téléchargé depuis DATAPULSE (« Télécharger la référence pour comparaison »).
                Par défaut, la référence du module (cancer du sein, régression logistique).
    """
    comparateur = _comparateur()
    espace = _espace()
    manquantes = _manquantes(espace, ["X_test", "y_pred"])
    if manquantes:
        _afficher(f"❓ Je ne trouve pas {', '.join(manquantes)} : exécute d'abord ton code du défi.")
        return
    if not os.path.exists(reference):
        _afficher(f"❓ Fichier `{reference}` introuvable : place-le à côté de ce notebook.")
        return
    with open(reference, encoding="utf-8") as f:
        ref = json.load(f)
    resultats = comparateur.resultats_etudiant(espace["X_test"], espace["y_pred"], espace.get("modele"))
    with open("mes_resultats.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False)
    resultat = comparateur.comparer(ref, resultats)

    progression = _charger()
    progression["comparaison"] = {**resultat.to_dict(), "reference": os.path.basename(reference),
                                  "date": datetime.now().isoformat(timespec="seconds")}
    _sauver(progression)

    lignes = [("✅ **" if resultat.conforme else "❌ **") + resultat.verdict + "**", ""]
    lignes += [f"- {'✅' if c.ok else '❌'} **{c.nom}** : {c.detail}" for c in resultat.controles]
    if resultat.pistes:
        lignes += ["", "**Pistes à vérifier :**", ""] + [f"- {p}" for p in resultat.pistes]
        lignes += ["", "Corrige ton code et relance la comparaison. Si tu ne trouves pas, explique l'écart "
                       "avec tes mots : `expliquer_ecart(\"...\")`."]
    lignes += ["", "_Tes résultats sont aussi dans `mes_resultats.json`, à déposer si tu veux dans la page "
                   "« 🎓 Comparer mon code » de DATAPULSE._"]
    _afficher("\n".join(lignes))


def expliquer_ecart(texte):
    """Enregistre l'explication de l'étudiant quand sa comparaison n'est pas conforme"""
    progression = _charger()
    comparaison = progression.get("comparaison", {})
    if not comparaison:
        _afficher("❓ Lance d'abord la comparaison : `comparer_avec_datapulse()`.")
        return
    if comparaison.get("conforme"):
        _afficher("✅ Ton résultat est identique à celui de DATAPULSE : il n'y a pas d'écart à expliquer.")
        return
    if _est_vide(texte):
        _afficher("✏️ Écris ton explication après « l'écart vient de : », puis exécute à nouveau la cellule.")
        return
    comparaison["explication"] = texte
    _sauver(progression)
    _afficher("📝 Explication enregistrée : ton enseignant pourra la relire.")


# ---------------------------------------------------------------------------
# Étape G : journal de bord et bilan
# ---------------------------------------------------------------------------

def journal(texte):
    """Ajoute une entrée datée au journal de bord du projet fil rouge"""
    if _est_vide(texte):
        _afficher("✏️ Complète au moins une ligne de ton journal, puis exécute à nouveau la cellule.")
        return
    nouveau = not os.path.exists(FICHIER_JOURNAL)
    with open(FICHIER_JOURNAL, "a", encoding="utf-8") as f:
        if nouveau:
            f.write("# Journal de bord du projet fil rouge\n")
        f.write(f"\n## {datetime.now().strftime('%d/%m/%Y %H:%M')} — Module 1\n\n{texte.strip()}\n")
    _afficher(f"📓 Entrée ajoutée à `{FICHIER_JOURNAL}`.")


def bilan():
    """Bilan du module : critères de réussite (plan, section 4.3) et détail par exercice"""
    p = _charger()
    score_quiz = sum(1 for r in p["quiz"].values() if r.get("premiere_reponse_correcte"))
    exercices = {e: p["exercices"].get(e, {"tentatives": 0, "indices": 0, "valide": False, "solution_vue": False})
                 for e in EXERCICES}
    comparaison = p.get("comparaison", {})
    criteres = [
        ("Quiz de la leçon", score_quiz >= 2, f"{score_quiz} / {len(QUIZ)} (minimum 2)"),
        ("Exercices D1 à D3", all(e["valide"] for e in exercices.values()),
         ", ".join(f"{n} {'✅' if e['valide'] else '❌'}" for n, e in exercices.items())),
        ("Défi E", p.get("defi", {}).get("valide", False),
         "code exécutable et conforme à la consigne" if p.get("defi", {}).get("valide") else "pas encore validé"),
        ("Comparaison F", bool(comparaison.get("conforme") or comparaison.get("explication")),
         "identique à DATAPULSE" if comparaison.get("conforme") else
         "écart expliqué" if comparaison.get("explication") else "pas encore faite ou non conforme"),
    ]
    lignes = ["### 🎓 Bilan du module 1", "", "| Critère | Résultat | Détail |", "|---|---|---|"]
    lignes += [f"| {nom} | {'✅' if ok else '❌'} | {detail} |" for nom, ok, detail in criteres]
    lignes += ["", "| Exercice | Tentatives | Indices | Solution vue |", "|---|---|---|---|"]
    lignes += [f"| {n} | {e['tentatives']} | {e['indices']} | {'oui' if e['solution_vue'] else 'non'} |"
               for n, e in exercices.items()]
    module_valide = all(ok for _, ok, _ in criteres)
    lignes += ["", "**Module 1 validé 🎉** Passe au module 2 : arbres, forêts et KNN." if module_valide
               else "**Module pas encore validé** : reprends les critères marqués ❌."]
    lignes += ["", "**Ce que tu sais maintenant faire :** séparer entraînement et test, entraîner une régression "
                   "logistique avec normalisation, lire une matrice de confusion, calculer accuracy et rappel, "
                   "et reproduire en code un résultat de DATAPULSE."]
    p["module_valide"] = module_valide
    _sauver(p)
    _afficher("\n".join(lignes))
