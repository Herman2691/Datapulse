"""
code_generator.py - Génération du code scikit-learn équivalent à une analyse DATAPULSE

Le code est produit uniquement à partir des paramètres exportés en JSON
(DataProcessor.run_parameters) : il peut donc être régénéré hors de DATAPULSE.

Usage en ligne de commande :
    python code_generator.py parametres.json

Auteur: Herman Kandolo
"""

import importlib
import json
import sys

# Sélection des colonnes, identique à DataProcessor._create_preprocessor
DTYPES_PAR_TRANSFORMATEUR = {
    "num": ("colonnes_numeriques", ["number", "bool"]),
    "cat": ("colonnes_categorielles", ["object", "category"]),
}

# Explication courte de chaque étape de prétraitement, pour les débutants
EXPLICATIONS = {
    ("SimpleImputer", "median"): "remplace les valeurs manquantes par la médiane",
    ("SimpleImputer", "most_frequent"): "remplace les valeurs manquantes par la valeur la plus fréquente",
    ("SimpleImputer", "mean"): "remplace les valeurs manquantes par la moyenne",
    ("StandardScaler", None): "centre et réduit : moyenne 0, écart-type 1",
    ("OneHotEncoder", None): "transforme chaque catégorie en colonne 0/1",
}

# Explication de l'encodage du fichier CSV (voir DataProcessor.load_data)
EXPLICATIONS_ENCODAGE = {
    "utf-8-sig": "utf-8-sig : UTF-8, avec ou sans la marque ajoutée par Excel",
    "utf-8": "encodage UTF-8",
    "cp1252": "cp1252 : fichier enregistré par Excel sous Windows",
    "latin-1": "latin-1 : ancien encodage d'Europe de l'Ouest",
}


def _import_public(chemin_classe):
    """Renvoie (module public, classe) : sklearn.linear_model._logistic.X -> sklearn.linear_model"""
    module, nom = chemin_classe.rsplit(".", 1)
    parties = module.split(".")
    for i in range(1, len(parties) + 1):
        candidat = ".".join(parties[:i])
        if any(p.startswith("_") for p in parties[:i]):
            break
        mod = importlib.import_module(candidat)
        if hasattr(mod, nom):
            return candidat, getattr(mod, nom)
    return module, getattr(importlib.import_module(module), nom)


def _normaliser(valeur):
    """Même conversion que DataProcessor._to_jsonable, pour comparer aux valeurs par défaut"""
    if isinstance(valeur, dict):
        return {str(k): _normaliser(v) for k, v in valeur.items()}
    if isinstance(valeur, (list, tuple)):
        return [_normaliser(v) for v in valeur]
    if hasattr(valeur, "item") and not isinstance(valeur, (list, dict, str)):
        try:
            valeur = valeur.item()
        except (TypeError, ValueError):
            pass
    if isinstance(valeur, float) and valeur != valeur:
        return "NaN"
    if valeur is None or isinstance(valeur, (bool, int, float, str)):
        return valeur
    if isinstance(valeur, type):
        return f"{valeur.__module__}.{valeur.__name__}"
    return str(valeur)


class _Contexte:
    """Mémorise les imports nécessaires (numpy) pendant le rendu"""

    def __init__(self):
        self.besoin_numpy = False


def _rendre_valeur(valeur, defaut, ctx):
    if valeur == "NaN":
        ctx.besoin_numpy = True
        return "np.nan"
    if isinstance(valeur, str) and valeur.startswith("numpy."):
        ctx.besoin_numpy = True
        return "np." + valeur.split(".", 1)[1]
    if isinstance(valeur, list):
        elements = ", ".join(_rendre_valeur(v, None, ctx) for v in valeur)
        if isinstance(defaut, tuple):
            return f"({elements},)" if len(valeur) == 1 else f"({elements})"
        return f"[{elements}]"
    if isinstance(valeur, str):
        return json.dumps(valeur, ensure_ascii=False)
    return repr(valeur)


def _rendre_estimateur(chemin_classe, parametres, ctx, indent=0):
    """Rend 'Classe(param=valeur, ...)' avec seulement les paramètres non par défaut"""
    _, classe = _import_public(chemin_classe)
    defauts = classe().get_params(deep=False)
    args = []
    for nom, valeur in parametres.items():
        defaut = defauts.get(nom)
        if nom in defauts and _normaliser(defaut) == valeur:
            continue
        args.append(f"{nom}={_rendre_valeur(valeur, defaut, ctx)}")

    en_ligne = f"{classe.__name__}({', '.join(args)})"
    if len(en_ligne) + indent <= 79 or not args:
        return en_ligne
    marge = " " * (indent + 4)
    return f"{classe.__name__}(\n" + "".join(f"{marge}{a},\n" for a in args) + " " * indent + ")"


def _imports_groupes(chemins):
    """{module public: [classes]} en conservant l'ordre d'apparition"""
    groupes = {}
    for chemin in chemins:
        module, classe = _import_public(chemin)
        noms = groupes.setdefault(module, [])
        if classe.__name__ not in noms:
            noms.append(classe.__name__)
    return [f"from {m} import {', '.join(n)}" for m, n in groupes.items()]


def _explication(etape):
    nom = etape["classe"].rsplit(".", 1)[1]
    strategie = etape["parametres"].get("strategy") if nom == "SimpleImputer" else None
    return EXPLICATIONS.get((nom, strategie))


def _attendu(valeur):
    if isinstance(valeur, float):
        return f"{valeur:.6f}"
    return json.dumps(valeur)


def generer_code(params, metriques=None):
    """Génère le code Python commenté qui reproduit exactement l'analyse décrite par params"""
    ctx = _Contexte()
    donnees, cible = params["donnees"], params["cible"]
    decoupage, modele = params["decoupage"], params["modele"]
    classification = cible["type_analyse"] == "Classification"
    metriques = metriques or {}
    lignes = []
    ajout = lignes.append

    # --- En-tête
    ajout("# " + "=" * 70)
    ajout("# Code scikit-learn équivalent à l'analyse DATAPULSE")
    ajout(f"# Modèle : {modele['nom']} • test {decoupage['test_size']:.0%} • graine {decoupage['random_state']}")
    ajout(f"# Généré le {params['date']} avec scikit-learn {params['versions']['scikit-learn']}")
    ajout("# Utilise la même version de scikit-learn pour retrouver exactement les mêmes résultats.")
    ajout("# " + "=" * 70)
    ajout("")

    # --- 1. Chargement
    lecture = donnees.get("lecture") or {"fonction": "pandas.read_csv"}
    fichier = json.dumps(donnees.get("fichier") or "mes_donnees.csv", ensure_ascii=False)
    ajout("# 1. Charger les données")
    ajout("import pandas as pd")
    if lecture["fonction"] == "pandas.read_excel":
        moteur = f", engine={json.dumps(lecture['engine'])}" if lecture.get("engine") else ""
        ajout(f"df = pd.read_excel({fichier}{moteur})")
    else:
        encodage = lecture.get("encoding", "utf-8")
        ajout(f"# sep=None : séparateur détecté automatiquement ; {EXPLICATIONS_ENCODAGE.get(encodage, 'encodage du fichier')}")
        ajout(f"df = pd.read_csv({fichier}, sep=None, engine=\"python\", encoding={json.dumps(encodage)})")
    if donnees.get("noms_colonnes_nettoyes"):
        ajout("df.columns = df.columns.str.strip()  # retire les espaces autour des noms de colonnes")
    ajout("")

    # --- 2. X et y
    nom_cible = json.dumps(cible["colonne"], ensure_ascii=False)
    ajout("# 2. Séparer les variables explicatives (X) et la variable cible (y)")
    ajout(f"X = df.drop(columns=[{nom_cible}])")
    ajout(f"y = df[{nom_cible}]")
    if cible["classes_encodees"] is not None:
        correspondance = ", ".join(f"{c} -> {i}" for i, c in enumerate(cible["classes_encodees"][:5]))
        if len(cible["classes_encodees"]) > 5:
            correspondance += ", ..."
        ajout("")
        ajout(f"# La cible est du texte : chaque classe devient un nombre ({correspondance})")
        ajout("from sklearn.preprocessing import LabelEncoder")
        ajout("y = LabelEncoder().fit_transform(y)")
    ajout("")

    # --- 3. Découpage
    stratify = ", stratify=y" if decoupage["stratify"] else ""
    ajout("# 3. Séparer entraînement / test (mêmes réglages que DATAPULSE)")
    ajout("from sklearn.model_selection import train_test_split")
    ajout("X_train, X_test, y_train, y_test = train_test_split(")
    ajout(f"    X, y, test_size={decoupage['test_size']!r}{stratify}, random_state={decoupage['random_state']!r}")
    ajout(")")
    ajout("")

    # --- 4. Prétraitement
    pre = params["pretraitement"]
    transformateurs = [t for t in pre["transformateurs"] if t["colonnes"]]
    etapes_classes = [e["classe"] for t in transformateurs for e in t["etapes"]]
    ajout("# 4. Prétraitement : appris sur le jeu d'entraînement uniquement (dans le Pipeline),")
    ajout("#    pour éviter toute fuite d'information du jeu de test")
    ajout("from sklearn.compose import ColumnTransformer")
    ajout("from sklearn.pipeline import Pipeline")
    lignes.extend(_imports_groupes(etapes_classes))
    ajout("")
    for t in transformateurs:
        variable, dtypes = DTYPES_PAR_TRANSFORMATEUR.get(t["nom"], (f"colonnes_{t['nom']}", None))
        if dtypes is not None:
            ajout(f"{variable} = X.select_dtypes(include={json.dumps(dtypes)}).columns.tolist()")
        else:
            ajout(f"{variable} = {json.dumps(t['colonnes'], ensure_ascii=False)}")
    if pre["colonnes_ignorees"]:
        ajout(f"# Attention : ces colonnes ne sont ni numériques, ni booléennes, ni texte, elles sont ignorées : "
              f"{json.dumps(pre['colonnes_ignorees'], ensure_ascii=False)}")
    ajout("")
    ajout("pretraitement = ColumnTransformer([")
    for t in transformateurs:
        variable = DTYPES_PAR_TRANSFORMATEUR.get(t["nom"], (f"colonnes_{t['nom']}", None))[0]
        ajout(f"    ({json.dumps(t['nom'])}, Pipeline([")
        noms_etapes = {"SimpleImputer": "imputer", "StandardScaler": "scaler", "OneHotEncoder": "onehot"}
        for e in t["etapes"]:
            nom_classe = e["classe"].rsplit(".", 1)[1]
            nom_etape = noms_etapes.get(nom_classe, nom_classe.lower())
            rendu = _rendre_estimateur(e["classe"], e["parametres"], ctx, indent=8)
            explication = _explication(e)
            commentaire = f"  # {explication}" if explication and "\n" not in rendu else ""
            ajout(f"        ({json.dumps(nom_etape)}, {rendu}),{commentaire}")
        ajout(f"    ]), {variable}),")
    ajout("])")
    ajout("")

    # --- 5. Modèle
    module_modele, _ = _import_public(modele["classe"])
    nom_classe_modele = modele["classe"].rsplit(".", 1)[1]
    ajout(f"# 5. Modèle : {modele['nom']}, précédé du prétraitement")
    ajout(f"from {module_modele} import {nom_classe_modele}")
    ajout("modele = Pipeline([")
    ajout("    (\"pretraitement\", pretraitement),")
    ajout(f"    (\"modele\", {_rendre_estimateur(modele['classe'], modele['hyperparametres'], ctx, indent=4)}),")
    ajout("])")
    ajout("")

    # --- 6. Entraînement
    ajout("# 6. Entraîner sur le jeu d'entraînement")
    ajout("modele.fit(X_train, y_train)")
    ajout("")

    # --- 7. Évaluation
    ajout("# 7. Évaluer sur le jeu de test (des données que le modèle n'a jamais vues)")
    ajout("y_pred = modele.predict(X_test)")
    if classification:
        ajout("from sklearn.metrics import accuracy_score, confusion_matrix")
        attendu = f"  # attendu : {_attendu(metriques['accuracy'])}" if "accuracy" in metriques else ""
        ajout(f"print(\"Accuracy :\", accuracy_score(y_test, y_pred)){attendu}")
        ajout("print(\"Matrice de confusion :\")")
        attendu = f"  # attendu : {_attendu(metriques['matrice_confusion'])}" if "matrice_confusion" in metriques else ""
        ajout(f"print(confusion_matrix(y_test, y_pred)){attendu}")
    else:
        ajout("from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error")
        for nom, fonction in [("r2", "r2_score"), ("mse", "mean_squared_error"), ("mae", "mean_absolute_error")]:
            attendu = f"  # attendu : {_attendu(metriques[nom])}" if nom in metriques else ""
            ajout(f"print(\"{nom.upper().replace('R2', 'R²')} :\", {fonction}(y_test, y_pred)){attendu}")
    ajout("")

    # --- 8. Validation croisée
    vc = params["validation_croisee"]
    attendu = f"  # attendu : {_attendu(metriques['cv_moyenne'])}" if "cv_moyenne" in metriques else ""
    ajout(f"# 8. Validation croisée ({vc['cv']} plis) sur le jeu d'entraînement : un score plus fiable")
    ajout("#    qu'un seul découpage, car calculé sur plusieurs partages différents")
    ajout("from sklearn.model_selection import cross_val_score")
    ajout(f"scores = cross_val_score(modele, X_train, y_train, cv={vc['cv']}, scoring={json.dumps(vc['scoring'])})")
    ajout(f"print(\"Validation croisée (moyenne) :\", scores.mean()){attendu}")

    if ctx.besoin_numpy:
        lignes.insert(lignes.index("import pandas as pd"), "import numpy as np")
    return "\n".join(lignes) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python code_generator.py parametres.json")
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        print(generer_code(json.load(f)))
