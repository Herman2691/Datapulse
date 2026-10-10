"""
comparateur.py - Moteur de comparaison entre le résultat d'un étudiant et la référence DATAPULSE

La référence est le fichier JSON exporté par DATAPULSE (DataProcessor.get_reference_json).
Le résultat de l'étudiant contient, pour chaque ligne de son jeu de test, la position de
la ligne dans le fichier (l'index de X_test), sa prédiction et, si possible, ses probabilités.

Côté étudiant, à la fin de son code :
    from comparateur import sauvegarder_resultats
    sauvegarder_resultats(X_test, y_pred, modele)

Puis, côté plateforme :
    python comparateur.py reference.json mes_resultats.json

Ce module ne dépend pas de DATAPULSE.

Auteur: Herman Kandolo
"""

import json
import sys
from dataclasses import dataclass, field

import numpy as np

# Tolérances par défaut (plan, section 5.3), si la référence n'en fournit pas
TOLERANCES_PAR_DEFAUT = {
    "accuracy": 1e-6,
    "matrice_confusion": 0,
    "probabilites": 1e-4,
    "regression": 1e-6,
}


# ---------------------------------------------------------------------------
# Côté étudiant
# ---------------------------------------------------------------------------

def resultats_etudiant(X_test, y_pred, modele=None):
    """Rassemble ce que la comparaison attend, à partir des variables de l'étudiant"""
    resultats = {
        "indices": [int(i) for i in X_test.index],
        "y_pred": np.asarray(y_pred).tolist(),
        "classes": None,
        "probabilites": None,
    }
    if modele is not None and hasattr(modele, "predict_proba"):
        try:
            resultats["probabilites"] = modele.predict_proba(X_test).tolist()
            resultats["classes"] = np.asarray(modele.classes_).tolist()
        except Exception:
            pass  # modèle de régression ou probabilités indisponibles
    return resultats


def sauvegarder_resultats(X_test, y_pred, modele=None, chemin="mes_resultats.json"):
    """Enregistre les résultats de l'étudiant dans un fichier JSON pour la comparaison"""
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resultats_etudiant(X_test, y_pred, modele), f, ensure_ascii=False)
    print(f"Résultats enregistrés dans {chemin}")
    return chemin


# ---------------------------------------------------------------------------
# Résultat de la comparaison
# ---------------------------------------------------------------------------

@dataclass
class Controle:
    nom: str
    ok: bool
    detail: str


@dataclass
class Comparaison:
    """Résultat de la comparaison : verdict, contrôles effectués et pistes pour l'étudiant"""
    conforme: bool
    verdict: str
    controles: list = field(default_factory=list)
    pistes: list = field(default_factory=list)

    def rapport(self):
        """Rapport lisible pour l'étudiant"""
        lignes = [("✅ " if self.conforme else "❌ ") + self.verdict, ""]
        for c in self.controles:
            lignes.append(f"{'✔' if c.ok else '✘'} {c.nom} : {c.detail}")
        if self.pistes:
            lignes += ["", "Pistes à vérifier :"]
            lignes += [f"  • {p}" for p in self.pistes]
        return "\n".join(lignes)

    def to_dict(self):
        return {
            "conforme": self.conforme,
            "verdict": self.verdict,
            "controles": [c.__dict__ for c in self.controles],
            "pistes": self.pistes,
        }


# ---------------------------------------------------------------------------
# Comparaison
# ---------------------------------------------------------------------------

def _encoder(valeurs, classes_encodees):
    """Ramène les étiquettes texte de l'étudiant aux numéros utilisés par DATAPULSE"""
    if classes_encodees is None:
        return list(valeurs)
    correspondance = {c: i for i, c in enumerate(classes_encodees)}
    return [correspondance.get(v, v) for v in valeurs]


# Modèles dont le résultat dépend fortement de la graine
MODELES_ALEATOIRES = ("RandomForestClassifier", "RandomForestRegressor", "MLPClassifier", "MLPRegressor")
# Modèles dont les prédictions ne changent pas quand on normalise les variables
MODELES_INSENSIBLES_ECHELLE = ("DecisionTreeClassifier", "DecisionTreeRegressor", "RandomForestClassifier",
                               "RandomForestRegressor", "LinearRegression")


def _pistes(params, sens):
    """Pistes quand le découpage est le même mais que les résultats diffèrent

    sens : "meilleur", "moins bon" ou "egal" (score de l'étudiant par rapport à DATAPULSE).
    Les pistes dépendent du modèle : la normalisation n'a aucun effet sur un arbre,
    la graine compte surtout pour une forêt ou un réseau de neurones.
    """
    modele = params["modele"]
    nom_classe = modele["classe"].rsplit(".", 1)[1]
    graine = modele["hyperparametres"].get("random_state", "absent")

    piste_graine = None
    if graine != "absent":
        piste_graine = (f"As-tu fixé random_state={graine} dans le modèle ? "
                        "Sans graine, le résultat change à chaque exécution.")

    if nom_classe in MODELES_INSENSIBLES_ECHELLE:
        piste_pretraitement = ("Ton prétraitement est-il le même : imputation des valeurs manquantes "
                               "(médiane, valeur la plus fréquente) et encodage des variables texte ?")
    else:
        piste_pretraitement = ("Ta normalisation (StandardScaler) est-elle oubliée, ou ajustée sur toutes "
                               "les données avant le découpage ? Elle doit être apprise sur X_train "
                               "uniquement, par exemple dans un Pipeline.")
        if sens == "meilleur":
            piste_pretraitement = ("Attention, un meilleur score peut venir d'une fuite de données. "
                                   + piste_pretraitement)

    piste_modele = (f"Le modèle est-il bien un {nom_classe}, avec les mêmes hyperparamètres "
                    f"(voir « Montre-moi le code » dans DATAPULSE) ?")

    if nom_classe in MODELES_ALEATOIRES:
        pistes = [piste_graine, piste_pretraitement, piste_modele]
    else:
        pistes = [piste_pretraitement, piste_modele, piste_graine]
    return [p for p in pistes if p]


def _comparer_decoupage(reference, etudiant):
    """Vérifie que l'étudiant a le même jeu de test ; renvoie (Controle, pistes)"""
    params = reference["parametres"]
    attendus = reference["jeu_de_test"]["indices"]
    obtenus = etudiant["indices"]
    n_lignes = params["donnees"]["n_lignes"]
    decoupage = params["decoupage"]

    if len(obtenus) != len(set(obtenus)):
        return Controle("Découpage", False, "des lignes apparaissent plusieurs fois dans ton jeu de test"), [
            "Ton X_test contient des doublons : vérifie le découpage."]

    if obtenus and max(obtenus) >= n_lignes:
        return Controle("Découpage", False, f"ton jeu de test contient la ligne {max(obtenus)}, "
                                            f"mais le fichier n'en a que {n_lignes}"), [
            "Utilises-tu bien le même fichier de données que DATAPULSE ?"]

    if sorted(obtenus) == sorted(attendus):
        return Controle("Découpage", True, f"même jeu de test ({len(attendus)} lignes)"), []

    communes = len(set(obtenus) & set(attendus))
    detail = f"{communes} ligne(s) de test en commun sur {len(attendus)} attendue(s)"
    if len(obtenus) != len(attendus):
        detail += f", et ton jeu de test compte {len(obtenus)} lignes"
        pistes = [f"Vérifie test_size : DATAPULSE utilise test_size={decoupage['test_size']}."]
    elif obtenus == list(range(len(obtenus))):
        pistes = ["L'index de ton X_test va de 0 à n : as-tu utilisé reset_index() ou recréé "
                  "un DataFrame ? Garde l'index d'origine pour que la comparaison retrouve les lignes."]
    else:
        pistes = [f"Vérifie random_state : DATAPULSE utilise random_state={decoupage['random_state']}."]
        if decoupage["stratify"]:
            pistes.append("Vérifie stratify : DATAPULSE utilise stratify=y.")
        else:
            pistes.append("Vérifie stratify : DATAPULSE n'en utilise pas (régression).")
    return Controle("Découpage", False, detail), pistes


def _comparer_classification(reference, etudiant, tol):
    params, test, metriques = reference["parametres"], reference["jeu_de_test"], reference["metriques"]
    classes_encodees = params["cible"]["classes_encodees"]
    controles, pistes = [], []

    # Aligner les prédictions de l'étudiant sur l'ordre des lignes de la référence
    position = {i: k for k, i in enumerate(etudiant["indices"])}
    ordre = [position[i] for i in test["indices"]]
    y_pred = np.array(_encoder(etudiant["y_pred"], classes_encodees), dtype=object)[ordre]
    y_vrai = np.array(test["y_vrai"], dtype=object)
    y_ref = np.array(test["y_pred"], dtype=object)

    accuracy = float(np.mean(y_pred == y_vrai))
    ecart_acc = accuracy - metriques["accuracy"]
    acc_ok = abs(ecart_acc) <= tol["accuracy"]
    controles.append(Controle("Accuracy", acc_ok,
                              f"{accuracy:.6f} (attendu {metriques['accuracy']:.6f})"))

    # Prédictions identiques ligne par ligne : plus strict qu'une même matrice de confusion
    differences = int((y_pred != y_ref).sum())
    matrice_ok = differences == 0
    controles.append(Controle("Prédictions", matrice_ok,
                              "identiques ligne par ligne, même matrice de confusion" if matrice_ok
                              else f"{differences} prédiction(s) différente(s) sur {len(y_ref)}"))

    proba_ok = None
    if test.get("probabilites") is not None and etudiant.get("probabilites") is not None:
        probas_etu = np.array(etudiant["probabilites"], dtype=float)[ordre]
        if etudiant.get("classes") is not None:
            # Remettre les colonnes de l'étudiant dans l'ordre des classes de la référence
            colonnes = _encoder(etudiant["classes"], classes_encodees)
            index_col = {c: k for k, c in enumerate(colonnes)}
            if all(c in index_col for c in test["classes"]):
                probas_etu = probas_etu[:, [index_col[c] for c in test["classes"]]]
        probas_ref = np.array(test["probabilites"], dtype=float)
        if probas_etu.shape != probas_ref.shape:
            proba_ok = False
            controles.append(Controle("Probabilités", False,
                                      f"{probas_etu.shape[1]} classe(s) au lieu de {probas_ref.shape[1]}"))
        else:
            ecart = float(np.abs(probas_etu - probas_ref).max())
            proba_ok = ecart <= tol["probabilites"]
            controles.append(Controle("Probabilités", proba_ok,
                                      f"écart maximal {ecart:.1e} (tolérance {tol['probabilites']:.0e})"))
    else:
        controles.append(Controle("Probabilités", True, "non comparées (absentes)"))

    if acc_ok and matrice_ok and proba_ok is not False:
        return controles, pistes, "Ton résultat est identique à celui de DATAPULSE."

    # Diagnostics (plan, section 5.4) : poser la bonne question plutôt que dire « faux »
    if not acc_ok and ecart_acc > 0:
        sens, verdict = "meilleur", "Ton score est meilleur que celui de DATAPULSE, avec le même découpage."
    elif not acc_ok:
        sens, verdict = "moins bon", "Ton score est moins bon que celui de DATAPULSE, avec le même découpage."
    elif not matrice_ok:
        sens, verdict = "egal", "Même score, mais certaines prédictions diffèrent de celles de DATAPULSE."
    else:
        sens, verdict = "egal", "Mêmes prédictions, mais tes probabilités diffèrent de celles de DATAPULSE."
    return controles, pistes + _pistes(params, sens), verdict


def _comparer_regression(reference, etudiant, tol):
    params, test, metriques = reference["parametres"], reference["jeu_de_test"], reference["metriques"]
    position = {i: k for k, i in enumerate(etudiant["indices"])}
    ordre = [position[i] for i in test["indices"]]
    y_pred = np.array(etudiant["y_pred"], dtype=float)[ordre]
    y_vrai = np.array(test["y_vrai"], dtype=float)
    y_ref = np.array(test["y_pred"], dtype=float)

    r2 = 1 - np.sum((y_vrai - y_pred) ** 2) / np.sum((y_vrai - y_vrai.mean()) ** 2)
    ecart_r2 = r2 - metriques["r2"]
    r2_ok = abs(ecart_r2) <= tol["regression"]
    controles = [Controle("R²", r2_ok, f"{r2:.6f} (attendu {metriques['r2']:.6f})")]

    echelle = max(float(np.abs(y_ref).max()), 1.0)
    ecart = float(np.abs(y_pred - y_ref).max())
    pred_ok = ecart <= tol["regression"] * echelle
    controles.append(Controle("Prédictions", pred_ok, f"écart maximal {ecart:.1e}"))

    if r2_ok and pred_ok:
        return controles, [], "Ton résultat est identique à celui de DATAPULSE."

    if not r2_ok and ecart_r2 > 0:
        sens, verdict = "meilleur", "Ton R² est meilleur que celui de DATAPULSE, avec le même découpage."
    elif not r2_ok:
        sens, verdict = "moins bon", "Ton R² est moins bon que celui de DATAPULSE, avec le même découpage."
    else:
        sens, verdict = "egal", "Même R², mais tes prédictions diffèrent légèrement de celles de DATAPULSE."
    return controles, _pistes(params, sens), verdict


def comparer(reference, etudiant):
    """Compare le résultat d'un étudiant à la référence DATAPULSE

    reference : dict (fichier de référence DATAPULSE)
    etudiant  : dict avec "indices", "y_pred" et éventuellement "probabilites" et "classes"
                (voir resultats_etudiant)
    """
    if reference.get("format") != "datapulse-reference":
        raise ValueError("Ce fichier n'est pas une référence DATAPULSE")
    if len(etudiant["indices"]) != len(etudiant["y_pred"]):
        return Comparaison(False, "Ton fichier de résultats est incohérent.",
                           [Controle("Résultats", False, f"{len(etudiant['indices'])} lignes de test "
                                                         f"mais {len(etudiant['y_pred'])} prédictions")],
                           ["Passe bien X_test et y_pred = modele.predict(X_test)."])

    tol = {**TOLERANCES_PAR_DEFAUT, **reference.get("tolerances_suggerees", {})}

    controle, pistes = _comparer_decoupage(reference, etudiant)
    if not controle.ok:
        return Comparaison(False, "Ton jeu de test n'est pas le même que celui de DATAPULSE : "
                                  "les scores ne sont pas comparables.", [controle], pistes)

    if reference["parametres"]["cible"]["type_analyse"] == "Classification":
        controles, pistes, verdict = _comparer_classification(reference, etudiant, tol)
    else:
        controles, pistes, verdict = _comparer_regression(reference, etudiant, tol)
    controles.insert(0, controle)
    conforme = all(c.ok for c in controles)
    return Comparaison(conforme, verdict, controles, pistes)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : python comparateur.py reference.json mes_resultats.json")
        sys.exit(2)
    with open(sys.argv[1], encoding="utf-8") as f:
        ref = json.load(f)
    with open(sys.argv[2], encoding="utf-8") as f:
        etu = json.load(f)
    resultat = comparer(ref, etu)
    print(resultat.rapport())
    sys.exit(0 if resultat.conforme else 1)
