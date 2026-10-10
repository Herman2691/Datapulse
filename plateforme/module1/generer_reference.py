"""
generer_reference.py - Prépare les fichiers du module 1 à partir de DATAPULSE

Crée, à côté de ce script :
- cancer_du_sein.csv      : le jeu de données du module, à charger dans DATAPULSE (étape B)
- reference_module1.json  : la référence DATAPULSE (régression logistique, test 20 %, graine 42),
                            utilisée pour la comparaison de l'étape F

À relancer si DATAPULSE ou scikit-learn change de version :
    python plateforme/module1/generer_reference.py

Auteur: Herman Kandolo
"""

import os
import sys

from sklearn.datasets import load_breast_cancer

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ICI, "..", ".."))

from data_processor import DataProcessor, CLASSIFICATION  # noqa: E402


def main():
    # Même ordre de lignes que load_breast_cancer : la ligne i du CSV est l'index i de X
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    donnees = X.copy()
    donnees["target"] = y
    chemin_csv = os.path.join(ICI, "cancer_du_sein.csv")
    donnees.to_csv(chemin_csv, index=False)

    processor = DataProcessor()
    processor.load_data(chemin_csv)
    processor.preprocess_data("target", CLASSIFICATION, test_size=0.2, random_state=42)
    processor.run_model("Régression Logistique", CLASSIFICATION)

    chemin_reference = os.path.join(ICI, "reference_module1.json")
    with open(chemin_reference, "w", encoding="utf-8") as f:
        f.write(processor.get_reference_json())

    print(f"Accuracy de référence : {processor.run_metrics['accuracy']:.6f}")
    print(f"Fichiers créés : {chemin_csv}, {chemin_reference}")


if __name__ == "__main__":
    main()
