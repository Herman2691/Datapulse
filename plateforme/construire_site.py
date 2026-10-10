"""
construire_site.py - Construit le site JupyterLite du module 1 (exécution dans le navigateur)

Rassemble les fichiers du module et comparateur.py dans un dossier de contenu, puis
lance « jupyter lite build ». Le site produit (dossier _site) est statique : il se
publie tel quel sur GitHub Pages, Netlify ou n'importe quel hébergement web.

Prérequis (de préférence dans un environnement virtuel séparé) :
    pip install jupyterlite-core jupyterlite-pyodide-kernel

Utilisation :
    python plateforme/construire_site.py            # construit plateforme/_site
    python -m http.server -d plateforme/_site 8000  # puis ouvrir http://localhost:8000

Auteur: Herman Kandolo
"""

import os
import shutil
import subprocess
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
CONTENU = os.path.join(ICI, "_contenu")
SITE = os.path.join(ICI, "_site")

# Fichiers visibles par l'étudiant dans le navigateur
FICHIERS_MODULE1 = [
    "module1_premiere_classification.ipynb",
    "module1_defi.ipynb",
    "module1_outils.py",
    "cancer_du_sein.csv",
    "reference_module1.json",
]


def main():
    shutil.rmtree(CONTENU, ignore_errors=True)
    os.makedirs(CONTENU)
    for nom in FICHIERS_MODULE1:
        shutil.copy(os.path.join(ICI, "module1", nom), CONTENU)
    shutil.copy(os.path.join(RACINE, "comparateur.py"), CONTENU)

    commande = [sys.executable, "-m", "jupyter", "lite", "build",
                "--contents", CONTENU, "--output-dir", SITE]
    try:
        subprocess.run(commande, check=True, cwd=ICI)
    finally:
        shutil.rmtree(CONTENU, ignore_errors=True)
    print(f"Site construit dans {SITE}")


if __name__ == "__main__":
    main()
