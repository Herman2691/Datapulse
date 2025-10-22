"""
data_processor.py - Module de Traitement de Données et Machine Learning
Version 3.0 - Avec AFC, ACM et support Excel

Auteur: Herman Kandolo
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Classification / Régression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Clustering / Réduction de dimension
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, mean_absolute_error

# AFC et ACM
from sklearn.preprocessing import LabelEncoder as LE

# Métriques et Visualisation
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                            ConfusionMatrixDisplay, roc_curve, roc_auc_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

CLASSIFICATION = "Classification"
REGRESSION = "Régression Linéaire"
CLUSTERING = "Clustering (K-Means/CAH)"
DIMENSION_REDUCTION = "Réduction de Dimension (ACP)"
AFC_ANALYSIS = "AFC (Analyse Factorielle des Correspondances)"
ACM_ANALYSIS = "ACM (Analyse des Correspondances Multiples)"

class DataProcessor:
    def __init__(self):
        self.data = None
        self.column_names = []

    # ✅ Fonction corrigée
    def load_data(self, file_path):
        """Charge un fichier CSV ou Excel avec détection automatique (corrigée)"""
        try:
            file_extension = file_path.lower().split('.')[-1]

            # --- Excel ---
            if file_extension in ['xlsx', 'xls', 'xlsm', 'xlsb']:
                try:
                    self.data = pd.read_excel(file_path, engine='openpyxl')
                except Exception:
                    try:
                        self.data = pd.read_excel(file_path, engine='xlrd')
                    except Exception:
                        try:
                            self.data = pd.read_excel(file_path, engine='pyxlsb')
                        except Exception:
                            self.data = pd.read_excel(file_path)
                if self.data is None or self.data.empty:
                    try:
                        self.data = pd.read_excel(file_path, engine='pyxlsb')
                        print("✅ Lecture alternative via pyxlsb réussie")
                    except Exception:
                        pass
                print("✅ Fichier Excel chargé")

            # --- CSV ---
            elif file_extension == 'csv':
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                separators = [',', ';', '\t', '|']
                loaded = False

                for encoding in encodings:
                    for sep in separators:
                        try:
                            self.data = pd.read_csv(file_path, sep=sep, encoding=encoding, engine='python')
                            if len(self.data.columns) > 1 or (len(self.data.columns) == 1 and sep not in str(self.data.columns[0])):
                                loaded = True
                                print(f"✅ Fichier CSV chargé (encodage={encoding}, séparateur='{sep}')")
                                break
                        except Exception:
                            continue
                    if loaded:
                        break

                # ✅ Correction si une seule colonne
                if loaded and (self.data.shape[1] == 1):
                    print("⚠️ Une seule colonne détectée — tentative de correction automatique…")
                    for sep in [';', ',', '\t', '|']:
                        try:
                            tmp_df = pd.read_csv(file_path, sep=sep, encoding=encoding, engine='python')
                            if tmp_df.shape[1] > 1:
                                self.data = tmp_df
                                print(f"✅ Correction automatique réussie (séparateur='{sep}')")
                                break
                        except Exception:
                            continue

                if not loaded:
                    self.data = pd.read_csv(file_path)
                    print("✅ Fichier CSV chargé (méthode par défaut)")

            else:
                print(f"❌ Format non supporté: {file_extension}")
                return False

            if self.data is None or self.data.empty:
                print("❌ Erreur: le fichier est vide ou non lisible")
                return False

            self.data.columns = self.data.columns.str.strip()
            self.column_names = self.data.columns.tolist()
            print(f"[DEBUG] Chargement terminé : {file_extension} | Colonnes = {len(self.column_names)} | Shape = {self.data.shape}")
            return True

        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            import traceback
            traceback.print_exc()
            self.data = None
            self.column_names = []
            return False

    # (autres fonctions inchangées...)
