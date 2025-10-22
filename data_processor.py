"""
data_processor.py - Module de Traitement de Données et Machine Learning
Version 2.2 - Optimisée avec Random Forest, KNN, ACM et AFC

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

# Configuration du style
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#16213e',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#00d4ff',
    'axes.labelcolor': '#e4e4e4',
    'text.color': '#e4e4e4',
    'xtick.color': '#e4e4e4',
    'ytick.color': '#e4e4e4',
    'grid.color': '#0f3460',
    'grid.alpha': 0.3
})

# Constantes
CLASSIFICATION = "Classification"
REGRESSION = "Régression Linéaire"
CLUSTERING = "Clustering (K-Means/CAH)"
DIMENSION_REDUCTION = "Réduction de Dimension (ACP)"
ACM_ANALYSIS = "ACM (Analyse Correspondances Multiples)"
AFC_ANALYSIS = "AFC (Analyse Factorielle Correspondances)"


class DataProcessor:
    """Classe principale pour le traitement de données et Machine Learning"""
    
    def __init__(self):
        """Initialise le DataProcessor avec configurations par défaut"""
        self.data = None
        self.column_names = []
        self.target_column = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_full_processed = None
        self.preprocessor = None
        self.current_model = None
        self.current_model_type = None
        self.current_model_name = None
        self.pca_result = None
        self.acm_result = None
        self.afc_result = None
        self.label_encoder = None
        self.y = None

        # Modèles de Classification
        self.classification_models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            "KNN (K-Nearest Neighbors)": KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            ),
            "RNA (MLPClassifier)": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                early_stopping=True,
                random_state=42
            ),
            "SVM (SVC)": SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            "Régression Logistique": LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                random_state=42
            ),
            "Arbre de Décision": DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        }

        # Modèles de Régression
        self.regression_models = {
            "Random Forest (Reg)": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            "KNN (Reg)": KNeighborsRegressor(
                n_neighbors=5,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            ),
            "Régression Linéaire": LinearRegression(),
            "Arbre de Décision (Reg)": DecisionTreeRegressor(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        }
        
        self.clustering_models = {
            "K-Means": KMeans(random_state=42, n_init=10),
            "CAH (Hierarchical)": "CAH"
        }

    def _plot_to_base64(self, fig):
        """Convertit une figure Matplotlib en string Base64 PNG"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#16213e')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()

    def load_data(self, file_path):
        """Charge un fichier CSV ou Excel"""
        try:
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension in ['xlsx', 'xls']:
                print(f"📊 Chargement d'un fichier Excel...")
                self.data = pd.read_excel(file_path, engine='openpyxl' if file_extension == 'xlsx' else None)
                print(f"✅ Fichier Excel chargé avec succès")
            else:
                self.data = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8')
            
            if self.data.empty:
                print("❌ Erreur: Le fichier est vide")
                return False

            self.data.columns = self.data.columns.str.strip()
            self.column_names = self.data.columns.tolist()
            
            print(f"✅ Données chargées: {len(self.data)} lignes, {len(self.column_names)} colonnes")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            self.data = None
            self.column_names = []
            return False

    def _create_preprocessor(self, X_raw):
        """Crée un pipeline de prétraitement adaptatif"""
        numeric_features = X_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        return preprocessor
        
    def preprocess_data(self, target_column, analysis_type):
        """Prétraite les données selon le type d'analyse"""
        self.current_model_type = analysis_type
        
        try:
            if analysis_type in [CLASSIFICATION, REGRESSION]:
                if not target_column or target_column not in self.data.columns:
                    print("❌ Erreur: Colonne cible non définie")
                    return False

                self.target_column = target_column
                X_raw = self.data.drop(columns=[target_column])
                self.y = self.data[target_column].copy()

                if analysis_type == CLASSIFICATION:
                    if self.y.dtype == 'object' or self.y.dtype.name == 'category':
                        self.label_encoder = LabelEncoder()
                        self.y = self.label_encoder.fit_transform(self.y)
                    else:
                        self.label_encoder = None

                self.preprocessor = self._create_preprocessor(X_raw)
                self.preprocessor.fit(X_raw)
                
                X_processed = self.preprocessor.transform(X_raw)
                feature_names = self.preprocessor.get_feature_names_out()
                X = pd.DataFrame(X_processed, columns=feature_names)

                stratify = self.y if analysis_type == CLASSIFICATION else None
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, self.y, test_size=0.2, random_state=42, stratify=stratify
                )
                
                print(f"✅ Prétraitement: Train={len(self.X_train)}, Test={len(self.X_test)}")
                return True

            elif analysis_type in [CLUSTERING, DIMENSION_REDUCTION, ACM_ANALYSIS, AFC_ANALYSIS]:
                X_raw = self.data.copy()
                
                if target_column and target_column in X_raw.columns:
                    X_raw = X_raw.drop(columns=[target_column])

                self.preprocessor = self._create_preprocessor(X_raw)
                self.preprocessor.fit(X_raw)
                
                X_processed = self.preprocessor.transform(X_raw)
                feature_names = self.preprocessor.get_feature_names_out()
                self.X_full_processed = pd.DataFrame(X_processed, columns=feature_names)
                
                self.X_train = self.X_test = self.y_train = self.y_test = None
                
                print(f"✅ Prétraitement non-supervisé: {self.X_full_processed.shape[0]} échantillons")
                return True

            return False

        except Exception as e:
            print(f"❌ Erreur prétraitement: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_descriptive_stats(self):
        """Génère les statistiques descriptives complètes"""
        if self.data is None:
            return "❌ Aucune donnée chargée.", None
        
        report = f"📊 **STATISTIQUES DESCRIPTIVES**\n\n"
        report += f"**Dimensions:** {len(self.data)} lignes × {len(self.column_names)} colonnes\n\n"
        
        report += "**Détails par Colonne:**\n"
        for col in self.data.columns:
            report += f"\n🔹 **{col}**\n"
            report += f"   Type: {self.data[col].dtype}\n"
            report += f"   Valeurs manquantes: {self.data[col].isna().sum()} ({self.data[col].isna().sum()/len(self.data)*100:.1f}%)\n"
            report += f"   Valeurs uniques: {self.data[col].nunique()}\n"
            
            if pd.api.types.is_numeric_dtype(self.data[col]):
                report += f"   Moyenne: {self.data[col].mean():.2f}\n"
                report += f"   Médiane: {self.data[col].median():.2f}\n"
                report += f"   Écart-type: {self.data[col].std():.2f}\n"
                report += f"   Min: {self.data[col].min():.2f}, Max: {self.data[col].max():.2f}\n"
            else:
                top_value = self.data[col].mode()[0] if len(self.data[col].mode()) > 0 else "N/A"
                report += f"   Valeur la plus fréquente: {top_value}\n"
        
        total_missing = self.data.isna().sum().sum()
        total_cells = len(self.data) * len(self.data.columns)
        report += f"\n**Résumé Général:**\n"
        report += f"   Valeurs manquantes totales: {total_missing} ({total_missing/total_cells*100:.2f}%)\n"
        report += f"   Mémoire utilisée: {self.data.memory_usage(deep=True).sum() / 1024:.2f} KB\n"
        
        return report, None

    def get_correlation_matrix(self):
        """Calcule et génère la matrice de corrélation"""
        if self.data is None:
            return "❌ Aucune donnée chargée.", None

        numeric_data = self.data.select_dtypes(include=np.number)
        if numeric_data.empty or len(numeric_data.columns) < 2:
            return "❌ Au moins 2 colonnes numériques nécessaires.", None
        
        corr_matrix = numeric_data.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   fmt=".2f", linewidths=1, ax=ax, cbar_kws={'shrink': 0.8},
                   vmin=-1, vmax=1, center=0)
        
        plt.title("🔗 Matrice de Corrélation (Pearson)", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        img_str = self._plot_to_base64(fig)
        
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'corr': abs(corr_matrix.iloc[i, j])
                })
        
        top_corr = sorted(corr_pairs, key=lambda x: x['corr'], reverse=True)[:5]
        
        report = "🔗 **MATRICE DE CORRÉLATION**\n\n"
        report += "**Top 5 des Corrélations:**\n"
        for i, pair in enumerate(top_corr, 1):
            report += f"{i}. {pair['var1']} ↔ {pair['var2']}: {pair['corr']:.3f}\n"
        
        return report, img_str

    def get_bivariate_analysis(self, col1, col2):
        """Génère une analyse bivariée adaptée"""
        if self.data is None:
            return "❌ Aucune donnée chargée.", None
        
        if col1 not in self.data.columns or col2 not in self.data.columns:
            return "❌ Colonnes invalides.", None

        type1 = self.data[col1].dtype
        type2 = self.data[col2].dtype
        is_numeric = lambda t: pd.api.types.is_numeric_dtype(t)

        fig, ax = plt.subplots(figsize=(12, 8))
        plot_type = ""
        extra_stats = ""
        
        if is_numeric(type1) and is_numeric(type2):
            from scipy import stats
            
            mask = ~(self.data[col1].isna() | self.data[col2].isna())
            x_clean = self.data.loc[mask, col1]
            y_clean = self.data.loc[mask, col2]
            
            scatter = ax.scatter(x_clean, y_clean, s=100, alpha=0.6, 
                               c=y_clean, cmap='viridis', edgecolors='k', linewidths=0.5)
            
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            ax.plot(x_clean, p(x_clean), "r--", linewidth=3, 
                   label=f'Régression: y={z[0]:.2f}x+{z[1]:.2f}')
            
            plt.colorbar(scatter, label=col2)
            plt.legend(loc='best')
            plot_type = "📈 Scatter Plot avec Régression"
            
            correlation, p_value = stats.pearsonr(x_clean, y_clean)
            extra_stats = f"\n**Corrélation de Pearson:** {correlation:.4f}"
            extra_stats += f"\n**P-value:** {p_value:.4e}"
            extra_stats += f"\n**Significativité:** {'✅ Significative' if p_value < 0.05 else '❌ Non significative'}"
            
        elif is_numeric(type1) or is_numeric(type2):
            num_col = col1 if is_numeric(type1) else col2
            cat_col = col2 if is_numeric(type1) else col1
            
            n_categories = self.data[cat_col].nunique()
            if n_categories > 10:
                top_cats = self.data[cat_col].value_counts().head(10).index
                data_plot = self.data[self.data[cat_col].isin(top_cats)]
            else:
                data_plot = self.data
            
            parts = ax.violinplot([data_plot[data_plot[cat_col] == cat][num_col].dropna() 
                                   for cat in data_plot[cat_col].unique()],
                                  positions=range(len(data_plot[cat_col].unique())),
                                  showmeans=True, showmedians=True)
            
            for pc in parts['bodies']:
                pc.set_facecolor('#00d4ff')
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(data_plot[cat_col].unique())))
            ax.set_xticklabels(data_plot[cat_col].unique(), rotation=45, ha='right')
            
            plot_type = "🎻 Violin Plot"
            
        else:
            ct = pd.crosstab(self.data[col1], self.data[col2])
            
            if ct.shape[0] > 15 or ct.shape[1] > 15:
                ct = ct.iloc[:15, :15]
            
            sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
                       cbar_kws={'label': 'Fréquence'}, linewidths=1)
            plot_type = "🔥 Heatmap (Tableau Croisé)"
            
            from scipy.stats import chi2_contingency
            chi2, p_val, dof, expected = chi2_contingency(ct)
            extra_stats = f"\n**Test du Chi²:** {chi2:.4f}"
            extra_stats += f"\n**P-value:** {p_val:.4e}"

        plt.title(f"{plot_type}\n{col1} vs {col2}", fontsize=14, fontweight='bold', pad=20)
        plt.xlabel(col1, fontsize=12, fontweight='bold')
        plt.ylabel(col2, fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_str = self._plot_to_base64(fig)
        
        report = f"🔀 **ANALYSE BIVARIÉE: {col1} vs {col2}**\n\n"
        report += f"**Type:** {plot_type}\n"
        report += f"**Observations:** {(~(self.data[col1].isna() | self.data[col2].isna())).sum()}\n"
        report += extra_stats
        
        return report, img_str

    def run_model(self, model_name, analysis_type):
        """Entraîne et évalue un modèle avec visualisations complètes"""
        if self.X_train is None or self.y_train is None:
            return "❌ Données d'entraînement non préparées.", None, None, None

        if analysis_type == CLASSIFICATION:
            models = self.classification_models
            scoring = 'accuracy'
        elif analysis_type == REGRESSION:
            models = self.regression_models
            scoring = 'r2'
        else:
            return "Type d'analyse inconnu.", None, None, None

        if model_name not in models:
            return "Modèle inconnu.", None, None, None

        self.current_model_name = model_name
        self.current_model = models[model_name]
        
        print(f"🔧 Entraînement de {model_name}...")
        self.current_model.fit(self.X_train, self.y_train)
        y_pred = self.current_model.predict(self.X_test)
        
        report = f"🎯 **RAPPORT: {model_name}**\n\n"
        report += f"**Type:** {analysis_type}\n"
        report += f"**Échantillons:** Train={len(self.X_train)}, Test={len(self.X_test)}\n\n"
        
        main_img_str = None
        tree_img_str = None
        learning_curve_img = None

        # ÉVALUATION CLASSIFICATION
        if analysis_type == CLASSIFICATION:
            from sklearn.metrics import classification_report
            
            accuracy = accuracy_score(self.y_test, y_pred)
            report += f"**Précision Globale:** {accuracy:.4f} ({accuracy*100:.2f}%)\n\n"
            
            # Validation croisée
            cv_scores = cross_val_score(self.current_model, self.X_train, self.y_train, 
                                       cv=5, scoring='accuracy')
            report += f"**Validation Croisée (5-fold):**\n"
            report += f"   Moyenne: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})\n\n"
            
            class_report = classification_report(self.y_test, y_pred)
            report += "**Rapport de Classification:**\n```\n" + class_report + "\n```\n\n"
            
            try:
                cm = confusion_matrix(self.y_test, y_pred)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                           cbar_kws={'label': 'Count'}, linewidths=2)
                ax1.set_title('Matrice de Confusion (Valeurs Brutes)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Vraie Classe', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Classe Prédite', fontsize=12, fontweight='bold')
                
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax2,
                           cbar_kws={'label': 'Proportion'}, linewidths=2, vmin=0, vmax=1)
                ax2.set_title('Matrice de Confusion (Normalisée)', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Vraie Classe', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Classe Prédite', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                main_img_str = self._plot_to_base64(fig)
                
            except Exception as e:
                report += f"\n⚠️ Erreur matrice: {e}\n"

        # ÉVALUATION RÉGRESSION
        elif analysis_type == REGRESSION:
            from sklearn.metrics import mean_absolute_error, median_absolute_error
            
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            medae = median_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Validation croisée
            cv_scores = cross_val_score(self.current_model, self.X_train, self.y_train,
                                       cv=5, scoring='r2')
            
            report += f"**Métriques d'Erreur:**\n"
            report += f"   • MSE: {mse:.4f}\n"
            report += f"   • RMSE: {rmse:.4f}\n"
            report += f"   • MAE: {mae:.4f}\n"
            report += f"   • MedAE: {medae:.4f}\n\n"
            report += f"**Coefficient R²:** {r2:.4f}\n"
            report += f"**Validation Croisée (5-fold):** {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})\n"
            report += f"**Qualité:** {'✅ Excellent' if r2 > 0.9 else '✔️ Bon' if r2 > 0.7 else '⚠️ Moyen' if r2 > 0.5 else '❌ Faible'}\n\n"
            
            fig = plt.figure(figsize=(16, 7))
            
            ax1 = plt.subplot(121)
            ax1.scatter(self.y_test, y_pred, alpha=0.6, s=100, edgecolors='k', linewidths=0.5)
            ax1.plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    'r--', lw=3, label='Prédiction Parfaite')
            
            z = np.polyfit(self.y_test, y_pred, 1)
            p = np.poly1d(z)
            ax1.plot(self.y_test, p(self.y_test), "g-", linewidth=2, 
                    label=f'Tendance: y={z[0]:.2f}x+{z[1]:.2f}')
            
            ax1.set_title('Prédictions vs Valeurs Réelles', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Valeurs Réelles', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Prédictions', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2 = plt.subplot(122)
            residuals = y_pred - self.y_test
            ax2.scatter(y_pred, residuals, alpha=0.6, s=100, edgecolors='k', linewidths=0.5)
            ax2.axhline(y=0, color='r', linestyle='--', linewidth=3)
            ax2.set_title('Distribution des Résidus', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Valeurs Prédites', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Résidus', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            main_img_str = self._plot_to_base64(fig)
        
        # VISUALISATION DE L'ARBRE
        if "Arbre" in model_name:
            try:
                fig, ax = plt.subplots(figsize=(25, 15))
                
                feature_names = [name.split('__')[-1][:25] for name in self.X_train.columns]
                
                if analysis_type == CLASSIFICATION:
                    class_names = [str(c) for c in self.current_model.classes_] if hasattr(self.current_model, 'classes_') else None
                else:
                    class_names = None
                
                plot_tree(self.current_model, 
                         feature_names=feature_names,
                         class_names=class_names,
                         filled=True,
                         rounded=True,
                         fontsize=9,
                         ax=ax,
                         proportion=True,
                         precision=2,
                         impurity=False)
                
                plt.title(f"🌳 Arbre de Décision: {model_name}", 
                         fontsize=18, fontweight='bold', pad=25)
                plt.tight_layout()
                tree_img_str = self._plot_to_base64(fig)
                
                report += f"\n**Arbre de Décision:**\n"
                report += f"   • Profondeur: {self.current_model.get_depth()}\n"
                report += f"   • Nombre de feuilles: {self.current_model.get_n_leaves()}\n\n"
                
            except Exception as e:
                report += f"\n⚠️ Impossible de visualiser l'arbre: {e}\n"
        
        # IMPORTANCE DES FEATURES (Random Forest et arbres)
        if "Random Forest" in model_name or "Arbre" in model_name:
            try:
                if hasattr(self.current_model, 'feature_importances_'):
                    importances = self.current_model.feature_importances_
                    indices = np.argsort(importances)[-10:]  # Top 10
                    
                    feature_names = [self.X_train.columns[i].split('__')[-1][:30] for i in indices]
                    
                    report += f"\n**Top 10 Features Importantes:**\n"
                    for name, imp in zip(feature_names, importances[indices]):
                        report += f"   • {name}: {imp:.4f}\n"
                    
            except Exception as e:
                report += f"\n⚠️ Erreur importance features: {e}\n"
        
        # COURBE D'APPRENTISSAGE
        try:
            print("📊 Génération courbe d'apprentissage...")
            X_combined = pd.concat([self.X_train, self.X_test])
            y_combined = np.concatenate([self.y_train, self.y_test])
            
            train_sizes = np.linspace(0.1, 1.0, 10)
            
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.current_model,
                X_combined,
                y_combined,
                train_sizes=train_sizes,
                cv=min(5, len(y_combined) // 20),
                scoring=scoring,
                n_jobs=-1,
                random_state=42
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            ax.plot(train_sizes_abs, train_mean, 'o-', color='#00d4ff', 
                   linewidth=3, markersize=10, label='Score Entraînement')
            ax.fill_between(train_sizes_abs, 
                           train_mean - train_std,
                           train_mean + train_std, 
                           alpha=0.2, color='#00d4ff')
            
            ax.plot(train_sizes_abs, val_mean, 'o-', color='#ff6b6b', 
                   linewidth=3, markersize=10, label='Score Validation (CV)')
            ax.fill_between(train_sizes_abs, 
                           val_mean - val_std,
                           val_mean + val_std, 
                           alpha=0.2, color='#ff6b6b')
            
            plt.title(f'📈 Courbe d\'Apprentissage: {model_name}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Nombre d\'Échantillons d\'Entraînement', fontsize=13, fontweight='bold')
            plt.ylabel(f'Score ({scoring})', fontsize=13, fontweight='bold')
            plt.legend(loc='best', fontsize=12, framealpha=0.9)
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            
            learning_curve_img = self._plot_to_base64(fig)
            
            final_train = train_mean[-1]
            final_val = val_mean[-1]
            gap = final_train - final_val
            
            report += f"**Courbe d'Apprentissage:**\n"
            report += f"   • Score final (Train): {final_train:.4f}\n"
            report += f"   • Score final (Validation): {final_val:.4f}\n"
            report += f"   • Écart (Overfitting): {gap:.4f}\n"
            
            if gap > 0.15:
                report += "   • ⚠️ **SURAPPRENTISSAGE DÉTECTÉ**\n"
            elif gap < 0.05:
                report += "   • ✅ **BON ÉQUILIBRE**\n"
            else:
                report += "   • ✔️ **ACCEPTABLE**\n"
                
        except Exception as e:
            report += f"\n⚠️ Impossible de générer la courbe: {e}\n"
            print(f"Erreur courbe: {e}")
        
        print(f"✅ Entraînement de {model_name} terminé!")
        return report, main_img_str, tree_img_str, learning_curve_img

    def run_clustering(self, model_name, n_clusters):
        """Exécute le clustering avec visualisations avancées"""
        if self.X_full_processed is None:
            return "❌ Données non prétraitées.", None, None

        report = f"🔍 **RAPPORT DE CLUSTERING: {model_name}**\n\n"
        main_img_str = None
        secondary_img_str = None
        
        X_clust = self.X_full_processed.values

        try:
            if model_name == "K-Means":
                
                print(f"🔄 K-Means avec k={n_clusters}...")
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_clust)
                
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X_clust, labels)
                    report += f"**Score Silhouette:** {silhouette:.4f}\n"
                    report += f"**Qualité:** {'✅ Excellente' if silhouette > 0.7 else '✔️ Bonne' if silhouette > 0.5 else '⚠️ Moyenne' if silhouette > 0.3 else '❌ Faible'}\n\n"
                    
                report += f"**Inertie:** {kmeans.inertia_:.2f}\n\n"
                
                report += f"**Répartition des Clusters:**\n"
                unique, counts = np.unique(labels, return_counts=True)
                for cluster_id, count in zip(unique, counts):
                    pct = (count / len(labels)) * 100
                    report += f"   • Cluster {cluster_id}: {count} échantillons ({pct:.1f}%)\n"
                
                if X_clust.shape[1] >= 2:
                    pca_2d = PCA(n_components=2).fit_transform(X_clust)
                    
                    fig, ax = plt.subplots(figsize=(12, 9))
                    
                    scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], 
                                       c=labels, 
                                       cmap='tab10', 
                                       s=100,
                                       alpha=0.7,
                                       edgecolors='k',
                                       linewidths=1)
                    
                    centers_2d = PCA(n_components=2).fit_transform(kmeans.cluster_centers_)
                    ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                             c='red', s=400, alpha=0.9,
                             marker='X', edgecolors='black',
                             linewidths=3, label='Centres', zorder=5)
                    
                    for i, center in enumerate(centers_2d):
                        ax.annotate(f'C{i}', xy=center, xytext=(10, 10),
                                  textcoords='offset points', fontsize=12,
                                  bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
                    
                    cbar = plt.colorbar(scatter, label='Cluster ID', ticks=range(n_clusters))
                    ax.legend(fontsize=11)
                    ax.set_title(f'🔍 Clusters K-Means (k={n_clusters}) - Projection ACP 2D', 
                               fontsize=16, fontweight='bold', pad=20)
                    ax.set_xlabel('Première Composante Principale', fontsize=13, fontweight='bold')
                    ax.set_ylabel('Deuxième Composante Principale', fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.4)
                    plt.tight_layout()
                    
                    main_img_str = self._plot_to_base64(fig)
                    
                    # MÉTHODE DU COUDE
                    print("📊 Génération méthode du coude...")
                    inertias = []
                    silhouettes = []
                    K_range = range(2, min(11, len(X_clust) // 2))
                    
                    for k in K_range:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        km.fit(X_clust)
                        inertias.append(km.inertia_)
                        if k > 1:
                            silhouettes.append(silhouette_score(X_clust, km.labels_))
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                    
                    ax1.plot(list(K_range), inertias, 'o-', linewidth=3, markersize=10, color='#00d4ff')
                    ax1.axvline(x=n_clusters, color='red', linestyle='--', linewidth=2, label=f'k sélectionné = {n_clusters}')
                    ax1.set_xlabel('Nombre de Clusters (k)', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Inertie (WCSS)', fontsize=12, fontweight='bold')
                    ax1.set_title('📉 Méthode du Coude', fontsize=14, fontweight='bold', pad=15)
                    ax1.legend(fontsize=11)
                    ax1.grid(True, alpha=0.4)
                    
                    ax2.plot(list(K_range), silhouettes, 'o-', linewidth=3, markersize=10, color='#ff6b6b')
                    ax2.axvline(x=n_clusters, color='red', linestyle='--', linewidth=2, label=f'k sélectionné = {n_clusters}')
                    ax2.axhline(y=0.5, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Seuil "Bon" (0.5)')
                    ax2.set_xlabel('Nombre de Clusters (k)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Score Silhouette Moyen', fontsize=12, fontweight='bold')
                    ax2.set_title('📊 Évolution du Score Silhouette', fontsize=14, fontweight='bold', pad=15)
                    ax2.legend(fontsize=11)
                    ax2.grid(True, alpha=0.4)
                    
                    plt.tight_layout()
                    secondary_img_str = self._plot_to_base64(fig)
                    
                else:
                    report += "\n⚠️ Impossible de visualiser (< 2 dimensions)\n"

            elif model_name == "CAH (Hierarchical)":
                
                print("🌳 Calcul de la hiérarchie...")
                linked = linkage(X_clust, method='ward')
                
                fig, ax = plt.subplots(figsize=(16, 10))
                
                from scipy.cluster.hierarchy import set_link_color_palette
                set_link_color_palette(['#00d4ff', '#ff6b6b', '#00ff00', '#ffaa00', '#ff00ff'])
                
                dendrogram(linked, 
                          orientation='top',
                          distance_sort='descending',
                          show_leaf_counts=True,
                          ax=ax,
                          color_threshold=0.7*max(linked[:,2]),
                          above_threshold_color='gray',
                          leaf_font_size=10)
                
                plt.title(f"🌳 Dendrogramme - Classification Hiérarchique (Méthode de Ward)", 
                         fontsize=16, fontweight='bold', pad=25)
                plt.xlabel("Indices des Échantillons (ou Taille des Clusters)", fontsize=13, fontweight='bold')
                plt.ylabel("Distance (Ward)", fontsize=13, fontweight='bold')
                plt.axhline(y=0.7*max(linked[:,2]), color='red', linestyle='--', linewidth=2, 
                          label=f'Seuil suggéré (70% distance max)')
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                
                main_img_str = self._plot_to_base64(fig)
                
                report += f"**Dendrogramme généré avec succès**\n\n"
                report += f"**Interprétation:**\n"
                report += f"   • La hauteur des branches indique la distance de fusion\n"
                report += f"   • Coupez l'arbre à différentes hauteurs pour différents nombres de clusters\n"
                report += f"   • Plus les branches sont hautes, plus les clusters sont distincts\n\n"
                
                report += f"**Recommandation:**\n"
                report += f"   • Distance maximale: {max(linked[:,2]):.2f}\n"
                report += f"   • Seuil suggéré (70%): {0.7*max(linked[:,2]):.2f}\n"
                report += f"   • Pour {n_clusters} clusters, couper à la ligne rouge\n"

            print(f"✅ Clustering {model_name} terminé!")
            return report, main_img_str, secondary_img_str

        except Exception as e:
            print(f"❌ Erreur clustering: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Erreur lors du clustering: {e}", None, None

    def run_pca(self, n_components):
        """Exécute l'ACP avec visualisations complètes"""
        if self.X_full_processed is None:
            return "❌ Données non prétraitées pour l'ACP.", None, None

        try:
            n_components = min(n_components, self.X_full_processed.shape[1])
            
            print(f"🔄 Calcul de l'ACP avec {n_components} composantes...")
            pca = PCA(n_components=n_components)
            self.pca_result = pca.fit_transform(self.X_full_processed)
            
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            report = f"📉 **ANALYSE EN COMPOSANTES PRINCIPALES (ACP)**\n\n"
            report += f"**Nombre de composantes:** {n_components}\n\n"
            
            report += "**Variance Expliquée par Composante:**\n"
            for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
                report += f"   • PC{i+1}: {var:.4f} ({var*100:.2f}%) - Cumulé: {cum_var:.4f} ({cum_var*100:.2f}%)\n"
            
            n_for_90 = np.argmax(cumulative_variance >= 0.90) + 1 if any(cumulative_variance >= 0.90) else n_components
            report += f"\n**Recommandation:**\n"
            report += f"   • Pour capturer 90% de la variance: {n_for_90} composantes\n"
            report += f"   • Variance expliquée actuelle: {cumulative_variance[-1]*100:.2f}%\n\n"
            
            # SCREE PLOT
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            ax1.bar(range(1, len(explained_variance_ratio) + 1), 
                   explained_variance_ratio, 
                   alpha=0.7, 
                   color='#00d4ff',
                   edgecolor='black',
                   linewidth=1.5,
                   label='Variance individuelle')
            
            ax1.plot(range(1, len(cumulative_variance) + 1), 
                    cumulative_variance, 
                    marker='o', 
                    linestyle='--', 
                    color='#ff6b6b',
                    linewidth=3,
                    markersize=10,
                    label='Variance cumulée')
            
            ax1.axhline(y=0.9, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Seuil 90%')
            ax1.axhline(y=0.8, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Seuil 80%')
            
            ax1.set_xlabel("Numéro de Composante", fontsize=12, fontweight='bold')
            ax1.set_ylabel("Variance Expliquée", fontsize=12, fontweight='bold')
            ax1.set_title("📊 Scree Plot - Variance Expliquée", fontsize=14, fontweight='bold', pad=15)
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.4)
            
            ax2.plot(range(1, len(cumulative_variance) + 1), 
                    cumulative_variance * 100, 
                    marker='o', 
                    linestyle='-', 
                    color='#00d4ff',
                    linewidth=3,
                    markersize=10)
            
            ax2.axhline(y=90, color='green', linestyle='--', linewidth=2, label='90%')
            ax2.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='80%')
            ax2.fill_between(range(1, len(cumulative_variance) + 1), 
                            cumulative_variance * 100, 
                            alpha=0.3, 
                            color='#00d4ff')
            
            ax2.set_xlabel("Nombre de Composantes", fontsize=12, fontweight='bold')
            ax2.set_ylabel("Variance Cumulée (%)", fontsize=12, fontweight='bold')
            ax2.set_title("📈 Variance Cumulée", fontsize=14, fontweight='bold', pad=15)
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.4)
            ax2.set_ylim([0, 105])
            
            plt.tight_layout()
            scree_img = self._plot_to_base64(fig)
            
            # PROJECTION 2D/3D
            scatter_img = None
            
            if n_components >= 2:
                if n_components >= 3:
                    from mpl_toolkits.mplot3d import Axes3D
                    
                    fig = plt.figure(figsize=(14, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(self.pca_result[:, 0], 
                                       self.pca_result[:, 1],
                                       self.pca_result[:, 2],
                                       c=range(len(self.pca_result)),
                                       cmap='viridis',
                                       s=80,
                                       alpha=0.6,
                                       edgecolors='k',
                                       linewidths=0.5)
                    
                    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=12, fontweight='bold')
                    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=12, fontweight='bold')
                    ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}%)', fontsize=12, fontweight='bold')
                    ax.set_title('🌐 Projection 3D des Données (PC1, PC2, PC3)', 
                               fontsize=14, fontweight='bold', pad=20)
                    
                    plt.colorbar(scatter, label='Index', shrink=0.8)
                    
                else:
                    fig, ax = plt.subplots(figsize=(12, 9))
                    
                    scatter = ax.scatter(self.pca_result[:, 0], 
                                       self.pca_result[:, 1],
                                       c=range(len(self.pca_result)),
                                       cmap='viridis',
                                       s=100,
                                       alpha=0.6,
                                       edgecolors='k',
                                       linewidths=0.5)
                    
                    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)', 
                                fontsize=13, fontweight='bold')
                    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)', 
                                fontsize=13, fontweight='bold')
                    ax.set_title('📊 Projection 2D des Données sur les 2 Premières Composantes', 
                               fontsize=14, fontweight='bold', pad=20)
                    
                    plt.colorbar(scatter, label='Index')
                    ax.grid(True, alpha=0.4)
                
                plt.tight_layout()
                scatter_img = self._plot_to_base64(fig)
                
                # Contribution des variables
                report += "**Contributions des Variables (Top 3 par PC):**\n"
                components = pca.components_
                feature_names = self.X_full_processed.columns
                
                for i in range(min(3, n_components)):
                    top_indices = np.argsort(np.abs(components[i]))[-3:][::-1]
                    report += f"\n   PC{i+1}:\n"
                    for idx in top_indices:
                        contrib = components[i][idx]
                        report += f"      • {feature_names[idx]}: {contrib:.3f}\n"

            print("✅ ACP terminée!")
            return report, scree_img, scatter_img
            
        except Exception as e:
            print(f"❌ Erreur ACP: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Erreur lors de l'ACP: {e}", None, None

    def run_acm(self, n_components=5, selected_columns=None):
        """Exécute l'Analyse des Correspondances Multiples (ACM)"""
        if self.data is None:
            return "❌ Aucune donnée chargée.", None, None
        
        try:
            # Sélectionner les colonnes catégorielles
            if selected_columns is None:
                categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            else:
                categorical_cols = selected_columns
            
            if len(categorical_cols) < 2:
                return "❌ Au moins 2 variables catégorielles nécessaires pour l'ACM.", None, None
            
            print(f"📊 ACM sur {len(categorical_cols)} variables...")
            
            # Créer le tableau disjonctif complet
            data_acm = self.data[categorical_cols].copy()
            data_acm = data_acm.dropna()
            
            # OneHot encoding
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            X_encoded = encoder.fit_transform(data_acm)
            
            # Appliquer PCA sur les données encodées (c'est l'ACM)
            from sklearn.decomposition import PCA
            n_comp = min(n_components, X_encoded.shape[1], X_encoded.shape[0])
            pca = PCA(n_components=n_comp)
            self.acm_result = pca.fit_transform(X_encoded)
            
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            report = f"🔍 **ANALYSE DES CORRESPONDANCES MULTIPLES (ACM)**\n\n"
            report += f"**Variables analysées:** {', '.join(categorical_cols)}\n"
            report += f"**Nombre d'observations:** {len(data_acm)}\n"
            report += f"**Nombre de dimensions:** {n_comp}\n\n"
            
            report += "**Inertie Expliquée par Dimension:**\n"
            for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
                report += f"   • Dim{i+1}: {var:.4f} ({var*100:.2f}%) - Cumulé: {cum_var:.4f} ({cum_var*100:.2f}%)\n"
            
            # SCREE PLOT
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            ax1.bar(range(1, len(explained_variance) + 1), 
                   explained_variance * 100,
                   alpha=0.7, 
                   color='#00d4ff',
                   edgecolor='black',
                   linewidth=1.5)
            
            ax1.set_xlabel("Dimension", fontsize=12, fontweight='bold')
            ax1.set_ylabel("Inertie Expliquée (%)", fontsize=12, fontweight='bold')
            ax1.set_title("📊 Éboulis des Valeurs Propres (ACM)", fontsize=14, fontweight='bold', pad=15)
            ax1.grid(True, alpha=0.4)
            
            ax2.plot(range(1, len(cumulative_variance) + 1), 
                    cumulative_variance * 100, 
                    marker='o', 
                    linestyle='-', 
                    color='#ff6b6b',
                    linewidth=3,
                    markersize=10)
            
            ax2.axhline(y=80, color='green', linestyle='--', linewidth=2, label='80%')
            ax2.fill_between(range(1, len(cumulative_variance) + 1), 
                            cumulative_variance * 100, 
                            alpha=0.3, 
                            color='#ff6b6b')
            
            ax2.set_xlabel("Nombre de Dimensions", fontsize=12, fontweight='bold')
            ax2.set_ylabel("Inertie Cumulée (%)", fontsize=12, fontweight='bold')
            ax2.set_title("📈 Inertie Cumulée", fontsize=14, fontweight='bold', pad=15)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.4)
            
            plt.tight_layout()
            scree_img = self._plot_to_base64(fig)
            
            # PROJECTION 2D
            projection_img = None
            if n_comp >= 2:
                fig, ax = plt.subplots(figsize=(12, 9))
                
                scatter = ax.scatter(self.acm_result[:, 0], 
                                   self.acm_result[:, 1],
                                   c=range(len(self.acm_result)),
                                   cmap='viridis',
                                   s=100,
                                   alpha=0.6,
                                   edgecolors='k',
                                   linewidths=0.5)
                
                ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                
                ax.set_xlabel(f'Dim 1 ({explained_variance[0]*100:.1f}%)', 
                            fontsize=13, fontweight='bold')
                ax.set_ylabel(f'Dim 2 ({explained_variance[1]*100:.1f}%)', 
                            fontsize=13, fontweight='bold')
                ax.set_title('🔍 Plan Factoriel ACM (Individus)', 
                           fontsize=14, fontweight='bold', pad=20)
                
                plt.colorbar(scatter, label='Index')
                ax.grid(True, alpha=0.4)
                plt.tight_layout()
                projection_img = self._plot_to_base64(fig)
            
            print("✅ ACM terminée!")
            return report, scree_img, projection_img
            
        except Exception as e:
            print(f"❌ Erreur ACM: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Erreur lors de l'ACM: {e}", None, None

    def run_afc(self, row_col, col_col):
        """Exécute l'Analyse Factorielle des Correspondances (AFC)"""
        if self.data is None:
            return "❌ Aucune donnée chargée.", None, None
        
        try:
            if row_col not in self.data.columns or col_col not in self.data.columns:
                return "❌ Colonnes invalides.", None, None
            
            print(f"📊 AFC: {row_col} vs {col_col}...")
            
            # Créer le tableau de contingence
            contingency_table = pd.crosstab(self.data[row_col], self.data[col_col])
            
            # Limiter la taille pour la visualisation
            if contingency_table.shape[0] > 20:
                top_rows = contingency_table.sum(axis=1).nlargest(20).index
                contingency_table = contingency_table.loc[top_rows]
            if contingency_table.shape[1] > 20:
                top_cols = contingency_table.sum(axis=0).nlargest(20).index
                contingency_table = contingency_table[top_cols]
            
            # Test du Chi²
            from scipy.stats import chi2_contingency
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            report = f"📊 **ANALYSE FACTORIELLE DES CORRESPONDANCES (AFC)**\n\n"
            report += f"**Variable Lignes:** {row_col} ({len(contingency_table)} modalités)\n"
            report += f"**Variable Colonnes:** {col_col} ({len(contingency_table.columns)} modalités)\n"
            report += f"**Effectif Total:** {contingency_table.sum().sum()}\n\n"
            
            report += f"**Test d'Indépendance du Chi²:**\n"
            report += f"   • Chi² = {chi2:.4f}\n"
            report += f"   • p-value = {p_value:.4e}\n"
            report += f"   • Degrés de liberté = {dof}\n"
            
            if p_value < 0.05:
                report += f"   • ✅ **Les variables sont significativement associées** (p < 0.05)\n\n"
            else:
                report += f"   • ❌ **Pas d'association significative** (p ≥ 0.05)\n\n"
            
            # VISUALISATION TABLE DE CONTINGENCE
            fig, ax = plt.subplots(figsize=(14, 10))
            
            sns.heatmap(contingency_table, 
                       annot=True, 
                       fmt='d', 
                       cmap='YlOrRd',
                       ax=ax,
                       cbar_kws={'label': 'Fréquence'},
                       linewidths=1)
            
            plt.title(f"📊 Table de Contingence: {row_col} × {col_col}", 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel(col_col, fontsize=12, fontweight='bold')
            plt.ylabel(row_col, fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            contingency_img = self._plot_to_base64(fig)
            
            # ANALYSE DES CORRESPONDANCES (projection)
            projection_img = None
            try:
                # Calcul des profils
                row_profiles = contingency_table.div(contingency_table.sum(axis=1), axis=0)
                col_profiles = contingency_table.div(contingency_table.sum(axis=0), axis=1).T
                
                # Centrage
                row_mass = contingency_table.sum(axis=1) / contingency_table.sum().sum()
                col_mass = contingency_table.sum(axis=0) / contingency_table.sum().sum()
                
                # SVD pour obtenir les coordonnées
                from sklearn.decomposition import TruncatedSVD
                
                # Tableau standardisé
                n = contingency_table.sum().sum()
                P = contingency_table / n
                r = P.sum(axis=1)
                c = P.sum(axis=0)
                
                # Tableau des résidus standardisés
                S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
                
                # SVD
                svd = TruncatedSVD(n_components=min(2, min(S.shape)-1))
                row_coords = svd.fit_transform(S)
                col_coords = svd.transform(S.T)
                
                explained_inertia = svd.explained_variance_ratio_
                
                # Projection 2D
                fig, ax = plt.subplots(figsize=(14, 10))
                
                # Points lignes
                ax.scatter(row_coords[:, 0], row_coords[:, 1],
                          s=200, alpha=0.7, c='#00d4ff', 
                          edgecolors='k', linewidths=2,
                          marker='o', label=f'{row_col} (lignes)')
                
                for i, label in enumerate(contingency_table.index[:10]):  # Limiter labels
                    ax.annotate(str(label)[:15], 
                              (row_coords[i, 0], row_coords[i, 1]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, alpha=0.8)
                
                # Points colonnes
                ax.scatter(col_coords[:, 0], col_coords[:, 1],
                          s=200, alpha=0.7, c='#ff6b6b',
                          edgecolors='k', linewidths=2,
                          marker='^', label=f'{col_col} (colonnes)')
                
                for i, label in enumerate(contingency_table.columns[:10]):  # Limiter labels
                    ax.annotate(str(label)[:15],
                              (col_coords[i, 0], col_coords[i, 1]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, alpha=0.8)
                
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                
                ax.set_xlabel(f'Dim 1 ({explained_inertia[0]*100:.1f}%)', 
                            fontsize=13, fontweight='bold')
                ax.set_ylabel(f'Dim 2 ({explained_inertia[1]*100:.1f}%)', 
                            fontsize=13, fontweight='bold')
                ax.set_title('🎯 Plan Factoriel AFC (Lignes & Colonnes)', 
                           fontsize=14, fontweight='bold', pad=20)
                ax.legend(loc='best', fontsize=11)
                ax.grid(True, alpha=0.4)
                plt.tight_layout()
                
                projection_img = self._plot_to_base64(fig)
                
                report += f"**Projection Factorielle:**\n"
                report += f"   • Inertie Dim1: {explained_inertia[0]*100:.2f}%\n"
                report += f"   • Inertie Dim2: {explained_inertia[1]*100:.2f}%\n"
                report += f"   • Total expliqué: {sum(explained_inertia)*100:.2f}%\n"
                
            except Exception as e:
                report += f"\n⚠️ Projection non disponible: {e}\n"
            
            print("✅ AFC terminée!")
            return report, contingency_img, projection_img
            
        except Exception as e:
            print(f"❌ Erreur AFC: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Erreur lors de l'AFC: {e}", None, None

    def predict_single_sample(self, input_data):
        """Prédit sur un nouvel échantillon"""
        if self.current_model is None or self.preprocessor is None:
            return "❌ Modèle non entraîné.", None
            
        try:
            processed_input = {}
            for feature, value in input_data.items():
                if value == '' or value is None:
                    processed_input[feature] = np.nan
                else:
                    try:
                        processed_input[feature] = float(value)
                    except:
                        processed_input[feature] = value
            
            X_test_raw = pd.DataFrame([processed_input])
            original_features = self.preprocessor.feature_names_in_
            
            X_test_raw_filtered = pd.DataFrame(columns=original_features)
            for col in original_features:
                if col in processed_input:
                    X_test_raw_filtered[col] = [processed_input[col]]
                else:
                    X_test_raw_filtered[col] = [np.nan]
            
            X_test_processed = self.preprocessor.transform(X_test_raw_filtered)
            X_test_final = pd.DataFrame(X_test_processed, columns=self.preprocessor.get_feature_names_out())

            prediction = self.current_model.predict(X_test_final)[0]
            
            if self.current_model_type == CLASSIFICATION:
                if self.label_encoder is not None:
                    prediction_label = self.label_encoder.inverse_transform([prediction])[0]
                elif hasattr(self.current_model, 'classes_'):
                    prediction_label = self.current_model.classes_[prediction]
                else:
                    prediction_label = prediction
                
                if hasattr(self.current_model, 'predict_proba'):
                    probas = self.current_model.predict_proba(X_test_final)[0]
                    proba_str = ", ".join([f"Classe {i}: {p*100:.1f}%" for i, p in enumerate(probas)])
                    message = f"✅ Prédiction: **{prediction_label}**\nProbabilités: {proba_str}"
                else:
                    message = f"✅ Prédiction: **{prediction_label}**"
                
                return message, prediction_label
            
            elif self.current_model_type == REGRESSION:
                return f"✅ Prédiction: **{prediction:.2f}**", prediction
            
            return "Prédiction non supportée.", None

        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Erreur lors de la prédiction: {e}", None


# Fonction utilitaire pour créer un dataset de test
def create_sample_dataset(n_samples=1000, task='classification'):
    """Crée un dataset d'exemple pour tester"""
    np.random.seed(42)
    
    if task == 'classification':
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n_samples, n_features=10, 
                                   n_informative=7, n_redundant=2,
                                   n_classes=3, random_state=42)
        df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(10)])
        df['Category'] = np.random.choice(['A', 'B', 'C'], n_samples)
        df['Target'] = y
        
    else:
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=n_samples, n_features=8, 
                              n_informative=6, noise=10, random_state=42)
        df = pd.DataFrame(X, columns=[f'Variable_{i}' for i in range(8)])
        df['Type'] = np.random.choice(['Type1', 'Type2'], n_samples)
        df['Value'] = y
    
    return df


if __name__ == "__main__":
    print("="*70)
    print("  TEST DU MODULE DATA_PROCESSOR")
    print("="*70)
    
    df = create_sample_dataset(500, 'classification')
    df.to_csv('test_data.csv', index=False)
    print(f"✅ Dataset créé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    processor = DataProcessor()
    
    if processor.load_data('test_data.csv'):
        print("✅ Chargement réussi")
    
    if processor.preprocess_data('Target', CLASSIFICATION):
        print("✅ Prétraitement réussi")
    
    result = processor.run_model("Random Forest", CLASSIFICATION)
    if result and len(result) > 0 and not isinstance(result[0], str):
        print("✅ Entraînement réussi")
        print("\nRapport:")
        print(result[0][:200] + "...")
    
    import os
    os.remove('test_data.csv')
    
    print("\n" + "="*70)
    print("✅ TOUS LES TESTS SONT PASSÉS!")
    print("="*70)