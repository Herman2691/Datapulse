"""
Datapulse.py -DATAPULSE Pro Web Application
Application Streamlit pour l'analyse de données et Machine Learning

⚠️ FICHIER À LANCER AVEC: streamlit run Datapulse.py

Auteur: Herman Kandolo chercheur en Intelligence Artificielle et Data Science
Date: 2025
Version: 3.0 Professional Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
from PIL import Image

# Import du DataProcessor
from data_processor import DataProcessor, CLASSIFICATION, REGRESSION, CLUSTERING, DIMENSION_REDUCTION, ACM_ANALYSIS, AFC_ANALYSIS

# Configuration de la page
st.set_page_config(
    page_title="DATAPULSE Pro - ML Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS professionnel moderne - Thème Mode Clair (Style Gemini)
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Variables CSS - Palette Mode Clair */
    :root {
        --primary-color: #6B5CE7;
        --secondary-color: #8B7EF5;
        --accent-color: #A59BF9;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --bg-primary: #F0F4F9;
        --bg-secondary: #FFFFFF;
        --text-primary: #1F1F1F;
        --text-secondary: #5E5E5E;
        --border-color: #D1D9E6;
        --qwen-gradient-start: #6B5CE7;
        --qwen-gradient-end: #4C3FD9;
    }
    
    /* Reset et base */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: var(--bg-primary);
        padding: 2rem;
    }
    
    .stApp {
        background: var(--bg-primary);
    }
    
    /* Header professionnel - Style Mode Clair */
    .pro-header {
        background: linear-gradient(135deg, var(--qwen-gradient-start) 0%, var(--qwen-gradient-end) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(107, 92, 231, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .pro-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .pro-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.15);
        letter-spacing: -0.5px;
    }
    
    .pro-header .subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.25rem;
        font-weight: 400;
        margin: 0.5rem 0;
    }
    
    .pro-header .version {
        color: rgba(255,255,255,0.85);
        font-size: 0.875rem;
        font-weight: 300;
    }
    
    /* Titres et sections */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    h2 {
        font-size: 2rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid var(--primary-color);
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h3 {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: var(--text-primary);
    }
    
    /* Cards professionnelles */
    .pro-card {
        background: var(--bg-secondary);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .pro-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(107, 92, 231, 0.12);
        border-color: var(--primary-color);
    }
    
    .pro-card h4 {
        color: var(--primary-color);
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .pro-card p {
        color: var(--text-secondary);
        line-height: 1.6;
        margin: 0.5rem 0;
    }
    
    /* Boutons professionnels - Style Mode Clair */
    .stButton > button {
        background: linear-gradient(135deg, var(--qwen-gradient-start) 0%, var(--qwen-gradient-end) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(107, 92, 231, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(107, 92, 231, 0.3);
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Metrics professionnelles */
    .stMetric {
        background: var(--bg-secondary);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        border-color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(107, 92, 231, 0.12);
    }
    
    .stMetric label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: var(--primary-color);
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Messages de statut */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.04) 100%);
        border-left: 4px solid var(--success-color);
        border-radius: 8px;
        padding: 1rem;
        color: var(--text-primary);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.04) 100%);
        border-left: 4px solid var(--error-color);
        border-radius: 8px;
        padding: 1rem;
        color: var(--text-primary);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(245, 158, 11, 0.04) 100%);
        border-left: 4px solid var(--warning-color);
        border-radius: 8px;
        padding: 1rem;
        color: var(--text-primary);
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(107, 92, 231, 0.08) 0%, rgba(107, 92, 231, 0.04) 100%);
        border-left: 4px solid var(--primary-color);
        border-radius: 8px;
        padding: 1rem;
        color: var(--text-primary);
    }
    
    /* Sidebar professionnel */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FAFBFC 0%, var(--bg-secondary) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: var(--primary-color);
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-secondary);
        border-radius: 10px;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: var(--primary-color);
        background: linear-gradient(135deg, var(--bg-secondary) 0%, rgba(107, 92, 231, 0.05) 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-secondary);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(107, 92, 231, 0.08);
        color: var(--primary-color);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }
    
    /* File uploader */
    .stFileUploader {
        background: var(--bg-secondary);
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-color);
        background: rgba(107, 92, 231, 0.03);
    }
    
    /* Select boxes */
    .stSelectbox, .stMultiSelect {
        color: var(--text-primary);
    }
    
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: var(--bg-secondary);
        border-color: var(--border-color);
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }
    
    /* Feature badge */
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Prediction result */
    .prediction-result {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 32px rgba(107, 92, 231, 0.2);
        margin: 2rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #E8EDF5;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-color);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de la session state
if 'processor' not in st.session_state:
    st.session_state.processor = DataProcessor()
    st.session_state.data_loaded = False
    st.session_state.model_trained = False
    st.session_state.analysis_history = []
    st.session_state.current_model_info = None

def main():
    """Fonction principale de l'application"""
    
    # En-tête professionnel
    st.markdown("""
    <div class='pro-header'>
        <h1>🚀 DATAPULSE PRO</h1>
        <p class='subtitle'>Plateforme Professionnelle d'Analyse ML & Data Science</p>
        <p class='version'>Conçu par Herman Kandolo • Version 3.0 Professional • Excel & CSV Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar pour la navigation
    with st.sidebar:
        st.markdown("### 📋 NAVIGATION")
        
        page = st.radio(
            "",
            ["🏠 Accueil", 
             "📂 Chargement & EDA", 
             "🎯 Modélisation Supervisée",
             "🔍 Analyse Non-Supervisée",
             "📊 ACM & AFC",
             "🧪 Test du Modèle",
             "⚖️ Comparaison",
             "📊 Historique",
             "📤 Export"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Statut des données
        st.markdown("### 📈 STATUT")
        
        if st.session_state.data_loaded:
            st.success("✅ Données chargées")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📋 Lignes", len(st.session_state.processor.data))
            with col2:
                st.metric("📊 Colonnes", len(st.session_state.processor.column_names))
        else:
            st.warning("⚠️ Aucune donnée")
        
        if st.session_state.model_trained:
            st.success("✅ Modèle entraîné")
            if st.session_state.current_model_info:
                st.info(f"**{st.session_state.current_model_info['model']}**")
        
        st.markdown("---")
        st.markdown("### 🔧 ACTIONS RAPIDES")
        
        if st.button("🔄 Nouveau Projet", use_container_width=True):
            if st.session_state.data_loaded:
                st.session_state.processor = DataProcessor()
                st.session_state.data_loaded = False
                st.session_state.model_trained = False
                st.session_state.analysis_history = []
                st.rerun()
    
    # Routage des pages
    if page == "🏠 Accueil":
        page_accueil()
    elif page == "📂 Chargement & EDA":
        page_chargement_eda()
    elif page == "🎯 Modélisation Supervisée":
        page_supervise()
    elif page == "🔍 Analyse Non-Supervisée":
        page_non_supervise()
    elif page == "📊 ACM & AFC":
        page_acm_afc()
    elif page == "🧪 Test du Modèle":
        page_test_modele()
    elif page == "⚖️ Comparaison":
        page_comparaison()
    elif page == "📊 Historique":
        page_historique()
    elif page == "📤 Export":
        page_export()

def page_accueil():
    """Page d'accueil professionnelle"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class='pro-card' style='text-align: center;'>
            <h2 style='border: none; background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                👋 Bienvenue sur DATAPULSE Pro
            </h2>
            <p style='font-size: 1.1rem; color: var(--text-secondary); line-height: 1.8;'>
                Votre plateforme complète et professionnelle pour l'analyse de données 
                et le Machine Learning de nouvelle génération
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fonctionnalités principales
    st.markdown("## ✨ Fonctionnalités Principales")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        <div class='pro-card'>
            <h4>📊 Analyse Exploratoire Avancée</h4>
            <p>
                • Statistiques descriptives complètes et détaillées<br>
                • Matrices de corrélation interactives haute définition<br>
                • Visualisations bivariées automatiques<br>
                • Détection intelligente de valeurs manquantes<br>
                • <span class='feature-badge'>CSV</span> <span class='feature-badge'>EXCEL</span> Support complet
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='pro-card'>
            <h4>🎯 Machine Learning Supervisé</h4>
            <p>
                • <strong>Random Forest</strong> - Classification & Régression<br>
                • <strong>KNN</strong> - K-Nearest Neighbors optimisé<br>
                • <strong>RNA</strong> - Réseaux de neurones profonds<br>
                • <strong>SVM</strong> - Support Vector Machines<br>
                • Courbes d'apprentissage automatiques<br>
                • Analyse d'importance des features
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col2:
        st.markdown("""
        <div class='pro-card'>
            <h4>🔍 Machine Learning Non-Supervisé</h4>
            <p>
                • Clustering K-Means avec optimisation automatique<br>
                • Classification Hiérarchique Ascendante (CAH)<br>
                • Réduction de dimension (ACP avancée)<br>
                • <span class='feature-badge'>ACM</span> Analyse Correspondances Multiples<br>
                • <span class='feature-badge'>AFC</span> Analyse Factorielle Correspondances<br>
                • Méthode du coude et silhouette score
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='pro-card'>
            <h4>🧪 Test & Export Professionnel</h4>
            <p>
                • Interface de test interactive et intuitive<br>
                • Comparaison multi-modèles en temps réel<br>
                • Historique détaillé des analyses<br>
                • Export <span class='feature-badge'>CSV</span> <span class='feature-badge'>JSON</span> <span class='feature-badge'>EXCEL</span><br>
                • Sauvegarde de modèles entraînés
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Modèles disponibles
    st.markdown("## 🤖 Algorithmes de Machine Learning")
    
    models_col1, models_col2, models_col3 = st.columns(3)
    
    with models_col1:
        st.markdown("""
        <div class='pro-card'>
            <h4>Classification</h4>
            <p>
                🌳 Random Forest<br>
                🎯 KNN (K-Nearest Neighbors)<br>
                🧠 RNA (MLPClassifier)<br>
                ⚡ SVM (Support Vector Machine)<br>
                📈 Régression Logistique<br>
                🌲 Arbre de Décision
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with models_col2:
        st.markdown("""
        <div class='pro-card'>
            <h4>Régression</h4>
            <p>
                🌳 Random Forest Regressor<br>
                🎯 KNN Regressor<br>
                📉 Régression Linéaire<br>
                🌲 Arbre de Décision<br>
                ⚡ SVR (Support Vector Regression)<br>
                🧠 RNA pour Régression
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with models_col3:
        st.markdown("""
        <div class='pro-card'>
            <h4>Analyses Factorielles</h4>
            <p>
                📊 ACP (Analyse Composantes)<br>
                🔍 ACM (Correspondances Multiples)<br>
                🎯 AFC (Factorielle Correspondances)<br>
                📈 K-Means Clustering<br>
                🌳 CAH (Classification Hiérarchique)<br>
                🎨 t-SNE & UMAP
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bouton de démarrage
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 COMMENCER L'ANALYSE", use_container_width=True, type="primary"):
            st.rerun()

def page_chargement_eda():
    """Page de chargement et EDA"""
    
    st.markdown("## 📂 Chargement & Analyse Exploratoire")
    
    st.markdown("### 1️⃣ Importer vos Données")
    
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV ou Excel (CSV, XLSX, XLS)",
        type=['csv', 'xlsx', 'xls'],
        help="Uploadez votre fichier CSV, XLSX ou XLS pour commencer l'analyse"
    )
    
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.session_state.processor.load_data(temp_path):
                st.session_state.data_loaded = True
                
                file_type = "Excel" if file_extension in ['xlsx', 'xls'] else "CSV"
                st.success(f"✅ Fichier {file_type} chargé avec succès : {len(st.session_state.processor.data)} lignes, {len(st.session_state.processor.column_names)} colonnes")
                
                st.markdown("### 👀 Aperçu des Données")
                st.dataframe(
                    st.session_state.processor.data.head(10),
                    use_container_width=True,
                    height=300
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📋 Types de Colonnes")
                    types_df = pd.DataFrame({
                        'Colonne': st.session_state.processor.data.columns,
                        'Type': st.session_state.processor.data.dtypes,
                        'Manquantes': st.session_state.processor.data.isna().sum(),
                        'Uniques': st.session_state.processor.data.nunique()
                    })
                    st.dataframe(types_df, use_container_width=True)
                
                with col2:
                    st.markdown("### 📊 Répartition des Types")
                    type_counts = st.session_state.processor.data.dtypes.value_counts()
                    fig = px.pie(
                        values=type_counts.values,
                        names=type_counts.index.astype(str),
                        title="Distribution des Types de Données",
                        color_discrete_sequence=['#6B5CE7', '#8B7EF5', '#A59BF9', '#10B981', '#F59E0B']
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(240, 244, 249, 0.8)',
                        font=dict(color='#1F1F1F'),
                        title_font_size=16
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error("❌ Erreur lors du chargement du fichier")
                
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")
    
    if st.session_state.data_loaded:
        st.markdown("---")
        
        st.markdown("### 2️⃣ Définir la Variable Cible")
        target_col = st.selectbox(
            "Choisissez la colonne cible (Y) pour l'analyse supervisée",
            options=st.session_state.processor.column_names,
            help="Cette variable sera celle que vous voulez prédire"
        )
        
        if target_col:
            st.session_state.target_column = target_col
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Type", str(st.session_state.processor.data[target_col].dtype))
            with col2:
                st.metric("Valeurs Uniques", st.session_state.processor.data[target_col].nunique())
            with col3:
                st.metric("Manquantes", st.session_state.processor.data[target_col].isna().sum())
            
            fig = px.histogram(
                st.session_state.processor.data,
                x=target_col,
                title=f"Distribution de {target_col}",
                color_discrete_sequence=['#6B5CE7']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(240, 244, 249, 0.8)',
                font=dict(color='#1F1F1F')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 3️⃣ Analyse Exploratoire des Données")
        
        eda_tabs = st.tabs(["📊 Statistiques", "🔗 Corrélation", "🔀 Analyse Bivariée"])
        
        with eda_tabs[0]:
            if st.button("📊 Afficher les Statistiques", use_container_width=True):
                with st.spinner("Calcul en cours..."):
                    report, _ = st.session_state.processor.get_descriptive_stats()
                    st.text(report)
                    
                    numeric_cols = st.session_state.processor.data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        stats_df = st.session_state.processor.data[numeric_cols].describe().T
                        st.dataframe(stats_df, use_container_width=True)
        
        with eda_tabs[1]:
            if st.button("🔗 Générer la Matrice de Corrélation", use_container_width=True):
                with st.spinner("Génération en cours..."):
                    report, img_base64 = st.session_state.processor.get_correlation_matrix()
                    
                    if img_base64:
                        img_data = base64.b64decode(img_base64)
                        img = Image.open(io.BytesIO(img_data))
                        st.image(img, use_container_width=True)
                    
                    st.info(report)
        
        with eda_tabs[2]:
            st.markdown("**Analysez la relation entre une variable et la cible**")
            
            var_to_analyze = st.selectbox(
                "Variable à analyser",
                options=[col for col in st.session_state.processor.column_names if col != st.session_state.target_column],
                key="bivar_select"
            )
            
            if st.button("▶ Lancer l'Analyse Bivariée", use_container_width=True):
                if var_to_analyze and st.session_state.target_column:
                    with st.spinner("Analyse en cours..."):
                        report, img_base64 = st.session_state.processor.get_bivariate_analysis(
                            var_to_analyze,
                            st.session_state.target_column
                        )
                        
                        if img_base64:
                            img_data = base64.b64decode(img_base64)
                            img = Image.open(io.BytesIO(img_data))
                            st.image(img, use_container_width=True)
                        
                        st.info(report)

def page_supervise():
    """Page de modélisation supervisée"""
    
    st.markdown("## 🎯 Modélisation Supervisée")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger des données dans la section 'Chargement & EDA'")
        return
    
    if 'target_column' not in st.session_state:
        st.warning("⚠️ Veuillez définir une variable cible")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='pro-card'>
            <h4>Type d'Analyse</h4>
        </div>
        """, unsafe_allow_html=True)
        analysis_type = st.radio(
            "Type",
            [CLASSIFICATION, REGRESSION],
            help="Classification pour catégories, Régression pour valeurs continues",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
        <div class='pro-card'>
            <h4>Algorithme de ML</h4>
        </div>
        """, unsafe_allow_html=True)
        if analysis_type == CLASSIFICATION:
            models = list(st.session_state.processor.classification_models.keys())
        else:
            models = list(st.session_state.processor.regression_models.keys())
        
        selected_model = st.selectbox(
            "Algorithme",
            options=models,
            help="Sélectionnez l'algorithme de Machine Learning à utiliser",
            label_visibility="collapsed"
        )
    
    if selected_model:
        model_info = {
            "Random Forest": "🌳 Ensemble d'arbres de décision. Excellence en précision et robustesse.",
            "KNN (K-Nearest Neighbors)": "🎯 Classification par k plus proches voisins. Efficace pour données non-linéaires.",
            "Random Forest (Reg)": "🌳 Random Forest pour régression. Gestion optimale des relations complexes.",
            "KNN (Reg)": "🎯 KNN pour régression. Prédiction par moyenne des k voisins."
        }
        
        if selected_model in model_info:
            st.info(f"ℹ️ {model_info[selected_model]}")
    
    st.markdown("---")
    
    if st.button("🚀 ENTRAÎNER & ÉVALUER LE MODÈLE", use_container_width=True, type="primary"):
        
        with st.spinner("⏳ Prétraitement des données..."):
            success = st.session_state.processor.preprocess_data(
                target_column=st.session_state.target_column,
                analysis_type=analysis_type
            )
        
        if not success:
            st.error("❌ Échec du prétraitement")
            return
        
        st.success("✅ Prétraitement terminé")
        
        with st.spinner(f"🧠 Entraînement du modèle {selected_model}..."):
            result = st.session_state.processor.run_model(selected_model, analysis_type)
        
        if isinstance(result, str):
            st.error(result)
            return
        
        report_text, main_img, tree_img, curve_img = result if len(result) == 4 else (*result, None, None)
        
        st.success(f"✅ Modèle {selected_model} entraîné avec succès!")
        
        st.session_state.model_trained = True
        st.session_state.current_model_info = {
            'model': selected_model,
            'type': analysis_type,
            'target': st.session_state.target_column,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.session_state.analysis_history.append(st.session_state.current_model_info.copy())
        
        st.markdown("### 📊 Résultats de l'Évaluation")
        st.text(report_text)
        
        if main_img:
            st.markdown("#### 📈 Visualisation Principale")
            img_data = base64.b64decode(main_img)
            img = Image.open(io.BytesIO(img_data))
            st.image(img, use_container_width=True)
        
        if tree_img:
            st.markdown("#### 🌳 Arbre de Décision")
            img_data = base64.b64decode(tree_img)
            img = Image.open(io.BytesIO(img_data))
            st.image(img, use_container_width=True)
        
        if curve_img:
            st.markdown("#### 📊 Courbe d'Apprentissage")
            img_data = base64.b64decode(curve_img)
            img = Image.open(io.BytesIO(img_data))
            st.image(img, use_container_width=True)

def page_non_supervise():
    """Page d'analyse non-supervisée"""
    
    st.markdown("## 🔍 Analyse Non-Supervisée")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger des données")
        return
    
    tabs = st.tabs(["🔍 Clustering", "📉 Réduction de Dimension (ACP)"])
    
    with tabs[0]:
        st.markdown("### Clustering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_model = st.selectbox(
                "Algorithme de Clustering",
                options=list(st.session_state.processor.clustering_models.keys())
            )
        
        with col2:
            n_clusters = st.number_input(
                "Nombre de Clusters (k)",
                min_value=2,
                max_value=10,
                value=3
            )
        
        if st.button("▶ LANCER LE CLUSTERING", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                st.session_state.processor.preprocess_data(target_column=None, analysis_type=CLUSTERING)
                
                report, main_img, secondary_img = st.session_state.processor.run_clustering(cluster_model, n_clusters)
                
                st.success("✅ Clustering terminé!")
                st.text(report)
                
                if main_img:
                    img_data = base64.b64decode(main_img)
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, use_container_width=True)
                
                if secondary_img:
                    st.markdown("#### Méthode du Coude")
                    img_data = base64.b64decode(secondary_img)
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### Analyse en Composantes Principales")
        
        n_components = st.number_input(
            "Nombre de composantes",
            min_value=2,
            max_value=min(10, len(st.session_state.processor.column_names)),
            value=2
        )
        
        if st.button("▶ LANCER L'ACP", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                st.session_state.processor.preprocess_data(target_column=None, analysis_type=DIMENSION_REDUCTION)
                
                report, scree_img, scatter_img = st.session_state.processor.run_pca(n_components)
                
                st.success("✅ ACP terminée!")
                st.text(report)
                
                if scree_img:
                    st.markdown("#### Scree Plot")
                    img_data = base64.b64decode(scree_img)
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, use_container_width=True)
                
                if scatter_img:
                    st.markdown("#### Projection 2D")
                    img_data = base64.b64decode(scatter_img)
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, use_container_width=True)

def page_acm_afc():
    """Page ACM et AFC"""
    
    st.markdown("## 📊 Analyses Factorielles: ACM & AFC")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger des données")
        return
    
    tabs = st.tabs(["🔍 ACM", "🎯 AFC"])
    
    with tabs[0]:
        st.markdown("### ACM - Analyse des Correspondances Multiples")
        st.info("💡 L'ACM analyse les relations entre plusieurs variables catégorielles")
        
        categorical_cols = st.session_state.processor.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) < 2:
            st.warning("⚠️ L'ACM nécessite au moins 2 variables catégorielles")
        else:
            selected_cols = st.multiselect(
                "Sélectionnez les variables catégorielles",
                options=categorical_cols,
                default=categorical_cols[:min(5, len(categorical_cols))]
            )
            
            n_components_acm = st.number_input("Nombre de dimensions", min_value=2, max_value=10, value=5, key="acm_comp")
            
            if st.button("▶ LANCER L'ACM", use_container_width=True):
                if len(selected_cols) < 2:
                    st.error("❌ Sélectionnez au moins 2 variables")
                else:
                    with st.spinner("Analyse ACM..."):
                        st.session_state.processor.preprocess_data(target_column=None, analysis_type=ACM_ANALYSIS)
                        report, scree_img, projection_img = st.session_state.processor.run_acm(n_components_acm, selected_cols)
                        
                        st.success("✅ ACM terminée!")
                        st.text(report)
                        
                        if scree_img:
                            img_data = base64.b64decode(scree_img)
                            img = Image.open(io.BytesIO(img_data))
                            st.image(img, use_container_width=True)
                        
                        if projection_img:
                            img_data = base64.b64decode(projection_img)
                            img = Image.open(io.BytesIO(img_data))
                            st.image(img, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### AFC - Analyse Factorielle des Correspondances")
        st.info("💡 L'AFC analyse la relation entre 2 variables catégorielles")
        
        categorical_cols = st.session_state.processor.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_cols) < 2:
            st.warning("⚠️ L'AFC nécessite au moins 2 variables catégorielles")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                row_variable = st.selectbox("Variable LIGNES", options=categorical_cols, key="afc_row")
            
            with col2:
                col_variable = st.selectbox("Variable COLONNES", options=[col for col in categorical_cols if col != row_variable], key="afc_col")
            
            if st.button("▶ LANCER L'AFC", use_container_width=True):
                with st.spinner("Analyse AFC..."):
                    st.session_state.processor.preprocess_data(target_column=None, analysis_type=AFC_ANALYSIS)
                    report, contingency_img, projection_img = st.session_state.processor.run_afc(row_variable, col_variable)
                    
                    st.success("✅ AFC terminée!")
                    st.text(report)
                    
                    if contingency_img:
                        img_data = base64.b64decode(contingency_img)
                        img = Image.open(io.BytesIO(img_data))
                        st.image(img, use_container_width=True)
                    
                    if projection_img:
                        img_data = base64.b64decode(projection_img)
                        img = Image.open(io.BytesIO(img_data))
                        st.image(img, use_container_width=True)

def page_test_modele():
    """Page de test du modèle"""
    
    st.markdown("## 🧪 Test du Modèle")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Veuillez d'abord entraîner un modèle")
        return
    
    model_info = st.session_state.current_model_info
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🤖 Modèle", model_info['model'])
    with col2:
        st.metric("📋 Type", model_info['type'])
    with col3:
        st.metric("🎯 Cible", model_info['target'])
    
    st.markdown("---")
    
    all_features = [col for col in st.session_state.processor.data.columns if col != model_info['target']]
    
    st.markdown("### 🔍 Saisir les Valeurs")
    
    input_values = {}
    col_left, col_right = st.columns(2)
    
    for i, feature in enumerate(all_features):
        col = col_left if i % 2 == 0 else col_right
        
        with col:
            dtype = st.session_state.processor.data[feature].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                input_values[feature] = st.number_input(
                    feature,
                    value=float(st.session_state.processor.data[feature].mean()),
                    key=f"input_{feature}"
                )
            else:
                unique_vals = st.session_state.processor.data[feature].unique()
                input_values[feature] = st.selectbox(
                    feature,
                    options=unique_vals,
                    key=f"input_{feature}"
                )
    
    st.markdown("---")
    
    if st.button("🎯 LANCER LA PRÉDICTION", use_container_width=True, type="primary"):
        with st.spinner("Calcul en cours..."):
            result_msg, prediction = st.session_state.processor.predict_single_sample(input_values)
            
            if prediction is not None:
                st.balloons()
                st.success("✅ Prédiction réussie!")
                
                st.markdown(f"""
                <div class='prediction-result'>
                    <h2 style='color: white; margin: 0;'>🎯 Résultat de la Prédiction</h2>
                    <h1 style='color: white; font-size: 4rem; margin: 1.5rem 0; font-weight: 700;'>{prediction}</h1>
                    <p style='color: rgba(255,255,255,0.9); font-size: 1.25rem;'>
                        Variable cible : {model_info['target']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(result_msg)
            else:
                st.error(result_msg)

def page_comparaison():
    """Page de comparaison"""
    
    st.markdown("## ⚖️ Comparaison de Modèles")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Veuillez d'abord charger des données")
        return
    
    st.info("🚧 Comparez les performances de plusieurs modèles")
    
    analysis_type = st.radio("Type d'Analyse", [CLASSIFICATION, REGRESSION])
    
    if analysis_type == CLASSIFICATION:
        all_models = list(st.session_state.processor.classification_models.keys())
    else:
        all_models = list(st.session_state.processor.regression_models.keys())
    
    selected_models = st.multiselect(
        "Sélectionnez les modèles (minimum 2)",
        options=all_models,
        default=all_models[:2] if len(all_models) >= 2 else all_models
    )
    
    if len(selected_models) < 2:
        st.warning("⚠️ Sélectionnez au moins 2 modèles")
        return
    
    if st.button("▶ LANCER LA COMPARAISON", use_container_width=True, type="primary"):
        if 'target_column' not in st.session_state:
            st.error("❌ Définissez une variable cible")
            return
        
        results = []
        
        with st.spinner("Prétraitement..."):
            st.session_state.processor.preprocess_data(
                target_column=st.session_state.target_column,
                analysis_type=analysis_type
            )
        
        progress_bar = st.progress(0)
        
        for i, model_name in enumerate(selected_models):
            with st.spinner(f"Entraînement de {model_name}..."):
                result = st.session_state.processor.run_model(model_name, analysis_type)
                
                if not isinstance(result, str):
                    report_text = result[0]
                    
                    if analysis_type == CLASSIFICATION:
                        import re
                        match = re.search(r'Précision Globale.*?: ([\d.]+)', report_text)
                        score = float(match.group(1)) if match else 0
                        metric_name = "Accuracy"
                    else:
                        match = re.search(r'Coefficient R².*?: ([\d.]+)', report_text)
                        score = float(match.group(1)) if match else 0
                        metric_name = "R²"
                    
                    results.append({'Modèle': model_name, metric_name: score})
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        st.success("✅ Comparaison terminée!")
        
        results_df = pd.DataFrame(results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Tableau Comparatif")
            st.dataframe(results_df, use_container_width=True)
            
            best_idx = results_df[metric_name].idxmax()
            best_model = results_df.loc[best_idx, 'Modèle']
            best_score = results_df.loc[best_idx, metric_name]
            
            st.markdown(f"""
            <div class='pro-card' style='border-left: 4px solid var(--success-color);'>
                <h4 style='color: var(--success-color);'>🏆 Meilleur Modèle</h4>
                <p style='font-size: 1.25rem;'>
                    <strong>{best_model}</strong><br>
                    {metric_name}: {best_score:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📈 Graphique")
            fig = px.bar(results_df, x='Modèle', y=metric_name, color=metric_name, color_continuous_scale=['#6B5CE7', '#8B7EF5', '#A59BF9'])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(240, 244, 249, 0.8)',
                font=dict(color='#1F1F1F')
            )
            st.plotly_chart(fig, use_container_width=True)

def page_historique():
    """Page historique"""
    
    st.markdown("## 📊 Historique des Analyses")
    
    if not st.session_state.analysis_history:
        st.info("ℹ️ Aucune analyse effectuée")
        return
    
    st.markdown(f"**Total : {len(st.session_state.analysis_history)} analyses**")
    
    for i, analysis in enumerate(reversed(st.session_state.analysis_history), 1):
        with st.expander(f"Analyse #{len(st.session_state.analysis_history) - i + 1} - {analysis['model']}"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("**Type**")
                st.write(analysis['type'])
            with col2:
                st.markdown("**Modèle**")
                st.write(analysis['model'])
            with col3:
                st.markdown("**Cible**")
                st.write(analysis['target'])
            with col4:
                st.markdown("**Date**")
                st.write(analysis['timestamp'])
    
    st.markdown("---")
    st.markdown("### 📈 Statistiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        models_used = [a['model'] for a in st.session_state.analysis_history]
        model_counts = pd.Series(models_used).value_counts()
        
        fig = px.pie(values=model_counts.values, names=model_counts.index, title="Modèles Utilisés", color_discrete_sequence=['#6B5CE7', '#8B7EF5', '#A59BF9', '#10B981', '#F59E0B'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#1F1F1F'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        types_used = [a['type'] for a in st.session_state.analysis_history]
        type_counts = pd.Series(types_used).value_counts()
        
        fig = px.bar(x=type_counts.index, y=type_counts.values, title="Types d'Analyses", color=type_counts.values, color_continuous_scale=['#6B5CE7', '#8B7EF5', '#A59BF9'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(240, 244, 249, 0.8)', font=dict(color='#1F1F1F'))
        st.plotly_chart(fig, use_container_width=True)

def page_export():
    """Page export"""
    
    st.markdown("## 📤 Export des Résultats")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Aucune donnée à exporter")
        return
    
    export_tabs = st.tabs(["📄 CSV", "📊 Excel", "📋 JSON", "💾 Modèle"])
    
    with export_tabs[0]:
        st.markdown("### Export CSV")
        if st.button("⬇️ Télécharger CSV", use_container_width=True):
            csv = st.session_state.processor.data.to_csv(index=False)
            st.download_button(
                label="📥 TÉLÉCHARGER CSV",
                data=csv,
                file_name=f"appanaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with export_tabs[1]:
        st.markdown("### Export Excel")
        if st.button("⬇️ Télécharger Excel", use_container_width=True):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.processor.data.to_excel(writer, index=False, sheet_name='Données')
            excel_data = output.getvalue()
            st.download_button(
                label="📥 TÉLÉCHARGER EXCEL",
                data=excel_data,
                file_name=f"appanaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    with export_tabs[2]:
        st.markdown("### Export JSON")
        if st.session_state.analysis_history:
            if st.button("⬇️ Télécharger Historique", use_container_width=True):
                import json
                json_data = json.dumps(st.session_state.analysis_history, indent=2)
                st.download_button(
                    label="📥 TÉLÉCHARGER JSON",
                    data=json_data,
                    file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("ℹ️ Aucun historique")
    
    with export_tabs[3]:
        st.markdown("### Sauvegarde Modèle")
        if st.session_state.model_trained:
            st.info("🚧 Fonctionnalité en développement")
            st.markdown("""
            <div class='pro-card'>
                <p>Cette fonctionnalité permettra :</p>
                <p>
                    • Sauvegarde du modèle (pickle)<br>
                    • Export des paramètres<br>
                    • Pipeline réutilisable<br>
                    • Partage avec l'équipe
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Aucun modèle entraîné")

if __name__ == "__main__":
    main()