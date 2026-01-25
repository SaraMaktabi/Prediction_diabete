"""
INTERFACE WEB MODERNE - PRÉDICTION DIABÈTE
==========================================
Application Streamlit avec design moderne dark theme
Modèle corrigé (96.1% accuracy - sans data leakage)

Installation:
    pip install streamlit plotly pandas numpy joblib scikit-learn

Lancement:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# ============================================
# CONFIGURATION PAGE
# ============================================
st.set_page_config(
    page_title="DiabAI - Prédiction Intelligente",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DARK THEME CSS MODERNE
# ============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    h1, h2, h3 {
        color: #00d4ff !important;
        font-weight: 700;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 8px;
        color: white !important;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
        border: 2px solid #ef4444;
    }
    
    .negative {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.2) 100%);
        border: 2px solid #22c55e;
    }
    
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# CHARGER LES MODÈLES
# ============================================
@st.cache_resource
def load_models():
    models = {}
    model_paths = {
        'clinical': 'models/clinical/xgboost_fixed.pkl',
        'symptoms': 'models/symptoms/random_forest.pkl',
        'pima': 'models/pima/gradient_boosting.pkl'
    }
    
    for name, path in model_paths.items():
        try:
            if os.path.exists(path):
                models[name] = joblib.load(path)
            else:
                st.sidebar.error(f"⚠️ {name}: {path} introuvable")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur {name}: {e}")
    
    return models if models else None

models = load_models()

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-title">🏥 DiabAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Système Intelligent de Prédiction du Diabète</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    prediction_type = st.selectbox(
        "Type de Prédiction",
        ["🏥 Données Cliniques", "🩺 Symptômes", "📊 Données Pima"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Performance")
    
    st.markdown("""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #94a3b8;">Clinical XGBoost</div>
            <div style="font-size: 1.5rem; color: #22c55e; font-weight: 700;">96.1%</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #94a3b8;">Symptoms RF</div>
            <div style="font-size: 1.5rem; color: #3b82f6; font-weight: 700;">92.1%</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #94a3b8;">Pima GB</div>
            <div style="font-size: 1.5rem; color: #f59e0b; font-weight: 700;">74.1%</div>
        </div>
    """, unsafe_allow_html=True)

if not models:
    st.error("❌ Modèles non chargés")
    st.stop()

# ============================================
# CLINICAL
# ============================================
if prediction_type == "🏥 Données Cliniques":
    st.markdown("## 🏥 Analyse Clinique")
    st.info("💡 XGBoost - 96.1% Accuracy")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Démographie")
        age = st.slider("Âge", 18, 90, 45)
        sex = st.selectbox("Sexe", ["Homme", "Femme"])
        ethnicity = st.selectbox("Ethnicité", ["Caucasien", "Africain", "Asiatique", "Autre"])
        
        st.markdown("#### 🏃 Mode de Vie")
        physical = st.selectbox("Activité", ["Sédentaire", "Faible", "Modéré", "Élevé"])
        smoking = st.selectbox("Tabac", ["Non", "Ex", "Oui"])
        alcohol = st.slider("Alcool/sem", 0, 20, 2)
    
    with col2:
        st.markdown("#### 📏 Mesures")
        bmi = st.number_input("IMC", 15.0, 50.0, 25.0, 0.1)
        waist = st.number_input("Tour taille (cm)", 50, 150, 85)
        bp_sys = st.number_input("PA Systolique", 80, 200, 120)
        bp_dia = st.number_input("PA Diastolique", 50, 130, 80)
        
        st.markdown("#### 🧪 Tests")
        glucose = st.number_input("Glucose (mg/dL)", 50, 300, 100)
        hba1c = st.number_input("HbA1c (%)", 4.0, 15.0, 5.5, 0.1)
    
    with col3:
        st.markdown("#### 🩸 Analyses")
        chol_t = st.number_input("Chol Total", 100, 400, 200)
        chol_h = st.number_input("HDL", 20, 100, 50)
        chol_l = st.number_input("LDL", 50, 300, 130)
        ggt = st.number_input("GGT", 5, 200, 25)
        urate = st.number_input("Urate", 2.0, 10.0, 5.0, 0.1)
        calories = st.number_input("Calories/j", 1000, 4000, 2000)
        
        st.markdown("#### 🧬 Historique")
        family = st.radio("Antécédents", ["Non", "Oui"])
        gest = st.radio("Diabète gest.", ["Non", "Oui", "N/A"])
    
    if st.button("🔍 ANALYSER", use_container_width=True):
        with st.spinner("Analyse..."):
            try:
                data = pd.DataFrame({
                    'Age': [(age - 50) / 20],
                    'Sex': [1 if sex == "Homme" else 0],
                    'Ethnicity': [{"Caucasien": 0, "Africain": 1, "Asiatique": 2, "Autre": 3}[ethnicity]],
                    'BMI': [(bmi - 28) / 6],
                    'Waist_Circumference': [(waist - 90) / 15],
                    'Fasting_Blood_Glucose': [(glucose - 100) / 30],
                    'HbA1c': [(hba1c - 5.5) / 2],
                    'Blood_Pressure_Systolic': [(bp_sys - 120) / 20],
                    'Blood_Pressure_Diastolic': [(bp_dia - 80) / 15],
                    'Cholesterol_Total': [(chol_t - 200) / 50],
                    'Cholesterol_HDL': [(chol_h - 50) / 20],
                    'Cholesterol_LDL': [(chol_l - 130) / 40],
                    'GGT': [(ggt - 30) / 30],
                    'Serum_Urate': [(urate - 5) / 2],
                    'Physical_Activity_Level': [["Sédentaire", "Faible", "Modéré", "Élevé"].index(physical)],
                    'Dietary_Intake_Calories': [(calories - 2000) / 500],
                    'Alcohol_Consumption': [alcohol / 10],
                    'Smoking_Status': [["Non", "Ex", "Oui"].index(smoking)],
                    'Family_History_of_Diabetes': [1 if family == "Oui" else 0],
                    'Previous_Gestational_Diabetes': [1 if gest == "Oui" else 0]
                })
                
                pred = models['clinical'].predict(data)[0]
                proba = models['clinical'].predict_proba(data)[0]
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Non-Diabétique", f"{proba[0]*100:.1f}%")
                with col2:
                    st.metric("Diabétique", f"{proba[1]*100:.1f}%")
                with col3:
                    risk = "ÉLEVÉ" if proba[1] > 0.7 else "MODÉRÉ" if proba[1] > 0.3 else "FAIBLE"
                    st.metric("Risque", risk)
                
                if pred == 1:
                    st.markdown(f"""
                    <div class='prediction-box positive'>
                        <h2>⚠️ RISQUE DIABÈTE</h2>
                        <p style='font-size: 24px;'>Probabilité: {proba[1]*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='prediction-box negative'>
                        <h2>✅ PAS DE DIABÈTE</h2>
                        <p style='font-size: 24px;'>Probabilité: {proba[0]*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba[1]*100,
                    title={'text': "Risque (%)", 'font': {'color': 'white'}},
                    number={'font': {'color': 'white'}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ef4444" if proba[1] > 0.5 else "#22c55e"},
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(34, 197, 94, 0.3)"},
                            {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.3)"},
                            {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.3)"}
                        ]
                    }
                ))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# ============================================
# SYMPTOMS
# ============================================
elif prediction_type == "🩺 Symptômes":
    st.markdown("## 🩺 Analyse Symptômes")
    st.info("💡 Random Forest - 92.1% Accuracy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_s = st.slider("Âge", 16, 90, 40)
        gender = st.radio("Sexe", ["Homme", "Femme"])
        
        st.markdown("#### Symptômes")
        poly_u = st.checkbox("Polyurie")
        poly_d = st.checkbox("Polydipsie")
        weight = st.checkbox("Perte poids")
        weak = st.checkbox("Faiblesse")
        poly_p = st.checkbox("Polyphagie")
        obese = st.checkbox("Obésité")
    
    with col2:
        st.markdown("#### Autres")
        thrush = st.checkbox("Candidose")
        blur = st.checkbox("Vision trouble")
        itch = st.checkbox("Démangeaisons")
        irrit = st.checkbox("Irritabilité")
        heal = st.checkbox("Cicatrisation lente")
        pares = st.checkbox("Parésie")
        stiff = st.checkbox("Raideur")
        alop = st.checkbox("Alopécie")
    
    if st.button("🔍 ANALYSER", use_container_width=True):
        with st.spinner("Analyse..."):
            try:
                data = pd.DataFrame({
                    'Age': [(age_s - 40) / 15],
                    'Gender': [1 if gender == "Homme" else 0],
                    'Polyuria': [int(poly_u)],
                    'Polydipsia': [int(poly_d)],
                    'sudden weight loss': [int(weight)],
                    'weakness': [int(weak)],
                    'Polyphagia': [int(poly_p)],
                    'Genital thrush': [int(thrush)],
                    'visual blurring': [int(blur)],
                    'Itching': [int(itch)],
                    'Irritability': [int(irrit)],
                    'delayed healing': [int(heal)],
                    'partial paresis': [int(pares)],
                    'muscle stiffness': [int(stiff)],
                    'Alopecia': [int(alop)],
                    'Obesity': [int(obese)]
                })
                
                pred = models['symptoms'].predict(data)[0]
                proba = models['symptoms'].predict_proba(data)[0]
                count = sum([poly_u, poly_d, weight, weak, poly_p, thrush, blur, itch, irrit, heal, pares, stiff, alop, obese])
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Symptômes", f"{count}/14")
                with col2:
                    st.metric("Probabilité", f"{proba[1]*100:.1f}%")
                with col3:
                    st.metric("Confiance", f"{max(proba)*100:.1f}%")
                
                if pred == 1:
                    st.markdown(f"""
                    <div class='prediction-box positive'>
                        <h2>⚠️ COMPATIBLES</h2>
                        <p style='font-size: 24px;'>{proba[1]*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='prediction-box negative'>
                        <h2>✅ PEU COMPATIBLES</h2>
                        <p style='font-size: 24px;'>{proba[0]*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# ============================================
# PIMA
# ============================================
else:
    st.markdown("## 📊 Analyse Pima")
    st.info("💡 Gradient Boosting - 74.1%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        preg = st.number_input("Grossesses", 0, 20, 0)
        age_p = st.slider("Âge", 21, 81, 33)
        gluc_p = st.number_input("Glucose", 0, 200, 120)
        ins = st.number_input("Insuline", 0, 850, 80)
    
    with col2:
        bp_p = st.number_input("Pression", 0, 122, 72)
        skin = st.number_input("Pli cutané", 0, 99, 20)
        bmi_p = st.number_input("IMC", 0.0, 67.0, 32.0)
        dpf = st.number_input("Pedigree", 0.0, 2.5, 0.5)
    
    if st.button("🔍 PRÉDIRE", use_container_width=True):
        with st.spinner("Analyse..."):
            try:
                data = pd.DataFrame({
                    'Pregnancies': [(preg - 3.8) / 3.4],
                    'Glucose': [(gluc_p - 120.9) / 31.9],
                    'BloodPressure': [(bp_p - 69.1) / 19.4],
                    'SkinThickness': [(skin - 20.5) / 16.0],
                    'Insulin': [(ins - 79.8) / 115.2],
                    'BMI': [(bmi_p - 31.9) / 7.9],
                    'DiabetesPedigreeFunction': [dpf],
                    'Age': [(age_p - 33.2) / 11.8]
                })
                
                pred = models['pima'].predict(data)[0]
                proba = models['pima'].predict_proba(data)[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Non-Diabétique", f"{proba[0]*100:.1f}%")
                with col2:
                    st.metric("Diabétique", f"{proba[1]*100:.1f}%")
                
                if pred == 1:
                    st.markdown(f"""
                    <div class='prediction-box positive'>
                        <h2>⚠️ RISQUE</h2>
                        <p>{proba[1]*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='prediction-box negative'>
                        <h2>✅ FAIBLE</h2>
                        <p>{proba[0]*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>⚠️ Avertissement</strong></p>
    <p>Outil éducatif uniquement - Ne remplace pas un diagnostic médical</p>
    <p style='font-size: 12px;'>© 2026 DiabAI - Master IA</p>
</div>
""", unsafe_allow_html=True)