# Projet Prédiction du Diabète

## 1. Objectif du projet
L’objectif de ce projet est d’explorer, nettoyer et prétraiter trois datasets liés au diabète afin de préparer les données pour un futur apprentissage automatique (supervisé ou non supervisé). Le projet est structuré autour des étapes suivantes : exploration, analyse exploratoire (EDA), visualisations, nettoyage et prétraitement.

---

## 2. Datasets utilisés

### 2.1 Dataset Symptômes diabète (Dataset_600_Lignes)
Contient des informations sur les patients et leurs symptômes liés au diabète.  
**Variables principales :**
- Age : Âge du patient
- Gender : Sexe
- Polyuria, Polydipsia, sudden weight loss, weakness, Polyphagia, Genital thrush, visual blurring, Itching, Irritability, delayed healing, partial paresis, muscle stiffness, Alopecia, Obesity : Symptômes binaires (Oui/Non)
- Class : Variable cible (Positive = diabétique, Negative = non diabétique)

### 2.2 Dataset Clinique / Médical (Dataset_10000_Lignees)
Contient des mesures biologiques et des informations sur le mode de vie des patients.  
**Variables principales :**
- Données démographiques : Age, Sex, Ethnicity
- Mesures physiologiques : BMI, Waist_Circumference, BloodPressure, Glucose, HbA1c, Cholesterol, GGT, Serum_Urate
- Mode de vie : Physical_Activity_Level, Dietary_Intake_Calories, Alcohol_Consumption, Smoking_Status
- Historique familial : Family_History_of_Diabetes, Previous_Gestational_Diabetes
- Score : Diabetes_Score
- Cible : Diabetes (1 = diabétique, 0 = non diabétique)

### 2.3 Dataset Pima Indians (Dataset_Pregnancies)
Dataset classique pour la prédiction du diabète.  
**Variables principales :**
- Pregnancies : Nombre de grossesses
- Glucose, BloodPressure, SkinThickness, Insulin, BMI : Mesures physiologiques
- DiabetesPedigreeFunction : Score génétique/familial
- Age : Âge du patient
- Outcome : Variable cible (1 = diabétique, 0 = non diabétique)

---

## 3. Étapes réalisées pour chaque dataset

### 3.1 Exploration et analyse statistique
- Affichage de l’aperçu général du dataset (head, info)
- Statistiques descriptives pour variables numériques et catégorielles
- Vérification des valeurs manquantes et doublons
- Comptage des valeurs à zéro pour les variables importantes
- Analyse de la distribution de la variable cible

### 3.2 Analyse exploratoire (EDA) et visualisations
- Histogrammes pour chaque variable numérique
- Boxplots pour détecter les valeurs aberrantes (outliers)
- Heatmap de corrélation détaillée pour identifier les relations entre variables
- Graphiques catégoriels (symptômes ou mode de vie) versus la variable cible
- Tous les graphiques sont exportés automatiquement dans un document Word pour chaque dataset :
  - `EDA_Visualisations_Diabetes.docx` pour le dataset Symptômes
  - `EDA_Clinical_Diabetes.docx` pour le dataset Clinique
  - `EDA_Pima_Diabetes.docx` pour le dataset Pima

### 3.3 Nettoyage et prétraitement
- Suppression des doublons
- Gestion des valeurs manquantes :
  - Suppression des lignes manquantes si peu nombreuses
  - Remplacement des valeurs aberrantes ou nulles par la médiane (ex. Glucose, Insulin)
- Encodage des variables catégorielles (Male/Female, Yes/No, niveau d’activité, Ethnicity, etc.)
- Normalisation des variables numériques avec `StandardScaler`
- Séparation des features (`X`) et de la cible (`y`) pour préparation ML
- Création de fichiers CSV prêts à l’emploi :
  - `dataset_cleaned.csv` et `dataset_ready_for_ml.csv` pour le dataset Symptômes
  - `dataset_clinical_cleaned.csv` et `dataset_clinical_ready.csv` pour le dataset Clinique
  - `pima_cleaned.csv` et `pima_ready_for_ml.csv` pour le dataset Pima




