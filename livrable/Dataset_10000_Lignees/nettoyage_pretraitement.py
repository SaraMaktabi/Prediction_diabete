import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Dataset\Dataset_with_Diabetes.csv")
df_clean = df.copy()

# ===============================
# 1. Suppression doublons
# ===============================
df_clean = df_clean.drop_duplicates()

# ===============================
# 2. Valeurs manquantes
# ===============================
df_clean = df_clean.dropna()  # on peut remplacer par imputation si nécessaire

# ===============================
# 3. Encodage variables catégorielles
# ===============================
categorical_cols = ['Sex', 'Ethnicity', 'Physical_Activity_Level', 'Alcohol_Consumption', 'Smoking_Status', 'Previous_Gestational_Diabetes']
for col in categorical_cols:
    df_clean[col] = df_clean[col].astype('category').cat.codes

# ===============================
# 4. Normalisation variables numériques
# ===============================
numeric_cols = ['Age','BMI','Waist_Circumference','Fasting_Blood_Glucose','HbA1c','Blood_Pressure_Systolic','Blood_Pressure_Diastolic','Cholesterol_Total','Cholesterol_HDL','Cholesterol_LDL','GGT','Serum_Urate','Dietary_Intake_Calories','Diabetes_Score']

scaler = StandardScaler()
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

# ===============================
# 5. Séparation X / y
# ===============================
X = df_clean.drop('Diabetes', axis=1)
y = df_clean['Diabetes']

# ===============================
# 6. Sauvegarde datasets
# ===============================
df_clean.to_csv("dataset_10000_cleaned.csv", index=False)
pd.concat([X,y], axis=1).to_csv("dataset_10000_ready.csv", index=False)

print("✔ Nettoyage et prétraitement terminés")
print("✔ Fichiers générés : dataset_10000_cleaned.csv, dataset_10000_ready.csv")