import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ===============================
# 1. Chargement du dataset
# ===============================
df = pd.read_csv("diabetes_data_upload.csv")

# ===============================
# 2. Copie de travail
# ===============================
df_clean = df.copy()

# ===============================
# 3. Suppression des doublons
# ===============================
df_clean = df_clean.drop_duplicates()

# ===============================
# 4. Gestion des valeurs manquantes
# ===============================
# Vérification
missing = df_clean.isnull().sum()

# Suppression si très peu de valeurs manquantes
df_clean = df_clean.dropna()

# (Alternative possible : imputation si nécessaire)

# ===============================
# 5. Nettoyage des colonnes texte
# ===============================
# Uniformiser les chaînes
for col in df_clean.columns:
    if df_clean[col].dtype == "object":
        df_clean[col] = df_clean[col].str.strip()

# ===============================
# 6. Conversion Yes / No → 1 / 0
# ===============================
binary_mapping = {
    "Yes": 1,
    "No": 0
}

binary_columns = [
    'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
    'Irritability', 'delayed healing', 'partial paresis',
    'muscle stiffness', 'Alopecia', 'Obesity'
]

for col in binary_columns:
    df_clean[col] = df_clean[col].map(binary_mapping)

# ===============================
# 7. Encodage du genre
# ===============================
df_clean['Gender'] = df_clean['Gender'].map({
    'Male': 1,
    'Female': 0
})

# ===============================
# 8. Encodage de la classe (si utilisée)
# ===============================
df_clean['class'] = df_clean['class'].map({
    'Positive': 1,
    'Negative': 0
})

# ===============================
# 9. Vérification valeurs aberrantes (Age)
# ===============================
# Suppression âges non réalistes
df_clean = df_clean[(df_clean['Age'] >= 1) & (df_clean['Age'] <= 120)]

# ===============================
# 10. Séparation features / cible
# ===============================
X = df_clean.drop('class', axis=1)
y = df_clean['class']

# ===============================
# 11. Normalisation des variables numériques
# ===============================
scaler = StandardScaler()
X[['Age']] = scaler.fit_transform(X[['Age']])

# ===============================
# 12. Dataset final prêt ML
# ===============================
df_ready = pd.concat([X, y], axis=1)

# ===============================
# 13. Sauvegarde des fichiers
# ===============================
df_clean.to_csv("dataset_cleaned.csv", index=False)
df_ready.to_csv("dataset_ready_for_ml.csv", index=False)

print("✔ Nettoyage et prétraitement terminés")
print("✔ Fichiers générés :")
print("- dataset_cleaned.csv")
print("- dataset_ready_for_ml.csv")
