from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

df = pd.read_csv(r"diabetes.csv")
df_clean = df.copy()

# ===============================
# 1. Suppression des doublons
# ===============================
df_clean = df_clean.drop_duplicates()

# ===============================
# 2. Remplacement des zéros par NaN pour certaines colonnes
# ===============================
cols_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df_clean[cols_zero] = df_clean[cols_zero].replace(0, np.nan)

# Imputation avec la médiane
df_clean[cols_zero] = df_clean[cols_zero].fillna(df_clean[cols_zero].median())

# ===============================
# 3. Séparation X / y
# ===============================
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

# ===============================
# 4. Normalisation des variables numériques
# ===============================
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ===============================
# 5. Dataset final prêt ML
# ===============================
df_ready = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

# ===============================
# 6. Sauvegarde des fichiers
# ===============================
df_clean.to_csv("dataset_preg_cleaned.csv", index=False)
df_ready.to_csv("dataset_preg_ready_for_ml.csv", index=False)

print("✔ Nettoyage et prétraitement terminés")
print("✔ Fichiers générés : dataset_preg_cleaned.csv, dataset_preg_ready_for_ml.csv")