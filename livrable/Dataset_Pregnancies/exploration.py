import pandas as pd
import numpy as np

# ===============================
# 1. Chargement du dataset
# ===============================
df = pd.read_csv("diabetes.csv")

# ===============================
# 2. Aperçu général
# ===============================
print("APERÇU GÉNÉRAL DU DATASET")
print(df.head())
print("\nDimensions du dataset :", df.shape)
print("\nInfos générales")
print(df.info())

# ===============================
# 3. Statistiques descriptives
# ===============================
print("\nSTATISTIQUES DESCRIPTIVES")
print(df.describe())

# ===============================
# 4. Valeurs manquantes
# ===============================
print("\nVALEURS MANQUANTES PAR COLONNE")
print(df.isnull().sum())

# ===============================
# 5. Valeurs à zéro (à vérifier pour certaines colonnes)
# ===============================
cols_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
print("\nVALEURS À ZÉRO")
for col in cols_zero:
    print(f"{col} : {(df[col] == 0).sum()}")

# ===============================
# 6. Distribution de la variable cible
# ===============================
print("\nDISTRIBUTION DE LA VARIABLE CIBLE (Outcome)")
print(df['Outcome'].value_counts())
print("\nPourcentage")
print(df['Outcome'].value_counts(normalize=True)*100)

# ===============================
# 7. Doublons
# ===============================
print("\nDOUBLONS")
print("Nombre de doublons :", df.duplicated().sum())
