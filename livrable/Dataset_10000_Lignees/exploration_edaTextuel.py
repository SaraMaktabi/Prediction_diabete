import pandas as pd
import numpy as np

# ===============================
# 1. Chargement du dataset
# ===============================
df = pd.read_csv("Dataset\Dataset_with_Diabetes.csv")

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
print("\nSTATISTIQUES DESCRIPTIVES (numériques)")
print(df.describe())

print("\nSTATISTIQUES DESCRIPTIVES (catégorielles)")
print(df.describe(include="object"))

# ===============================
# 4. Valeurs manquantes
# ===============================
print("\nVALEURS MANQUANTES PAR COLONNE")
print(df.isnull().sum())

# ===============================
# 5. Valeurs à zéro (numériques)
# ===============================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
print("\nVALEURS À ZÉRO (variables numériques)")
for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    print(f"{col} : {zero_count}")

# ===============================
# 6. Distribution de la variable cible
# ===============================
print("\nDISTRIBUTION DE LA VARIABLE CIBLE (Diabetes)")
print(df['Diabetes'].value_counts())
print("\nPourcentage")
print(df['Diabetes'].value_counts(normalize=True)*100)

# ===============================
# 7. Doublons
# ===============================
print("\nDOUBLONS")
print("Nombre de doublons :", df.duplicated().sum())
