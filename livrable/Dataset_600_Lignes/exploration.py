import pandas as pd

# ===============================
# 1. Chargement du dataset
# ===============================
df = pd.read_csv(r"diabetes_data_upload.csv")

# ===============================
# 2. Aperçu général
# ===============================
print("APERÇU GÉNÉRAL DU DATASET")
print(df.head())
print("\nDimensions du dataset :", df.shape)

print("\nInformations générales")
print(df.info())

# ===============================
# 3. Statistiques descriptives
# ===============================
print("\nSTATISTIQUES DESCRIPTIVES (variables numériques)")
print(df.describe())

print("\nSTATISTIQUES DESCRIPTIVES (variables catégorielles)")
print(df.describe(include="object"))

# ===============================
# 4. Valeurs manquantes
# ===============================
print("\nVALEURS MANQUANTES PAR COLONNE")
print(df.isnull().sum())

# ===============================
# 5. Valeurs à zéro (variables numériques)
# ===============================
print("\nVALEURS À ZÉRO (variables numériques)")
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    print(f"{col} : {zero_count}")

# ===============================
# 6. Distribution de la variable cible
# ===============================
print("\nDISTRIBUTION DE LA VARIABLE CIBLE (class)")
print(df['class'].value_counts())
print("\nDistribution en pourcentage")
print(df['class'].value_counts(normalize=True) * 100)

# ===============================
# 7. Doublons
# ===============================
print("\nDOUBLONS")
print("Nombre de doublons :", df.duplicated().sum())
