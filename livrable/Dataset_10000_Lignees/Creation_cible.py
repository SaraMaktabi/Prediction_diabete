import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ------------------------
# 1. Charger le dataset
# ------------------------
df = pd.read_csv(r"Dataset\diabetes_dataset.csv")

# Supprimer id (inutile)
df = df.drop(columns=["id"])

# ------------------------
# 2. Choisir les variables importantes pour le score
# ------------------------
variables_score = [
    "Fasting_Blood_Glucose", 
    "HbA1c", 
    "BMI", 
    "Waist_Circumference",
    "Blood_Pressure_Systolic", 
    "Blood_Pressure_Diastolic",
    "Cholesterol_Total", 
    "Cholesterol_LDL",
    "GGT"
]

# ------------------------
# 3. Normaliser ces variables
# ------------------------
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[variables_score]), columns=variables_score)

# ------------------------
# 4. Créer un score composite
# ------------------------
df["Diabetes_Score"] = (
    0.4*df_scaled["Fasting_Blood_Glucose"] + 
    0.3*df_scaled["HbA1c"] +
    0.1*df_scaled["BMI"] +
    0.1*df_scaled["Waist_Circumference"] +
    0.05*df_scaled["Blood_Pressure_Systolic"] +
    0.05*df_scaled["Cholesterol_LDL"]
)

# ------------------------
# 5. Créer la variable binaire
# ------------------------
# Ici on prend le top 40% comme "diabétique"
threshold = df["Diabetes_Score"].quantile(0.6)
df["Diabetes"] = (df["Diabetes_Score"] >= threshold).astype(int)

# ------------------------
# 6. Vérification
# ------------------------
print("Distribution de la variable cible composite :")
print(df["Diabetes"].value_counts())

# ------------------------
# 7. Sauvegarder le dataset final
# ------------------------
output_file = r"Dataset/Dataset_with_Diabetes.csv"
df.to_csv(output_file, index=False)
print(f"\nDataset final sauvegardé dans : {output_file}")
