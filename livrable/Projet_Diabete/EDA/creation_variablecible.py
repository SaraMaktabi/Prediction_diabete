import pandas as pd

# Charger le dataset
df = pd.read_csv("diabetes_dataset.csv")

# Création de la variable cible Diabetes
df["Diabetes"] = (
    (df["Fasting_Blood_Glucose"] >= 126) &
    (df["HbA1c"] >= 6.5)
).astype(int)

# Vérification
print("Distribution de la variable Diabetes :")
print(df["Diabetes"].value_counts())

# Sauvegarde du dataset avec la nouvelle variable
df.to_csv("diabetes_dataset_with_diabetes.csv", index=False)
