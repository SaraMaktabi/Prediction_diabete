import pandas as pd

# Charger le dataset
df = pd.read_csv("..\Dataset\Dataset_before_cleaning.csv")
# Suppression de la variable HbA1c
df = df.drop(columns=["Diabetes"])
print("Colonnes finales utilisées :")
print(df.columns)
# Sauvegarde du dataset sans la variable HbA1c
df.to_csv("..\Dataset\Dataset_nonsup.csv", index=False)