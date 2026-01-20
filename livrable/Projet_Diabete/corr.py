import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# 1. Charger le dataset
# ------------------------
df = pd.read_csv(r"..\Dataset\diabetes_dataset.csv")


# ------------------------
# 2. Sélection des variables numériques
# ------------------------
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
X = df[numerical_cols]

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------
# 3. Réduction dimensionnelle pour visualisation
# ------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.5)
plt.title("Projection PCA des patients")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# ------------------------
# 4. Clustering K-Means
# ------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['KMeans_Cluster'] = kmeans_labels

# Visualisation des clusters K-Means
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels, cmap='viridis', alpha=0.6)
plt.title("Clusters K-Means (2 clusters)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# ------------------------
# 5. Clustering DBSCAN
# ------------------------
dbscan = DBSCAN(eps=2, min_samples=50)
dbscan_labels = dbscan.fit_predict(X_scaled)
df['DBSCAN_Cluster'] = dbscan_labels

# Visualisation des clusters DBSCAN
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=dbscan_labels, cmap='plasma', alpha=0.6)
plt.title("Clusters DBSCAN")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# ------------------------
# 6. Analyse des clusters (moyennes des variables)
# ------------------------
print("\n--- Moyennes des variables par cluster K-Means ---")
print(df.groupby('KMeans_Cluster')[numerical_cols].mean())

print("\n--- Moyennes des variables par cluster DBSCAN ---")
print(df.groupby('DBSCAN_Cluster')[numerical_cols].mean())
