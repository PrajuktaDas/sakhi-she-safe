"""KMeans Clustering helps in clustering similar areas together
PCA simplifies our dataset and reduces the dimension.It creates a core feature by capturing all other features' essence  """





import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
df=pd.read_csv("data/processed/final_features.csv")

print(df.shape)
print(df.head())

cluster_features=df.filter(regex="^norm_") # selecting columns that start with "norm_"
print(cluster_features.columns)

#KMeans
kmeans=KMeans(n_clusters=3,random_state=42) # initialize KMeans
kmeans.fit(cluster_features) #train KMeans
print(kmeans.cluster_centers_)
df["cluster"]=kmeans.labels_ # creates a new cluster column
print(df["cluster"].value_counts()) # shows how many rows belong to each cluster
print(df.groupby("cluster").mean(numeric_only=True)) # helps to understand what each cluster checks

#PCA
pca=PCA(n_components=2) # reduce features to 2 dimensions
pca_features=pca.fit_transform(cluster_features) # applies PCA
print(pca.explained_variance_ratio_)
df["pca1"]=pca_features[:,0]    # creates the pca columns
df["pca2"]=pca_features[:,1]


plt.scatter(df["pca1"],df["pca2"],c=df["cluster"],cmap="viridis",alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Crime Risk Clusters (KMeans + PCA)")
plt.show()

df.to_csv("data/processed/clustered_data.csv",index=False)# saves updated dataset
