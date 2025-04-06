import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load and preprocess the marketing campaign data."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the data for clustering."""
    # Select relevant columns
    features = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
               'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
               'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    
    return X, features

def perform_pca(X, n_components=2):
    """Perform PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def find_optimal_k(X, max_k=10):
    """Find the optimal number of clusters using the elbow method."""
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    return wcss

def perform_kmeans(X, n_clusters):
    """Perform K-means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans

def perform_dbscan(X, eps=0.5, min_samples=5):
    """Perform DBSCAN clustering."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels, dbscan

def plot_clusters(X, labels, title):
    """Plot the clusters."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

def analyze_clusters(df, labels):
    """Analyze the characteristics of each cluster."""
    df['Cluster'] = labels
    cluster_analysis = df.groupby('Cluster').mean()
    return cluster_analysis 