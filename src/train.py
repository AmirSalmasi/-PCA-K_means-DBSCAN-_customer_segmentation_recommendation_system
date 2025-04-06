import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

from .customer_segmentation import load_data, preprocess_data

def train_models(data_path, output_dir='models'):
    """
    Train and save the clustering models.
    
    Args:
        data_path (str): Path to the input data file
        output_dir (str): Directory to save the trained models
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    df = load_data(data_path)
    X, features = preprocess_data(df)
    
    # Train and save scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    
    # Train and save PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, os.path.join(output_dir, 'pca.joblib'))
    
    # Train and save K-means
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_pca)
    joblib.dump(kmeans, os.path.join(output_dir, 'kmeans.joblib'))
    
    # Train and save DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_pca)
    joblib.dump(dbscan, os.path.join(output_dir, 'dbscan.joblib'))
    
    # Save feature names
    joblib.dump(features, os.path.join(output_dir, 'features.joblib'))
    
    # Calculate and print silhouette scores
    kmeans_score = silhouette_score(X_pca, kmeans_labels)
    dbscan_score = silhouette_score(X_pca, dbscan_labels)
    
    print(f"K-means Silhouette Score: {kmeans_score:.4f}")
    print(f"DBSCAN Silhouette Score: {dbscan_score:.4f}")
    
    return {
        'scaler': scaler,
        'pca': pca,
        'kmeans': kmeans,
        'dbscan': dbscan,
        'features': features,
        'kmeans_score': kmeans_score,
        'dbscan_score': dbscan_score
    }

if __name__ == "__main__":
    # Example usage
    data_path = "../data/marketing_campaign.csv"
    train_models(data_path) 