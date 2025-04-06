import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from .customer_segmentation import load_data, preprocess_data
from .logging_config import logger
from .database import Database

def train_models(data_path, output_dir='models', user_id=None):
    """
    Train and save the clustering models.
    
    Args:
        data_path (str): Path to the input data file
        output_dir (str): Directory to save the trained models
        user_id (int): ID of the user training the models
    """
    try:
        logger.info(f"Starting model training process with data from {data_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Created/verified output directory: {output_dir}")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df = load_data(data_path)
        X, features = preprocess_data(df)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        logger.debug(f"Features used: {features}")
        
        # Train and save scaler
        logger.info("Training StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Train and save PCA
        logger.info("Training PCA...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_path = os.path.join(output_dir, 'pca.joblib')
        joblib.dump(pca, pca_path)
        logger.info(f"PCA saved to {pca_path}")
        logger.debug(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        
        # Train and save K-means
        logger.info("Training K-means model...")
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_pca)
        kmeans_path = os.path.join(output_dir, 'kmeans.joblib')
        joblib.dump(kmeans, kmeans_path)
        logger.info(f"K-means model saved to {kmeans_path}")
        
        # Train and save DBSCAN
        logger.info("Training DBSCAN model...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_pca)
        dbscan_path = os.path.join(output_dir, 'dbscan.joblib')
        joblib.dump(dbscan, dbscan_path)
        logger.info(f"DBSCAN model saved to {dbscan_path}")
        
        # Save feature names
        features_path = os.path.join(output_dir, 'features.joblib')
        joblib.dump(features, features_path)
        logger.info(f"Feature names saved to {features_path}")
        
        # Calculate performance metrics
        logger.info("Calculating model performance metrics...")
        kmeans_score = silhouette_score(X_pca, kmeans_labels)
        dbscan_score = silhouette_score(X_pca, dbscan_labels)
        
        logger.info(f"K-means Silhouette Score: {kmeans_score:.4f}")
        logger.info(f"DBSCAN Silhouette Score: {dbscan_score:.4f}")
        
        # Save model versions to database if user_id is provided
        if user_id:
            db = Database()
            try:
                # Save K-means model version
                kmeans_version = f"kmeans_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                kmeans_metrics = json.dumps({
                    'silhouette_score': float(kmeans_score),
                    'n_clusters': 4,
                    'features': features
                })
                kmeans_model_id = db.save_model_version(
                    kmeans_version, 'kmeans', kmeans_metrics, user_id
                )
                
                # Save DBSCAN model version
                dbscan_version = f"dbscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                dbscan_metrics = json.dumps({
                    'silhouette_score': float(dbscan_score),
                    'eps': 0.5,
                    'min_samples': 5,
                    'features': features
                })
                dbscan_model_id = db.save_model_version(
                    dbscan_version, 'dbscan', dbscan_metrics, user_id
                )
                
                # Save customer segments
                kmeans_segments = pd.DataFrame({
                    'customer_id': df.index,
                    'segment_id': kmeans_labels,
                    'confidence_score': None  # Add confidence scores if available
                })
                db.save_customer_segments(kmeans_segments, kmeans_model_id)
                
                dbscan_segments = pd.DataFrame({
                    'customer_id': df.index,
                    'segment_id': dbscan_labels,
                    'confidence_score': None
                })
                db.save_customer_segments(dbscan_segments, dbscan_model_id)
                
                db.close()
            except Exception as e:
                logger.error(f"Error saving to database: {str(e)}", exc_info=True)
                if db:
                    db.close()
        
        return {
            'scaler': scaler,
            'pca': pca,
            'kmeans': kmeans,
            'dbscan': dbscan,
            'features': features,
            'kmeans_score': kmeans_score,
            'dbscan_score': dbscan_score
        }
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Example usage
        data_path = "../data/marketing_campaign.csv"
        logger.info(f"Starting training script with data path: {data_path}")
        train_models(data_path)
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}", exc_info=True)
        raise 