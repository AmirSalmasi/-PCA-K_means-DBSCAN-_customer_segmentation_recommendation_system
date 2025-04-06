import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
from datetime import datetime

from .logging_config import logger

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        logger.info("Loading trained models...")
        models_dir = Path("models")
        models = {
            'scaler': joblib.load(models_dir / 'scaler.joblib'),
            'pca': joblib.load(models_dir / 'pca.joblib'),
            'kmeans': joblib.load(models_dir / 'kmeans.joblib'),
            'dbscan': joblib.load(models_dir / 'dbscan.joblib'),
            'features': joblib.load(models_dir / 'features.joblib')
        }
        logger.info("Models loaded successfully")
        return models
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        raise

# Title
st.title("Customer Segmentation Dashboard")

# Sidebar
st.sidebar.header("Model Selection")
model_type = st.sidebar.radio(
    "Select Clustering Model",
    ["K-means", "DBSCAN"]
)
logger.info(f"User selected model type: {model_type}")

# Load data and models
try:
    models = load_models()
    logger.info("Loading customer data...")
    df = pd.read_csv("data/marketing_campaign.csv")
    logger.info(f"Loaded {len(df)} customer records")
    
    # Preprocess data
    X = df[models['features']]
    X_scaled = models['scaler'].transform(X)
    X_pca = models['pca'].transform(X_scaled)
    
    # Get predictions
    if model_type == "K-means":
        labels = models['kmeans'].predict(X_pca)
        logger.info("Applied K-means clustering")
    else:
        labels = models['dbscan'].fit_predict(X_pca)
        logger.info("Applied DBSCAN clustering")
    
    # Add predictions to dataframe
    df['Cluster'] = labels
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Distribution")
        fig = px.pie(df, names='Cluster', title='Customer Distribution by Cluster')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Cluster Characteristics")
        cluster_stats = df.groupby('Cluster')[models['features']].mean()
        st.dataframe(cluster_stats.style.background_gradient())
        logger.debug("Displayed cluster statistics")
    
    with col2:
        st.subheader("Cluster Visualization")
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=labels.astype(str),
            title='Customer Clusters (PCA)',
            labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Feature Importance")
        pca_components = pd.DataFrame(
            models['pca'].components_,
            columns=models['features'],
            index=['PC1', 'PC2']
        )
        fig = px.imshow(
            pca_components,
            title='PCA Component Loadings',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
        logger.debug("Displayed cluster visualizations")
    
    # Customer details section
    st.subheader("Customer Details")
    customer_id = st.selectbox("Select Customer ID", df.index)
    customer_data = df.loc[customer_id]
    logger.info(f"User viewing details for customer ID: {customer_id}")
    
    col3, col4 = st.columns(2)
    with col3:
        st.write("### Personal Information")
        st.write(f"Cluster: {customer_data['Cluster']}")
        st.write(f"Income: ${customer_data['Income']:,.2f}")
        st.write(f"Education: {customer_data['Education']}")
        st.write(f"Marital Status: {customer_data['Marital_Status']}")
    
    with col4:
        st.write("### Purchase Behavior")
        purchase_cols = [col for col in df.columns if col.startswith('Mnt') or col.startswith('Num')]
        purchase_data = customer_data[purchase_cols]
        fig = px.bar(
            x=purchase_data.index,
            y=purchase_data.values,
            title='Purchase History'
        )
        st.plotly_chart(fig, use_container_width=True)
        logger.debug(f"Displayed purchase history for customer {customer_id}")

except Exception as e:
    error_msg = f"Error loading models or data: {str(e)}"
    logger.error(error_msg, exc_info=True)
    st.error(error_msg)
    st.info("Please make sure you have run the training script first.") 