import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
from datetime import datetime
import json

from .logging_config import logger
from .database import Database

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'db' not in st.session_state:
    st.session_state.db = Database()

def login_page():
    """Display login page."""
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            user = st.session_state.db.authenticate_user(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.db.log_audit(
                    user['id'],
                    'login',
                    f"User {username} logged in successfully"
                )
                st.rerun()
            else:
                st.error("Invalid username or password")

def main_dashboard():
    """Display main dashboard."""
    # Sidebar
    st.sidebar.title("Navigation")
    
    # User info
    st.sidebar.write(f"Logged in as: {st.session_state.user['username']}")
    if st.sidebar.button("Logout"):
        st.session_state.db.log_audit(
            st.session_state.user['id'],
            'logout',
            f"User {st.session_state.user['username']} logged out"
        )
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()
    
    # Model selection
    st.sidebar.header("Model Selection")
    model_type = st.sidebar.radio(
        "Select Clustering Model",
        ["K-means", "DBSCAN"]
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
    
    # Main content
    st.title("Customer Segmentation Dashboard")
    
    # Model version info
    latest_model = st.session_state.db.get_latest_model_version(model_type.lower())
    if latest_model:
        st.info(f"Using {model_type} model version: {latest_model[1]}")
        metrics = json.loads(latest_model[3])
        st.write(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    
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
        
        # Log user action
        st.session_state.db.log_audit(
            st.session_state.user['id'],
            'view_customer',
            f"Viewed details for customer {customer_id}"
        )
    
    except Exception as e:
        error_msg = f"Error loading models or data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        st.info("Please make sure you have run the training script first.")

# Main app logic
if not st.session_state.authenticated:
    login_page()
else:
    main_dashboard() 