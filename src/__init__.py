"""
Customer Segmentation and Recommendation System

This package provides functionality for customer segmentation using various clustering techniques.
"""

from .customer_segmentation import (
    load_data,
    preprocess_data,
    perform_pca,
    find_optimal_k,
    perform_kmeans,
    perform_dbscan,
    plot_clusters,
    analyze_clusters
) 