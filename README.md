# Customer Segmentation and Recommendation System

This project implements a customer segmentation system using various clustering techniques (PCA, K-means, and DBSCAN) to analyze customer behavior and provide recommendations.

## Project Structure

```
.
├── data/                   # Data files
│   └── marketing_campaign.csv
├── notebooks/              # Jupyter notebooks
│   └── customer_segmentation.ipynb
├── src/                    # Source code
├── images/                 # Generated visualizations
└── requirements.txt        # Project dependencies
```

## Features

- Data preprocessing and feature engineering
- Principal Component Analysis (PCA) for dimensionality reduction
- K-means clustering for customer segmentation
- DBSCAN clustering for outlier detection
- Customer behavior analysis
- Recommendation system based on customer segments

## Requirements

- Python 3.x
- UV (fast Python package installer)
- Virtual environment (recommended)

## Installation

### 1. Install UV

#### Windows
```bash
pip install uv
```

#### Linux/macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install dependencies using UV
uv pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook in the notebooks directory
2. Run the cells to perform the analysis
3. View the results and visualizations

## Results

The analysis provides insights into customer segments and their behaviors, which can be used for targeted marketing campaigns and personalized recommendations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
