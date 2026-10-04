# Human Action Detection using MHEALTH Dataset

## Overview
This project builds an end-to-end Machine Learning pipeline to classify human physical activities using multi-sensor body measurements (accelerometer, gyroscope, and ECG data) from the MHEALTH dataset. The pipeline classifies 12 distinct physical activities alongside a null state.

## Pipeline Architecture

- **`1_data_processing.py`**: Loads raw sensor data, handles class imbalance by downsampling activity `0` to 40,000 rows, performs a 75/25 train-test split, scales features using `sklearn.preprocessing.RobustScaler`, and exports numpy array artifacts (`.npy`).
- **`2_model_training.py`**: Loads preprocessed arrays, trains three benchmark models (Logistic Regression, KNN, and Decision Tree), evaluates performance with macro metrics (Accuracy, Precision, Recall, F1), and saves Seaborn confusion matrix heatmaps (`.png`).
- **`3_model_export.py`**: Trains and serializes the winning KNN model (`best_knn_model.pkl`) using `joblib` and generates `requirements.txt`.

## Key Results

| Model | Test Accuracy | Macro F1 Score | Status |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | ~76.95% | 0.7423 | Baseline |
| **Decision Tree (max_depth=14)** | ~88.94% | 0.8778 | Tree Baseline |
| **K-Neighbors Classifier (k=5)** | **99.92%** | **0.9990** | **Best Model** |

## How to Run

Install dependencies and run the scripts sequentially in your terminal:

```bash
# Step 0: Install dependencies
pip install -r requirements.txt

# Step 1: Preprocess dataset and generate .npy array artifacts
python 1_data_processing.py

# Step 2: Train and evaluate models
python 2_model_training.py

# Step 3: Train and export the winning model
python 3_model_export.py
```
