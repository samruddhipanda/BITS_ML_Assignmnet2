"""
ML Assignment 2 - Classification Models Training Script
This script trains 6 classification models and saves them for use in the Streamlit app.

Models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Evaluation Metrics:
- Accuracy, AUC, Precision, Recall, F1, MCC
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the dataset.
    Modify this function based on your chosen dataset.
    """
    df = pd.read_csv(filepath)
    
    # Display basic info
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    
    return df


def prepare_features(df, target_column):
    """
    Prepare features and target variable.
    Handle missing values and encode categorical variables.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        # Save label encoder
        with open('model/label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.median())
    
    return X, y


def calculate_metrics(y_true, y_pred, y_prob, average='weighted'):
    """
    Calculate all required evaluation metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'F1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # Calculate AUC (handle multi-class)
    try:
        if len(np.unique(y_true)) == 2:
            metrics['AUC'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics['AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except:
        metrics['AUC'] = 0.0
    
    return metrics


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train all 6 models and calculate evaluation metrics.
    """
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        results[name] = metrics
        trained_models[name] = model
        
        # Print results
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  AUC: {metrics['AUC']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
        print(f"  F1: {metrics['F1']:.4f}")
        print(f"  MCC: {metrics['MCC']:.4f}")
        
        # Save the model
        model_filename = f"model/{name.lower().replace(' ', '_')}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Model saved to {model_filename}")
    
    return results, trained_models


def create_comparison_table(results):
    """
    Create a comparison table of all model metrics.
    """
    df_results = pd.DataFrame(results).T
    df_results = df_results[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    df_results = df_results.round(4)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(df_results.to_string())
    print("="*80)
    
    # Save to CSV
    df_results.to_csv('model/model_comparison.csv')
    print("\nComparison table saved to model/model_comparison.csv")
    
    return df_results


def main():
    """
    Main function to run the training pipeline.
    
    INSTRUCTIONS:
    1. Replace 'your_dataset.csv' with your actual dataset file
    2. Replace 'target_column' with your target column name
    3. Run this script to train all models
    """
    
    # ============================================
    # MODIFY THESE VARIABLES FOR YOUR DATASET
    # ============================================
    DATASET_PATH = 'data/your_dataset.csv'  # Change this to your dataset path
    TARGET_COLUMN = 'target'  # Change this to your target column name
    # ============================================
    
    print("="*80)
    print("ML CLASSIFICATION MODELS - TRAINING SCRIPT")
    print("="*80)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\nDataset not found at: {DATASET_PATH}")
        print("Please update DATASET_PATH variable with your dataset location.")
        print("\nUsing sample data for demonstration...")
        
        # Create sample data for demonstration
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        TARGET_COLUMN = 'target'
        
        # Create data directory and save sample data
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/breast_cancer.csv', index=False)
        print("Sample breast cancer dataset created at: data/breast_cancer.csv")
    else:
        df = load_and_preprocess_data(DATASET_PATH)
    
    # Prepare features
    print("\n" + "-"*40)
    print("PREPARING FEATURES")
    print("-"*40)
    X, y = prepare_features(df, TARGET_COLUMN)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('model/feature_names.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    # Train and evaluate models
    print("\n" + "-"*40)
    print("TRAINING MODELS")
    print("-"*40)
    results, trained_models = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    
    # Save test data for Streamlit app
    test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_data['target'] = y_test.values
    test_data.to_csv('data/test_data.csv', index=False)
    print("\nTest data saved to data/test_data.csv (for use in Streamlit app)")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nAll models have been saved to the 'model/' directory.")
    print("You can now run the Streamlit app using: streamlit run app.py")


if __name__ == "__main__":
    main()
