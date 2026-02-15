"""
ML Assignment 2 - Streamlit Web Application
Interactive ML Classification Demo App

Features:
- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Classification App",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1F77B4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all trained models from pickle files."""
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'kNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
    
    return models


@st.cache_resource
def load_scaler():
    """Load the saved scaler."""
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)
    return None


@st.cache_resource
def load_feature_names():
    """Load the saved feature names."""
    if os.path.exists('feature_names.pkl'):
        with open('feature_names.pkl', 'rb') as f:
            return pickle.load(f)
    return None


def calculate_metrics(y_true, y_pred, y_prob, average='weighted'):
    """Calculate all evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average=average, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # Calculate AUC
    try:
        if len(np.unique(y_true)) == 2:
            metrics['AUC'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics['AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except:
        metrics['AUC'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 ML Classification Models Demo</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📊 Configuration")
    
    # Load models
    models = load_models()
    scaler = load_scaler()
    feature_names = load_feature_names()
    
    if not models:
        st.warning("⚠️ No trained models found. Please run the training script first.")
        st.info("""
        **Instructions:**
        1. Place your dataset in the `data/` folder
        2. Update `model/train_models.py` with your dataset path and target column
        3. Run: `python model/train_models.py`
        4. Restart this app
        """)
        return
    
    # Model Selection Dropdown (Requirement b)
    st.sidebar.subheader("🎯 Select Model")
    selected_model = st.sidebar.selectbox(
        "Choose a classification model:",
        list(models.keys()),
        help="Select one of the 6 trained classification models"
    )
    
    st.sidebar.markdown("---")
    
    # File Upload Section (Requirement a)
    st.sidebar.subheader("📁 Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file (test data only)",
        type=['csv'],
        help="Upload a CSV file containing test data. The file should include feature columns and a 'target' column."
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"📈 Model: {selected_model}")
        st.markdown(f"**Currently selected model:** {selected_model}")
        
        # Model description
        model_descriptions = {
            'Logistic Regression': "A linear model for binary/multi-class classification using logistic function.",
            'Decision Tree': "A tree-based model that makes decisions based on feature conditions.",
            'kNN': "Classifies based on the majority class of k nearest neighbors.",
            'Naive Bayes': "Probabilistic classifier based on Bayes' theorem with feature independence assumption.",
            'Random Forest': "Ensemble of decision trees using bagging for improved accuracy.",
            'XGBoost': "Gradient boosting ensemble method known for high performance."
        }
        st.info(model_descriptions.get(selected_model, ""))
    
    with col2:
        st.subheader("📋 Data Status")
        if uploaded_file is not None:
            st.success("✅ Test data uploaded successfully!")
        else:
            st.warning("⚠️ Please upload test data to see predictions and metrics.")
    
    st.markdown("---")
    
    # Process uploaded data
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            
            st.subheader("📊 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            st.caption(f"Dataset shape: {data.shape[0]} rows × {data.shape[1]} columns")
            
            # Check for target column
            target_col = None
            possible_targets = ['target', 'Target', 'label', 'Label', 'class', 'Class', 'y']
            for col in possible_targets:
                if col in data.columns:
                    target_col = col
                    break
            
            if target_col is None:
                st.warning("⚠️ No target column found. Please ensure your CSV has a column named 'target', 'label', or 'class'.")
                target_col = st.selectbox("Select target column:", data.columns)
            
            if target_col:
                # Separate features and target
                X = data.drop(columns=[target_col])
                y = data[target_col]
                
                # Encode target if needed
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    class_names = le.classes_
                else:
                    class_names = [str(i) for i in sorted(y.unique())]
                
                # Handle categorical features
                for col in X.select_dtypes(include=['object']).columns:
                    le_temp = LabelEncoder()
                    X[col] = le_temp.fit_transform(X[col].astype(str))
                
                # Handle missing values
                X = X.fillna(X.median())
                
                # Scale features
                if scaler is not None:
                    X_scaled = scaler.transform(X)
                else:
                    temp_scaler = StandardScaler()
                    X_scaled = temp_scaler.fit_transform(X)
                
                # Get model and make predictions
                model = models[selected_model]
                y_pred = model.predict(X_scaled)
                y_prob = model.predict_proba(X_scaled)
                
                # Calculate metrics (Requirement c)
                metrics = calculate_metrics(y, y_pred, y_prob)
                
                st.markdown("---")
                st.subheader("📊 Evaluation Metrics")
                
                # Display metrics in columns
                metric_cols = st.columns(6)
                metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
                
                for i, (col, metric) in enumerate(zip(metric_cols, metric_names)):
                    with col:
                        value = metrics.get(metric, 0)
                        st.metric(
                            label=metric,
                            value=f"{value:.4f}",
                            delta=None
                        )
                
                st.markdown("---")
                
                # Confusion Matrix and Classification Report (Requirement d)
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🔢 Confusion Matrix")
                    fig = plot_confusion_matrix(y, y_pred, class_names)
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.subheader("📋 Classification Report")
                    report = classification_report(y, y_pred, target_names=class_names, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(4), use_container_width=True)
                
                st.markdown("---")
                
                # Predictions Table
                st.subheader("🔮 Predictions")
                predictions_df = data.copy()
                predictions_df['Predicted'] = y_pred
                predictions_df['Actual'] = y
                predictions_df['Correct'] = predictions_df['Predicted'] == predictions_df['Actual']
                
                # Show prediction summary
                correct_count = predictions_df['Correct'].sum()
                total_count = len(predictions_df)
                st.success(f"✅ Correctly classified: {correct_count}/{total_count} ({correct_count/total_count*100:.2f}%)")
                
                # Show predictions table
                st.dataframe(predictions_df.head(20), use_container_width=True)
                
                # Download predictions
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            st.info("Please ensure your CSV file has the correct format.")
    
    else:
        # Show instructions when no file is uploaded
        st.info("""
        ### 📝 How to use this app:
        
        1. **Select a Model**: Choose one of the 6 classification models from the sidebar dropdown.
        
        2. **Upload Test Data**: Upload a CSV file containing your test data. The file should include:
           - All feature columns used during training
           - A 'target' column with the actual labels
        
        3. **View Results**: After uploading, you'll see:
           - Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
           - Confusion matrix visualization
           - Classification report
           - Prediction results
        
        ### 📊 Available Models:
        - **Logistic Regression**: Linear classifier using logistic function
        - **Decision Tree**: Tree-based classifier
        - **kNN**: K-Nearest Neighbors classifier
        - **Naive Bayes**: Probabilistic classifier (Gaussian)
        - **Random Forest**: Ensemble of decision trees
        - **XGBoost**: Gradient boosting ensemble
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>ML Assignment 2 - Classification Models Demo</p>
            <p>Built with Streamlit 🚀</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
