# ML Assignment 2 - Classification Models

## 📌 Problem Statement

This project implements and compares six machine learning classification models on a chosen dataset. The goal is to build an end-to-end ML pipeline including data preprocessing, model training, evaluation, and deployment through an interactive Streamlit web application.

**Objective:** Predict the target class using various classification algorithms and compare their performance using multiple evaluation metrics.

---

## 📊 Dataset Description

**Dataset:** [Your Dataset Name Here]

**Source:** [Kaggle / UCI - Add Link Here]

| Attribute | Description |
|-----------|-------------|
| Feature Count | XX features |
| Instance Count | XXX samples |
| Target Classes | [Class 1, Class 2, ...] |
| Task Type | [Binary/Multi-class] Classification |

**Features:**
- [Feature 1]: Description
- [Feature 2]: Description
- ... (list all features)

**Target Variable:** [Target column name] - [Description of what is being predicted]

---

## 🤖 Models Implemented

The following six classification models were implemented and evaluated:

1. **Logistic Regression** - Linear classifier using logistic function
2. **Decision Tree Classifier** - Tree-based classifier using feature splits
3. **K-Nearest Neighbors (kNN)** - Instance-based classifier using distance metrics
4. **Naive Bayes (Gaussian)** - Probabilistic classifier based on Bayes' theorem
5. **Random Forest (Ensemble)** - Bagging ensemble of decision trees
6. **XGBoost (Ensemble)** - Gradient boosting ensemble method

---

## 📈 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.9825 | 0.9954 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9161 | 0.9123 | 0.9130 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9939 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |
| XGBoost (Ensemble) | 0.9474 | 0.9917 | 0.9474 | 0.9474 | 0.9471 | 0.8864 |

> **Note:** Metrics shown above are from the Breast Cancer Wisconsin dataset (sample). Update with your chosen dataset metrics.

---

## 🔍 Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Best overall performer (98.25% accuracy). Effective for linearly separable data, fast training, and works well on high-dimensional data like medical features. |
| Decision Tree | Lowest performance (91.23% accuracy). Prone to overfitting without pruning. Easy to interpret but less generalizable on complex feature interactions. |
| kNN | Strong performance (95.61% accuracy). Benefits from standardized features. Sensitive to k value selection; works well with the scaled medical data. |
| Naive Bayes | Moderate performance (92.98% accuracy). Despite independence assumption violations, achieved high AUC (0.9868). Fast training and good baseline model. |
| Random Forest (Ensemble) | Strong performance (95.61% accuracy). Robust to overfitting through bagging. Higher AUC (0.9939) shows better ranking ability than single Decision Tree. |
| XGBoost (Ensemble) | Good performance (94.74% accuracy). Excellent AUC (0.9917). Gradient boosting provides strong generalization with regularization benefits. |

---

## 🚀 Streamlit App Features

The deployed Streamlit application includes:

- ✅ **Dataset Upload Option (CSV)** - Upload test data for evaluation
- ✅ **Model Selection Dropdown** - Choose from 6 trained models
- ✅ **Evaluation Metrics Display** - View Accuracy, AUC, Precision, Recall, F1, MCC
- ✅ **Confusion Matrix** - Visual representation of predictions
- ✅ **Classification Report** - Detailed per-class metrics

---

## 🔗 Links

- **GitHub Repository:** [Add your GitHub repo link here]
- **Live Streamlit App:** [Add your deployed Streamlit app link here]

---

## 📁 Project Structure

```
project-folder/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/
│   ├── your_dataset.csv      # Original dataset
│   └── test_data.csv         # Test data for Streamlit app
└── model/
    ├── train_models.py       # Model training script
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl
    ├── feature_names.pkl
    └── model_comparison.csv
```

---

## 🛠️ Installation & Usage

### Local Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <project-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the models (update dataset path in `model/train_models.py` first):
   ```bash
   python model/train_models.py
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New App" and select your repository
5. Choose branch (main) and select `app.py`
6. Click Deploy

---

## 📝 Assignment Submission Checklist

- [ ] GitHub repo link works
- [ ] Streamlit app link opens correctly
- [ ] App loads without errors
- [ ] All required features implemented
- [ ] README.md updated with actual metrics

---

## 👤 Author

**Name:** [Your Name]  
**ID:** [Your Student ID]  
**Course:** Machine Learning  
**Assignment:** ML Assignment 2

---

*This project was completed as part of the ML Assignment 2 - Classification Models requirement.*
