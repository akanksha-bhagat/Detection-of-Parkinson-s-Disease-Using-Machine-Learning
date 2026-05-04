# Detection-of-Parkinson-s-Disease-Using-Machine-Learning
# Parkinson's Disease Detection Using Machine Learning

This repository contains the implementation of a machine learning-based system for detecting Parkinson’s Disease using biomedical voice features. The project aims to assist in early and non-invasive diagnosis of Parkinson's Disease by analyzing speech patterns with machine learning algorithms.

## 🔬 Project Objective

To develop a predictive model that classifies whether a patient has Parkinson’s Disease based on voice measurements, enabling early diagnosis and improved healthcare outcomes.

## 🧠 Technologies & Tools

- **Python**
- **Pandas / NumPy**
- **Scikit-learn**
- **Matplotlib / Seaborn**
- **Jupyter Notebook**
- **Machine Learning Models:** SVM, Random Forest, Logistic Regression, KNN

## 📁 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **Attributes:** Biomedical voice measurements such as jitter, shimmer, and harmonic-to-noise ratio.
- **Target Variable:** `status` (1 = Parkinson’s, 0 = Healthy)

## 📊 Data Preprocessing

- Missing value handling
- Feature scaling (StandardScaler)
- Correlation analysis
- Train-test split (80-20)

## 🤖 Models Trained

| Model               | Accuracy |
|--------------------|----------|
| Support Vector Machine (SVM) | 94% |
| Random Forest       | 92% |
| Logistic Regression | 88% |
| K-Nearest Neighbors | 85% |

SVM performed the best and was selected as the final model for deployment.

## 📈 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve and AUC Score



