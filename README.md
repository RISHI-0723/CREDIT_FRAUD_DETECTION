# 🛡️ Credit Card Fraud Detection using Machine Learning

An end-to-end Machine Learning project that detects fraudulent credit card transactions using classification algorithms. The project handles severe class imbalance using **SMOTE**, compares multiple ML models, and provides an interactive **Streamlit dashboard** for visualization and prediction.

---

## 📌 Project Overview

Credit card fraud is one of the biggest challenges in the banking industry. Since fraudulent transactions make up only a very small percentage of all transactions, traditional machine learning models tend to become biased toward the majority class.

This project addresses this problem by:

- Performing Exploratory Data Analysis (EDA)
- Handling class imbalance using SMOTE
- Training multiple classification models
- Comparing model performance
- Selecting the best model based on F1-Score
- Deploying an interactive Streamlit dashboard

---

## 🚀 Features

- 📊 Exploratory Data Analysis
- 🧹 Duplicate Detection & Removal
- ⚖️ SMOTE for Class Balancing
- 🤖 Multiple ML Models
  - Logistic Regression
  - Decision Tree
  - Random Forest
- 📈 Performance Evaluation
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- 📉 Confusion Matrix Visualization
- 📊 Feature Importance
- 🔍 Real-Time Fraud Prediction
- 🌐 Interactive Streamlit Dashboard

---

## 📂 Project Structure

```text
CREDIT_FRAUD_DETECTION/
│
├── app.py
├── credit_card_fraud_eda.py
├── fraud_detection_model.pkl
├── results_metadata.json
├── requirements.txt
├── README.md
│
├── plots/
│   ├── class_distribution.png
│   ├── smote_balancing.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_random_forest.png
│   └── feature_importance.png
│
└── creditcard.csv (Not included due to GitHub file size limit)
```

---

## 🧠 Machine Learning Workflow

1. Load Dataset
2. Perform Exploratory Data Analysis
3. Check Missing Values
4. Remove Duplicate Records
5. Split Dataset into Training & Testing Sets
6. Apply SMOTE to Balance Classes
7. Train Multiple Classification Models
8. Evaluate Models
9. Compare Performance
10. Select Best Model
11. Save Trained Model
12. Deploy Streamlit Dashboard

---

## 📊 Models Used

- Logistic Regression
- Decision Tree
- Random Forest

The best-performing model is selected using the **F1-Score**, which provides the best balance between Precision and Recall for fraud detection.

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/RISHI-0723/CREDIT_FRAUD_DETECTION.git
```

Move into the project directory:

```bash
cd CREDIT_FRAUD_DETECTION
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

---

## 📸 Dashboard

The dashboard includes:

- Home
- Dataset Analysis
- SMOTE Visualization
- Model Performance Comparison
- Confusion Matrices
- Feature Importance
- Real-Time Prediction
- Project Summary

---

## 📊 Dataset

Dataset Used:

**Kaggle Credit Card Fraud Detection Dataset**

The original dataset is **not included** in this repository because it exceeds GitHub's file size limit.

You can download it from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🎯 Results

Among the evaluated models, **Random Forest** achieved the best overall performance based on the F1-Score, making it the most suitable model for detecting fraudulent transactions while balancing Precision and Recall.

---

## 👨‍💻 Author

**Rishi**

GitHub:
https://github.com/RISHI-0723

---

## 📜 License

This project is intended for educational and learning purposes.
