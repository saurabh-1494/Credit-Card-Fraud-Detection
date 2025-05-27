# Credit-Card-Fraud-Detection
# 💳 Credit Card Fraud Detection using Isolation Forest

This project uses **unsupervised machine learning** techniques—specifically, the **Isolation Forest** algorithm—to detect fraudulent transactions from the Credit Card Fraud dataset available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 📁 Dataset Information

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraudulent Transactions:** 492 (highly imbalanced)
- **Features:**
  - `Time`: Time in seconds since the first transaction
  - `Amount`: Transaction amount
  - `V1` to `V28`: PCA-transformed features
  - `Class`: Target variable (1 = Fraud, 0 = Genuine)

---

## ⚙️ Technologies & Libraries

- Python 3.x
- `pandas` – for data handling
- `numpy` – for numerical computations
- `matplotlib`, `seaborn` – for visualizations
- `scikit-learn` – for preprocessing, modeling, and evaluation
- `IsolationForest` – for anomaly detection

---

## 🚀 Steps Performed

### 🔹 1. Import Libraries
Necessary libraries are imported for data processing, modeling, and evaluation.

### 🔹 2. Load and Explore Dataset
The CSV file is loaded using pandas and basic statistics and class distributions are printed.

### 🔹 3. Preprocessing
- Scaled the `Amount` and `Time` columns using `StandardScaler`.
- Dropped original `Amount` and `Time`.
- Reordered columns for better readability.

### 🔹 4. Train Isolation Forest Model
- Model is trained on the entire dataset using `IsolationForest`.
- Predictions are converted: `-1 → 1 (Fraud)` and `1 → 0 (Genuine)`.

### 🔹 5. Evaluation
- Metrics Used: Accuracy, Precision, Recall, F1-Score.
- Confusion Matrix is printed and visualized using a heatmap.

---

## 📊 Results

The model's performance is evaluated based on:

- **Accuracy**: Percentage of correctly classified transactions.
- **Precision**: How many predicted frauds were actually fraud?
- **Recall**: How many actual frauds were detected?
- **F1-Score**: Harmonic mean of precision and recall.

Additionally, a **confusion matrix heatmap** is plotted for visual analysis.

---

## 📌 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in your project folder.
2. Make sure Python and required packages are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
