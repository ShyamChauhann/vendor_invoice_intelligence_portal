# vendor_invoice_intelligence_portal

## 📌 Project Overview
This project implements an end-to-end machine learning system designed to support finance teams. It automates key processes by predicting expected freight costs for vendor invoices and flagging high-risk invoices that require manual review due to abnormal costs, freight, or operational patterns. 

This isn't just a Jupyter Notebook analysis; it is a fully structured, production-ready ML pipeline deployed as a Streamlit web application.

## 🎯 Business Objectives
This project tackles two major problems in invoice and supply chain management:
1. **Freight Cost Prediction (Regression):** Predicts freight costs for a vendor invoice using the purchase dollar amount. This supports better forecasting, budgeting, and vendor negotiation by preemptively estimating non-trivial landed costs.
2. **Invoice Risk Flagging (Classification):** Predicts whether a vendor invoice should be flagged for manual approval based on abnormal delivery delays or cost patterns. This reduces financial risk, improves operational efficiency, and prioritizes human review where it adds the most value (instead of manually auditing 100% of invoices).

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation & EDA:** Pandas, NumPy, Seaborn, Matplotlib
* **Database:** SQLite3
* **Machine Learning:** Scikit-Learn (Linear Regression, Random Forest, Logistic Regression, Decision Trees), SciPy (T-Test Hypothesis Testing)
* **Model Serialization:** Joblib
* **Web Application:** Streamlit

## 🗄️ Data Source
The data is stored in a relational SQLite database (`inventory.db`). Relevant data is extracted and aggregated using SQL queries/JOINs via Python. Key tables include:
* `Purchases`: Purchase order numbers, dates, and receiving information.
* `PurchasePrices`: Item-level pricing details.
* `VendorInvoices`: Invoice generation dates, billed quantities, billed dollars, and freight costs.

## 🧠 Models Used & Evaluation Metrics
### 1. Freight Cost Prediction
* **Baseline Models Tested:** Decision Tree Regressor, Random Forest Regressor.
* **Final Model:** **Linear Regression** (Selected as the best-fit model due to its high accuracy and lowest error rates).
* **Evaluation Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score.

### 2. Invoice Risk Flagging
* **Baseline Models Tested:** Logistic Regression, Decision Tree Classifier.
* **Final Model:** **Random Forest Classifier**.
* **Optimization:** Hyperparameter tuning was performed using `GridSearchCV` optimized for the **F1-Score** to handle class imbalances and reduce False Positives.
* **Evaluation Metrics:** Precision, Recall, F1-Score, Confusion Matrix.

## 📁 Project Structure
```text
├── data/                       # Contains the inventory.db SQLite database
├── notebooks/                  # Jupyter notebooks for EDA, hypothesis testing, and model prototyping
├── freight_cost_prediction/    # Pipeline scripts for Regression model
│   ├── data_preprocessing.py   # Data extraction and feature engineering
│   ├── modeling_evaluation.py  # Model training and testing functions
│   └── train.py                # Main script to train and save the best regression model
├── invoice_flagging/           # Pipeline scripts for Classification model
│   ├── data_preprocessing.py   
│   ├── modeling_evaluation.py  
│   └── train.py                # Main script to train and save the classification model
├── inferencing/                # Scripts to load saved models and predict on unseen data
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
├── models/                     # Saved joblib (.pkl) model files and scalers
├── app.py                      # Streamlit web application portal
├── requirements.txt            # Required Python dependencies
└── README.md                   # Project documentation
