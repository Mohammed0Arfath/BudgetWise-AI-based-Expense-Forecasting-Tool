

# 💰 BudgetWise – AI-based Expense Forecasting Tool

BudgetWise is a **machine learning powered personal finance assistant** that helps users understand spending patterns, classify transactions, and forecast future expenses.
It leverages advanced **ML regression, classification, and time-series models** to deliver insights and predictions for smarter money management.

---

## ✨ Features

* 📊 **Data Preprocessing & Cleaning**

  * Duplicate removal, outlier treatment, standardization of categories/payment modes.
  * Feature engineering (date-based, frequency, notes length, transaction size).

* 🤖 **Machine Learning Models**

  * **Regression Models** → Predict transaction/expense amounts (Random Forest, Gradient Boosting, Ridge, LightGBM with GPU acceleration).
  * **Classification Models** → Income vs Expense categorization.
  * **Time-Series Forecasting** → Daily, weekly, and monthly expense trends + 30-day forecasting.
  * **Deep Learning Ready** → Extensions with LSTM/GRU for sequential forecasting.

* 🔮 **Forecasting & Insights**

  * Predict monthly and daily expenses.
  * Detect unusual spending spikes (anomaly detection).
  * User clustering for personalized forecasting.

* 📈 **Visualizations**

  * Actual vs Predicted comparisons.
  * Residual distribution.
  * Feature importance rankings.
  * Time-series plots with moving averages, trendlines, and forecasts.

* 📦 **Deployment Ready**

  * Models, scalers, and encoders saved with **joblib**.
  * Metadata exported for integration into dashboards or APIs.

---

## 🏗️ Tech Stack

* **Languages**: Python 3.11

* **Libraries**:

  * `pandas`, `numpy`, `matplotlib`, `seaborn` → Data handling & visualization
  * `scikit-learn` → Classical ML models & preprocessing
  * `lightgbm` (GPU enabled) → Accelerated regression/forecasting
  * `cuml` (optional, RAPIDS) → GPU Random Forests
  * `joblib` → Model persistence

* **Environment**:

  * Runs on **Kaggle GPU/TPU notebooks** or local JupyterLab

---

## 🚀 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Mohammed0Arfath/BudgetWise-AI-based-Expense-Forecasting-Tool
   cd BudgetWise AI based Expense Forecasting Tool
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. If running on Kaggle GPU:

   ```python
   import lightgbm as lgb
   print(lgb.__version__)   # should detect GPU
   ```

---

## 📊 Workflow

1. **Data Preprocessing** → Cleaning, feature extraction, encoding, scaling.
2. **Regression Models** → Train ML models for amount prediction.
3. **Classification Models** → Income vs Expense prediction.
4. **Time-Series Analysis** → Forecast next 30 days + seasonal/weekly trends.
5. **Hyperparameter Tuning** → GPU-accelerated LightGBM with GridSearch.
6. **Model Persistence** → Save trained models for deployment.
7. **Dashboard Integration** → Export forecasts as CSV/JSON.

---

## 📈 Example Visuals

* **Expense Trends with Moving Averages**
* **Monthly Spending Breakdown**
* **Actual vs Predicted Transaction Amounts**
* **Feature Importance (LightGBM)**

*(Screenshots/plots can be added here as images)*

---

## 📂 Project Structure

```
BudgetWise/
│── data/                     # Raw and processed datasets
│── notebooks/                # Jupyter notebooks for pipeline steps
│── models/                   # Saved trained models (.pkl)
│── outputs/                  # Forecast CSVs, JSONs, and reports
│── src/                      # Core ML pipeline scripts
│── README.md                 # Project documentation
│── requirements.txt          # Dependencies
```

---

## 📦 Deliverables

* Trained **regression models** (amount prediction)
* Trained **classification models** (transaction type)
* **Time-series forecasts** (30-day prediction)
* Feature scalers, encoders, and metadata
* Visualization outputs for insights
* Deployment-ready **saved models**

---

## 🔮 Future Enhancements

* ✅ LSTM/GRU-based sequential forecasting
* ✅ Anomaly detection for fraud/spikes
* ✅ Personalized financial recommendations
* ✅ Web dashboard for real-time predictions

---

## 👨‍💻 Contributors

* **Mohammed Arfath R** – AI/ML Engineering Student | Project Lead

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

