## Dataset Overview: Telco Customer Churn

This project uses the **Telco Customer Churn** dataset, originally released by IBM, which contains customer data for a fictional telecommunications company. The goal is to predict whether a customer is likely to **churn** (i.e., leave the service) based on their demographics, account details, and service usage.

---

### Dataset Summary

| Category          | Example Columns                               | Description |
|------------------|-----------------------------------------------|-------------|
| Customer ID       | `customerID`                                  | Unique identifier for each customer |
| Demographics      | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Basic personal attributes |
| Account Info      | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` | Subscription, billing, and tenure details |
| Services Signed Up| `PhoneService`, `InternetService`, `OnlineBackup`, `StreamingTV`, etc. | Services subscribed by the customer |
| Target Variable   | `Churn`                                       | Whether the customer has churned (`Yes` or `No`) |

---

### Objective

The objective is to **predict customer churn** using both structured data and additional features derived from simulated customer interactions (NLP, clustering, time-series). This enables businesses to:
- Identify customers at high risk of leaving
- Improve customer retention strategies
- Personalize offers and outreach based on behavior

---

### Target Variable

- `Churn`:  
  - `Yes` → The customer has left the company  
  - `No`  → The customer is still active  

---

### Dataset Details

- Rows: ~7,000  
- Columns: 21  
- Data Types: Mixed (numeric, categorical, boolean)  
- Format: CSV

---

### Example Use Cases

- Predicting churn using classification models (e.g., XGBoost, Random Forest)
- Segmenting users with clustering based on behavior
- Feature engineering from tenure, billing, and service usage
- Explainability using SHAP values to justify churn predictions

---

**Location:**  
The dataset is stored in the project at:  
`data/telco_churn.csv`
