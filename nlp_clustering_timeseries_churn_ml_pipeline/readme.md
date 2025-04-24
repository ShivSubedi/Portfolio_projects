# NLP + Clustering + Time-Series Churn ML Pipeline

This project presents a full-stack machine learning pipeline designed to model and forecast customer churn using multimodal data sources — including natural language customer interactions, behavioral time-series patterns, and structured metadata.

Here, we  integrate:
- **NLP for text mining**,
- **unsupervised clustering for segmentation**, and
- **supervised learning models** to build a predictive system that mimics how enterprise teams in **banking, credit cards, and customer service industries** approach churn risk analysis.

---

## Project Objective

**Can we predict whether a customer is likely to churn in the next 30 days using their recent interactions, behavior patterns, and sentiment trends?**

This project simulates a real-world scenario where companies want to proactively identify at-risk customers using a combination of:
- Support interaction logs (textual complaints, inquiries)
- Account usage or service activity over time
- Customer demographic and service metadata

---

## Key Components

| Module | Description |
|--------|-------------|
| `NLP Preprocessing` | Clean and analyze customer messages, extract sentiment and embeddings |
| `Clustering` | Segment customers by interaction patterns using KMeans or HDBSCAN |
| `Time-Series Features` | Engineer lag features, rolling trends, and behavioral metrics |
| `Supervised Modeling` | Train LogReg, Decision Trees, Random Forest, and XGBoost for churn prediction |
| `Interpretability` | Use SHAP to explain model outputs and identify key churn drivers |

---

## Data Sources

- **Telco Customer Churn Dataset** (IBM Sample via Kaggle)  
- **Simulated Customer Interaction Logs** with timestamped text messages for each customer

---

## Techniques Used

- Natural Language Processing (VADER, TF-IDF, BERT embeddings)
- Unsupervised Clustering (KMeans, DBSCAN, UMAP)
- Time-Series Feature Engineering (rolling stats, lag features)
- Supervised ML (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Explainability (SHAP)

---

##  Business Value

This pipeline demonstrates how modern machine learning workflows can be applied to:
- Identify and retain at-risk customers
- Understand customer sentiment over time
- Segment users by behavioral profiles
- Provide interpretable insights for operational teams

---

## Folder Structure
- **data:** Stores raw and processed datasets used throughout the project.
- **src:** Contains modular Python scripts for data loading, feature engineering (turning raw data into model-ready inputs), modeling, and utilities.
- **notebooks:** Houses exploratory and development notebooks organized by project phase.
- **outputs:** Saves model artifacts, evaluation plots, predictions, and SHAP (SHapley Additive exPlanations) explanations (i.e., how much each feature contributed).


