# 📊 ML Model Evaluation and Comparison Dashboard

This project is a Streamlit-based dashboard that allows users to:
- Load various real-world datasets (Diabetes, California Housing, etc.)
- Train and compare multiple machine learning models (Linear Regression, XGBoost, Random Forest, etc.)
- Visualize key evaluation metrics like R², MAE, RMSE, and training time
- Perform interactive model benchmarking using different datasets
- Containerized with Docker for easy deployment

---

## 🚀 Features

- ✅ Select from multiple datasets via dropdown (`sklearn.datasets`)
- ✅ Automatically split data and train multiple ML models
- ✅ Supports Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, and more
- ✅ Generates performance plots (bar charts for metrics, training time, etc.)
- ✅ Easy-to-read comparison table with R², MAE, RMSE
- ✅ Runs entirely in the browser via Streamlit
- ✅ Dockerized for consistent local or cloud deployment

---

## 📦 Installation

### 🔧 Run Locally with Docker

```bash
git clone https://github.com/yourusername/ml-eval-dashboard.git
cd ml-eval-dashboard
docker-compose up --build

visit http://localhost:8501
