import streamlit as st
from sklearn.model_selection import train_test_split
from utils.data_loader import get_available_datasets, load_dataset
from utils.model_trainer import get_models
from utils.evaluation import run_all_models
import seaborn as sns
import matplotlib.pyplot as plt
from utils.visualization import plot_metric_bar_chart

st.set_page_config(page_title="📈 Model Comparison Dashboard", layout="wide")
st.title("🧪 ML Model Comparison on Regression Datasets")

# --- Dataset Selection ---
st.sidebar.header("Dataset Selection")
dataset_name = st.sidebar.selectbox("Choose a Dataset", get_available_datasets())

X, y = load_dataset(dataset_name)
st.write(f"### Dataset Preview – {dataset_name}")
st.dataframe(X.head())

# --- Train-Test Split ---
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# --- Model Evaluation ---
models = get_models()
results_df = run_all_models(models, X_train, X_test, y_train, y_test)

st.write("### 📊 Model Performance Metrics")
st.dataframe(results_df)

# --- Plotting ---
st.write("### 📈 Metric Comparison")
metric = st.selectbox("Select Metric", results_df.columns)

fig = plot_metric_bar_chart(results_df, metric)
st.pyplot(fig)