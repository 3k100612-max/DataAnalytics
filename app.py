import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time
import requests
import psutil  # New import to monitor VPS RAM
import os

# ML Imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# --- NEW: MEMORY OPTIMIZATION FUNCTION ---
def optimize_memory(df):
    """ Downcast data types to save RAM for 2GB VPS limits """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

# --- PAGE CONFIG ---
st.set_page_config(page_title="DataScience Lite", layout="wide", page_icon="⚡")
st.title("🌐 Data Warehouse (VPS Optimized)")

# --- SIDEBAR: SYSTEM MONITOR ---
st.sidebar.header("🖥️ VPS Status")
process = psutil.Process(os.getpid())
mem_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
st.sidebar.progress(min(mem_usage / 2048, 1.0)) # 2048MB limit
st.sidebar.caption(f"RAM Usage: {mem_usage:.2f} MB / 2048 MB")

st.sidebar.markdown("---")
st.sidebar.header("📥 Data Source")
source_type = st.sidebar.radio("Select Source:", ["Upload CSV", "Website URL"])

# --- NEW: ROW LIMITER FOR 2GB RAM ---
st.sidebar.subheader("⚙️ Memory Safety")
row_limit = st.sidebar.number_input("Max Rows to Load:", min_value=1000, max_value=100000, value=50000, step=5000, help="For a 2GB VPS, keep this under 100k.")

df = None

# --- DATA LOADING ---
if source_type == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        try:
            # Load only the specified number of rows to prevent crashing
            df = pd.read_csv(uploaded_file, nrows=row_limit)
            df = optimize_memory(df)
            st.success(f"Loaded {len(df)} rows. Memory optimized.")
        except Exception as e:
            st.error(f"Memory Error: {e}")
else:
    url = st.text_input("Paste URL (Raw CSV):")
    if url:
        try:
            df = pd.read_csv(url, nrows=row_limit)
            df = optimize_memory(df)
            st.success("Data Extracted & Optimized!")
        except Exception as e: 
            st.error(f"Error: {e}")

if df is not None:
    # --- 1. DATA PREP ---
    st.header("1. Data Health")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Quick Imputation (inplace to save RAM)
    for col in all_cols:
        if col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna("Unknown")

    st.write("### 💎 Processed Data Preview", df.head())

    # --- 2. MACHINE LEARNING WORKSHOP ---
    st.divider()
    st.header("3. 🤖 Machine Learning Workshop")
    
    col_set, col_res = st.columns([1, 2])
    
    with col_set:
        target = st.selectbox("Target Variable (Y):", all_cols)
        features = st.multiselect("Features (X):", [c for c in numeric_cols if c != target], default=[c for c in numeric_cols if c != target][:2])
        task = st.radio("Task:", ["Classification", "Regression"])
        algo = st.selectbox("Algorithm:", ["Linear Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"])
        
        # Hyperparameter for Trees
        max_depth_val = 3
        if algo == "Decision Tree":
            max_depth_val = st.slider("Max Tree Depth:", 1, 10, 3)

        train_btn = st.button("🚀 Train Model")

    with col_res:
        if train_btn:
            if not features:
                st.error("Select features first.")
            else:
                with st.spinner('Training...'):
                    X = df[features]
                    y = df[target]
                    
                    if task == "Classification":
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Scaling (Crucial for KNN/SVM)
                    sc = StandardScaler()
                    X_train = sc.fit_transform(X_train)
                    X_test = sc.transform(X_test)

                    # Model selection
                    if algo == "Linear Regression": model = LinearRegression()
                    elif algo == "Decision Tree":
                        model = DecisionTreeClassifier(max_depth=max_depth_val) if task == "Classification" else DecisionTreeRegressor(max_depth=max_depth_val)
                    elif algo == "SVM":
                        model = SVC() if task == "Classification" else SVR()
                    elif algo == "KNN":
                        model = KNeighborsClassifier() if task == "Classification" else KNeighborsRegressor()
                    elif algo == "Naive Bayes":
                        model = GaussianNB()

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    st.success(f"{algo} Complete!")
                    if task == "Regression":
                        st.metric("R² Score", f"{r2_score(y_test, preds):.4f}")
                    else:
                        st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2%}")

                    # --- EDUCATIONAL BREAKDOWN ---
                    with st.expander("🧮 How was this computed?"):
                        if algo == "Linear Regression":
                            st.latex(r"Y = \beta_0 + \beta_1X_1 + \epsilon")
                            st.write("Calculates a 'Line of Best Fit' to predict continuous numbers.")
                        elif algo == "KNN":
                            st.latex(r"d = \sqrt{\sum(x_i - y_i)^2}")
                            st.write("Finds the closest neighbors in the data to make a prediction.")
                        elif algo == "Decision Tree":
                            st.write(f"Flowchart logic with depth {max_depth_val}. Splits data based on feature importance.")

    st.sidebar.markdown("---")
    st.sidebar.caption("© Timothy Bal-e 2026")
