import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
import psutil

# --- MEMORY MONITORING ---
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ML Imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# --- OPTIMIZATION: DATA CACHING ---
@st.cache_data
def load_and_optimize(file, rows):
    """Loads data and downcasts types to save RAM"""
    df = pd.read_csv(file, nrows=rows, low_memory=False)
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

# --- PAGE CONFIG ---
st.set_page_config(page_title="DataScience Pro 8GB", layout="wide")
st.title("🌐 Data Warehouse & ML (8GB VPS Optimized)")

# --- SIDEBAR: RESOURCE MONITOR ---
st.sidebar.header("🖥️ VPS Status (8GB)")
if PSUTIL_AVAILABLE:
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 * 1024)
    # Visualizing 8GB (8192MB) limit
    st.sidebar.progress(min(mem_usage / 8192, 1.0))
    st.sidebar.caption(f"RAM Usage: {mem_usage:.1f} MB / 8192 MB")

st.sidebar.markdown("---")
# Increased limit for 8GB RAM
row_limit = st.sidebar.slider("Max Rows to Load", 1000, 1000000, 500000)

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    try:
        # Using the cached loader
        df = load_and_optimize(uploaded_file, row_limit)
        st.success(f"Successfully loaded {len(df)} rows into 8GB RAM.")
        
        # --- 1. DATA EXPLORATION ---
        st.header("1. Data Overview")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        
        col_a, col_b = st.columns(2)
        col_a.dataframe(df.head(10))
        
        if len(numeric_cols) > 1:
            with col_b:
                st.write("Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(df[numeric_cols].corr(), cmap="viridis", ax=ax)
                st.pyplot(fig)

        # --- 2. MACHINE LEARNING ---
        st.divider()
        st.header("2. Machine Learning Workshop")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            target = st.selectbox("Target (Y):", all_cols)
            features = st.multiselect("Features (X):", [c for c in numeric_cols if c != target])
            task = st.radio("Task Type:", ["Classification", "Regression"])
            
            # Algorithm selection based on task
            if task == "Classification":
                algo = st.selectbox("Model:", ["Random Forest", "Logistic Regression", "Decision Tree"])
            else:
                algo = st.selectbox("Model:", ["Random Forest", "Linear Regression", "Decision Tree"])
            
            train_btn = st.button("🚀 Run Analysis")

        with c2:
            if train_btn and features:
                # Fill missing values for ML
                X = df[features].fillna(df[features].mean())
                y = df[target]
                
                if task == "Classification":
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                else:
                    y = y.fillna(y.mean())

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                
                # Model Initialization
                if algo == "Random Forest":
                    model = RandomForestClassifier() if task == "Classification" else RandomForestRegressor()
                elif algo == "Decision Tree":
                    model = DecisionTreeClassifier() if task == "Classification" else DecisionTreeRegressor()
                elif algo == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = LogisticRegression(max_iter=1000)

                with st.spinner("Training Model..."):
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                
                if task == "Classification":
                    acc = accuracy_score(y_test, preds)
                    st.metric("Model Accuracy", f"{acc:.2%}")
                else:
                    r2 = r2_score(y_test, preds)
                    st.metric("R² Prediction Score", f"{r2:.4f}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
