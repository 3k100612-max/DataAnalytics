import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import time
import requests

# ML Imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# --- HELPER FUNCTIONS ---
@st.cache_data
def convert_df(df, file_format="csv"):
    if file_format == "csv":
        return df.to_csv(index=False).encode('utf-8')

# --- PAGE CONFIG ---
st.set_page_config(page_title="DataScience", layout="wide", page_icon="🤖")
st.title("🌐 Data Warehouse & Machine Learning Prototype")

# --- SIDEBAR ---
st.sidebar.header("📥 Data Source")
source_type = st.sidebar.radio("Select Source:", ["Upload CSV", "Website URL"])
st.sidebar.markdown("---")
st.sidebar.caption("© Timothy Bal-e 2026")

df = None

# --- DATA LOADING ---
if source_type == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    url = st.text_input("Paste URL (Wikipedia/GitHub(Raw CSV SITES)):")
    if url:
        try:
            # Check content type for better detection
            response = requests.get(url, timeout=10)
            content_type = response.headers.get('Content-Type', '').lower()

            if url.endswith('.csv') or "raw" in url or 'text/csv' in content_type:
                df = pd.read_csv(url)
                st.success("CSV Data Extracted!")
            else:
                tables = pd.read_html(url)
                if len(tables) == 0:
                    st.warning("No tables found on this page.")
                elif len(tables) == 1:
                    df = tables[0]
                    st.success("Single table found and extracted!")
                else:
                    st.info(f"Found {len(tables)} tables. Please select the one for analysis:")
                    table_idx = st.selectbox(
                        "Select Table to Analyze", 
                        range(len(tables)),
                        format_func=lambda x: f"Table {x} ({len(tables[x])} rows, {len(tables[x].columns)} columns)"
                    )
                    df = tables[table_idx]
        except Exception as e: 
            st.error(f"Error: {e}")

if df is not None:
    # --- 1. DATA PREP & CLEANING ---
    st.header("1. Data Health & Preprocessing")
    
    # Initial Audit
    df_clean = df.drop_duplicates()
    total_cells = np.prod(df_clean.shape)
    null_count = df_clean.isnull().sum().sum()
    missing_percent_total = (null_count / total_cells) * 100
    
    # UI Layout for Health Stats
    col_h1, col_h2, col_h3 = st.columns(3)
    col_h1.metric("Total Rows", df_clean.shape[0])
    col_h2.metric("Total Columns", df_clean.shape[1])
    col_h3.metric("Initial Missingness", f"{missing_percent_total:.2f}%", delta=f"{null_count} cells", delta_color="inverse")

    # Missing Data Breakdown Table
    missing_data_table = pd.DataFrame({
        'Missing Values': df_clean.isnull().sum(),
        'Percentage (%)': (df_clean.isnull().sum() / len(df_clean)) * 100
    })
    
    with st.expander("🔍 Detailed Data Health Audit"):
        st.write("Column-wise Missing Data Summary:")
        st.dataframe(missing_data_table.style.format({'Percentage (%)': '{:.2f}%'}))

    # --- AUTOMATED IMPUTATION ENGINE ---
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df_clean.columns.tolist()
    df_imputed = df_clean.copy()

    with st.status("Cleaning and Imputing Data...", expanded=False) as status:
        st.write("Identifying missing patterns...")
        time.sleep(0.5)
        for col in all_cols:
            if col in numeric_cols:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
            else:
                mode_val = df_imputed[col].mode()
                df_imputed[col] = df_imputed[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
        st.write("Finalizing data integrity checks...")
        time.sleep(0.5)
        status.update(label="Data Preprocessing Complete!", state="complete", expanded=False)

    # Verification Message
    final_nulls = df_imputed.isnull().sum().sum()
    if final_nulls == 0:
        st.success(f"✅ **Data Health: 100% Clean.** All {null_count} missing values have been successfully imputed.")
    else:
        st.warning(f"⚠️ Processed with {final_nulls} values remaining.")

    st.write("### 💎 Processed Data Preview", df_imputed.head())

    # --- 2. DATA MINING (PATTERN DISCOVERY) ---
    st.header("2. Pattern Discovery")
    tab_corr, tab_pca = st.tabs(
