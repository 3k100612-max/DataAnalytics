import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import psutil

# ML & Stats Imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

# --- VPS RESOURCE MONITORING ---
try:
    import psutil
    PS_AVAILABLE = True
except ImportError:
    PS_AVAILABLE = False

# --- MEMORY-EFFICIENT DATA LOADER & IMPUTER ---
@st.cache_data
def load_and_clean(file, rows):
    df = pd.read_csv(file, nrows=rows, low_memory=False)
    
    # 1. Count missing values before filling
    total_missing = df.isnull().sum().sum()
    missing_per_col = df.isnull().sum().to_dict()
    
    # 2. Downcast to save 50% RAM
    for col in df.columns:
        if df[col].dtype == 'float64': df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64': df[col] = df[col].astype('int32')
    
    # 3. AUTO-FILL (Imputation)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
    return df, total_missing, missing_per_col

# --- UI CONFIG ---
st.set_page_config(page_title="DataScience Pro", layout="wide", page_icon="🧪")
st.title("🧪 Advanced Data Science Warehouse: PCA & ML Control")
st.caption("Optimized for 8GB VPS | Created by @timothymarkbale2026")

# Sidebar
st.sidebar.header("🖥️ VPS System Status")
if PS_AVAILABLE:
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    st.sidebar.progress(min(mem/8192, 1.0))
    st.sidebar.caption(f"RAM Usage: {mem:.1f}MB / 8192MB")

st.sidebar.markdown("---")
row_limit = st.sidebar.slider("Max Rows to Load", 1000, 1000000, 500000)
st.sidebar.write("© timothymarkbale2026")

# --- 1. DATA ENTRY & HEALTH REPORT ---
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

if uploaded_file:
    # Get data and missing value counts
    df, total_missing, missing_dict = load_and_clean(uploaded_file, row_limit)
    
    st.header("📋 1. Data Health Report")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows Loaded", len(df))
    m2.metric("Missing Values Found", total_missing, delta="Automatically Filled", delta_color="normal")
    m3.metric("Data Quality", f"{((1 - (total_missing/(len(df)*len(df.columns))))*100):.1f}%")

    with st.expander("🔍 View missing data breakdown per column"):
        missing_df = pd.DataFrame(list(missing_dict.items()), columns=['Column', 'Missing Values'])
        st.table(missing_df[missing_df['Missing Values'] > 0])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # --- 2. PCA MODULE ---
    st.divider()
    st.header("💎 2. Principal Component Analysis (PCA)")
    
    pca_col1, pca_col2 = st.columns([1, 2])
    
    with pca_col1:
        pca_features = st.multiselect("Select Features for PCA:", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
        pca_components = st.number_input("Number of Principal Components:", min_value=2, max_value=min(len(pca_features), 10), value=2)
        color_target = st.selectbox("Color points by (Categorical Target):", df.columns)
        run_pca = st.button("🚀 Run PCA Visualization")

    with pca_col2:
        if run_pca and len(pca_features) >= 2:
            X_pca = StandardScaler().fit_transform(df[pca_features])
            pca = PCA(n_components=pca_components)
            components = pca.fit_transform(X_pca)
            
            pca_df = pd.DataFrame(data=components, columns=[f'PC{i+1}' for i in range(pca_components)])
            pca_df[color_target] = df[color_target].values
            
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color=color_target, 
                                 title=f"PCA Map: Data simplified into 2D",
                                 template="plotly_dark", opacity=0.7)
            st.plotly_chart(fig_pca, use_container_width=True)
            
            var_exp = sum(pca.explained_variance_ratio_) * 100
            st.info(f"💡 **Information Retained:** This map captures **{var_exp:.2f}%** of the original data variance.")

    # --- 3. ML WORKSHOP ---
    st.divider()
    st.header("🤖 3. Machine Learning Workshop")
    
    ml_col1, ml_col2 = st.columns([1, 2])
    
    with ml_col1:
        target = st.selectbox("Predict Target (Y):", df.columns, index=len(df.columns)-1)
        features = st.multiselect("Predictor Features (X):", [c for c in numeric_cols if c != target])
        task = st.radio("Task Type:", ["Classification (Groups)", "Regression (Numbers)"])
        algo = st.selectbox("Model:", ["Decision Tree", "Linear Regression", "Logistic Regression"])
        
        selected_depth = 5
        if algo == "Decision Tree":
            st.markdown("---")
            st.subheader("🌳 Tree Configuration")
            selected_depth = st.number_input("Enter Max Depth (1-30):", min_value=1, max_value=30, value=5)
            st.info("**Non-Technical Tip:** Think of 'Depth' as the number of questions the computer asks to make a decision.")

        train_btn = st.button("📈 Train & Generate Graphics")

    with ml_col2:
        if train_btn and features:
            X = df[features]
            y = df[target]
            
            class_names = None
            if "Classification" in task:
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                class_names = [str(c) for c in le.classes_]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            if algo == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=selected_depth) if "Classification" in task else DecisionTreeRegressor(max_depth=selected_depth)
            elif algo == "Linear Regression": model = LinearRegression()
            else: model = LogisticRegression()

            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

            # Accuracy Visuals
            if "Classification" in task:
                cm = confusion_matrix(y_test, preds)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm, xticklabels=class_names, yticklabels=class_names)
                ax_cm.set_title(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
                ax_cm.set_xlabel(f"Predicted {target}")
                ax_cm.set_ylabel(f"Actual {target}")
                st.pyplot(fig_cm)
            else:
                fig_reg = px.scatter(x=y_test, y=preds, labels={'x': f'Actual {target}', 'y': f'Predicted {target}'}, 
                                     title=f"Precision Map (R² Score: {r2_score(y_test, preds):.4f})", template="plotly_dark")
                fig_reg.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="Red", dash="dash"))
                st.plotly_chart(fig_reg, use_container_width=True)

            if algo == "Decision Tree":
                st.write("#### Logic Flowchart (First 2 Levels)")
                fig_t, ax_t = plt.subplots(figsize=(15, 8))
                plot_tree(model, feature_names=features, class_names=class_names, filled=True, max_depth=2, rounded=True)
                st.pyplot(fig_t)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | VPS Optimized 8GB</p>", unsafe_allow_html=True)
