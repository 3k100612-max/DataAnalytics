import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
import time

# --- SAFE IMPORT FOR PSUTIL ---
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
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# --- MEMORY OPTIMIZATION ---
def optimize_memory(df):
    """ Downcast types to save RAM on 2GB VPS """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

# --- PAGE CONFIG ---
st.set_page_config(page_title="DataScience Pro VPS", layout="wide", page_icon="🤖")
st.title("🌐 Data Warehouse & ML Prototype (VPS Optimized)")

# --- SIDEBAR: RESOURCE MONITOR ---
st.sidebar.header("🖥️ VPS Status")
if PSUTIL_AVAILABLE:
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 * 1024)
    st.sidebar.progress(min(mem_usage / 2048, 1.0))
    st.sidebar.caption(f"RAM Usage: {mem_usage:.1f} MB / 2048 MB")
else:
    st.sidebar.warning("Run 'pip install psutil' to see RAM usage.")

st.sidebar.markdown("---")
st.sidebar.header("📥 Data Source")
row_limit = st.sidebar.number_input("Memory Safety: Max Rows", 1000, 100000, 50000, help="2GB VPS limit: 50k-70k rows recommended.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
df = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, nrows=row_limit)
        df = optimize_memory(df)
        st.success(f"Data Loaded: {len(df)} rows.")
    except Exception as e:
        st.error(f"Error: {e}")

if df is not None:
    # --- 1. DATA HEALTH & PREP ---
    st.header("1. Data Health & Preprocessing")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Fast Imputation (In-place to save RAM)
    df_imputed = df.copy()
    for col in all_cols:
        if col in numeric_cols:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
        else:
            df_imputed[col] = df_imputed[col].fillna("Unknown")

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Rows Loaded", len(df_imputed))
    col_m2.metric("Numeric Features", len(numeric_cols))
    st.write("### 💎 Data Preview", df_imputed.head())

    # --- 2. PATTERN DISCOVERY ---
    st.header("2. Pattern Discovery")
    tab_corr, tab_pca = st.tabs(["📊 Correlation", "🗺️ Similarity Map (PCA)"])
    
    with tab_corr:
        if len(numeric_cols) > 1:
            fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
            sns.heatmap(df_imputed[numeric_cols].corr(), annot=True, cmap="mako", ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.info("Need more numeric columns for Correlation.")

    with tab_pca:
        if len(numeric_cols) >= 2:
            pca_scaled = StandardScaler().fit_transform(df_imputed[numeric_cols])
            pca_res = PCA(n_components=2).fit_transform(pca_scaled)
            pca_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', template="plotly_dark")
            st.plotly_chart(fig_pca, use_container_width=True)

    # --- 3. MACHINE LEARNING WORKSHOP ---
    st.divider()
    st.header("3. 🤖 Machine Learning Workshop")
    
    ml_type = st.selectbox("Select Learning Type:", ["Supervised (Prediction)", "Unsupervised (Clustering)"])

    if ml_type == "Supervised (Prediction)":
        col_set, col_res = st.columns([1, 2])
        
        with col_set:
            target = st.selectbox("Target Variable (Y):", all_cols)
            features = st.multiselect("Features (X):", [c for c in numeric_cols if c != target])
            
            # --- TASK FILTERING ---
            task = st.radio("Goal:", ["Classification (Groups)", "Regression (Numbers)"])
            
            if task == "Classification (Groups)":
                algo = st.selectbox("Algorithm:", ["Random Forest", "Decision Tree", "Logistic Regression", "SVM", "KNN", "Naive Bayes"])
            else:
                algo = st.selectbox("Algorithm:", ["Random Forest", "Decision Tree", "Linear Regression", "SVM", "KNN"])
            
            # --- HYPERPARAMETER: MAX DEPTH ---
            max_depth_val = None
            if "Tree" in algo or "Forest" in algo:
                max_depth_val = st.slider("Max Tree Depth:", 1, 20, 3)
            
            train_btn = st.button("🚀 Train & Predict")

        with col_res:
            if train_btn and features:
                with st.spinner(f"Training {algo}..."):
                    X = df_imputed[features]
                    y = df_imputed[target]
                    
                    if task == "Classification (Groups)":
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Scaling
                    sc = StandardScaler()
                    X_train = sc.fit_transform(X_train)
                    X_test = sc.transform(X_test)

                    # Model Selection
                    if algo == "Random Forest":
                        model = RandomForestClassifier(max_depth=max_depth_val) if task == "Classification (Groups)" else RandomForestRegressor(max_depth=max_depth_val)
                    elif algo == "Decision Tree":
                        model = DecisionTreeClassifier(max_depth=max_depth_val) if task == "Classification (Groups)" else DecisionTreeRegressor(max_depth=max_depth_val)
                    elif algo == "Linear Regression": model = LinearRegression()
                    elif algo == "Logistic Regression": model = LogisticRegression()
                    elif algo == "SVM": model = SVC() if task == "Classification (Groups)" else SVR()
                    elif algo == "KNN": model = KNeighborsClassifier() if task == "Classification (Groups)" else KNeighborsRegressor()
                    elif algo == "Naive Bayes": model = GaussianNB()

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    st.success(f"{algo} Trained!")
                    if task == "Regression (Numbers)":
                        st.metric("R² Score", f"{r2_score(y_test, preds):.4f}")
                    else:
                        st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2%}")
                        st.code(classification_report(y_test, preds))

                    # --- EDUCATIONAL SECTION ---
                    with st.expander("🧮 How was this computed?"):
                        if "Tree" in algo:
                            st.latex(r"\text{Gini} = 1 - \sum (P_i)^2")
                            st.write(f"The model split data into branches up to depth **{max_depth_val}**.")
                        elif algo == "Linear Regression":
                            st.latex(r"Y = \beta_0 + \beta_1X_1 + \epsilon")
                        elif algo == "KNN":
                            st.latex(r"d = \sqrt{\sum(x_i - y_i)^2}")

    elif ml_type == "Unsupervised (Clustering)":
        c_features = st.multiselect("Clustering Features:", numeric_cols)
        k = st.slider("K (Clusters):", 2, 10, 3)
        if st.button("🧬 Run K-Means") and c_features:
            X_c = StandardScaler().fit_transform(df_imputed[c_features])
            df_imputed['Cluster'] = KMeans(n_clusters=k, n_init=10).fit_predict(X_c)
            st.plotly_chart(px.scatter(df_imputed, x=c_features[0], y=c_features[1], color='Cluster'))

st.sidebar.caption("© Timothy Bal-e 2026")
