import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import time

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
st.set_page_config(page_title="DataScience Pro Max", layout="wide", page_icon="🤖")
st.title("🌐 Advanced Data Warehouse & ML Suite")

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
    url = st.text_input("Paste Website URL (e.g., GitHub Raw CSV):")
    if url:
        try:
            df = pd.read_csv(url) if (url.endswith('.csv') or "raw" in url) else pd.read_html(url)[0]
            st.success("Data extracted successfully!")
        except Exception as e:
            st.error(f"Extraction Error: {e}")

if df is not None:
    # --- 1. DATA PREP & CLEANING ---
    st.header("1. Data Health & Preprocessing")
    df_clean = df.drop_duplicates()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df_clean.columns.tolist()
    
    # Imputation
    df_imputed = df_clean.copy()
    for col in all_cols:
        if col in numeric_cols:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
        else:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else "Unknown")

    st.write("### Data Preview", df_imputed.head())

    # --- 2. DATA MINING (PATTERN DISCOVERY) ---
    st.header("2. Pattern Discovery")
    tab_corr, tab_pca = st.tabs(["Correlation Heatmap", "PCA Analysis"])
    
    with tab_corr:
        corr = df_imputed[numeric_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="mako", ax=ax_corr)
        st.pyplot(fig_corr)

    with tab_pca:
        if len(numeric_cols) >= 2:
            scaler_pca = StandardScaler()
            pca_data = scaler_pca.fit_transform(df_imputed[numeric_cols])
            pca = PCA(n_components=2)
            components = pca.fit_transform(pca_data)
            pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', title="PCA Map", template="plotly_dark")
            st.plotly_chart(fig_pca, use_container_width=True)

    # --- 3. MACHINE LEARNING WORKSHOP ---
    st.divider()
    st.header("3. 🤖 Machine Learning Workshop")
    
    ml_mode = st.selectbox("Select Learning Type:", ["Supervised (Prediction)", "Unsupervised (Clustering)"])

    if ml_mode == "Supervised (Prediction)":
        col_set, col_res = st.columns([1, 2])
        
        with col_set:
            target = st.selectbox("Target Variable (Y):", all_cols)
            features = st.multiselect("Features (X):", [c for c in numeric_cols if c != target], default=[c for c in numeric_cols if c != target][:3])
            task = st.radio("Task:", ["Classification", "Regression"])
            
            if task == "Classification":
                algo = st.selectbox("Algorithm:", ["Logistic Regression", "Random Forest", "Decision Tree", "SVM", "KNN", "Naive Bayes"])
            else:
                algo = st.selectbox("Algorithm:", ["Linear Regression", "Random Forest", "Decision Tree", "SVM", "KNN"])
            
            train_btn = st.button("🚀 Train & Predict")

        with col_res:
            if train_btn:
                with st.spinner(f'Applying {algo} and analyzing patterns...'):
                    time.sleep(1.5) # Simulated loading for UX
                    
                    X = df_imputed[features]
                    y = df_imputed[target]
                    
                    if task == "Classification":
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Scaling
                    sc = StandardScaler()
                    X_train = sc.fit_transform(X_train)
                    X_test = sc.transform(X_test)

                    # Model Selection Logic
                    if algo == "Linear Regression": model = LinearRegression()
                    elif algo == "Logistic Regression": model = LogisticRegression()
                    elif algo == "Random Forest" and task == "Classification": model = RandomForestClassifier()
                    elif algo == "Random Forest" and task == "Regression": model = RandomForestRegressor()
                    elif algo == "Decision Tree" and task == "Classification": model = DecisionTreeClassifier()
                    elif algo == "Decision Tree" and task == "Regression": model = DecisionTreeRegressor()
                    elif algo == "SVM" and task == "Classification": model = SVC()
                    elif algo == "SVM" and task == "Regression": model = SVR()
                    elif algo == "KNN" and task == "Classification": model = KNeighborsClassifier()
                    elif algo == "KNN" and task == "Regression": model = KNeighborsRegressor()
                    elif algo == "Naive Bayes": model = GaussianNB()

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    # Output
                    st.success(f"Model {algo} trained successfully!")
                    if task == "Regression":
                        st.metric("R² Score", f"{r2_score(y_test, preds):.4f}")
                        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")
                    else:
                        st.metric("Accuracy Score", f"{accuracy_score(y_test, preds):.2%}")
                        st.text("Detailed Report:")
                        st.code(classification_report(y_test, preds))

    elif ml_mode == "Unsupervised (Clustering)":
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            cluster_features = st.multiselect("Select Features for Clustering:", numeric_cols, default=numeric_cols[:2])
            k_val = st.slider("Number of Clusters (K):", 2, 10, 3)
            cluster_btn = st.button("🧬 Run K-Means")
            
        with col_c2:
            if cluster_btn:
                with st.spinner('Calculating clusters...'):
                    X_clust = df_imputed[cluster_features]
                    X_clust_scaled = StandardScaler().fit_transform(X_clust)
                    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_clust_scaled)
                    
                    df_imputed['Cluster'] = clusters
                    fig_clust = px.scatter(df_imputed, x=cluster_features[0], y=cluster_features[1], 
                                         color='Cluster', title=f"K-Means Results (K={k_val})", template="plotly_white")
                    st.plotly_chart(fig_clust, use_container_width=True)
                    st.write("Cluster Distribution:", df_imputed['Cluster'].value_counts())

    # --- 4. EXPORT ---
    st.divider()
    st.header("4. Export Results")
    st.download_button("📥 Download Processed Dataset", data=convert_df(df_imputed), file_name="ai_processed_data.csv")

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© Timothy Bal-e 2026 | Smart Data Warehouse Pro Max</div>", unsafe_allow_html=True)
