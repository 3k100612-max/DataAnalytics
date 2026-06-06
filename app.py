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
st.set_page_config(page_title="DataScience Pro", layout="wide", page_icon="🤖")
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
    
    df_clean = df.drop_duplicates()
    total_cells = np.prod(df_clean.shape)
    null_count = df_clean.isnull().sum().sum()
    missing_percent_total = (null_count / total_cells) * 100
    
    col_h1, col_h2, col_h3 = st.columns(3)
    col_h1.metric("Total Rows", df_clean.shape[0])
    col_h2.metric("Total Columns", df_clean.shape[1])
    col_h3.metric("Initial Missingness", f"{missing_percent_total:.2f}%", delta=f"{null_count} cells", delta_color="inverse")

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df_clean.columns.tolist()
    df_imputed = df_clean.copy()

    with st.status("Cleaning and Imputing Data...", expanded=False) as status:
        for col in all_cols:
            if col in numeric_cols:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
            else:
                mode_val = df_imputed[col].mode()
                df_imputed[col] = df_imputed[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
        status.update(label="Data Preprocessing Complete!", state="complete", expanded=False)

    st.write("### 💎 Processed Data Preview", df_imputed.head())

    # --- 2. DATA MINING (PATTERN DISCOVERY) ---
    st.header("2. Pattern Discovery")
    tab_corr, tab_pca = st.tabs(["📊 Correlation Heatmap", "🗺️ Data Similarity Map (PCA)"])
    
    with tab_corr:
        if len(numeric_cols) > 1:
            corr = df_imputed[numeric_cols].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="mako", ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.info("Not enough numeric columns for a correlation heatmap.")

    with tab_pca:
        if len(numeric_cols) >= 2:
            scaler_pca = StandardScaler()
            pca_scaled = scaler_pca.fit_transform(df_imputed[numeric_cols])
            pca_engine = PCA(n_components=2)
            pca_results = pca_engine.fit_transform(pca_scaled)
            pca_plot_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
            pca_plot_df = pd.concat([pca_plot_df, df_imputed.reset_index(drop=True)], axis=1)
            fig_pca = px.scatter(pca_plot_df, x='PC1', y='PC2', color=all_cols[0], template="plotly_dark")
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.info("PCA requires at least 2 numeric columns.")

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
            
            # --- HYPERPARAMETER: MAX DEPTH ---
            max_depth_val = None
            if algo in ["Decision Tree", "Random Forest"]:
                max_depth_val = st.slider("Max Tree Depth:", 1, 20, 3, help="Controls how deep the tree grows. Higher depth can lead to overfitting.")

            train_btn = st.button("🚀 Train & Predict")

        with col_res:
            if train_btn:
                if not features:
                    st.error("Please select features (X) to train the model.")
                else:
                    with st.spinner(f'Training {algo}...'):
                        X = df_imputed[features]
                        y = df_imputed[target]
                        
                        if task == "Classification":
                            le = LabelEncoder()
                            y = le.fit_transform(y.astype(str))
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        sc = StandardScaler()
                        X_train_scaled = sc.fit_transform(X_train)
                        X_test_scaled = sc.transform(X_test)

                        # Model Selection Logic
                        if algo == "Linear Regression": model = LinearRegression()
                        elif algo == "Logistic Regression": model = LogisticRegression()
                        elif algo == "Random Forest":
                            model = RandomForestClassifier(max_depth=max_depth_val) if task == "Classification" else RandomForestRegressor(max_depth=max_depth_val)
                        elif algo == "Decision Tree":
                            model = DecisionTreeClassifier(max_depth=max_depth_val) if task == "Classification" else DecisionTreeRegressor(max_depth=max_depth_val)
                        elif algo == "SVM":
                            model = SVC(probability=True) if task == "Classification" else SVR()
                        elif algo == "KNN":
                            model = KNeighborsClassifier() if task == "Classification" else KNeighborsRegressor()
                        elif algo == "Naive Bayes":
                            model = GaussianNB()

                        model.fit(X_train_scaled, y_train)
                        preds = model.predict(X_test_scaled)

                        st.success(f"Model {algo} trained successfully!")
                        if task == "Regression":
                            m1, m2 = st.columns(2)
                            m1.metric("R² Score", f"{r2_score(y_test, preds):.4f}")
                            m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")
                        else:
                            st.metric("Accuracy Score", f"{accuracy_score(y_test, preds):.2%}")
                            with st.expander("View Classification Report"):
                                st.code(classification_report(y_test, preds))

                        # --- EDUCATIONAL COMPUTATION SECTION ---
                        st.markdown("---")
                        st.subheader("🧮 How was this computed?")
                        
                        with st.expander(f"Explain the logic behind {algo}"):
                            if algo == "Linear Regression":
                                st.write("**The Formula:**")
                                st.latex(r"Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \epsilon")
                                st.write("**Non-Technical Guide:**")
                                st.write("This model draws a straight 'Line of Best Fit'. It calculates how much $Y$ changes for every 1 unit increase in your $X$ variables.")
                                weights = pd.DataFrame({'Feature': features, 'Weight': model.coef_})
                                st.dataframe(weights)

                            elif algo == "Decision Tree":
                                st.write("**The Logic (CART):**")
                                st.latex(r"\text{Gini} = 1 - \sum (P_i)^2")
                                st.write(f"**Depth Used:** {max_depth_val}")
                                st.write("**Non-Technical Guide:**")
                                st.write("Think of this as a flowchart. The model asks 'Yes/No' questions to split your data into groups. It stops when it reaches the 'Max Depth' you selected or when the groups are pure.")

                            elif algo == "Naive Bayes":
                                st.write("**The Formula (Bayes Theorem):**")
                                st.latex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}")
                                st.write("**Non-Technical Guide:**")
                                st.write("This model calculates the probability of a result based on the features. It is 'Naive' because it assumes each feature is independent of the others.")

                            elif algo == "KNN":
                                st.write("**The Distance Formula (Euclidean):**")
                                st.latex(r"d(p,q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}")
                                st.write("**Non-Technical Guide:**")
                                st.write("KNN looks at the closest 'neighbors' to a data point. If most neighbors belong to Class A, the new point is also labeled Class A.")

                            elif algo == "SVM":
                                st.write("**The Goal: Optimal Hyperplane**")
                                st.write("SVM tries to find the widest possible 'road' (margin) that separates different classes.")
                                st.write("**Non-Technical Guide:**")
                                st.write("Imagine drawing a line between two groups of dots. SVM finds the line that stays as far away from the dots of both groups as possible.")

    elif ml_mode == "Unsupervised (Clustering)":
        cluster_features = st.multiselect("Select Features for Clustering:", numeric_cols, default=numeric_cols[:2])
        k_val = st.slider("Number of Clusters (K):", 2, 10, 3)
        if st.button("🧬 Run K-Means"):
            X_clust = StandardScaler().fit_transform(df_imputed[cluster_features])
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            df_imputed['Cluster'] = kmeans.fit_predict(X_clust)
            fig_clust = px.scatter(df_imputed, x=cluster_features[0], y=cluster_features[1], color='Cluster', template="plotly_white")
            st.plotly_chart(fig_clust, use_container_width=True)

    # --- 4. EXPORT ---
    st.divider()
    st.header("4. Export Results")
    st.download_button("📥 Download Cleaned & Processed Dataset", data=convert_df(df_imputed), file_name="ai_processed_data_2026.csv")

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© Timothy Bal-e 2026 | Smart Data Warehouse</div>", unsafe_allow_html=True)
