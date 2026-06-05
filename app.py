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
    url = st.text_input("Paste Website URL (e.g., ph.investing.com, GitHub, etc.):")
    if url:
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin
            import io

            # 1. Setup headers to mimic a real browser (Investing.com blocks default scrapers)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')

            # --- LOGIC A: SEARCH FOR CSV FILES ---
            csv_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.lower().endswith('.csv') or "download" in href.lower():
                    csv_links.append(urljoin(url, href))

            # --- LOGIC B: SEARCH FOR HTML TABLES ---
            # We use io.StringIO to prevent the 'File Not Found' error
            try:
                html_tables = pd.read_html(io.StringIO(response.text))
            except:
                html_tables = []

            # --- UI SELECTION ---
            if csv_links:
                st.success(f"📂 Found {len(csv_links)} potential CSV download(s)!")
                selected_csv = st.selectbox("Select CSV to load:", csv_links)
                if st.button("Load CSV Data"):
                    csv_res = requests.get(selected_csv, headers=headers)
                    df = pd.read_csv(io.StringIO(csv_res.text))
            
            elif html_tables:
                st.info(f"📊 Found {len(html_tables)} data tables on this page.")
                
                # Create a list of table summaries to help the user choose
                table_labels = []
                for i, t in enumerate(html_tables):
                    # Clean the table: remove empty columns/rows
                    t = t.dropna(axis=1, how='all').dropna(axis=0, how='all')
                    preview = ", ".join(list(t.columns.astype(str))[:2])
                    table_labels.append(f"Table {i} (Fields: {preview})")
                
                selected_table_label = st.selectbox("Choose the specific data table to analyze:", table_labels)
                table_index = table_labels.index(selected_table_label)
                
                # Load the selected table
                df = html_tables[table_index]
                
                # --- AUTO-CLEANING FOR INVESTING.COM ---
                # Financial sites use commas and % signs which break ML. We clean them here.
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            # Remove commas, plus signs, and percentage symbols
                            df[col] = df[col].str.replace(',', '').str.replace('%', '').str.replace('+', '')
                            df[col] = pd.to_numeric(df[col])
                        except:
                            pass # Keep as string if it's a name (e.g., "Apple Inc")
                
                st.success(f"Successfully extracted {df.shape[0]} rows from Table {table_index}!")

            else:
                st.error("No data detected. This page might be protected or uses dynamic JavaScript.")

        except Exception as e:
            st.error(f"Scraping Error: {e}")


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
            st.subheader("Visualizing Data Relationships")
            st.info("""
                **Non-Technical Guide:** Each dot is a record. Dots that are **closer together** share similar medical or statistical patterns. 
                We have condensed all your columns into this 2D 'Similarity Map'.
            """)
            
            # PCA UI Controls
            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                color_by = st.selectbox("Color Dots By:", all_cols, help="Select a column to see how groups separate on the map.")
                hover_toggle = st.checkbox("Show Detailed Hover Info", value=True)

            # PCA Calculation
            scaler_pca = StandardScaler()
            pca_scaled = scaler_pca.fit_transform(df_imputed[numeric_cols])
            pca_engine = PCA(n_components=2)
            pca_results = pca_engine.fit_transform(pca_scaled)
            
            # Prepare Dataframe for Plotly
            pca_plot_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
            # Re-attach original data for labels/colors
            pca_plot_df = pd.concat([pca_plot_df, df_imputed.reset_index(drop=True)], axis=1)

            # Enhanced Plotly Chart
            fig_pca = px.scatter(
                pca_plot_df, x='PC1', y='PC2', 
                color=color_by,
                title=f"Similarity Map grouped by {color_by}",
                template="plotly_dark",
                hover_data=all_cols if hover_toggle else None,
                labels={"PC1": "Primary Pattern Direction", "PC2": "Secondary Pattern Direction"},
                color_continuous_scale="Viridis"
            )
            
            st.plotly_chart(fig_pca, use_container_width=True)
            
            with st.expander("💡 Technical Breakdown: What drives this map?"):
                loadings = pd.DataFrame(
                    pca_engine.components_.T, 
                    columns=['PC1', 'PC2'], 
                    index=numeric_cols
                )
                st.write("This table shows which features influence the horizontal (PC1) and vertical (PC2) positions:")
                st.dataframe(loadings.style.background_gradient(cmap='mako'))
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
            
            train_btn = st.button("🚀 Train & Predict")

        with col_res:
            if train_btn:
                if not features:
                    st.error("Please select features (X) to train the model.")
                else:
                    with st.spinner(f'Training {algo}...'):
                        time.sleep(1)
                        X = df_imputed[features]
                        y = df_imputed[target]
                        
                        if task == "Classification":
                            le = LabelEncoder()
                            y = le.fit_transform(y.astype(str))
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        sc = StandardScaler()
                        X_train = sc.fit_transform(X_train)
                        X_test = sc.transform(X_test)

                        # Model selection
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

                        st.success(f"Model {algo} trained successfully!")
                        if task == "Regression":
                            m1, m2 = st.columns(2)
                            m1.metric("R² Score", f"{r2_score(y_test, preds):.4f}")
                            m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.2f}")
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
                if len(cluster_features) < 2:
                    st.error("Please select at least 2 features.")
                else:
                    X_clust_scaled = StandardScaler().fit_transform(df_imputed[cluster_features])
                    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_clust_scaled)
                    df_imputed['Cluster'] = clusters
                    fig_clust = px.scatter(df_imputed, x=cluster_features[0], y=cluster_features[1], 
                                         color='Cluster', title=f"K-Means Results (K={k_val})", template="plotly_white")
                    st.plotly_chart(fig_clust, use_container_width=True)

    # --- 4. EXPORT ---
    st.divider()
    st.header("4. Export Results")
    st.download_button("📥 Download Cleaned & Processed Dataset", data=convert_df(df_imputed), file_name="ai_processed_data_promax.csv")

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© Timothy Bal-e 2026 | Smart Data Warehouse</div>", unsafe_allow_html=True)
