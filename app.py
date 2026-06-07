import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import psutil
import streamlit.components.v1 as components

# ML Imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

# --- 1. SYSTEM MONITORING ---
def get_vps_ram():
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except: return 0

browser_ram_js = """
<div id="browser-mem" style="font-family: sans-serif; color: #808495; font-size: 0.8rem;">Detecting Browser RAM...</div>
<script>
    function updateRam() {
        const mem = window.performance.memory;
        if (mem) {
            const used = (mem.usedJSHeapSize / (1024 * 1024)).toFixed(1);
            const total = (mem.jsHeapSizeLimit / (1024 * 1024)).toFixed(1);
            document.getElementById('browser-mem').innerHTML = "🌐 Browser Tab: " + used + "MB / " + total + "MB";
        }
    }
    setInterval(updateRam, 2000); updateRam();
</script>
"""

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_fix(file, rows):
    df = pd.read_csv(file, nrows=rows, low_memory=False)
    total_missing = df.isnull().sum().sum()
    # Downcast for 8GB VPS efficiency
    for col in df.columns:
        if df[col].dtype == 'float64': df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64': df[col] = df[col].astype('int32')
    df = df.fillna(df.median(numeric_only=True))
    for col in df.select_dtypes(exclude=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    return df, total_missing

# --- 3. UI SETUP ---
st.set_page_config(page_title="GPAI Data Pro", layout="wide", page_icon="🧪")
st.title("🧪 Advanced ML Workshop & Data Warehouse")
st.caption("Supervised & Unsupervised Suite | Optimized for 8GB VPS | @timothymarkbale2026")

with st.sidebar:
    st.header("🖥️ System Status")
    vps_mem = get_vps_ram()
    st.write(f"💾 VPS RAM: {vps_mem:.1f}MB / 8192MB")
    st.progress(min(vps_mem/8192, 1.0))
    components.html(browser_ram_js, height=50)
    st.markdown("---")
    row_limit = st.sidebar.slider("Max Rows to Load", 1000, 1000000, 500000)
    st.sidebar.write("© timothymarkbale2026")

# --- STEP 1: UPLOAD & CLEAN ---
uploaded_file = st.file_uploader("1. Upload CSV Dataset", type="csv")

if uploaded_file:
    df, total_missing = load_and_fix(uploaded_file, row_limit)
    st.success(f"✅ Data Cleaned: {total_missing} missing values fixed.")
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Cleaned Data", data=csv_data, file_name="cleaned_data.csv", mime="text/csv")

    st.divider()

    # --- STEP 4: SELECTION PATH ---
    st.header("🧠 2. Select Training Environment")
    mode = st.radio("Choose Mode:", ["None", "Exploratory (PCA)", "Unsupervised (Clustering)", "Supervised (Prediction)"], horizontal=True)

    # --- PATH A: EXPLORATORY (PCA) ---
    if mode == "Exploratory (PCA)":
        st.subheader("💎 PCA Dimensionality Reduction")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        pca_feats = st.multiselect("Features:", num_cols, default=num_cols[:min(3, len(num_cols))])
        if st.button("Run PCA"):
            X_pca = StandardScaler().fit_transform(df[pca_feats])
            pca = PCA(n_components=2)
            comps = pca.fit_transform(X_pca)
            pdf = pd.DataFrame(comps, columns=['PC1', 'PC2'])
            st.plotly_chart(px.scatter(pdf, x='PC1', y='PC2', template="plotly_dark", title="2D Projection Map"), use_container_width=True)

    # --- PATH B: UNSUPERVISED (K-MEANS) ---
    elif mode == "Unsupervised (Clustering)":
        st.subheader("🧬 Unsupervised Learning: K-Means Clustering")
        with st.expander("📖 Logic: How K-Means Computes"):
            st.write("**Goal:** Group data points into $K$ clusters based on distance.")
            st.latex(r"J = \sum_{j=1}^{K} \sum_{i=1}^{n} ||x_i^{(j)} - c_j||^2")
            st.info("💡 **Analogy:** Imagine people standing in a room. K-Means finds 'meeting points' (centroids) so that everyone is as close as possible to their nearest leader.")
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cluster_feats = st.multiselect("Features to Cluster:", num_cols, default=num_cols[:min(2, len(num_cols))])
        k_val = st.slider("Number of Clusters (K):", 2, 10, 3)
        
        if st.button("Run Clustering") and len(cluster_feats) >= 2:
            X_cl = StandardScaler().fit_transform(df[cluster_feats])
            km = KMeans(n_clusters=k_val, n_init=10)
            df['Cluster'] = km.fit_predict(X_cl)
            fig = px.scatter(df, x=cluster_feats[0], y=cluster_feats[1], color='Cluster', 
                             title=f"K-Means: Data grouped into {k_val} Clusters", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # --- PATH C: SUPERVISED (FULL SUITE) ---
    elif mode == "Supervised (Prediction)":
        st.subheader("🤖 Supervised Learning Workshop")
        m_col1, m_col2 = st.columns([1, 2])
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        with m_col1:
            target = st.selectbox("Predict Target (Y):", df.columns)
            features = st.multiselect("Predictors (X):", [c for c in num_cols if c != target])
            task = st.radio("Task:", ["Classification", "Regression"])
            algo = st.selectbox("Algorithm:", ["Linear/Logistic Regression", "Decision Tree (CART)", "Naive Bayes", "KNN", "SVM"])
            
            # Dynamic Hyperparameters
            params = {}
            if algo == "Decision Tree (CART)":
                params['depth'] = st.number_input("Max Depth:", 1, 100, 5)
            elif algo == "KNN":
                params['n'] = st.slider("Neighbors (K):", 1, 21, 5)
            
            run_train = st.button("🚀 Start Model Training")

        with m_col2:
            if run_train and features:
                # --- ALGORITHM LOGIC EXPANDERS ---
                with st.expander("📖 Mathematical Computation Guide", expanded=True):
                    if "Regression" in algo and task == "Regression":
                        st.write("**Linear Regression:** Finds the best line fit.")
                        st.latex(r"Y = \beta_0 + \beta_1X_1 + \epsilon")
                    elif "Decision Tree" in algo:
                        st.write("**CART Logic:** Splits data using Gini Impurity.")
                        st.latex(r"Gini = 1 - \sum (P_i)^2")
                    elif "Naive Bayes" in algo:
                        st.write("**Bayesian Probability:** Predicts based on prior knowledge.")
                        st.latex(r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}")
                    elif "KNN" in algo:
                        st.write("**Euclidean Distance:** Looks at the closest 'K' neighbors.")
                        st.latex(r"d = \sqrt{\sum (x_i - y_i)^2}")
                    elif "SVM" in algo:
                        st.write("**Hyperplane Optimization:** Finds the widest 'street' between groups.")
                        st.latex(r"w \cdot x - b = 0")

                # --- TRAINING ENGINE ---
                X, y = df[features], df[target]
                if task == "Classification":
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                    class_names = [str(c) for c in le.classes_]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_train)
                X_te_s = sc.transform(X_test)

                # Model Selection
                if algo == "Linear/Logistic Regression":
                    model = LogisticRegression() if task == "Classification" else LinearRegression()
                elif algo == "Decision Tree (CART)":
                    model = DecisionTreeClassifier(max_depth=params['depth']) if task == "Classification" else DecisionTreeRegressor(max_depth=params['depth'])
                elif algo == "Naive Bayes":
                    model = GaussianNB()
                elif algo == "KNN":
                    model = KNeighborsClassifier(n_neighbors=params['n']) if task == "Classification" else KNeighborsRegressor(n_neighbors=params['n'])
                elif algo == "SVM":
                    model = SVC() if task == "Classification" else SVR()

                model.fit(X_tr_s, y_train)
                preds = model.predict(X_te_s)

                # Results Visualization
                if task == "Classification":
                    cm = confusion_matrix(y_test, preds.astype(int))
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
                    ax.set_title(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
                    ax.set_xlabel(f"Predicted {target}"); ax.set_ylabel(f"Actual {target}")
                    st.pyplot(fig)
                else:
                    fig_reg = px.scatter(x=y_test, y=preds, labels={'x': f'Actual {target}', 'y': f'Predicted {target}'}, 
                                         title=f"R² Score: {r2_score(y_test, preds):.4f}", template="plotly_dark")
                    fig_reg.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_reg, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | Comprehensive ML Suite</p>", unsafe_allow_html=True)
