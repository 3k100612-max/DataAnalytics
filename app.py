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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

# --- 1. VPS & BROWSER RAM MONITORING ---
def get_vps_ram():
    try:
        mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        return mem
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

# --- 2. DATA ENGINE (Step 2: Fix Missing Data) ---
@st.cache_data
def load_and_fix(file, rows):
    df = pd.read_csv(file, nrows=rows, low_memory=False)
    
    # Track missing data
    total_missing = df.isnull().sum().sum()
    
    # Optimization for 8GB VPS
    for col in df.columns:
        if df[col].dtype == 'float64': df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64': df[col] = df[col].astype('int32')
    
    # Fix Missing Values (Imputation)
    df = df.fillna(df.median(numeric_only=True))
    for col in df.select_dtypes(exclude=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
    return df, total_missing

# --- 3. UI SETUP ---
st.set_page_config(page_title="DataScience Pro", layout="wide", page_icon="📈")
st.title("📊 Pro Data Warehouse & ML Workshop")
st.caption("Sequential Workflow | Optimized for 8GB VPS | @timothymarkbale2026")

# Sidebar
with st.sidebar:
    st.header("🖥️ System Status")
    vps_mem = get_vps_ram()
    st.write(f"💾 VPS RAM: {vps_mem:.1f}MB / 8192MB")
    st.progress(min(vps_mem/8192, 1.0))
    components.html(browser_ram_js, height=50)
    st.markdown("---")
    row_limit = st.sidebar.slider("Max Rows to Load", 1000, 1000000, 500000)
    st.sidebar.write("© timothymarkbale2026")

# --- STEP 1: DATA UPLOAD ---
uploaded_file = st.file_uploader("1. Upload your CSV dataset", type="csv")

if uploaded_file:
    # --- STEP 2: CHECK & FIX ---
    df, total_missing = load_and_fix(uploaded_file, row_limit)
    
    st.success(f"✅ Data Health Check Complete: {total_missing} missing values fixed.")
    
    # --- STEP 3: DOWNLOAD FIXED DATA ---
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Fixed Dataset (CSV)",
        data=csv_data,
        file_name="fixed_dataset.csv",
        mime="text/csv",
        help="Click to download the version with no missing values."
    )

    st.divider()

    # --- STEP 4 & 5: CONDITIONAL RUN (PCA/HEATMAP vs ML) ---
    st.header("🧠 2. Choose Your Analysis Path")
    st.write("To save RAM, the system runs either Exploratory Analysis or ML Training, not both at once.")
    
    mode = st.radio("Select Action:", 
                    ["👀 Exploratory Analysis (PCA & Heatmap)", "🤖 Machine Learning Training"],
                    help="Switching to ML will stop PCA to save memory.")

    # --- PATH A: EXPLORATORY ANALYSIS ---
    if mode == "👀 Exploratory Analysis (PCA & Heatmap)":
        st.subheader("💎 Exploratory Visuals")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Principal Component Analysis (PCA)**")
            pca_feats = st.multiselect("Select PCA Features:", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
            target_color = st.selectbox("Color Map by:", df.columns)
            
            if st.button("Generate PCA Map") and len(pca_feats) >= 2:
                X_pca = StandardScaler().fit_transform(df[pca_feats])
                pca = PCA(n_components=2)
                comps = pca.fit_transform(X_pca)
                pdf = pd.DataFrame(comps, columns=['PC1', 'PC2'])
                pdf[target_color] = df[target_color].values
                fig_pca = px.scatter(pdf, x='PC1', y='PC2', color=target_color, template="plotly_dark", title="2D Projection")
                st.plotly_chart(fig_pca, use_container_width=True)
        
        with col_b:
            st.write("**Correlation Heatmap**")
            if st.button("Generate Feature Heatmap"):
                corr = df[numeric_cols].corr()
                fig_corr, ax_corr = plt.subplots()
                sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax_corr)
                ax_corr.set_title("Feature Relationships")
                st.pyplot(fig_corr)

    # --- PATH B: MACHINE LEARNING TRAINING ---
    else:
        st.subheader("🤖 Machine Learning Workshop")
        st.info("Exploratory tools are currently disabled to maximize RAM for Training.")
        
        m_col1, m_col2 = st.columns([1, 2])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        with m_col1:
            # --- STEP 6: DYNAMIC LABELS ---
            target = st.selectbox("Predict Target (Y):", df.columns, help="The app will use this label for all charts.")
            features = st.multiselect("Predictors (X):", [c for c in numeric_cols if c != target])
            task = st.radio("Goal:", ["Classification (Groups)", "Regression (Numbers)"])
            
            if task == "Classification (Groups)":
                algo = st.selectbox("Model:", ["Decision Tree", "Logistic Regression"])
            else:
                algo = st.selectbox("Model:", ["Decision Tree", "Linear Regression"])
            
            # --- STEP 7: CONDITIONAL DEPTH ---
            depth = 5
            if algo == "Decision Tree":
                st.markdown("---")
                depth = st.number_input("Select Max Depth for Tree:", min_value=1, max_value=100, value=5)
            
            run_ml = st.button("🚀 Start Training")

        with m_col2:
            if run_ml and features:
                with st.spinner(f"Training {algo}..."):
                    X, y = df[features], df[target]
                    
                    if "Classification" in task:
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                        class_names = [str(c) for c in le.classes_]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    sc = StandardScaler()
                    X_train = sc.fit_transform(X_train)
                    X_test = sc.transform(X_test)

                    if algo == "Decision Tree":
                        model = DecisionTreeClassifier(max_depth=depth) if "Classification" in task else DecisionTreeRegressor(max_depth=depth)
                    elif algo == "Linear Regression": model = LinearRegression()
                    else: model = LogisticRegression()

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    # Visualization using Dataset Labels
                    if "Classification" in task:
                        cm = confusion_matrix(y_test, preds.astype(int))
                        fig_h, ax_h = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", xticklabels=class_names, yticklabels=class_names)
                        ax_h.set_title(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
                        ax_h.set_xlabel(f"Predicted {target}") # Dynamic Label
                        ax_h.set_ylabel(f"Actual {target}")    # Dynamic Label
                        st.pyplot(fig_h)
                    else:
                        fig_r = px.scatter(x=y_test, y=preds, labels={'x': f'Actual {target}', 'y': f'Predicted {target}'}, 
                                           title=f"R² Score: {r2_score(y_test, preds):.4f}", template="plotly_dark")
                        st.plotly_chart(fig_r, use_container_width=True)
            elif run_ml:
                st.warning("Please select features to train.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | Optimized Sequential Workflow</p>", unsafe_allow_html=True)
