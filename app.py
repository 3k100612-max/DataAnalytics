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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

# --- 1. SYSTEM & RAM MONITORING ---
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
    for col in df.columns:
        if df[col].dtype == 'float64': df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64': df[col] = df[col].astype('int32')
    df = df.fillna(df.median(numeric_only=True))
    for col in df.select_dtypes(exclude=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    return df, total_missing

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="GPAI Data Pro", layout="wide", page_icon="🧪")
st.title("🧪 Advanced ML Workshop & Data Warehouse")
st.caption(f"Server Date: 2026-06-06 | RAM Management Active | @timothymarkbale2026")

if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None

with st.sidebar:
    st.header("🖥️ System Status")
    vps_mem = get_vps_ram()
    st.write(f"💾 VPS RAM: {vps_mem:.1f}MB / 8192MB")
    st.progress(min(vps_mem/8192, 1.0))
    components.html(browser_ram_js, height=50)
    st.markdown("---")
    row_limit = st.sidebar.slider("Max Rows to Load", 1000, 1000000, 100000)

uploaded_file = st.file_uploader("1. Upload CSV Dataset", type="csv")

if uploaded_file:
    df, total_missing = load_and_fix(uploaded_file, row_limit)
    st.success(f"✅ Data Ready. {total_missing} missing values handled.")
    
    mode = st.radio("Select Active Workspace:", ["None", "Exploratory Analysis (PCA & Heatmap)", "Machine Learning Workshop"], horizontal=True)

    if mode == "Machine Learning Workshop":
        st.subheader("🤖 Supervised Learning Workshop")
        m_col1, m_col2 = st.columns([1, 2])
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        with m_col1:
            st.write("### ⚙️ Model Configuration")
            target = st.selectbox("1. Target to Predict (Y):", df.columns)
            available_predictors = [col for col in num_cols if col != target]
            
            features = st.multiselect("2. Select Clues (X):", options=available_predictors)
            task = st.radio("3. Task Type:", ["Classification (Group)", "Regression (Value)"])
            algo = st.selectbox("4. Algorithm:", ["Linear/Logistic Regression", "Decision Tree (CART)", "Naive Bayes", "KNN", "SVM"])
            
            if st.button("🚀 Start Model Training"):
                try:
                    # SAFETY CHECK: Downsample for heavy algorithms to prevent 502
                    working_df = df.copy()
                    if (algo in ["KNN", "SVM"]) and len(working_df) > 5000:
                        working_df = working_df.sample(5000, random_state=42)
                        st.warning("⚠️ High-complexity algorithm detected. Using a 5,000 row sample to prevent server crash (502).")

                    X, y = working_df[features], working_df[target]
                    class_names = None
                    
                    if "Classification" in task:
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                        class_names = [str(c) for c in le.classes_]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler().fit(X_train)
                    X_tr_s, X_te_s = scaler.transform(X_train), scaler.transform(X_test)

                    if algo == "Linear/Logistic Regression":
                        model = LogisticRegression(max_iter=1000) if "Classification" in task else LinearRegression()
                    elif algo == "Decision Tree (CART)":
                        model = DecisionTreeClassifier(max_depth=5) if "Classification" in task else DecisionTreeRegressor(max_depth=5)
                    elif algo == "Naive Bayes": model = GaussianNB()
                    elif algo == "KNN": model = KNeighborsClassifier(n_neighbors=5) if "Classification" in task else KNeighborsRegressor(n_neighbors=5)
                    elif algo == "SVM": model = SVC(probability=True) if "Classification" in task else SVR()

                    model.fit(X_tr_s, y_train)
                    st.session_state.ml_results = {
                        'model': model, 'target': target, 'features': features, 'task': task,
                        'algo': algo, 'class_names': class_names, 'y_test': y_test, 'preds': model.predict(X_te_s)
                    }
                except Exception as e: st.error(f"⚠️ Error: {str(e)}")

        with m_col2:
            if st.session_state.ml_results:
                res = st.session_state.ml_results
                model = res['model']
                
                st.write(f"### 🎯 Results for {res['target']}")
                if "Classification" in res['task']:
                    acc = accuracy_score(res['y_test'], res['preds'].astype(int))
                    st.metric("Model Accuracy", f"{acc:.2%}")
                    cm = confusion_matrix(res['y_test'], res['preds'].astype(int))
                    fig, ax = plt.subplots(figsize=(5,4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", xticklabels=res['class_names'], yticklabels=res['class_names'], ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    r2 = r2_score(res['y_test'], res['preds'])
                    st.metric("R² Prediction Strength", f"{r2:.3f}")
                    fig_reg = px.scatter(x=res['y_test'], y=res['preds'], labels={'x': 'Actual', 'y': 'Predicted'}, template="plotly_dark")
                    fig_reg.add_shape(type="line", x0=min(res['y_test']), y0=min(res['y_test']), x1=max(res['y_test']), y1=max(res['y_test']), line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_reg, use_container_width=True)

                # Predictor Influence
                st.divider()
                st.write("### 📊 Predictor Influence")
                importance_data = None
                if hasattr(model, 'feature_importances_'): 
                    importance_data = model.feature_importances_
                elif hasattr(model, 'coef_'): 
                    importance_data = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)

                if importance_data is not None:
                    imp_df = pd.DataFrame({'Feature': res['features'], 'Value': importance_data}).sort_values(by='Value')
                    fig_imp = px.bar(imp_df, x='Value', y='Feature', orientation='h', color='Value', color_continuous_scale='Portland', theme="streamlit")
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info(f"💡 {res['algo']} does not use simple weights for influence, but it uses high-dimensional geometry to find patterns.")
            else:
                st.info("Select your clues and click 'Start Model Training'.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | Educational ML Workshop</p>", unsafe_allow_html=True)
