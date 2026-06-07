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

# --- 2. DATA ENGINE (Step 2: Cleaning) ---
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

# --- 3. UI SETUP ---
st.set_page_config(page_title="GPAI Data Pro", layout="wide", page_icon="🧪")
st.title("🧪 Advanced ML Workshop & Data Warehouse")
st.caption("Refined Dataset Labeling | Optimized for 8GB VPS | @timothymarkbale2026")

with st.sidebar:
    st.header("🖥️ System Status")
    vps_mem = get_vps_ram()
    st.write(f"💾 VPS RAM: {vps_mem:.1f}MB / 8192MB")
    st.progress(min(vps_mem/8192, 1.0))
    components.html(browser_ram_js, height=50)
    st.markdown("---")
    row_limit = st.sidebar.slider("Max Rows to Load", 1000, 1000000, 500000)

# --- STEP 1: UPLOAD ---
uploaded_file = st.file_uploader("1. Upload CSV Dataset", type="csv")

if uploaded_file:
    # --- STEP 2: FIX ---
    df, total_missing = load_and_fix(uploaded_file, row_limit)
    st.success(f"✅ Data Health Check: {total_missing} missing values fixed.")
    
    # --- STEP 3: DOWNLOAD ---
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Fixed Dataset", data=csv_data, file_name="fixed_data.csv", mime="text/csv")

    st.divider()

    # --- STEP 4 & 5: MODE SELECTION ---
    st.header("🧠 2. Choose Analysis Mode")
    mode = st.radio("Select Action:", ["None", "Exploratory Analysis (PCA & Heatmap)", "Machine Learning Training"], horizontal=True)

    if mode == "Exploratory Analysis (PCA & Heatmap)":
        st.subheader("💎 Dimensionality & Correlation Analysis")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("### PCA Projection Map")
            pca_feats = st.multiselect("Select Numeric Features:", num_cols, default=num_cols[:min(3, len(num_cols))])
            target_color = st.selectbox("Color Map by:", df.columns)
            if st.button("Generate PCA") and len(pca_feats) >= 2:
                X_pca = StandardScaler().fit_transform(df[pca_feats])
                pca = PCA(n_components=2)
                comps = pca.fit_transform(X_pca)
                pdf = pd.DataFrame(comps, columns=['Principal Component 1', 'Principal Component 2'])
                pdf[target_color] = df[target_color].values
                # Dynamic Title and Labels
                fig_pca = px.scatter(pdf, x='Principal Component 1', y='Principal Component 2', color=target_color, 
                                     template="plotly_dark", title=f"PCA Analysis: {target_color} Distribution")
                st.plotly_chart(fig_pca, use_container_width=True)

        with c2:
            st.write("### Correlation Heatmap")
            if st.button("Generate Heatmap"):
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[num_cols].corr(), annot=False, cmap="coolwarm", ax=ax, cbar_kws={'label': 'Correlation Strength'})
                ax.set_title("Feature Relationship Map")
                st.pyplot(fig)

    # --- PATH B: MACHINE LEARNING (STEP 6 & 7) ---
    elif mode == "Machine Learning Training":
        st.subheader("🤖 Supervised Learning Workshop")
        st.info("Exploratory visuals stopped to save RAM for model training.")
        
        m_col1, m_col2 = st.columns([1, 2])
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        with m_col1:
            target = st.selectbox("Predict Target (Y):", df.columns)
            # CRITICAL: Numeric features only for stability
            features = st.multiselect("Predictor Features (Numeric X only):", [c for c in num_cols if c != target])
            task = st.radio("Task Type:", ["Classification", "Regression"])
            algo = st.selectbox("Algorithm:", ["Linear/Logistic Regression", "Decision Tree (CART)", "Naive Bayes", "KNN", "SVM"])
            
            depth = 5
            if "Decision Tree" in algo:
                depth = st.number_input("Select Max Tree Depth:", 1, 100, 5)
            
            run_train = st.button("🚀 Start Training")

        with m_col2:
            if run_train and features:
                try:
                    # --- MATH & SCALING ---
                    with st.expander("📖 Computation Guide", expanded=True):
                        st.write("**Standard Scaling Active:** Normalizing data for math stability.")
                        st.latex(r"z = \frac{x - \mu}{\sigma}")

                    # --- PREP & TRAIN ---
                    X, y = df[features], df[target]
                    if task == "Classification":
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                        class_names = [str(c) for c in le.classes_]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Apply Scaler to prevent "Connect Error"
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_train)
                    X_te_s = scaler.transform(X_test)

                    if algo == "Linear/Logistic Regression":
                        model = LogisticRegression(max_iter=1000) if task == "Classification" else LinearRegression()
                    elif algo == "Decision Tree (CART)":
                        model = DecisionTreeClassifier(max_depth=depth) if task == "Classification" else DecisionTreeRegressor(max_depth=depth)
                    elif algo == "Naive Bayes": model = GaussianNB()
                    elif algo == "KNN": model = KNeighborsClassifier() if task == "Classification" else KNeighborsRegressor()
                    elif algo == "SVM": model = SVC() if task == "Classification" else SVR()

                    model.fit(X_tr_s, y_train)
                    preds = model.predict(X_te_s)

                    # --- DYNAMIC VISUALS WITH DATASET LABELS ---
                    if task == "Classification":
                        st.write(f"#### Result: Predicting {target}")
                        cm = confusion_matrix(y_test, preds.astype(int))
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", xticklabels=class_names, yticklabels=class_names)
                        ax.set_title(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
                        ax.set_xlabel(f"Predicted {target}"); ax.set_ylabel(f"Actual {target}")
                        st.pyplot(fig)
                    else:
                        st.write(f"#### Precision Map: Predicting {target}")
                        fig_reg = px.scatter(x=y_test, y=preds, labels={'x': f'Actual {target}', 'y': f'Predicted {target}'}, 
                                             title=f"Regression: Actual vs Predicted {target}", template="plotly_dark")
                        fig_reg.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="Red", dash="dash"))
                        st.plotly_chart(fig_reg, use_container_width=True)

                except Exception as e:
                    st.error(f"⚠️ Calculation Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | Enhanced Dataset Labeling Active</p>", unsafe_allow_html=True)
