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
from sklearn.inspection import permutation_importance  # Added for KNN/SVM influence

# --- 1. SYSTEM & RAM MONITORING ---
def get_vps_ram():
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except: 
        return 0

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
st.set_page_config(page_title="Machine Learning Intuition Lab", layout="wide", page_icon="🧪")
st.title("🧪 ML Modeling & Data Warehouse")

# Initialize Session State
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None

with st.sidebar:
    st.header("🖥️ System Status")
    vps_mem = get_vps_ram()
    st.write(f"💾 VPS RAM: {vps_mem:.1f}MB / 8192MB")
    st.progress(min(vps_mem/8192, 1.0))
    components.html(browser_ram_js, height=50)
    st.markdown("---")
    row_limit = st.sidebar.slider("Max Rows to Load", 1000, 1000000, 500000)

# --- STEP 1: DATA UPLOAD ---
uploaded_file = st.file_uploader("1. Upload CSV Dataset", type="csv")

if uploaded_file:
    df, total_missing = load_and_fix(uploaded_file, row_limit)
    st.success(f"✅ Data Health Check: {total_missing} missing values fixed automatically.")
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Fixed Dataset (CSV)", data=csv_data, file_name="cleaned_data.csv")

    st.divider()

    mode = st.radio("Select Active Workspace:", 
                    ["None", "Exploratory Analysis (PCA & Heatmap)", "Machine Learning Workshop"], 
                    horizontal=True)

    # --- PATH A: EXPLORATORY ANALYSIS ---
    if mode == "Exploratory Analysis (PCA & Heatmap)":
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.info("💡 Exploratory Analysis simplifies data and finds hidden relationships before training starts.")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("### 💎 PCA (Dimensionality Reduction)")
            with st.expander("📖 The Shadow Analogy (Explainer)"):
                st.write("Imagine holding a 3D teapot in front of a flashlight. The shadow on the wall is 2D. **PCA** finds the best angle to hold the teapot so the shadow captures the most detail.")
            
            pca_feats = st.multiselect("Select Numeric Columns to Compress:", num_cols, default=num_cols[:min(3, len(num_cols))])
            target_color = st.selectbox("Color Map by:", df.columns)
            
            if st.button("Generate PCA Insights") and len(pca_feats) >= 2:
                X_pca = StandardScaler().fit_transform(df[pca_feats])
                pca = PCA(n_components=2)
                comps = pca.fit_transform(X_pca)
                
                pdf = pd.DataFrame(comps, columns=['PC1', 'PC2'])
                pdf[target_color] = df[target_color].values
                fig_pca = px.scatter(pdf, x='PC1', y='PC2', color=target_color, title=f"PCA Visualization: {target_color}")
                st.plotly_chart(fig_pca, use_container_width=True, theme="streamlit")

                # Logic Visualizer
                st.write("#### 🧠 PCA Logic Visualizer")
                l_col1, l_col2 = st.columns(2)
                with l_col1:
                    fig_var = px.bar(x=['PC1', 'PC2'], y=pca.explained_variance_ratio_, title="Info Retained", labels={'y':'% Info'})
                    st.plotly_chart(fig_var, use_container_width=True)
                with l_col2:
                    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=pca_feats)
                    fig_load = px.bar(loadings, barmode='group', title="Feature Influence")
                    st.plotly_chart(fig_load, use_container_width=True)

        with col_b:
            st.write("### 🌡️ Relationship Heatmap")
            with st.expander("📖 The Dance Analogy (Explainer)"):
                st.write("Correlation measures if variables 'dance' together. **+1.0 (Blue)** means they move in sync; **-1.0 (Red)** means they move in opposite directions.")
            
            if st.button("Generate Relationship Heatmap"):
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = df[num_cols].corr()
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu", ax=ax, center=0)
                st.pyplot(fig)
                plt.close(fig)
                
                # Logic Insight
                if len(num_cols) > 1:
                    strongest = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
                    pair = strongest.index[1] 
                    # Use .iloc[1] to get the value safely
                    st.success(f"**Insight:** Strongest relationship found between **{pair[0]}** and **{pair[1]}** ({strongest.iloc[1]:.2f}).")
                else:
                    st.warning("Not enough numeric columns to find relationships.")

    # --- PATH B: MACHINE LEARNING WORKSHOP ---
    elif mode == "Train the Model - Machine Learning":
        st.subheader("🤖 Supervised Learning Workshop")
        m_col1, m_col2 = st.columns([1, 2])
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        with m_col1:
            st.write("### ⚙️ Model Configuration")
            target = st.selectbox("1. Target to Predict (Y):", df.columns)
            available_predictors = [col for col in num_cols if col != target]
            
            if st.checkbox("Rank Predictors by Relevance?", value=True):
                if target in num_cols:
                    correlations = df[num_cols].corr()[target].abs().sort_values(ascending=False)
                    available_predictors = correlations.drop(labels=[target]).index.tolist()
                    st.caption(f"💡 Best Clue: **{available_predictors[0]}**")

            features = st.multiselect("2. Select Clues (X):", options=available_predictors)
            task = st.radio("3. Task Type:", ["Classification (Group)", "Regression (Value)"])
            algo = st.selectbox("4. Algorithm:", ["Linear/Logistic Regression", "Decision Tree (CART)", "Naive Bayes", "KNN", "SVM"])
            
            # Decision Tree Depth Control
            depth = 5
            if "Decision Tree" in algo:
                depth = st.number_input("Select Max Tree Depth:", 1, 10, 5)
            
            if st.button("🚀 Start Model Training"):
                try:
                    X, y = df[features], df[target]
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
                        model = DecisionTreeClassifier(max_depth=depth) if "Classification" in task else DecisionTreeRegressor(max_depth=depth)
                    elif algo == "Naive Bayes": model = GaussianNB()
                    elif algo == "KNN": model = KNeighborsClassifier() if "Classification" in task else KNeighborsRegressor()
                    elif algo == "SVM": model = SVC() if "Classification" in task else SVR()

                    model.fit(X_tr_s, y_train)
                    st.session_state.ml_results = {
                        'model': model, 'target': target, 'features': features, 'task': task,
                        'algo': algo, 'class_names': class_names, 'y_test': y_test, 
                        'preds': model.predict(X_te_s), 'X_test_scaled': X_te_s
                    }
                except Exception as e: st.error(f"⚠️ Error: {str(e)}")

        with m_col2:
            if st.session_state.ml_results:
                res = st.session_state.ml_results
                model = res['model']
                
                st.write(f"### 🎯 Results for {res['target']}")
                if "Classification" in res['task']:
                    cm = confusion_matrix(res['y_test'], res['preds'].astype(int))
                    fig, ax = plt.subplots(figsize=(5,4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", xticklabels=res['class_names'], yticklabels=res['class_names'], ax=ax)
                    ax.set_title(f"Accuracy: {accuracy_score(res['y_test'], res['preds']):.2%}")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    fig_reg = px.scatter(x=res['y_test'], y=res['preds'], labels={'x': 'Actual', 'y': 'Predicted'}, title="Regression Accuracy")
                    fig_reg.add_shape(type="line", x0=min(res['y_test']), y0=min(res['y_test']), x1=max(res['y_test']), y1=max(res['y_test']), line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_reg, use_container_width=True, theme="streamlit")

                # Logic Explainer
                st.divider()
                st.write(f"### 🧠 {res['algo']} Logic Explainer")
                if "Decision Tree" in res['algo']:
                    fig_tree, ax_tree = plt.subplots(figsize=(12, 6))
                    # FIXED: Removed max_depth=2 so it reflects the actual trained tree depth
                    plot_tree(model, feature_names=res['features'], class_names=res['class_names'], filled=True, rounded=True, ax=ax_tree)
                    st.pyplot(fig_tree)
                    plt.close(fig_tree)
                elif "Regression" in res['algo']:
                    with st.expander("🔍 How it works"):
                        st.write("Regression assigns a 'Weight' to every clue. Higher weights mean that clue is more important.")
                        st.latex(r"y = w_1x_1 + w_2x_2 + b")
                elif "KNN" in res['algo']:
                    with st.expander("🔍 How it works"):
                        st.write("**K-Nearest Neighbors** finds the most similar rows (neighbors) in the training data to make a prediction.")
                elif "Naive Bayes" in res['algo']:
                    with st.expander("🔍 How it works"):
                        st.write("**Naive Bayes** uses probability to guess the category based on how clues are distributed.")
                        st.latex(r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}")
                elif "SVM" in res['algo']:
                    with st.expander("🔍 How it works"):
                        st.write("**Support Vector Machines** try to find the best boundary (hyperplane) that separates different groups.")

                # Predictor Influence
                st.divider()
                st.write("### 📊 Predictor Influence")
                importance_data = None
                
                # 1. Check for Tree Importance
                if hasattr(model, 'feature_importances_'): 
                    importance_data = model.feature_importances_
                # 2. Check for Linear Coefficients
                elif hasattr(model, 'coef_'): 
                    importance_data = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                # 3. Fallback for SVM/KNN/Naive Bayes (Permutation Importance)
                else:
                    with st.spinner("Calculating Influence (Permutation)..."):
                        perm = permutation_importance(model, res['X_test_scaled'], res['y_test'], n_repeats=5, random_state=42)
                        importance_data = perm.importances_mean

                if importance_data is not None:
                    imp_df = pd.DataFrame({'Feature': res['features'], 'Value': importance_data}).sort_values(by='Value')
                    fig_imp = px.bar(imp_df, x='Value', y='Feature', orientation='h', color='Value', color_continuous_scale='Portland')
                    fig_imp.update_traces(marker_line_color='black', marker_line_width=1)
                    st.plotly_chart(fig_imp, use_container_width=True, theme="streamlit")

                    if "Classification" in res['task'] and hasattr(model, 'coef_') and len(res['class_names']) > 1:
                        st.write("#### 🔍 Category Deep-Dive")
                        selected_class = st.selectbox("Select Category:", res['class_names'], key="dd_select")
                        class_idx = res['class_names'].index(selected_class)
                        class_coefs = model.coef_[class_idx] if len(model.coef_.shape) > 1 else model.coef_
                        class_imp_df = pd.DataFrame({'Feature': res['features'], 'Influence': class_coefs}).sort_values(by='Influence')
                        fig_class = px.bar(class_imp_df, x='Influence', y='Feature', orientation='h', color='Influence', color_continuous_scale='RdBu_r', title=f"Drivers for {selected_class}")
                        fig_class.update_traces(marker_line_color='black', marker_line_width=1)
                        st.plotly_chart(fig_class, use_container_width=True, theme="streamlit")
            else:
                st.info("Train a model to see the logic visualization here.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | Educational ML Workshop</p>", unsafe_allow_html=True)
