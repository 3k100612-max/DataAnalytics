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
st.caption(f"Server Date: 2026-06-06 | Sequential RAM Management | @timothymarkbale2026")

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

    # --- STEP 4: WORKSPACE SELECTION ---
    mode = st.radio("Select Active Workspace:", 
                    ["None", "Exploratory Analysis (PCA & Heatmap)", "Machine Learning Workshop"], 
                    horizontal=True)

    # --- PATH A: EXPLORATORY ANALYSIS ---
    if mode == "Exploratory Analysis (PCA & Heatmap)":
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("### PCA Projection")
            pca_feats = st.multiselect("Select Numeric Features for PCA:", num_cols, default=num_cols[:min(3, len(num_cols))])
            target_color = st.selectbox("Color Map by (Label):", df.columns)
            
            if st.button("Generate PCA Map") and len(pca_feats) >= 2:
                X_pca = StandardScaler().fit_transform(df[pca_feats])
                pca = PCA(n_components=2)
                comps = pca.fit_transform(X_pca)
                pdf = pd.DataFrame(comps, columns=['PC1', 'PC2'])
                pdf[target_color] = df[target_color].values
                fig_pca = px.scatter(pdf, x='PC1', y='PC2', color=target_color, template="plotly_dark", title=f"PCA: {target_color}")
                st.plotly_chart(fig_pca, use_container_width=True)

        with col_b:
            st.write("### Numerical Relationship Map")
            if st.button("Generate Relationship Heatmap"):
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                ax.set_title("Feature Correlation Heatmap")
                st.pyplot(fig)

    # --- PATH B: MACHINE LEARNING WORKSHOP ---
        # --- PATH B: MACHINE LEARNING WORKSHOP ---
    elif mode == "Machine Learning Workshop":
        st.subheader("🤖 Supervised Learning Workshop")
        m_col1, m_col2 = st.columns([1, 2])
        
        # Define numeric columns for math operations
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        with m_col1:
            st.write("### ⚙️ Model Configuration")
            
            # 1. Target Selection
            target = st.selectbox("1. What do you want to predict? (Target Y)", df.columns)
            
            # 2. Base Predictor List (All numeric columns except the target)
            available_predictors = [col for col in num_cols if col != target]
            
            # 3. SMART RANKING (Fixed Indentation)
            if st.checkbox("Rank Predictors by Relevance?", value=True):
                if target in num_cols:
                    try:
                        # Calculate correlation and sort
                        correlations = df[num_cols].corr()[target].abs().sort_values(ascending=False)
                        correlations = correlations.drop(labels=[target], errors='ignore')
                        available_predictors = correlations.index.tolist()
                        if available_predictors:
                            st.caption(f"💡 Best numeric clue: **{available_predictors[0]}**")
                    except Exception:
                        available_predictors = [col for col in num_cols if col != target]
                else:
                    st.info("💡 Ranking is optimized for numeric targets.")
                    available_predictors = [col for col in num_cols if col != target]

            # 4. Predictor Selection
            features = st.multiselect(f"2. Select Clues for {target} (Predictors X):", options=available_predictors)
            
            task = st.radio("3. Task Type:", ["Classification (Group)", "Regression (Value)"])
            algo = st.selectbox("4. Algorithm:", ["Linear/Logistic Regression", "Decision Tree (CART)", "Naive Bayes", "KNN", "SVM"])
            
            depth = 5
            if "Decision Tree" in algo:
                depth = st.number_input("Select Max Tree Depth:", 1, 100, 5)
            
            run_train = st.button("🚀 Start Model Training")

        with m_col2:
            if run_train and features:
                try:
                    # --- 1. DATA PREPARATION ---
                    X, y = df[features], df[target]
                    if "Classification" in task:
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                        class_names = [str(c) for c in le.classes_]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_train)
                    X_te_s = scaler.transform(X_test)

                    # --- 2. MODEL SELECTION ---
                    if algo == "Linear/Logistic Regression":
                        model = LogisticRegression(max_iter=1000) if "Classification" in task else LinearRegression()
                    elif algo == "Decision Tree (CART)":
                        model = DecisionTreeClassifier(max_depth=depth) if "Classification" in task else DecisionTreeRegressor(max_depth=depth)
                    elif algo == "Naive Bayes": model = GaussianNB()
                    elif algo == "KNN": model = KNeighborsClassifier() if "Classification" in task else KNeighborsRegressor()
                    elif algo == "SVM": model = SVC() if "Classification" in task else SVR()

                    model.fit(X_tr_s, y_train)
                    preds = model.predict(X_te_s)

                    # --- 3. PERFORMANCE VISUALS ---
                    st.write(f"### 🎯 Results for {target}")
                    if "Classification" in task:
                        cm = confusion_matrix(y_test, preds.astype(int))
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", xticklabels=class_names, yticklabels=class_names)
                        ax.set_title(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
                        ax.set_xlabel(f"Predicted {target}"); ax.set_ylabel(f"Actual {target}")
                        st.pyplot(fig)
                    else:
                        fig_reg = px.scatter(x=y_test, y=preds, labels={'x': f'Actual {target}', 'y': f'Predicted {target}'}, 
                                             title=f"Regression Accuracy", template="plotly_dark")
                        st.plotly_chart(fig_reg, use_container_width=True)

                    # --- 4. PREDICTOR INFLUENCE ---
                   st.write(f"### 📊 Predictor Influence for {target}")
                    st.caption(f"This analysis shows which clues are most responsible for determining the {target}.")

                    importance_data = None
                    if hasattr(model, 'feature_importances_'):
                        importance_data = model.feature_importances_
                        label_type = "Importance Score"
                    elif hasattr(model, 'coef_'):
                        importance_data = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                        label_type = "Impact Weight (Absolute)"

                    if importance_data is not None:
                        # Global Importance Chart
                        imp_df = pd.DataFrame({'Feature': features, 'Value': importance_data}).sort_values(by='Value')
                        fig_imp = px.bar(
                            imp_df, x='Value', y='Feature', orientation='h', 
                            title=f"Global Drivers: Which features define {target}?",
                            labels={'Value': label_type, 'Feature': 'Predictor Name'},
                            template="plotly_dark", color='Value', color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)

                        # --- CATEGORY-SPECIFIC INSIGHTS ---
                        if "Classification" in task and hasattr(model, 'coef_') and len(class_names) > 1:
                            st.divider()
                            st.write(f"#### 🔍 Deep Dive: What pushes someone into a specific {target}?")
                            
                            selected_class = st.selectbox(f"Select a category to see its specific drivers:", class_names)
                            class_idx = class_names.index(selected_class)
                            class_coefs = model.coef_[class_idx] if len(model.coef_.shape) > 1 else model.coef_
                            
                            class_imp_df = pd.DataFrame({'Feature': features, 'Influence': class_coefs}).sort_values(by='Influence')
                            
                            fig_class = px.bar(
                                class_imp_df, x='Influence', y='Feature', orientation='h',
                                title=f"Influence Map for category: '{selected_class}'",
                                labels={'Influence': 'Directional Influence (Negative = Away, Positive = Towards)'},
                                template="plotly_dark", color='Influence', color_continuous_scale='RdBu'
                            )
                            st.plotly_chart(fig_class, use_container_width=True)
                            st.info(f"💡 **How to read this:** Blue bars push the prediction **towards** {selected_class}. Red bars push it **away**.")

                except Exception as e:
                    st.error(f"⚠️ Training Error: {str(e)}")


# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | Refined Production Logic</p>", unsafe_allow_html=True)
