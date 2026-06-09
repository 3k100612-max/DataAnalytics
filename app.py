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
from sklearn.inspection import permutation_importance
from sklearn.tree import export_graphviz

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
st.set_page_config(page_title="Machine Learning Intuition Lab", layout="wide", page_icon="🧪",menu_items={ 'Get Help': 'https://www.extremelycoolapp.com/help','Report a bug': "https://www.extremelycoolapp.com/bug",'About': "# This is a header. Machine Learning Intuition Lab"})
st.title("🧪 Machine Learning Intuition Lab")

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
            target_color = st.selectbox("Color Map by:", df.columns, key="pca_color")
        
            if st.button("Generate PCA Insights") and len(pca_feats) >= 2:
                # 1. Scaling and Transformation
                X_pca = StandardScaler().fit_transform(df[pca_feats])
                pca = PCA(n_components=2)
                comps = pca.fit_transform(X_pca)
            
                # 2. Extract Loadings (Feature Influence)
                loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=pca_feats)
            
                # 3. IDENTIFY TOP DRIVERS
                top_driver_pc1 = loadings['PC1'].abs().idxmax()
                top_driver_pc2 = loadings['PC2'].abs().idxmax()
            
                # 4. CREATE DYNAMIC LABELS
                var_pc1 = pca.explained_variance_ratio_[0] * 100
                var_pc2 = pca.explained_variance_ratio_[1] * 100
                label_x = f"PC1 ({var_pc1:.1f}%) — Primary Driver: {top_driver_pc1}"
                label_y = f"PC2 ({var_pc2:.1f}%) — Primary Driver: {top_driver_pc2}"
            
                # 5. Generate Main Scatter Plot
                pdf = pd.DataFrame(comps, columns=['PC1', 'PC2'])
                pdf[target_color] = df[target_color].values
            
                fig_pca = px.scatter(
                    pdf, x='PC1', y='PC2', color=target_color, 
                    title=f"PCA: {target_color} Distribution",
                    labels={'PC1': label_x, 'PC2': label_y},
                    template="plotly_white"
                )
                st.plotly_chart(fig_pca, use_container_width=True)

                # 6. Logic Visualizer (Nested inside the button check)
                st.write("#### 🧠 PCA Logic Visualizer")
                l_col1, l_col2 = st.columns(2)
            
                with l_col1:
                    fig_var = px.bar(
                        x=['PC1', 'PC2'], 
                        y=pca.explained_variance_ratio_, 
                        title="Information Retention", 
                        labels={'y':'% Info Retained', 'x': 'Component'},
                        color_discrete_sequence=['#636EFA']
                    )
                    st.plotly_chart(fig_var, use_container_width=True)
                    
                with l_col2:
                    fig_load = px.bar(
                        loadings, 
                        barmode='group', 
                        title="Feature Influence (Loadings)",
                        labels={'index': 'Features', 'value': 'Weight'}
                    )
                    st.plotly_chart(fig_load, use_container_width=True)

                # 7. Narrative Summary
                st.info(f"""
                **Insight Summary:**
                * **PC1** represents **{var_pc1:.1f}%** of the dataset's variance and is most heavily influenced by **{top_driver_pc1}**.
                * **PC2** represents **{var_pc2:.1f}%** of the variance and is primarily driven by **{top_driver_pc2}**.
                * If points are grouped by color, it means **{target_color}** is strongly related to these two drivers.
                """)

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
                
                if len(num_cols) > 1:
                    strongest = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
                    pair = strongest.index[1] 
                    st.success(f"**Insight:** Strongest relationship found between **{pair[0]}** and **{pair[1]}** ({strongest.iloc[1]:.2f}).")
                else:
                    st.warning("Not enough numeric columns to find relationships.")

        # --- PATH B: MACHINE LEARNING WORKSHOP ---
    elif mode == "Machine Learning Workshop":
        st.subheader("🤖 Supervised Learning Workshop")
        m_col1, m_col2 = st.columns([1, 2])
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        with m_col1:
            st.write("### ⚙️ Model Configuration")
            target = st.selectbox("1. Target to Predict (Y):", df.columns)
            
            # --- NEW SAFETY CHECK ---
            unique_count = df[target].nunique()
            is_numeric_target = pd.api.types.is_numeric_dtype(df[target])
            
            task = st.radio("2. Task Type:", ["Classification (Group)", "Regression (Value)"])
            
            if task == "Classification (Group)" and unique_count > 20:
                st.warning(f"⚠️ **High Complexity:** '{target}' has {unique_count} unique categories. Classification works best with < 20 groups. Consider using Regression or a different column.")

            available_predictors = [col for col in num_cols if col != target]
            
            if st.checkbox("Rank Predictors by Relevance?", value=True):
                if is_numeric_target:
                    correlations = df[num_cols].corr()[target].abs().sort_values(ascending=False)
                    available_predictors = correlations.drop(labels=[target]).index.tolist()
                    st.caption(f"💡 Best Clue: **{available_predictors[0]}**")

            features = st.multiselect("3. Select Clues (X):", options=available_predictors)
            algo = st.selectbox("4. Algorithm:", ["Linear/Logistic Regression", "Decision Tree", "Naive Bayes", "KNN", "SVM"])
            
            depth = 5
            if "Decision Tree" in algo:
                depth = st.number_input("Select Max Tree Depth:", 1, 10, 5)
            
            if st.button("🚀 Start Model Training"):
                if not features:
                    st.error("Please select at least one feature (Clue) to train.")
                else:
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
                        elif algo == "Decision Tree":
                            model = DecisionTreeClassifier(max_depth=depth) if "Classification" in task else DecisionTreeRegressor(max_depth=depth)
                        elif algo == "Naive Bayes": model = GaussianNB()
                        elif algo == "KNN": model = KNeighborsClassifier() if "Classification" in task else KNeighborsRegressor()
                        elif algo == "SVM": model = SVC() if "Classification" in task else SVR()

                        model.fit(X_tr_s, y_train)
                        st.session_state.ml_results = {
                            'model': model, 'target': target, 'features': features, 'task': task,
                            'algo': algo, 'class_names': class_names, 'y_test': y_test, 
                            'preds': model.predict(X_te_s), 'X_test_scaled': X_te_s ,'tree_depth': depth
                        }
                    except Exception as e: st.error(f"⚠️ Error: {str(e)}")

        with m_col2:
            if st.session_state.ml_results:
                res = st.session_state.ml_results
                model = res['model']
                
                st.write(f"### 🎯 Results for {res['target']}")
                if "Classification" in res['task']:
                    acc = accuracy_score(res['y_test'], res['preds'])
                    cm = confusion_matrix(res['y_test'], res['preds'].astype(int))
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # IMPROVED HEATMAP LOGIC
                    show_labels = len(res['class_names']) < 15 # Hide labels if too many
                    sns.heatmap(cm, annot=show_labels, fmt='d', cmap="Purples", 
                                xticklabels=res['class_names'] if show_labels else False, 
                                yticklabels=res['class_names'] if show_labels else False, ax=ax)
                    
                    plt.xticks(rotation=45)
                    ax.set_title(f"Accuracy: {acc:.2%}")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    if not show_labels:
                        st.info(f"💡 Labels hidden because there are {len(res['class_names'])} categories. Try a simpler target.")
                else:
                    # Regression Plot
                    fig_reg = px.scatter(x=res['y_test'], y=res['preds'], 
                                       labels={'x': 'Actual Value', 'y': 'Predicted Value'}, 
                                       title=f"Regression: Actual vs Predicted (R²: {r2_score(res['y_test'], res['preds']):.2f})")
                    fig_reg.add_shape(type="line", x0=min(res['y_test']), y0=min(res['y_test']), 
                                    x1=max(res['y_test']), y1=max(res['y_test']), line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_reg, use_container_width=True)


                st.divider()
                st.write(f"### 🧠 {res['algo']} Logic Explainer")
                
                if "Decision Tree" in res['algo']:
                    st.write(f"#### 🌳 Interactive Decision Path (Depth: {res['tree_depth']})")
                    
                    # 1. Export the tree to DOT format (Vector Data)
                    dot_data = export_graphviz(
                        model, 
                        out_file=None, 
                        feature_names=res['features'],
                        class_names=res['class_names'],
                        filled=True, 
                        rounded=True,  
                        special_characters=True,
                        precision=2,
                        leaves_parallel=False # Keeps the layout clean
                    )
                    
                    # 2. Use st.graphviz_chart for an SVG-based, expandable visual
                    # This allows the user to hover and zoom without losing quality
                    st.graphviz_chart(dot_data, use_container_width=True)
                    
                    with st.expander("📝 How to read this expanded view"):
                        st.markdown("""
                        * **Color Intensity:** Darker colors mean the model is more 'confident' in that group.
                        * **Gini/Entropy:** A lower number means the group is 'pure' (mostly one category).
                        * **Samples:** Shows how many rows from your dataset fell into that specific branch.
                        * **Fullscreen:** Hover over the top-right of the chart and click the arrows to expand.
                        """)

                elif "Regression" in res['algo']:
                    with st.expander("🔍 The 'Weight' Logic (Formula)"):
                        st.write("Regression treats data like a scale. It assigns a **Weight ($w$)** to every clue to calculate the result.")
                        st.latex(r"y = w_1x_1 + w_2x_2 + ... + b")
                        st.info("Check the **Predictor Influence** chart below to see which clues have the heaviest weights!")

                elif "KNN" in res['algo']:
                    with st.expander("🔍 The 'Neighbor' Logic (Analogy)"):
                        st.write("### 👥 'Show me who your friends are...'")
                        st.write("**K-Nearest Neighbors** doesn't learn rules. Instead, it memorizes the entire dataset.")
                        st.write("To make a prediction, it looks for the **K** (e.g., 5) most similar rows in the past data and averages their results.")
                        st.success("It's like asking your 5 closest neighbors for advice before making a choice.")

                elif "Naive Bayes" in res['algo']:
                    with st.expander("🔍 The 'Probability' Logic (Math)"):
                        st.write("### 📊 The 'Odds' Calculator")
                        st.write("**Naive Bayes** calculates the probability of each category based on the clues provided.")
                        st.latex(r"P(Category | Clues) = \frac{P(Clues | Category) \times P(Category)}{P(Clues)}")
                        st.write("It is called **'Naive'** because it assumes every clue is 100% independent of the others.")

                elif "SVM" in res['algo']:
                    with st.expander("🔍 The 'Boundary' Logic (Visual)"):
                        st.write("### 🚧 The Great Divide")
                        st.write("**Support Vector Machines** try to find the best boundary (hyperplane) that separates different groups.")
                        st.write("It looks for the 'Support Vectors'—the data points closest to the edge—and builds a wall to separate them.")
                
                st.divider()
                st.write("### 📊 Predictor Influence")
                importance_data = None
                
                if hasattr(model, 'feature_importances_'): 
                    importance_data = model.feature_importances_
                elif hasattr(model, 'coef_'): 
                    importance_data = np.abs(model.coef_).mean(axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                else:
                    with st.spinner("Calculating Influence..."):
                        perm = permutation_importance(model, res['X_test_scaled'], res['y_test'], n_repeats=5, random_state=42)
                        importance_data = perm.importances_mean

                if importance_data is not None:
                    imp_df = pd.DataFrame({'Feature': res['features'], 'Value': importance_data}).sort_values(by='Value')
                    fig_imp = px.bar(imp_df, x='Value', y='Feature', orientation='h', color='Value', color_continuous_scale='Portland')
                    st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Train a model to see the logic visualization here.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | Educational ML Laboratory</p>", unsafe_allow_html=True)
