import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import psutil

# ML Imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, mean_absolute_error

# --- VPS RESOURCE MONITORING ---
try:
    import psutil
    PS_AVAILABLE = True
except ImportError:
    PS_AVAILABLE = False

# --- MEMORY-EFFICIENT DATA LOADER ---
@st.cache_data
def load_and_clean(file, rows):
    # Load with row limit to protect 8GB RAM
    df = pd.read_csv(file, nrows=rows, low_memory=False)
    
    # Downcast types (Float64 -> Float32) to save 50% RAM
    for col in df.columns:
        if df[col].dtype == 'float64': df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64': df[col] = df[col].astype('int32')
    
    # AUTO-CLEAN: Fill missing data using Median (robust to outliers)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    return df

# --- PAGE SETUP ---
st.set_page_config(page_title="DataScience Pro", layout="wide", page_icon="📈")
st.title("📊 Advanced ML Workshop & Data Warehouse")
st.caption("Optimized for 8GB VPS | Updated: June 2026 | Created by @timothymarkbale2026")

# Sidebar
st.sidebar.header("🖥️ System Status")
if PS_AVAILABLE:
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    st.sidebar.progress(min(mem/8192, 1.0))
    st.sidebar.caption(f"RAM Usage: {mem:.1f}MB / 8192MB")

st.sidebar.markdown("---")
row_limit = st.sidebar.slider("Max Rows to Load", 1000, 1000000, 500000)
st.sidebar.write("© timothymarkbale2026")

# --- 1. DATA UPLOAD ---
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")

if uploaded_file:
    df = load_and_clean(uploaded_file, row_limit)
    st.success(f"✅ Dataset Ready: {len(df)} rows loaded. All missing values filled and memory optimized.")

    # --- 2. DATA VISUALS ---
    st.header("🔍 1. Exploratory Data Analysis")
    c1, c2 = st.columns(2)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    with c1:
        st.subheader("Feature Correlation Network")
        if len(numeric_cols) > 1:
            fig_h, ax_h = plt.subplots(figsize=(10, 7))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_h)
            ax_h.set_title("How different columns relate to each other", fontsize=14)
            st.pyplot(fig_h)

    with c2:
        st.subheader("Deep Variable Distribution")
        sel_col = st.selectbox("Select a column to analyze:", numeric_cols)
        fig_p = px.histogram(df, x=sel_col, marginal="box", title=f"Statistical Spread of {sel_col}", template="plotly_dark")
        fig_p.update_layout(xaxis_title=f"Value of {sel_col}", yaxis_title="Total Count")
        st.plotly_chart(fig_p, use_container_width=True)

    # --- 3. ML WORKSHOP ---
    st.divider()
    st.header("🤖 2. Machine Learning Workshop")
    
    set_col, res_col = st.columns([1, 2])
    
    with set_col:
        target = st.selectbox("Target to Predict (Y):", df.columns)
        features = st.multiselect("Predictor Features (X):", [c for c in numeric_cols if c != target])
        task = st.radio("Goal Type:", ["Groups (Classification)", "Numbers (Regression)"])
        
        if task == "Groups (Classification)":
            algo = st.selectbox("Intelligence Model:", ["Logistic Regression", "Decision Tree", "Naive Bayes", "KNN", "SVM"])
        else:
            algo = st.selectbox("Intelligence Model:", ["Linear Regression", "Decision Tree", "KNN", "SVM"])
        
        train_btn = st.button("🚀 Train & Visualize Results")

    with res_col:
        if train_btn and features:
            X = df[features]
            y = df[target]
            
            # Categorical encoding for targets
            if task == "Groups (Classification)":
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                class_names = [str(c) for c in le.classes_]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Essential scaling for distance-based models
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Model Selection
            if algo == "Linear Regression": model = LinearRegression()
            elif algo == "Logistic Regression": model = LogisticRegression()
            elif algo == "Decision Tree":
                model = DecisionTreeClassifier(max_depth=5) if "Groups" in task else DecisionTreeRegressor(max_depth=5)
            elif algo == "Naive Bayes": model = GaussianNB()
            elif algo == "KNN":
                model = KNeighborsClassifier() if "Groups" in task else KNeighborsRegressor()
            elif algo == "SVM":
                model = SVC() if "Groups" in task else SVR()

            with st.spinner(f"Computing {algo}..."):
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)

            # --- DYNAMIC VISUAL REPRESENTATION ---
            st.subheader(f"📈 {algo} Prediction Visuals")
            
            if "Groups" in task:
                # Labeled Confusion Matrix
                cm = confusion_matrix(y_test, preds)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm, 
                            xticklabels=class_names, yticklabels=class_names)
                ax_cm.set_title(f"Accuracy: {accuracy_score(y_test, preds):.2%} | Target: {target}")
                ax_cm.set_xlabel(f"Predicted {target}")
                ax_cm.set_ylabel(f"Actual {target}")
                st.pyplot(fig_cm)
            else:
                # Labeled Regression Accuracy Map
                fig_reg = px.scatter(x=y_test, y=preds, labels={'x': f'Actual {target}', 'y': f'Predicted {target}'}, 
                                     title=f"Precision Map (R² Score: {r2_score(y_test, preds):.4f})", template="plotly_dark")
                fig_reg.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), 
                                  line=dict(color="Red", dash="dash"), name="Perfect Prediction")
                st.plotly_chart(fig_reg, use_container_width=True)

            # --- 4. NON-TECHNICAL GUIDE & MATH ---
            st.divider()
            with st.expander("📖 Logic & Computation Breakdown", expanded=True):
                t_col1, t_col2 = st.columns(2)
                
                with t_col1:
                    st.write("### 🧮 Mathematical Model")
                    if algo == "Linear Regression":
                        st.latex(r"Y = \beta_0 + \beta_1X_1 + \dots + \epsilon")
                    elif algo == "Logistic Regression":
                        st.latex(r"P(Y=1) = \frac{1}{1 + e^{-z}}")
                    elif algo == "Decision Tree":
                        st.latex(r"Gini = 1 - \sum (P_i)^2")
                    elif algo == "Naive Bayes":
                        st.latex(r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}")
                    elif algo == "KNN":
                        st.latex(r"d(p, q) = \sqrt{\sum(p_i - q_i)^2}")
                    elif algo == "SVM":
                        st.latex(r"\text{Margin} = \frac{2}{||w||}")
                
                with t_col2:
                    st.write("### 💡 Plain English Intuition")
                    if algo == "Linear Regression":
                        st.info(f"The computer is drawing a straight line to find a trend between your features and **{target}**.")
                    elif algo == "Decision Tree":
                        st.info(f"It's like a flowchart. The computer asks Yes/No questions about your data to sort it into groups.")
                    elif algo == "KNN":
                        st.info(f"It looks for the 'closest neighbors'. If most similar rows belong to Group A, it predicts Group A.")
                    else:
                        st.info(f"This model uses patterns in your data to calculate the most likely outcome for **{target}**.")

                if algo == "Decision Tree":
                    st.write("#### Visual Logic Tree")
                    fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
                    plot_tree(model, feature_names=features, class_names=class_names if "Groups" in task else None, 
                              filled=True, max_depth=2, rounded=True, precision=2)
                    st.pyplot(fig_tree)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | Engine: 8GB VPS Scaled</p>", unsafe_allow_html=True)
