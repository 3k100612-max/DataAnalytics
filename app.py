import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

# --- MEMORY MONITORING ---
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# --- OPTIMIZATION: CACHED DATA LOADER ---
@st.cache_data
def load_and_optimize(file, rows):
    df = pd.read_csv(file, nrows=rows, low_memory=False)
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

# --- PAGE CONFIG ---
st.set_page_config(page_title="DataScience Pro 8GB", layout="wide", page_icon="📊")
st.title("📊 Advanced ML Workshop (8GB VPS)")

# --- SIDEBAR: RESOURCE MONITOR & COPYRIGHT ---
st.sidebar.header("🖥️ VPS Status")
if PSUTIL_AVAILABLE:
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024 * 1024)
    st.sidebar.progress(min(mem_usage / 8192, 1.0))
    st.sidebar.caption(f"RAM Usage: {mem_usage:.1f} MB / 8192 MB")

st.sidebar.markdown("---")
row_limit = st.sidebar.slider("Max Rows to Load", 1000, 1000000, 500000)

# Sidebar Copyright
st.sidebar.markdown("---")
st.sidebar.write("© timothymarkbale2026")

# --- 1. DATA LOADING & CLEANING ---
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    try:
        raw_df = load_and_optimize(uploaded_file, row_limit)
        
        # Auto-Imputation (Filling Missing Data)
        df = raw_df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "N/A")

        st.success("✅ Data loaded and missing values filled automatically.")

        # --- 2. MACHINE LEARNING WORKSHOP ---
        st.header("🤖 Machine Learning Workshop")
        
        all_cols = df.columns.tolist()
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        col_set, col_res = st.columns([1, 2])
        
        with col_set:
            target = st.selectbox("Select Target (Y):", all_cols)
            features = st.multiselect("Select Features (X):", [c for c in numeric_features if c != target])
            task = st.radio("Goal:", ["Classification (Groups)", "Regression (Numbers)"])
            
            # Algorithm Selection
            if task == "Classification (Groups)":
                algo = st.selectbox("Algorithm:", ["Logistic Regression", "Decision Tree", "Naive Bayes", "KNN", "SVM"])
            else:
                algo = st.selectbox("Algorithm:", ["Linear Regression", "Decision Tree", "KNN", "SVM"])
            
            train_btn = st.button("🚀 Train & Visualize")

        with col_res:
            if train_btn and features:
                X = df[features]
                y = df[target]
                
                if task == "Classification (Groups)":
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scaling (Mandatory for distance-based algorithms)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # --- MODEL INITIALIZATION ---
                if algo == "Linear Regression": model = LinearRegression()
                elif algo == "Logistic Regression": model = LogisticRegression()
                elif algo == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=5) if task == "Classification (Groups)" else DecisionTreeRegressor(max_depth=5)
                elif algo == "Naive Bayes": model = GaussianNB()
                elif algo == "KNN":
                    model = KNeighborsClassifier() if task == "Classification (Groups)" else KNeighborsRegressor()
                elif algo == "SVM":
                    model = SVC() if task == "Classification (Groups)" else SVR()

                with st.spinner(f"Computing {algo}..."):
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)
                
                # Metrics
                if task == "Classification (Groups)":
                    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2%}")
                else:
                    st.metric("R² Score", f"{r2_score(y_test, preds):.4f}")

                # --- GRAPHICAL REPRESENTATION ---
                st.subheader("📈 Graphical Representation")
                
                if task == "Regression (Numbers)" and len(features) == 1:
                    plot_df = pd.DataFrame({features[0]: X_test[features[0]], "Actual": y_test, "Predicted": preds})
                    fig = px.scatter(plot_df, x=features[0], y="Actual", title="Actual vs Predicted", template="plotly_dark")
                    fig.add_traces(px.line(plot_df.sort_values(features[0]), x=features[0], y="Predicted").data)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif task == "Classification (Groups)":
                    cm = confusion_matrix(y_test, preds)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
                
                else:
                    fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title="Model Precision Map", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                # --- EDUCATIONAL BREAKDOWN ---
                st.divider()
                st.subheader("🧮 How was this computed?")
                
                with st.expander("See Mathematical Explanation"):
                    if algo == "Linear Regression":
                        st.latex(r"Y = \beta_0 + \beta_1X_1 + \dots + \epsilon")
                        st.write("**Non-Technical:** The computer draws a 'Line of Best Fit' through your data.")
                    
                    elif algo == "Decision Tree":
                        st.latex(r"Gini = 1 - \sum (P_i)^2")
                        st.write("**Non-Technical:** Works like a flowchart using 'Yes/No' logic.")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        plot_tree(model, feature_names=features, filled=True, max_depth=2)
                        st.pyplot(fig)

                    elif algo == "Naive Bayes":
                        st.latex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}")
                        st.write("**Non-Technical:** Predicts based on the probability of past evidence.")

                    elif algo == "KNN":
                        st.latex(r"d = \sqrt{\sum(p_i - q_i)^2}")
                        st.write("**Non-Technical:** Looks at the 5 closest 'neighbors' to make a prediction.")

                    elif algo == "SVM":
                        st.latex(r"\text{Minimize: } \frac{1}{2} ||w||^2")
                        st.write("**Non-Technical:** Finds the widest possible boundary between groups.")

    except Exception as e:
        st.error(f"Error: {e}")

# --- FOOTER COPYRIGHT ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026</p>", unsafe_allow_html=True)
