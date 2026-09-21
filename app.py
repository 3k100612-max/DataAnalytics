import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import psutil
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

# ML Imports
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz,
)

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20            # 80/20 split
MAX_CATEGORIES = 30         # categorical columns above this are treated as ID/free-text
MIN_CLASS_ROWS = 10         # classes rarer than this cannot be split/validated reliably
TUNE_ROWS = 6000            # rows used for hyperparameter search (keeps big CSVs fast)
CAP_ROWS = {"SVM": 25000, "KNN": 100000}   # final-fit caps for algorithms that scale badly
TARGET_SCORE = 0.90         # quality gate shown to the user
AUTO = "🏆 Auto-Compare All (Recommended)"


# ----------------------------------------------------------------------------
# 1. SYSTEM & RAM MONITORING
# ----------------------------------------------------------------------------
def get_vps_ram():
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
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


# ----------------------------------------------------------------------------
# 2. DATA ENGINE
# ----------------------------------------------------------------------------
@st.cache_data
def load_raw(file, rows):
    """Load the CSV and shrink dtypes. Missing values are KEPT here (they are
    imputed inside the ML pipeline, after the split, to avoid data leakage)."""
    df = pd.read_csv(file, nrows=rows, low_memory=False)
    total_missing = int(df.isnull().sum().sum())
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            # int32 only when it is safe
            if df[col].abs().max() < 2**31 - 1:
                df[col] = df[col].astype("int32")
    return df, total_missing


@st.cache_data
def fix_missing(df):
    """Fully imputed copy - used for exploration and the CSV download only."""
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    for col in df.select_dtypes(exclude=[np.number]).columns:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
    return df


# ----------------------------------------------------------------------------
# 3. ML HELPERS
# ----------------------------------------------------------------------------
def is_clf(task):
    return task.startswith("Classification")


def screen_features(df, features):
    """Drop columns that can only hurt: constants, IDs, free-text."""
    keep, dropped = [], []
    n = len(df)
    for c in features:
        s = df[c]
        nun = s.nunique(dropna=True)
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if nun <= 1:
            dropped.append((c, "constant column"))
        elif not numeric and nun > MAX_CATEGORIES:
            dropped.append((c, f"{nun} categories (ID / free-text like)"))
        elif pd.api.types.is_integer_dtype(s) and nun == n and n > 50:
            dropped.append((c, "different value on every row (ID-like)"))
        else:
            keep.append(c)
    return keep, dropped


def prepare_X(df, features):
    X = df[features].copy()
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            X[c] = s.astype(object).where(s.notna(), "Missing").astype(str)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if num_cols:
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
    return X


def make_preprocessor(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                              dtype=np.float32), cat),
    ], remainder="drop")


def build_candidates(clf, balanced):
    """name -> (estimator, hyper-parameter grid). Grid keys target the pipeline step 'model'."""
    cw = "balanced" if balanced else None
    if clf:
        return {
            "Linear/Logistic Regression": (
                LogisticRegression(max_iter=2000, class_weight=cw),
                {"model__C": [0.1, 1, 10]}),
            "Decision Tree": (
                DecisionTreeClassifier(random_state=SEED, class_weight=cw),
                {"model__max_depth": [3, 5, 7, 10], "model__min_samples_leaf": [1, 5, 20]}),
            "Naive Bayes": (
                GaussianNB(),
                {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}),
            "KNN": (
                KNeighborsClassifier(),
                {"model__n_neighbors": [3, 5, 11, 21], "model__weights": ["uniform", "distance"]}),
            "SVM": (
                SVC(class_weight=cw, random_state=SEED),
                {"model__C": [0.5, 1, 5, 20]}),
            "Random Forest": (
                RandomForestClassifier(n_estimators=150, class_weight=cw, random_state=SEED, n_jobs=1),
                {"model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 3]}),
            "Gradient Boosting": (
                HistGradientBoostingClassifier(class_weight=cw, random_state=SEED),
                {"model__learning_rate": [0.05, 0.1, 0.2], "model__max_depth": [None, 5]}),
        }
    return {
        "Linear/Logistic Regression": (LinearRegression(), {}),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=SEED),
            {"model__max_depth": [3, 5, 7, 10], "model__min_samples_leaf": [1, 5, 20]}),
        "KNN": (
            KNeighborsRegressor(),
            {"model__n_neighbors": [3, 5, 11, 21], "model__weights": ["uniform", "distance"]}),
        "SVM": (
            TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()),
            {"model__regressor__C": [0.5, 1, 5, 20], "model__regressor__epsilon": [0.05, 0.1, 0.2]}),
        "Random Forest": (
            RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=1),
            {"model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (
            HistGradientBoostingRegressor(random_state=SEED),
            {"model__learning_rate": [0.05, 0.1, 0.2], "model__max_depth": [None, 5]}),
    }


def subsample(X, y, n, clf):
    if len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, random_state=SEED,
                                    stratify=y if clf else None)
    return Xs, ys


def rank_predictors(df, target, candidates, clf):
    """Mutual-information ranking (works for numeric AND categorical clues)."""
    sample = df[[target] + candidates].dropna(subset=[target])
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=SEED)
    Xs = prepare_X(sample, candidates)
    disc = []
    for c in Xs.columns:
        if pd.api.types.is_numeric_dtype(Xs[c]):
            Xs[c] = Xs[c].fillna(Xs[c].median())
            disc.append(False)
        else:
            Xs[c] = pd.factorize(Xs[c])[0]
            disc.append(True)
    if clf:
        ys = LabelEncoder().fit_transform(sample[target].astype(str))
        mi = mutual_info_classif(Xs, ys, discrete_features=disc, random_state=SEED)
    else:
        ys = pd.to_numeric(sample[target], errors="coerce").fillna(0)
        mi = mutual_info_regression(Xs, ys, discrete_features=disc, random_state=SEED)
    return pd.Series(mi, index=candidates).sort_values(ascending=False)


def train_and_compare(X, y, clf, algos, tune, manual_depth, on_progress):
    """Stratified 80/20 split -> leak-free pipelines -> CV tuning on the TRAIN set
    only -> honest evaluation on the untouched 20% TEST set."""
    # ---- 1. balanced 80/20 split -------------------------------------------------
    if clf:
        strat = y
        split_note = "Stratified by class: every class keeps the same share in train and test."
    else:
        strat = pd.qcut(pd.Series(y).rank(method="first"), q=10, labels=False).values
        split_note = "Stratified by target deciles: train and test cover the same value range."
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=strat)

    # ---- 2. imbalance handling & scoring rule ------------------------------------
    balanced, ratio = False, 1.0
    if clf:
        counts = np.bincount(y_tr)
        counts = counts[counts > 0]
        ratio = counts.max() / counts.min()
        balanced = ratio > 1.5
        scoring = "balanced_accuracy" if balanced else "accuracy"
    else:
        scoring = "r2"

    # ---- 3. tuning subset + CV ----------------------------------------------------
    X_t, y_t = subsample(X_tr, y_tr, TUNE_ROWS, clf)
    if clf:
        n_splits = int(max(2, min(5, np.bincount(y_t)[np.bincount(y_t) > 0].min())))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    pre = make_preprocessor(X_tr)
    cands = build_candidates(clf, balanced)
    algos = [a for a in algos if a in cands]

    idx_train = np.random.RandomState(SEED).choice(len(X_tr), size=min(len(X_tr), 20000), replace=False)
    X_tr_probe, y_tr_probe = X_tr.iloc[idx_train], y_tr[idx_train]

    rows, best, tree_pipe = [], None, None
    for i, name in enumerate(algos):
        on_progress(i / len(algos), f"Training {name} ({i + 1}/{len(algos)})...")
        t0 = time.time()
        est, grid = cands[name]
        pipe = Pipeline([("prep", clone(pre)), ("model", est)])
        best_params = {}

        if tune and grid:
            search = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=2,
                                  refit=False, error_score=np.nan)
            search.fit(X_t, y_t)
            best_params = search.best_params_
            cv_mean = float(search.best_score_)
            cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
            pipe.set_params(**best_params)
        else:
            if name == "Decision Tree":
                pipe.set_params(model__max_depth=manual_depth)
            sc = cross_val_score(pipe, X_t, y_t, cv=cv, scoring=scoring, n_jobs=2)
            cv_mean, cv_std = float(sc.mean()), float(sc.std())

        cap = CAP_ROWS.get(name)
        Xf, yf = subsample(X_tr, y_tr, cap, clf) if cap else (X_tr, y_tr)
        pipe.fit(Xf, yf)

        preds = pipe.predict(X_te)
        tr_preds = pipe.predict(X_tr_probe)
        if clf:
            test_score = accuracy_score(y_te, preds)
            train_score = accuracy_score(y_tr_probe, tr_preds)
            extra = balanced_accuracy_score(y_te, preds)
        else:
            test_score = r2_score(y_te, preds)
            train_score = r2_score(y_tr_probe, tr_preds)
            extra = np.sqrt(mean_squared_error(y_te, preds))

        rows.append({
            "Algorithm": name,
            f"CV score ({scoring})": cv_mean,
            "CV ± std": cv_std,
            "Test score": test_score,
            "Train score": train_score,
            "Overfit gap": train_score - test_score,
            "Balanced acc." if clf else "RMSE": extra,
            "Time (s)": time.time() - t0,
            "Best settings": ", ".join(f"{k.split('__')[-1]}={v}" for k, v in best_params.items()) or "defaults",
        })

        # winner is chosen on CROSS-VALIDATION (never on the test set)
        key = (cv_mean if not np.isnan(cv_mean) else -np.inf, -cv_std)
        if best is None or key > best["key"]:
            best = {"key": key, "name": name, "pipe": pipe, "preds": preds,
                    "cv_mean": cv_mean, "cv_std": cv_std, "params": best_params}
        if name == "Decision Tree":
            tree_pipe = pipe

    on_progress(1.0, "Done!")
    board = pd.DataFrame(rows).sort_values(f"CV score ({scoring})", ascending=False).reset_index(drop=True)

    # ---- 4. baseline & importance -------------------------------------------------
    if clf:
        baseline = accuracy_score(y_te, DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te))
    else:
        baseline = 0.0   # predicting the mean always gives R² = 0

    importance = None
    try:
        n_imp = min(len(X_te), 2000)
        sel = np.random.RandomState(SEED).choice(len(X_te), size=n_imp, replace=False)
        perm = permutation_importance(
            best["pipe"], X_te.iloc[sel], y_te[sel], n_repeats=5, random_state=SEED,
            scoring="accuracy" if clf else "r2", n_jobs=1)
        importance = pd.DataFrame({"Feature": X_te.columns, "Value": perm.importances_mean})
    except Exception:
        pass

    return {
        "pipe": best["pipe"], "algo": best["name"], "preds": best["preds"], "y_test": y_te,
        "X_test": X_te, "leaderboard": board, "baseline": baseline, "scoring": scoring,
        "imbalance_ratio": ratio, "balanced": balanced, "split_note": split_note,
        "n_train": len(X_tr), "n_test": len(X_te), "tune_rows": len(X_t),
        "cv_mean": best["cv_mean"], "cv_std": best["cv_std"], "params": best["params"],
        "tree_pipe": tree_pipe, "importance": importance,
        "y_train": y_tr, "y_test_full": y_te,
    }


def feature_names_of(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    return [str(n).split("__", 1)[-1] for n in names]


def render_tree(pipe, class_names):
    model = pipe.named_steps["model"]
    dot_data = export_graphviz(model, out_file=None, feature_names=feature_names_of(pipe),
                               class_names=class_names, filled=True, rounded=True, precision=2)
    st.write("🎮 **Interactive Logic Explorer**")
    st.caption("🖱️ **Zoom:** Mouse wheel | **Pan:** Click & Drag | **Reset:** Double-click")
    dot_json = json.dumps(dot_data)
    chart_html = f"""
    <div id="graph-container" style="width: 100%; height: 600px; border: 1px solid #d1d5db; border-radius: 8px; background: white; cursor: move; overflow: hidden;">
        <div id="placeholder" style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #666;">
            Rendering Interactive Tree...
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
    <script type="module">
        import {{ Graphviz }} from "https://cdn.jsdelivr.net/npm/@hpcc-js/wasm/dist/index.js";
        async function render() {{
            try {{
                const graphviz = await Graphviz.load();
                const dot = {dot_json};
                const svgString = graphviz.dot(dot);
                const container = document.getElementById("placeholder");
                container.innerHTML = svgString;
                const svgElement = container.querySelector("svg");
                svgElement.style.width = "100%";
                svgElement.style.height = "100%";
                window.panZoom = svgPanZoom(svgElement, {{
                    zoomEnabled: true, controlIconsEnabled: true, fit: true,
                    center: true, minZoom: 0.1, maxZoom: 10
                }});
            }} catch (e) {{
                document.getElementById("placeholder").innerHTML = "❌ Error: " + e.message;
            }}
        }}
        render();
    </script>
    """
    components.html(chart_html, height=620)


# ----------------------------------------------------------------------------
# 4. UI CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Machine Learning Intuition Lab", layout="wide", page_icon="🧪",
    menu_items={'About': " Machine Learning Intuition Lab A Project in Fullfillment with the Requirement of MSIT643 Submitted by Timothy Mark A. Bal-e"})
st.title("🧪 Machine Learning Intuition Lab")
hide_branding_style = """
    <style>
    footer {display: none !important;}
    div[data-testid="stFooter"] {display: none !important;}
    ul[data-testid="main-menu-list"] > div:last-child {display: none !important;}
    ul[data-testid="main-menu-list"] > li:nth-child(1),
    ul[data-testid="main-menu-list"] > li:nth-child(2),
    ul[data-testid="main-menu-list"] > li:nth-child(3) {display: none !important;}
    </style>
    """
st.markdown(hide_branding_style, unsafe_allow_html=True)

if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None

with st.sidebar:
    st.header("🖥️ System Status")
    vps_mem = get_vps_ram()
    st.write(f"💾 VPS RAM: {vps_mem:.1f}MB / 8192MB")
    st.progress(min(vps_mem / 8192, 1.0))
    components.html(browser_ram_js, height=50)
    st.markdown("---")
    row_limit = st.slider("Max Rows to Load", 1000, 1000000, 500000)

# --- STEP 1: DATA UPLOAD ---
uploaded_file = st.file_uploader("1. Upload CSV Dataset", type="csv")

if uploaded_file:
    df_raw, total_missing = load_raw(uploaded_file, row_limit)
    df = fix_missing(df_raw)      # imputed copy for exploration + download
    st.success(f"✅ Data Health Check: {total_missing} missing values fixed automatically "
               f"(inside the model pipeline they are imputed AFTER the split, so no test data leaks into training).")

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
                X_pca = StandardScaler().fit_transform(df[pca_feats])
                pca = PCA(n_components=2)
                comps = pca.fit_transform(X_pca)

                loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=pca_feats)
                top_driver_pc1 = loadings['PC1'].abs().idxmax()
                top_driver_pc2 = loadings['PC2'].abs().idxmax()

                var_pc1 = pca.explained_variance_ratio_[0] * 100
                var_pc2 = pca.explained_variance_ratio_[1] * 100
                label_x = f"PC1 ({var_pc1:.1f}%) — Primary Driver: {top_driver_pc1}"
                label_y = f"PC2 ({var_pc2:.1f}%) — Primary Driver: {top_driver_pc2}"

                pdf = pd.DataFrame(comps, columns=['PC1', 'PC2'])
                pdf[target_color] = df[target_color].values

                fig_pca = px.scatter(
                    pdf, x='PC1', y='PC2', color=target_color,
                    title=f"PCA: {target_color} Distribution",
                    labels={'PC1': label_x, 'PC2': label_y}, template="plotly_white")
                st.plotly_chart(fig_pca, use_container_width=True)

                st.write("#### 🧠 PCA Logic Visualizer")
                l_col1, l_col2 = st.columns(2)
                with l_col1:
                    fig_var = px.bar(x=['PC1', 'PC2'], y=pca.explained_variance_ratio_,
                                     title="Information Retention",
                                     labels={'y': '% Info Retained', 'x': 'Component'},
                                     color_discrete_sequence=['#636EFA'])
                    st.plotly_chart(fig_var, use_container_width=True)
                with l_col2:
                    fig_load = px.bar(loadings, barmode='group', title="Feature Influence (Loadings)",
                                      labels={'index': 'Features', 'value': 'Weight'})
                    st.plotly_chart(fig_load, use_container_width=True)

                st.info(f"""
                **Insight Summary:**
                * **PC1** represents **{var_pc1:.1f}%** of the dataset's variance and is most heavily influenced by **{top_driver_pc1}**.
                * **PC2** represents **{var_pc2:.1f}%** of the variance and is primarily driven by **{top_driver_pc2}**.
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
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)).stack()
                    pair = upper.abs().sort_values(ascending=False).index[0]
                    st.success(f"**Insight:** Strongest relationship found between **{pair[0]}** and **{pair[1]}** ({corr_matrix.loc[pair]:.2f}).")

    # --- PATH B: MACHINE LEARNING WORKSHOP ---
    elif mode == "Machine Learning Workshop":
        st.subheader("🤖 Supervised Learning Workshop")
        m_col1, m_col2 = st.columns([1, 2])

        with m_col1:
            st.write("### ⚙️ Model Configuration")
            target = st.selectbox("1. Target to Predict (Y):", df_raw.columns)

            unique_count = df_raw[target].nunique()
            is_numeric_target = pd.api.types.is_numeric_dtype(df_raw[target]) and not pd.api.types.is_bool_dtype(df_raw[target])
            default_task = 1 if (is_numeric_target and unique_count > 20) else 0

            task = st.radio("2. Task Type:", ["Classification (Group)", "Regression (Value)"], index=default_task)
            clf = is_clf(task)

            if clf and unique_count > 20:
                st.warning(f"⚠️ **High Complexity:** '{target}' has {unique_count} unique categories. "
                           "Accuracy usually drops as classes increase. Consider Regression if it is a measurement.")
            if not clf and not is_numeric_target:
                st.error("Regression needs a numeric target. Switch to Classification.")

            all_predictors = [c for c in df_raw.columns if c != target]
            usable, dropped = screen_features(df_raw, all_predictors)
            if dropped:
                with st.expander(f"🧹 {len(dropped)} column(s) automatically excluded"):
                    for c, why in dropped:
                        st.write(f"* **{c}** — {why}")

            use_all = st.checkbox("Use all usable clues (recommended)", value=True)
            if use_all:
                features = usable
                st.caption(f"✅ Using {len(features)} clues (numeric + categorical).")
            else:
                ordered = usable
                if st.checkbox("Rank clues by relevance?", value=True) and usable:
                    try:
                        mi = rank_predictors(df_raw, target, usable, clf)
                        ordered = mi.index.tolist()
                        st.caption(f"💡 Best Clue: **{ordered[0]}**")
                    except Exception:
                        pass
                features = st.multiselect("3. Select Clues (X):", options=ordered)

            # --- DISTRIBUTION VISUALIZER ---
            if features:
                with st.expander("📊 Clue Distribution Analysis"):
                    st.info("💡 **Why check this?**\n* **Linear Models & Naive Bayes:** Love bell curves.\n* **KNN & SVM:** Hate outliers.\n* **Trees:** Don't care about the shape, but seeing overlaps helps!")
                    selected_feat = st.selectbox("Select Clue to Inspect:", features)
                    is_num_feat = pd.api.types.is_numeric_dtype(df_raw[selected_feat])
                    color_arg = target if (clf and unique_count <= 20) else None
                    fig_dist = px.histogram(df_raw.sample(min(len(df_raw), 20000), random_state=SEED),
                                            x=selected_feat, color=color_arg,
                                            marginal="box" if is_num_feat else None,
                                            title=f"Distribution of {selected_feat}",
                                            barmode="overlay", template="plotly_white")
                    st.plotly_chart(fig_dist, use_container_width=True)

            algo_names = list(build_candidates(clf, False).keys())
            algo = st.selectbox("4. Algorithm:", [AUTO] + algo_names)

            if algo == "Naive Bayes":
                st.warning("⚠️ **Gaussian Assumption:** Naive Bayes assumes your clues follow a Bell Curve. If data is skewed, accuracy will be low.")

            tune = True
            depth = 5
            if algo == AUTO:
                st.caption("Every algorithm is tuned with 5-fold cross-validation on the 80% training data, "
                           "then scored once on the untouched 20% test data. The winner is picked by cross-validation, not by the test set.")
            else:
                tune = st.checkbox("Auto-tune hyperparameters (recommended)", value=True)
                if algo == "Decision Tree" and not tune:
                    depth = st.number_input("Select Max Tree Depth:", 1, 10, 5)

            if st.button("🚀 Start Model Training"):
                if not features:
                    st.error("Please select at least one feature (Clue) to train.")
                elif not clf and not is_numeric_target:
                    st.error("Regression needs a numeric target.")
                else:
                    try:
                        work = df_raw.dropna(subset=[target]).copy()
                        class_names = None

                        if clf:
                            y_txt = work[target].astype(str)
                            vc = y_txt.value_counts()
                            rare = vc[vc < MIN_CLASS_ROWS].index
                            if len(rare):
                                st.warning(f"Removed {len(rare)} class(es) with fewer than {MIN_CLASS_ROWS} rows "
                                           "(too few to split and validate reliably).")
                                work = work[~y_txt.isin(rare)]
                            if work[target].nunique() < 2:
                                raise ValueError("Need at least 2 classes with enough rows.")
                            le = LabelEncoder()
                            y = le.fit_transform(work[target].astype(str))
                            class_names = [str(c) for c in le.classes_]
                        else:
                            y_num = pd.to_numeric(work[target], errors="coerce")
                            work = work[y_num.notna()]
                            y = y_num[y_num.notna()].astype(float).values

                        if len(work) < 50:
                            raise ValueError("Need at least 50 usable rows to train and validate reliably.")

                        X = prepare_X(work, features)
                        algos = algo_names if algo == AUTO else [algo]

                        prog = st.progress(0.0, text="Starting...")
                        res = train_and_compare(
                            X, y, clf, algos, tune, depth,
                            lambda p, msg: prog.progress(min(p, 1.0), text=msg))
                        prog.empty()

                        res.update({"target": target, "features": features, "task": task,
                                    "class_names": class_names, "auto": algo == AUTO})
                        st.session_state.ml_results = res
                    except Exception as e:
                        st.error(f"⚠️ Error: {str(e)}")

        with m_col2:
            if st.session_state.ml_results:
                res = st.session_state.ml_results
                clf_r = is_clf(res['task'])
                pipe = res['pipe']

                st.write(f"### 🎯 Results for {res['target']}")
                headline = "Best algorithm for this dataset" if res['auto'] else "Selected algorithm"
                settings = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in res['params'].items())
                st.success(f"🏆 **{headline}: {res['algo']}**" + (f" — settings: {settings}" if settings else ""))

                # ---- split summary ----
                s1, s2, s3 = st.columns(3)
                s1.metric("Training rows (80%)", f"{res['n_train']:,}")
                s2.metric("Test rows (20%)", f"{res['n_test']:,}")
                s3.metric("Cross-val score", f"{res['cv_mean']:.2%}" if clf_r else f"{res['cv_mean']:.3f}",
                          f"± {res['cv_std']:.3f}", delta_color="off")
                st.caption(f"🔀 {res['split_note']}")
                if clf_r and res['balanced']:
                    st.info(f"⚖️ Classes are imbalanced (largest/smallest ≈ {res['imbalance_ratio']:.1f}x). "
                            "Class weights were applied and models were ranked by **balanced accuracy**.")

                # ---- classification ----
                if clf_r:
                    y_test, preds = res['y_test'], res['preds']
                    acc = accuracy_score(y_test, preds)
                    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

                    met1, met2, met3, met4 = st.columns(4)
                    met1.metric("Accuracy", f"{acc:.2%}")
                    met2.metric("Precision", f"{prec:.2%}")
                    met3.metric("Recall", f"{rec:.2%}")
                    met4.metric("F1-Score", f"{f1:.2%}")
                    main_score, score_name = acc, "accuracy"

                    with st.expander("📖 What do these scores mean?"):
                        st.markdown("""
                        * **Accuracy:** Overall correctness.
                        * **Precision:** Quality of 'Positive' guesses (Low precision = many false alarms).
                        * **Recall:** Ability to find all 'Positive' cases (Low recall = many missed cases).
                        * **F1-Score:** The 'Harmonic Mean' of Precision and Recall. Best for imbalanced data.
                        """)

                    cm = confusion_matrix(y_test, preds, labels=list(range(len(res['class_names']))))
                    fig, ax = plt.subplots(figsize=(8, 6))
                    show_labels = len(res['class_names']) < 15
                    sns.heatmap(cm, annot=show_labels, fmt='d', cmap="Purples",
                                xticklabels=res['class_names'] if show_labels else False,
                                yticklabels=res['class_names'] if show_labels else False, ax=ax)
                    plt.xticks(rotation=45)
                    ax.set_title("Confusion Matrix: Where did the model get confused?")
                    st.pyplot(fig)
                    plt.close(fig)

                    with st.expander("🔀 Split balance check (train vs test class share)"):
                        names = res['class_names']
                        tr_share = np.bincount(res['y_train'], minlength=len(names)) / len(res['y_train'])
                        te_share = np.bincount(y_test, minlength=len(names)) / len(y_test)
                        share_df = pd.DataFrame({"Class": names * 2,
                                                 "Share": np.concatenate([tr_share, te_share]),
                                                 "Set": ["Train"] * len(names) + ["Test"] * len(names)})
                        st.plotly_chart(px.bar(share_df, x="Class", y="Share", color="Set", barmode="group",
                                               template="plotly_white"), use_container_width=True)

                # ---- regression ----
                else:
                    y_test, preds = res['y_test'], res['preds']
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    mse = mean_squared_error(y_test, preds)
                    rmse = np.sqrt(mse)

                    reg_met1, reg_met2, reg_met3, reg_met4 = st.columns(4)
                    reg_met1.metric("R² Score", f"{r2:.3f}")
                    reg_met2.metric("MAE", f"{mae:.2f}")
                    reg_met3.metric("MSE", f"{mse:.2f}")
                    reg_met4.metric("RMSE", f"{rmse:.2f}")
                    main_score, score_name = r2, "R²"

                    with st.expander("📖 What do these regression scores mean?"):
                        st.markdown("""
                        * **R² Score:** How well the model fits the data (1.0 is perfect).
                        * **MAE (Mean Absolute Error):** The average 'distance' your prediction is from the truth.
                        * **MSE (Mean Squared Error):** Similar to MAE, but punishes large errors more heavily.
                        * **RMSE (Root Mean Squared Error):** The standard deviation of the residuals (errors).
                        """)

                    fig_reg = px.scatter(x=y_test, y=preds, labels={'x': 'Actual Value', 'y': 'Predicted Value'},
                                         title="Actual vs Predicted Comparison")
                    fig_reg.add_shape(type="line", x0=float(np.min(y_test)), y0=float(np.min(y_test)),
                                      x1=float(np.max(y_test)), y1=float(np.max(y_test)),
                                      line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_reg, use_container_width=True)

                # ---- honest quality gate ----
                best_row = res['leaderboard'].iloc[0]
                gap = float(best_row["Overfit gap"])
                if main_score >= 0.95:
                    st.success(f"✅ Excellent: {score_name} {main_score:.2%}" if clf_r else f"✅ Excellent: R² {main_score:.3f}")
                elif main_score >= TARGET_SCORE:
                    st.success(f"✅ Meets the 90% target: {score_name} {main_score:.2%}" if clf_r else f"✅ Meets the 0.90 target: R² {main_score:.3f}")
                else:
                    st.warning(f"⚠️ Below the 90% target ({score_name} = {main_score:.2%})." if clf_r else
                               f"⚠️ Below the 0.90 target (R² = {main_score:.3f}).")
                    tips = []
                    if clf_r and main_score - res['baseline'] < 0.05:
                        tips.append(f"The model barely beats always guessing the most common class ({res['baseline']:.2%}). "
                                    "The selected clues probably do not contain enough signal about the target.")
                    if gap > 0.10:
                        tips.append(f"Overfitting: train score is {gap:.0%} higher than test. More rows or fewer/cleaner clues would help.")
                    if clf_r and len(res['class_names']) > 10:
                        tips.append("Many classes make high accuracy much harder; merge similar classes if that makes sense.")
                    tips.append("Add more informative columns, fix label noise, or check that the target is really predictable from these clues. "
                                "No algorithm can score higher than the data allows.")
                    for t in tips:
                        st.write(f"* {t}")
                if main_score > 0.995:
                    st.info("🔎 Score is almost perfect. Double-check that no clue is a copy or a direct result of the target (data leakage).")

                # ---- leaderboard ----
                if len(res['leaderboard']) > 1:
                    st.divider()
                    st.write("### 🏁 Algorithm Leaderboard")
                    st.caption("Ranked by cross-validation on the training data. 'Overfit gap' = train score − test score (large = memorising).")
                    lb = res['leaderboard'].copy()
                    pct_cols = [c for c in lb.columns if c.startswith("CV score") or c in ("CV ± std", "Test score", "Train score", "Overfit gap")]
                    st.dataframe(lb.style.format({**{c: "{:.3f}" for c in pct_cols},
                                                  "Time (s)": "{:.1f}",
                                                  ("Balanced acc." if clf_r else "RMSE"): "{:.3f}"}),
                                 use_container_width=True, hide_index=True)

                # ---- logic explainer ----
                st.divider()
                st.write(f"### 🧠 {res['algo']} Logic Explainer")
                algo_r = res['algo']

                if algo_r == "Decision Tree":
                    render_tree(pipe, res['class_names'])
                elif algo_r == "Linear/Logistic Regression":
                    with st.expander("🔍 The 'Weight' Logic", expanded=True):
                        st.latex(r"y = w_1x_1 + w_2x_2 + ... + b")
                elif algo_r == "KNN":
                    with st.expander("🔍 The 'Neighbor' Logic", expanded=True):
                        st.write("Looks for the **K** most similar rows and votes (or averages) their results.")
                elif algo_r == "Naive Bayes":
                    with st.expander("🔍 The 'Probability' Logic", expanded=True):
                        st.latex(r"P(C | Clues) = \frac{P(Clues | C) \times P(C)}{P(Clues)}")
                elif algo_r == "SVM":
                    with st.expander("🔍 The 'Boundary' Logic", expanded=True):
                        st.write("Finds the best boundary (hyperplane) that separates different groups.")
                elif algo_r == "Random Forest":
                    with st.expander("🔍 The 'Crowd of Trees' Logic", expanded=True):
                        st.write("Grows many decision trees on random slices of the data and lets them vote. "
                                 "One tree can be fooled; a crowd of different trees rarely is.")
                elif algo_r == "Gradient Boosting":
                    with st.expander("🔍 The 'Learn From Mistakes' Logic", expanded=True):
                        st.write("Builds small trees one after another, each one focusing on the rows the previous trees got wrong.")

                if algo_r != "Decision Tree" and res.get('tree_pipe') is not None:
                    with st.expander("🌳 See the Decision Tree from this comparison"):
                        render_tree(res['tree_pipe'], res['class_names'])

                # ---- importance ----
                st.divider()
                st.write("### 📊 Predictor Influence")
                st.caption("Permutation importance on the test set: how much the score drops when a clue is shuffled.")
                if res['importance'] is not None:
                    imp_df = res['importance'].sort_values(by='Value').tail(25)
                    fig_imp = px.bar(imp_df, x='Value', y='Feature', orientation='h',
                                     color='Value', color_continuous_scale='Portland')
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Importance could not be computed for this model.")
            else:
                st.info("Train a model to see the logic visualization here.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | Educational ML Laboratory</p>", unsafe_allow_html=True)
