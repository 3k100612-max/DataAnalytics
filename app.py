import json
import os
import re
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
from sklearn.cluster import (
    DBSCAN, AgglomerativeClustering, Birch, KMeans, MiniBatchKMeans,
)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier,
    ExtraTreesRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, IsolationForest, RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge,
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, calinski_harabasz_score,
    confusion_matrix, classification_report, davies_bouldin_score, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (
    GridSearchCV, KFold, StratifiedKFold, cross_val_score, learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
TARGET_SCORE = 0.80         # quality gate shown to the user
EXCELLENT_SCORE = min(0.95, TARGET_SCORE + 0.10)
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
            if df[col].abs().max() < 2 ** 31 - 1:
                df[col] = df[col].astype("int32")
    return df, total_missing

def add_datetime_features(df, dayfirst=False, parse_threshold=0.80):
    """
    Detect likely date/time columns and add numeric features for modeling.

    Returns:
        df_features: copy of df with date/time features added
        raw_date_cols: original date/time columns detected
    """
    df_features = df.copy()
    raw_date_cols = []

    name_hint = re.compile(r"date|time|timestamp|datetime", re.IGNORECASE)
    date_pattern = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")

    for col in df.columns:
        already_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
        is_text = (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        )

        # Do not guess that numeric columns are dates; they might be IDs.
        if not already_datetime and not is_text:
            continue

        sample = df[col].dropna()
        if len(sample) > 500:
            sample = sample.sample(500, random_state=42)

        if already_datetime:
            parsed_sample = pd.to_datetime(sample, errors="coerce")
        else:
            parsed_sample = pd.to_datetime(
                sample.astype(str),
                errors="coerce",
                dayfirst=dayfirst,
            )

        parse_rate = parsed_sample.notna().mean() if len(sample) else 0.0

        looks_like_date_text = (
            is_text
            and sample.astype(str).str.match(date_pattern).mean() >= parse_threshold
        )

        # Detect by column name or by recognizable date-formatted values.
        if not (
            already_datetime
            or name_hint.search(str(col))
            or (looks_like_date_text and parse_rate >= parse_threshold)
        ):
            continue

        parsed = pd.to_datetime(
            df[col],
            errors="coerce",
            dayfirst=dayfirst,
        )

        # Skip the column if too many values could not be parsed.
        if parsed.notna().mean() < parse_threshold:
            continue

        raw_date_cols.append(col)
        prefix = re.sub(r"\W+", "_", str(col)).strip("_") or "date"

        # Calendar features
        df_features[f"{prefix}_year"] = parsed.dt.year
        df_features[f"{prefix}_month"] = parsed.dt.month
        df_features[f"{prefix}_day"] = parsed.dt.day
        df_features[f"{prefix}_weekday"] = parsed.dt.dayofweek
        df_features[f"{prefix}_is_weekend"] = (
            parsed.dt.dayofweek >= 5
        ).astype("float32")

        # Add time-of-day features if the column contains times.
        has_time = (
            parsed.dt.hour.ne(0).any()
            or parsed.dt.minute.ne(0).any()
            or parsed.dt.second.ne(0).any()
        )
        if has_time:
            df_features[f"{prefix}_hour"] = parsed.dt.hour
            df_features[f"{prefix}_minute"] = parsed.dt.minute

        # Cyclical features represent repeating patterns such as months and weekdays.
        month_angle = 2 * np.pi * (parsed.dt.month - 1) / 12
        weekday_angle = 2 * np.pi * parsed.dt.dayofweek / 7

        df_features[f"{prefix}_month_sin"] = np.sin(month_angle)
        df_features[f"{prefix}_month_cos"] = np.cos(month_angle)
        df_features[f"{prefix}_weekday_sin"] = np.sin(weekday_angle)
        df_features[f"{prefix}_weekday_cos"] = np.cos(weekday_angle)

    return df_features, raw_date_cols

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
# 3. DATASET UNDERSTANDING HELPERS (steps 2-6)
# ----------------------------------------------------------------------------
def research_profile(df):
    """A quick 'research profile' of the raw dataset: shape, column types,
    memory footprint, duplicate rows, and overall completeness."""
    n_rows, n_cols = df.shape
    bool_cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in bool_cols]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in bool_cols]
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    duplicate_rows = int(df.duplicated().sum())
    missing_total = int(df.isnull().sum().sum())
    total_cells = n_rows * n_cols if n_rows and n_cols else 1
    missing_pct = missing_total / total_cells
    return {
        "n_rows": n_rows, "n_cols": n_cols,
        "numeric_cols": numeric_cols, "categorical_cols": categorical_cols, "bool_cols": bool_cols,
        "memory_mb": memory_mb, "duplicate_rows": duplicate_rows,
        "missing_total": missing_total, "missing_pct": missing_pct,
        "completeness": 1 - missing_pct,
    }


def missing_data_table(df):
    """Per-column missing-value counts and percentages, sorted worst-first."""
    miss = df.isnull().sum()
    pct = (miss / len(df) * 100) if len(df) else miss * 0
    out = pd.DataFrame({"Column": df.columns, "Missing": miss.values, "Missing %": pct.values})
    return out.sort_values("Missing", ascending=False).reset_index(drop=True)


def detect_duplicates_and_ids(df):
    """Duplicate row count plus a list of columns that look like identifiers,
    constants, or free text — the columns the ML step will screen out."""
    n = len(df)
    dup_rows = int(df.duplicated().sum())
    dup_pct = dup_rows / n if n else 0
    id_like = []
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if nun <= 1:
            id_like.append((c, "constant — every row has the same value, so it carries no information"))
        elif pd.api.types.is_integer_dtype(s) and nun == n and n > 50:
            id_like.append((c, "a unique value on every row — looks like a record identifier"))
        elif not numeric and nun > MAX_CATEGORIES and nun > n * 0.5:
            id_like.append((c, f"{nun} distinct text values — looks like free text or an identifier"))
    return dup_rows, dup_pct, id_like


def variable_summary_table(df):
    """One-row-per-column summary: type, missing %, and either numeric stats
    or the most common category, whichever fits the column."""
    rows = []
    for c in df.columns:
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        missing_pct = s.isnull().mean() * 100
        row = {"Column": c, "Type": "Numeric" if numeric else "Categorical",
               "Missing %": round(missing_pct, 1), "Unique values": int(s.nunique(dropna=True)),
               "Mean": None, "Std": None, "Min": None, "Max": None,
               "Top value": None, "Top value share %": None}
        if numeric and s.notna().any():
            row.update({"Mean": round(float(s.mean()), 3), "Std": round(float(s.std()), 3),
                       "Min": round(float(s.min()), 3), "Max": round(float(s.max()), 3)})
        elif not numeric and s.notna().any():
            vc = s.astype(str).value_counts(normalize=True)
            if len(vc):
                row.update({"Top value": str(vc.index[0]), "Top value share %": round(float(vc.iloc[0]) * 100, 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def build_methodology_report(res, mode_label, target=None, mode_extra=None):
    """Compose a plain-text methodology summary for the '13. Download results
    and methodology' step — a short write-up of what was done and why."""
    lines = [f"# Methodology Report — {mode_label}", ""]
    if target is not None:
        settings_txt = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in res['params'].items()) or "default settings"
        lines += [
            f"**Target variable:** {target}",
            f"**Task type:** {res['task']}",
            f"**Features (clues) used:** {', '.join(res['features'])}",
            "",
            "## Data Split",
            f"- {res['split_note']}",
            f"- Training rows: {res['n_train']:,} (80%)",
            f"- Test rows: {res['n_test']:,} (20%, held out and untouched until final evaluation)",
            "",
            "## Algorithm Selection",
            f"- Algorithm(s) considered: {len(res['leaderboard'])}",
            f"- Selected algorithm: {res['algo']} — {settings_txt}",
            f"- Selection was based on cross-validation on the training data only ({res['scoring']}); "
            "the test set was never used to choose the winner.",
            f"- Cross-validation score: {res['cv_mean']:.3f} (± {res['cv_std']:.3f})",
            "",
            "## Final Evaluation (test set, used once)",
        ]
        if is_clf(res['task']):
            acc = accuracy_score(res['y_test'], res['preds'])
            lines.append(f"- Accuracy: {acc:.2%}")
            lines.append(f"- Baseline (always guess the most common class): {res['baseline']:.2%}")
        else:
            r2 = r2_score(res['y_test'], res['preds'])
            lines.append(f"- R² score: {r2:.3f}")
        lines.append(f"- Missing values fixed automatically: {res.get('total_missing', 0):,}")
    elif mode_extra is not None:
        lines += [
            f"**Method:** {mode_extra.get('method')}",
            f"**Features used:** {', '.join(mode_extra.get('features', []))}",
            f"**Standardized features:** {'Yes' if mode_extra.get('scale_features') else 'No'}",
            "",
            "## Discovery Summary",
        ]
        m = mode_extra.get("metrics", {})
        for key, label in [("groups", "Groups found"), ("anomalies", "Anomalies found"),
                           ("silhouette", "Silhouette score"), ("davies_bouldin", "Davies-Bouldin score"),
                           ("calinski_harabasz", "Calinski-Harabasz score")]:
            if key in m and m[key] is not None:
                lines.append(f"- {label}: {m[key]}")
    lines += ["", "---", "Generated by the Machine Learning Intuition Lab."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 4. SUPERVISED ML HELPERS
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
    """Build supervised-learning candidates: {name: (estimator, hyperparameter_grid)}."""
    cw = "balanced" if balanced else None

    if clf:
        return {
            "Linear/Logistic Regression": (
                LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED),
                {"model__C": [0.1, 1, 10]}),
            "Linear Discriminant Analysis": (
                LinearDiscriminantAnalysis(),
                {"model__solver": ["svd", "lsqr"]}),
            "Decision Tree": (
                DecisionTreeClassifier(random_state=SEED, class_weight=cw),
                {"model__max_depth": [3, 5, 7, 10, None], "model__min_samples_leaf": [1, 5, 20]}),
            "Naive Bayes": (
                GaussianNB(),
                {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}),
            "KNN": (
                KNeighborsClassifier(),
                {"model__n_neighbors": [3, 5, 11, 21], "model__weights": ["uniform", "distance"]}),
            "SVM": (
                SVC(class_weight=cw, probability=True, random_state=SEED),
                {"model__C": [0.5, 1, 5, 20], "model__kernel": ["rbf", "linear"]}),
            "Random Forest": (
                RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=SEED, n_jobs=1),
                {"model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 3, 10]}),
            "Extra Trees": (
                ExtraTreesClassifier(n_estimators=200, class_weight=cw, random_state=SEED, n_jobs=1),
                {"model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 3, 10]}),
            "Gradient Boosting": (
                GradientBoostingClassifier(random_state=SEED),
                {"model__learning_rate": [0.03, 0.1, 0.2], "model__n_estimators": [100, 200],
                 "model__max_depth": [2, 3, 5]}),
            "HistGradient Boosting": (
                HistGradientBoostingClassifier(random_state=SEED),
                {"model__learning_rate": [0.05, 0.1], "model__max_iter": [100, 200],
                 "model__max_leaf_nodes": [15, 31]}),
            "AdaBoost": (
                AdaBoostClassifier(random_state=SEED),
                {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.05, 0.1, 1.0]}),
            "Neural Network": (
                MLPClassifier(max_iter=500, early_stopping=True, random_state=SEED),
                {"model__hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)],
                 "model__alpha": [0.0001, 0.001, 0.01]}),
        }

    return {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge Regression": (
            Ridge(),
            {"model__alpha": [0.01, 0.1, 1, 10, 100]}),
        "Lasso Regression": (
            Lasso(max_iter=5000),
            {"model__alpha": [0.001, 0.01, 0.1, 1]}),
        "Elastic Net": (
            ElasticNet(max_iter=5000),
            {"model__alpha": [0.001, 0.01, 0.1, 1], "model__l1_ratio": [0.1, 0.5, 0.9]}),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=SEED),
            {"model__max_depth": [3, 5, 7, 10, None], "model__min_samples_leaf": [1, 5, 20]}),
        "KNN": (
            KNeighborsRegressor(),
            {"model__n_neighbors": [3, 5, 11, 21], "model__weights": ["uniform", "distance"]}),
        "SVM": (
            TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()),
            {"model__regressor__C": [0.5, 1, 5, 20], "model__regressor__epsilon": [0.05, 0.1, 0.2],
             "model__regressor__kernel": ["rbf", "linear"]}),
        "Random Forest": (
            RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
            {"model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 3, 10]}),
        "Extra Trees": (
            ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
            {"model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 3, 10]}),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=SEED),
            {"model__learning_rate": [0.03, 0.1, 0.2], "model__n_estimators": [100, 200],
             "model__max_depth": [2, 3, 5]}),
        "HistGradient Boosting": (
            HistGradientBoostingRegressor(random_state=SEED),
            {"model__learning_rate": [0.05, 0.1], "model__max_iter": [100, 200],
             "model__max_leaf_nodes": [15, 31]}),
        "AdaBoost": (
            AdaBoostRegressor(random_state=SEED),
            {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.05, 0.1, 1.0]}),
        "Neural Network": (
            MLPRegressor(max_iter=500, early_stopping=True, random_state=SEED),
            {"model__hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)],
             "model__alpha": [0.0001, 0.001, 0.01]}),
    }


def subsample(X, y, n, clf):
    if n is None or len(X) <= n:
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
            if balanced:
                test_score = balanced_accuracy_score(y_te, preds)
                train_score = balanced_accuracy_score(y_tr_probe, tr_preds)
            else:
                test_score = accuracy_score(y_te, preds)
                train_score = accuracy_score(y_tr_probe, tr_preds)
            extra = balanced_accuracy_score(y_te, preds)
        else:
            test_score = r2_score(y_te, preds)
            train_score = r2_score(y_tr_probe, tr_preds)
            extra = float(np.sqrt(mean_squared_error(y_te, preds)))

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
            scoring=scoring, n_jobs=1)
        importance = pd.DataFrame({"Feature": X_te.columns, "Value": perm.importances_mean})
    except Exception:
        pass

    # ---- 5. learning curve (uses the winning pipeline's own settings) -------------
    learning_curve_data = None
    try:
        lc_pipe = clone(best["pipe"])
        lc_X, lc_y = subsample(X_tr, y_tr, min(len(X_tr), 8000), clf)
        lc_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED) if clf else KFold(n_splits=3, shuffle=True, random_state=SEED)
        train_sizes, train_scores, test_scores = learning_curve(
            lc_pipe, lc_X, lc_y, cv=lc_cv, scoring=scoring, n_jobs=2,
            train_sizes=np.linspace(0.2, 1.0, 5), random_state=SEED)
        learning_curve_data = pd.DataFrame({
            "Training rows": train_sizes,
            "Train score": train_scores.mean(axis=1),
            "Validation score": test_scores.mean(axis=1),
        })
    except Exception:
        pass

    return {
        "pipe": best["pipe"], "algo": best["name"], "preds": best["preds"], "y_test": y_te,
        "X_test": X_te, "leaderboard": board, "baseline": baseline, "scoring": scoring,
        "imbalance_ratio": ratio, "balanced": balanced, "split_note": split_note,
        "n_train": len(X_tr), "n_test": len(X_te), "tune_rows": len(X_t),
        "cv_mean": best["cv_mean"], "cv_std": best["cv_std"], "params": best["params"],
        "tree_pipe": tree_pipe, "importance": importance, "learning_curve": learning_curve_data,
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


def explain_algo(algo_r):
    """Show a short plain-English explanation for whichever algorithm won."""
    linear_names = {"Linear/Logistic Regression", "Linear Regression", "Ridge Regression",
                    "Lasso Regression", "Elastic Net", "Linear Discriminant Analysis"}
    tree_boost_names = {"Random Forest", "Extra Trees", "Gradient Boosting",
                        "HistGradient Boosting", "AdaBoost"}
    if algo_r in linear_names:
        with st.expander("🔍 The 'Weight' Logic", expanded=True):
            st.latex(r"y = w_1x_1 + w_2x_2 + ... + b")
            if algo_r in {"Ridge Regression", "Lasso Regression", "Elastic Net"}:
                st.write("Adds a penalty that shrinks weights toward zero, which reduces overfitting "
                         "and helps when clues are correlated with each other.")
    elif algo_r == "KNN":
        with st.expander("🔍 The 'Neighbor' Logic", expanded=True):
            st.write("Looks for the **K** most similar rows and votes (or averages) their results.")
    elif algo_r == "Naive Bayes":
        with st.expander("🔍 The 'Probability' Logic", expanded=True):
            st.latex(r"P(C | Clues) = \frac{P(Clues | C) \times P(C)}{P(Clues)}")
    elif algo_r == "SVM":
        with st.expander("🔍 The 'Boundary' Logic", expanded=True):
            st.write("Finds the best boundary (hyperplane) that separates different groups, "
                     "or the tightest tube around the trend for regression.")
    elif algo_r in tree_boost_names:
        with st.expander("🔍 The 'Crowd of Trees' Logic", expanded=True):
            if "Boosting" in algo_r or algo_r == "AdaBoost":
                st.write("Builds small trees one after another, each one focusing on the rows "
                         "the previous trees got wrong.")
            else:
                st.write("Grows many decision trees on random slices of the data and lets them vote. "
                         "One tree can be fooled; a crowd of different trees rarely is.")
    elif algo_r == "Neural Network":
        with st.expander("🔍 The 'Layered Weights' Logic", expanded=True):
            st.write("Passes the clues through layers of weighted connections, learning curved "
                     "(non-linear) patterns that straight-line models miss.")


# ----------------------------------------------------------------------------
# 5. UNSUPERVISED HELPERS
# ----------------------------------------------------------------------------
def create_cluster_labels(result_df, features, group_column="Group"):
    """Creates human-readable labels for clusters based on the selected features.
    Each feature is compared with the overall mean: High / Low / Average."""
    profile = result_df.groupby(group_column)[features].mean()
    overall_mean = result_df[features].mean()
    overall_std = result_df[features].std().replace(0, np.nan)

    profile_z = (profile - overall_mean) / overall_std
    profile_z = profile_z.replace([np.inf, -np.inf], np.nan).fillna(0)

    descriptive_labels = {}
    for group in profile_z.index:
        values = profile_z.loc[group].sort_values()
        high_features = [f for f, v in values.items() if v >= 0.50][:2]
        low_features = [f for f, v in values.items() if v <= -0.50][:2]
        group_number = str(group)
        if high_features and low_features:
            label = f"Group {group_number}: High {', '.join(high_features)}; Low {', '.join(low_features)}"
        elif high_features:
            label = f"Group {group_number}: High {', '.join(high_features)}"
        elif low_features:
            label = f"Group {group_number}: Low {', '.join(low_features)}"
        else:
            label = f"Group {group_number}: Average profile"
        descriptive_labels[group_number] = label

    return descriptive_labels, profile, profile_z


def run_unsupervised_analysis(df, features, method, n_clusters, eps, min_samples,
                              scale_features, random_state):
    """Clean the chosen numeric columns, run the selected unsupervised method,
    and package a PCA projection plus quality metrics for display."""
    data = df[features].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.median(numeric_only=True))

    X = StandardScaler().fit_transform(data) if scale_features else data.to_numpy()
    n_rows = len(data)

    if method == "K-Means Clustering":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
    elif method == "Mini-Batch K-Means":
        model = MiniBatchKMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
    elif method == "Hierarchical Clustering":
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)
    elif method == "Birch Clustering":
        labels = Birch(n_clusters=n_clusters).fit_predict(X)
    elif method == "Gaussian Mixture Clustering":
        labels = GaussianMixture(n_components=n_clusters, random_state=random_state).fit_predict(X)
    elif method == "DBSCAN Clustering":
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    elif method == "Isolation Forest":
        labels = IsolationForest(random_state=random_state, contamination="auto").fit_predict(X)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 2D PCA projection purely for visualization
    n_comp = min(2, X.shape[1])
    coords = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

    result_df = data.copy()
    result_df["Row"] = np.arange(n_rows)
    result_df["PCA 1"] = coords[:, 0]
    result_df["PCA 2"] = coords[:, 1]

    metrics = {"rows": n_rows, "features": len(features)}

    if method == "Isolation Forest":
        result_df["Status"] = np.where(labels == -1, "Anomaly", "Normal")
        metrics["anomalies"] = int((labels == -1).sum())
        return {"result": result_df, "metrics": metrics}

    # ---- clustering methods: group column, quality metrics, human-readable labels ----
    result_df["Group"] = labels.astype(str)
    unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]
    metrics["groups"] = len(unique_labels)
    metrics["silhouette"] = None
    metrics["davies_bouldin"] = None
    metrics["calinski_harabasz"] = None

    # Quality metrics computed independently of whether labeling below succeeds
    try:
        mask = labels != -1  # exclude DBSCAN noise from scoring
        if mask.sum() >= 2 and 2 <= len(set(labels[mask])) < mask.sum():
            metrics["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
            metrics["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X[mask], labels[mask]))
    except Exception:
        pass

    # Human-readable group descriptions
    try:
        cluster_labels, cluster_profile, cluster_profile_z = create_cluster_labels(
            result_df=result_df, features=features, group_column="Group")
        result_df["Group Label"] = result_df["Group"].map(cluster_labels)
        metrics["cluster_labels"] = cluster_labels
        metrics["cluster_profile"] = cluster_profile
        metrics["cluster_profile_z"] = cluster_profile_z
    except Exception:
        result_df["Group Label"] = result_df["Group"]
        metrics["cluster_labels"] = {}
        metrics["cluster_profile"] = None
        metrics["cluster_profile_z"] = None

    return {"result": result_df, "metrics": metrics}


# ----------------------------------------------------------------------------
# 6. UI CONFIGURATION
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Machine Learning Intuition Lab", layout="wide", page_icon="🧪",
    menu_items={'About': " Machine Learning Intuition Lab A Project in Fullfillment with the Requirement of MSIT643 Submitted by Timothy Mark A. Bal-e"})
st.title("🧪 Machine Learning Intuition Lab")
st.caption("A guided, 13-step research pipeline: upload a dataset and walk through every stage "
          "from a health check to a downloadable methodology report.")
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
    st.markdown("---")
    st.caption("📋 Pipeline steps:\n1. Upload · 2. Health check · 3. Profile · 4. Missing data · "
              "5. Duplicates/IDs · 6. Variable summaries · 7. Target · 8. Exploratory · "
              "9. Supervised/Unsupervised · 10. Comparison · 11. Validation · "
              "12. Interpretation · 13. Download")

# =============================================================================
# STEP 1 — UPLOAD DATASET
# =============================================================================
st.header("1️⃣ Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file to begin the research pipeline.", type="csv")

if uploaded_file:
    df_raw, total_missing = load_raw(uploaded_file, row_limit)

    # Create a separate copy with model-ready features derived from date/time columns.
    # Keep df_raw for the original data, target selector, and dataset summaries.
    df_model, raw_date_cols = add_datetime_features(
        df_raw,
        dayfirst=False,  # Set True if dates look like 31/12/2025.
    )

    df = fix_missing(df_raw)      # imputed copy for exploration + download
    profile = research_profile(df_raw)


    # =========================================================================
    # STEP 2 — DATASET HEALTH CHECK
    # =========================================================================
    st.divider()
    st.header("2️⃣ Dataset Health Check")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Rows", f"{profile['n_rows']:,}")
    h2.metric("Columns", f"{profile['n_cols']:,}")
    h3.metric("Completeness", f"{profile['completeness']:.1%}")
    h4.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")

    if profile['completeness'] >= 0.98 and profile['duplicate_rows'] == 0:
        st.success("✅ This dataset looks healthy — very few missing values and no duplicate rows.")
    elif profile['completeness'] >= 0.90:
        st.info(f"ℹ️ {profile['missing_total']:,} missing values were found and have been automatically "
               "fixed for exploration (median for numbers, most common value for categories). "
               "Inside the model pipeline, missing values are always fixed *after* the train/test "
               "split so no test information leaks into training.")
    else:
        st.warning(f"⚠️ This dataset is fairly sparse ({profile['completeness']:.0%} complete). "
                  "Results may be less reliable — consider checking the source data.")

    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Auto-Fixed Dataset (CSV)", data=csv_data, file_name="cleaned_data.csv")

    # =========================================================================
    # STEP 3 — RESEARCH PROFILE
    # =========================================================================
    st.divider()
    st.header("3️⃣ Research Profile")
    with st.expander("📋 Column types and memory footprint", expanded=True):
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Numeric columns", len(profile['numeric_cols']))
        p2.metric("Categorical columns", len(profile['categorical_cols']))
        p3.metric("Boolean columns", len(profile['bool_cols']))
        p4.metric("Memory footprint", f"{profile['memory_mb']:.1f} MB")

        fig_types = px.pie(
            names=["Numeric", "Categorical", "Boolean"],
            values=[len(profile['numeric_cols']), len(profile['categorical_cols']), len(profile['bool_cols'])],
            title="Column type breakdown", hole=0.45, template="plotly_white")
        st.plotly_chart(fig_types, use_container_width=True)
        st.dataframe(pd.DataFrame({"Column": df_raw.columns, "Data type": df_raw.dtypes.astype(str).values}),
                    use_container_width=True, hide_index=True)

    # =========================================================================
    # STEP 4 — MISSING-DATA ANALYSIS
    # =========================================================================
    st.divider()
    st.header("4️⃣ Missing-Data Analysis")
    with st.expander("🕳️ Where the gaps are", expanded=(profile['missing_total'] > 0)):
        if profile['missing_total'] == 0:
            st.success("✅ No missing values found in this dataset.")
        else:
            miss_tbl = missing_data_table(df_raw)
            miss_tbl_nonzero = miss_tbl[miss_tbl["Missing"] > 0]
            fig_missing = px.bar(miss_tbl_nonzero.head(20), x="Column", y="Missing %",
                                 title="Top columns by % missing", template="plotly_white",
                                 color="Missing %", color_continuous_scale="Reds")
            fig_missing.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_missing, use_container_width=True)
            st.dataframe(miss_tbl_nonzero, use_container_width=True, hide_index=True)
            st.caption("These gaps are filled automatically for exploration (median/most-common value), "
                      "and filled again — separately — inside the model pipeline after each train/test split.")

    # =========================================================================
    # STEP 5 — DUPLICATE AND IDENTIFIER DETECTION
    # =========================================================================
    st.divider()
    st.header("5️⃣ Duplicate & Identifier Detection")
    with st.expander("🔍 Rows and columns that don't belong in a model", expanded=True):
        dup_rows, dup_pct, id_like = detect_duplicates_and_ids(df_raw)
        dcol1, dcol2 = st.columns(2)
        dcol1.metric("Duplicate rows", f"{dup_rows:,}", f"{dup_pct:.1%} of rows", delta_color="off")
        dcol2.metric("ID-like / constant columns", len(id_like))
        if dup_rows:
            st.warning(f"⚠️ {dup_rows:,} duplicate rows were found. They aren't removed automatically, "
                      "but duplicated rows can inflate performance if they land in both halves of a split.")
        else:
            st.success("✅ No exact duplicate rows found.")
        if id_like:
            st.write("These columns look like identifiers, constants, or free text, and are "
                    "automatically screened out before model training:")
            for c, why in id_like:
                st.write(f"- **{c}** — {why}")
        else:
            st.success("✅ No obvious identifier or constant columns detected.")

    # =========================================================================
    # STEP 6 — VARIABLE SUMMARIES
    # =========================================================================
    st.divider()
    st.header("6️⃣ Variable Summaries")
    with st.expander("📊 Per-column statistics", expanded=False):
        var_tbl = variable_summary_table(df_raw)
        st.dataframe(var_tbl, use_container_width=True, hide_index=True)

        inspect_col = st.selectbox("Inspect one column's distribution:", df_raw.columns, key="var_inspect")
        s = df_raw[inspect_col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            fig_var = px.histogram(df_raw, x=inspect_col, marginal="box", template="plotly_white",
                                   title=f"Distribution of {inspect_col}")
        else:
            vc = s.astype(str).value_counts().reset_index().head(20)
            vc.columns = [inspect_col, "Count"]
            fig_var = px.bar(vc, x=inspect_col, y="Count", template="plotly_white",
                             title=f"Top categories in {inspect_col}")
            fig_var.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_var, use_container_width=True)

    # =========================================================================
    # STEP 7 — TARGET ANALYSIS
    # =========================================================================
    st.divider()
    st.header("7️⃣ Target Analysis")
    st.caption("Pick the variable you want to understand or predict. Everything from here on "
              "(exploratory analysis, modeling, and interpretation) is centered on this choice.")
    target = st.selectbox("Target variable:", df_raw.columns, key="target_step7")
    tcol = df_raw[target]
    unique_count = int(tcol.nunique())
    is_numeric_target = pd.api.types.is_numeric_dtype(tcol) and not pd.api.types.is_bool_dtype(tcol)

    if is_numeric_target and unique_count > 20:
        fig_target = px.histogram(df_raw, x=target, marginal="box", template="plotly_white",
                                  title=f"Distribution of {target}")
        suggested_task = "Regression"
    else:
        vc = tcol.astype(str).value_counts().reset_index()
        vc.columns = [target, "Count"]
        fig_target = px.bar(vc, x=target, y="Count", template="plotly_white",
                            title=f"Category counts for {target}", color="Count",
                            color_continuous_scale="Blues")
        suggested_task = "Classification"
    st.plotly_chart(fig_target, use_container_width=True)

    t1, t2, t3 = st.columns(3)
    t1.metric("Unique values", unique_count)
    t2.metric("Missing", f"{tcol.isnull().sum():,} ({tcol.isnull().mean():.1%})")
    t3.metric("Suggested task", suggested_task)
    if suggested_task == "Classification" and unique_count > 20:
        st.warning(f"⚠️ '{target}' has {unique_count} categories — high complexity for classification. "
                  "Consider regression if it's really a measurement.")

    # =========================================================================
    # STEP 8 — EXPLORATORY ANALYSIS
    # =========================================================================
    st.divider()
    st.header("8️⃣ Exploratory Analysis (PCA & Correlations)")
    with st.expander("🔬 Explore relationships before modeling", expanded=False):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.info("💡 Exploratory analysis simplifies data and finds hidden relationships before training starts.")

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("### 💎 PCA (Dimensionality Reduction)")
            with st.expander("📖 The Shadow Analogy (Explainer)"):
                st.write("Imagine holding a 3D teapot in front of a flashlight. The shadow on the wall is 2D. **PCA** finds the best angle to hold the teapot so the shadow captures the most detail.")

            pca_feats = st.multiselect("Select Numeric Columns to Compress:", num_cols, default=num_cols[:min(3, len(num_cols))])
            target_color = st.selectbox("Color Map by:", df.columns, index=list(df.columns).index(target) if target in df.columns else 0, key="pca_color")

            if st.button("Generate PCA Insights") and len(pca_feats) >= 2:
                pca_source = df.sample(min(len(df), 20000), random_state=SEED)
                pca_input = pca_source[pca_feats].replace([np.inf, -np.inf], np.nan)
                pca_input = pca_input.fillna(pca_input.median(numeric_only=True))
                X_pca = StandardScaler().fit_transform(pca_input)

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
                pdf[target_color] = pca_source[target_color].astype(str).values

                fig_pca = px.scatter(
                    pdf, x='PC1', y='PC2', color=target_color,
                    title=f"PCA: {target_color} Distribution",
                    labels={'PC1': label_x, 'PC2': label_y}, template="plotly_white")
                st.plotly_chart(fig_pca, use_container_width=True)

                st.write("#### 🧠 PCA Logic Visualizer")
                l_col1, l_col2 = st.columns(2)
                with l_col1:
                    fig_var2 = px.bar(x=['PC1', 'PC2'], y=pca.explained_variance_ratio_,
                                      title="Information Retention",
                                      labels={'y': '% Info Retained', 'x': 'Component'},
                                      color_discrete_sequence=['#636EFA'])
                    st.plotly_chart(fig_var2, use_container_width=True)
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

    # =========================================================================
    # STEP 9 — SUPERVISED OR UNSUPERVISED ANALYSIS
    # =========================================================================
    st.divider()
    st.header("9️⃣ Supervised or Unsupervised Analysis")
    analysis_type = st.radio(
        "Choose your analysis approach:",
        ["Supervised — predict the target from Step 7", "Unsupervised — discover structure, no target needed"],
        horizontal=True)
    is_supervised = analysis_type.startswith("Supervised")

    # =========================================================================
    # SUPERVISED PATH
    # =========================================================================
    if is_supervised:
        m_col1, m_col2 = st.columns([1, 2])

        with m_col1:
            st.write("#### ⚙️ Model Configuration")
            st.caption(f"Target: **{target}**")
            default_task = 1 if (is_numeric_target and unique_count > 20) else 0
            task = st.radio("Task Type:", ["Classification (Group)", "Regression (Value)"], index=default_task)
            clf = is_clf(task)

            if not clf and not is_numeric_target:
                st.error("Regression needs a numeric target. Switch to Classification, or pick a "
                        "numeric target in Step 7.")

            all_predictors = [c for c in df_model.columns
                              if c != target and c not in raw_date_cols
            ]
            usable, dropped = screen_features(df_model, all_predictors)
            
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
                features = st.multiselect("Select Clues (X):", options=ordered)

            algo_names = list(build_candidates(clf, False).keys())
            algo = st.selectbox("Algorithm:", [AUTO] + algo_names)

            if algo == "Naive Bayes":
                st.warning("⚠️ **Gaussian Assumption:** Naive Bayes assumes your clues follow a Bell Curve. If data is skewed, accuracy will be low.")
            if algo == AUTO:
                st.caption(f"Will train and compare all {len(algo_names)} algorithms — this can take a while on large datasets.")

            tune = True
            depth = 5
            if algo == AUTO:
                st.caption("Every algorithm is tuned with cross-validation on the 80% training data, "
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
                        work = df_model.dropna(subset=[target]).copy()
                        class_names = None

                        if clf:
                            y_txt = work[target].astype(str)
                            vc2 = y_txt.value_counts()
                            rare = vc2[vc2 < MIN_CLASS_ROWS].index
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
                                  "class_names": class_names, "auto": algo == AUTO,
                                  "total_missing": total_missing})
                        st.session_state.ml_results = res
                    except Exception as e:
                        st.error(f"⚠️ Error: {str(e)}")

        with m_col2:
            if st.session_state.ml_results:
                res = st.session_state.ml_results
                clf_r = is_clf(res['task'])
                pipe = res['pipe']
                y_test, preds = res['y_test'], res['preds']

                # =================================================================
                # STEP 10 — ALGORITHM AND FEATURE COMPARISON
                # =================================================================
                st.write(f"## 🔟 Algorithm & Feature Comparison — {res['target']}")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Training rows (80%)", f"{res['n_train']:,}")
                d2.metric("Test rows (20%)", f"{res['n_test']:,}")
                d3.metric("Clues used", f"{len(res['features'])}")
                d4.metric("Missing values fixed", f"{res.get('total_missing', 0):,}")

                headline = "Best algorithm out of everything compared" if res['auto'] else "Selected algorithm"
                settings = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in res['params'].items())
                st.success(f"🏆 **{headline}: {res['algo']}**" + (f" — tuned settings: {settings}" if settings else ""))

                lb = res['leaderboard']
                cv_col = f"CV score ({res['scoring']})"
                if res['auto'] and len(lb) > 1:
                    runner_up = lb.iloc[1]
                    margin = res['cv_mean'] - float(runner_up[cv_col])
                    st.write(
                        f"Out of **{len(lb)}** algorithms compared, **{res['algo']}** scored highest on "
                        f"cross-validation ({res['cv_mean']:.3f} vs **{runner_up['Algorithm']}**'s "
                        f"{float(runner_up[cv_col]):.3f} — a margin of {margin:.3f}). The winner is always "
                        "chosen from cross-validation on the training data; the test set is never used to pick it.")
                    pct_cols = [c for c in lb.columns if c.startswith("CV score") or c in ("CV ± std", "Test score", "Train score", "Overfit gap")]
                    fig_model_comparison = px.bar(
                        lb, x="Algorithm", y=cv_col, color="Overfit gap",
                        hover_data=["Test score", "Train score", "CV ± std", "Time (s)"],
                        title="Cross-Validation Score and Overfitting Gap",
                        color_continuous_scale="RdYlBu_r", template="plotly_white")
                    fig_model_comparison.update_layout(xaxis_tickangle=-35)
                    st.plotly_chart(fig_model_comparison, use_container_width=True)
                    with st.expander("🏁 See the full algorithm comparison table"):
                        st.dataframe(lb.style.format({**{c: "{:.3f}" for c in pct_cols},
                                                      "Time (s)": "{:.1f}",
                                                      ("Balanced acc." if clf_r else "RMSE"): "{:.3f}"}),
                                     use_container_width=True, hide_index=True)
                elif res['auto']:
                    st.write("Only one algorithm could be compared, so it was selected by default.")
                else:
                    st.write(f"You manually chose **{res['algo']}**" +
                             (f", tuned automatically to: {settings}." if settings else " with default settings."))

                if clf_r:
                    st.write(f"- {len(res['class_names'])} classes; baseline "
                            f"('always guess the most common class') scores {res['baseline']:.2%}.")

                # =================================================================
                # STEP 11 — MODEL VALIDATION
                # =================================================================
                st.divider()
                st.write("## 1️⃣1️⃣ Model Validation")
                st.caption("Does this model's score hold up, and can we trust it on new data?")

                if clf_r:
                    test_main = balanced_accuracy_score(y_test, preds) if res['balanced'] else accuracy_score(y_test, preds)
                    score_name = "balanced accuracy" if res['balanced'] else "accuracy"
                else:
                    test_main = r2_score(y_test, preds)
                    score_name = "R²"
                gap = float(lb.iloc[0]["Overfit gap"]) if len(lb) else 0.0

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Cross-validation score", f"{res['cv_mean']:.2%}" if clf_r else f"{res['cv_mean']:.3f}",
                          f"± {res['cv_std']:.3f}", delta_color="off")
                sc2.metric(f"Test score ({score_name})", f"{test_main:.2%}" if clf_r else f"{test_main:.3f}")
                sc3.metric("Train − test gap", f"{gap:.3f}")
                st.caption("The cross-validation score picked the winner on training data only; the test score is "
                          "this model's performance on the 20% of rows it never saw during training or tuning.")

                bits = [res['split_note']]
                if clf_r and res['balanced']:
                    bits.append(f"⚖️ Classes are imbalanced (largest/smallest ≈ {res['imbalance_ratio']:.1f}x), "
                               "so class weights were applied and models were ranked by balanced accuracy.")
                if gap > 0.10:
                    bits.append(f"⚠️ The train/test gap is {gap:.1%} — a sign of overfitting. Treat the score with caution.")
                elif gap > 0.05:
                    bits.append(f"The train/test gap is {gap:.1%}, a normal amount of overfitting.")
                else:
                    bits.append(f"✅ The train/test gap is only {gap:.1%}, so the model should generalize well to new data.")
                if clf_r:
                    beat_baseline = test_main - res['baseline']
                    bits.append(f"It beats the 'always guess the most common class' baseline "
                               f"({res['baseline']:.2%}) by {beat_baseline:.2%} points.")
                for b in bits:
                    st.write(f"- {b}")

                if clf_r:
                    acc = accuracy_score(y_test, preds)
                    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

                    met1, met2, met3, met4 = st.columns(4)
                    met1.metric("Accuracy", f"{acc:.2%}")
                    met2.metric("Precision", f"{prec:.2%}")
                    met3.metric("Recall", f"{rec:.2%}")
                    met4.metric("F1-Score", f"{f1:.2%}")

                    with st.expander("📖 What do these scores mean?"):
                        st.markdown("""
                        * **Accuracy:** Overall correctness.
                        * **Precision:** Quality of 'Positive' guesses (Low precision = many false alarms).
                        * **Recall:** Ability to find all 'Positive' cases (Low recall = many missed cases).
                        * **F1-Score:** The 'Harmonic Mean' of Precision and Recall. Best for imbalanced data.
                        """)

                    report = classification_report(y_test, preds, target_names=res['class_names'],
                                                    zero_division=0, output_dict=True)
                    report_df = pd.DataFrame(report).T
                    report_df = report_df.drop(index=[i for i in ["accuracy"] if i in report_df.index])
                    st.write("**Class-by-class performance:**")
                    st.dataframe(report_df.style.format({"precision": "{:.2%}", "recall": "{:.2%}",
                                                          "f1-score": "{:.2%}", "support": "{:.0f}"}),
                                 use_container_width=True)

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

                    mis_df = res['X_test'].copy().reset_index(drop=True)
                    mis_df["Actual"] = [res["class_names"][i] for i in y_test]
                    mis_df["Predicted"] = [res["class_names"][i] for i in preds]
                    mis_df = mis_df[mis_df["Actual"] != mis_df["Predicted"]]
                    st.caption(f"{len(mis_df):,} of {len(y_test):,} test rows were misclassified "
                             f"({len(mis_df) / len(y_test):.1%}).")
                    if len(mis_df):
                        with st.expander("Show misclassified rows"):
                            st.dataframe(mis_df.head(200), use_container_width=True)
                    else:
                        st.success("No misclassifications on the test set!")
                else:
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    mse = mean_squared_error(y_test, preds)
                    rmse = np.sqrt(mse)

                    reg_met1, reg_met2, reg_met3, reg_met4 = st.columns(4)
                    reg_met1.metric("R² Score", f"{r2:.3f}")
                    reg_met2.metric("MAE", f"{mae:.2f}")
                    reg_met3.metric("MSE", f"{mse:.2f}")
                    reg_met4.metric("RMSE", f"{rmse:.2f}")

                    with st.expander("📖 What do these regression scores mean?"):
                        st.markdown("""
                        * **R² Score:** How well the model fits the data (1.0 is perfect).
                        * **MAE (Mean Absolute Error):** The average 'distance' your prediction is from the truth.
                        * **MSE (Mean Squared Error):** Similar to MAE, but punishes large errors more heavily.
                        * **RMSE (Root Mean Squared Error):** The standard deviation of the residuals (errors).
                        """)

                    try:
                        buckets = pd.qcut(pd.Series(y_test).rank(method="first"), q=4,
                                          labels=["Lowest 25%", "Mid-low 25%", "Mid-high 25%", "Highest 25%"])
                        bucket_df = pd.DataFrame({"Bucket": buckets, "Actual": y_test, "Predicted": preds})
                        bucket_df["Abs Error"] = (bucket_df["Actual"] - bucket_df["Predicted"]).abs()
                        by_bucket = bucket_df.groupby("Bucket", observed=True).agg(
                            Rows=("Actual", "size"), MAE=("Abs Error", "mean"),
                            RMSE=("Abs Error", lambda s: np.sqrt((s ** 2).mean())))
                        st.write("**Performance by value range:**")
                        st.dataframe(by_bucket.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}),
                                    use_container_width=True)
                    except Exception:
                        st.info("Not enough spread in the target to break it into value ranges.")

                    fig_reg = px.scatter(x=y_test, y=preds, labels={'x': 'Actual Value', 'y': 'Predicted Value'},
                                        title="Actual vs Predicted Comparison")
                    fig_reg.add_shape(type="line", x0=float(np.min(y_test)), y0=float(np.min(y_test)),
                                      x1=float(np.max(y_test)), y1=float(np.max(y_test)),
                                      line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_reg, use_container_width=True)

                    residuals = y_test - preds
                    residual_df = pd.DataFrame({"Actual": y_test, "Predicted": preds, "Residual": residuals})
                    fig_residuals = px.scatter(
                        residual_df, x="Predicted", y="Residual", color="Residual",
                        color_continuous_scale="RdBu", title="Residual Plot",
                        labels={"Predicted": "Predicted value", "Residual": "Residual (Actual - Predicted)"},
                        template="plotly_white")
                    fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
                    st.plotly_chart(fig_residuals, use_container_width=True)
                    st.info("A good residual plot has points scattered randomly around zero. "
                           "Curves or funnel shapes may indicate nonlinearity, unequal variance, or missing features.")

                    err_df = res['X_test'].copy().reset_index(drop=True)
                    err_df["Actual"] = y_test
                    err_df["Predicted"] = preds
                    err_df["Abs Error"] = np.abs(y_test - preds)
                    err_df = err_df.sort_values("Abs Error", ascending=False)
                    with st.expander(f"Rows with the largest errors (top {min(200, len(err_df))})"):
                        st.dataframe(err_df.head(200), use_container_width=True)

                if res.get('learning_curve') is not None:
                    st.write("**Learning curve — does more data help?**")
                    lc = res['learning_curve']
                    fig_lc = px.line(lc, x="Training rows", y=["Train score", "Validation score"],
                                     markers=True, title="Score vs. Training Set Size",
                                     template="plotly_white")
                    st.plotly_chart(fig_lc, use_container_width=True)
                    gap_end = float(lc.iloc[-1]["Train score"] - lc.iloc[-1]["Validation score"])
                    if gap_end > 0.10:
                        st.info("The curves stay far apart even with more data — more rows alone probably "
                               "won't close the gap; better or cleaner features may help more.")
                    elif lc["Validation score"].iloc[-1] - lc["Validation score"].iloc[0] > 0.02:
                        st.info("The validation score is still rising — collecting more rows would likely help.")
                    else:
                        st.info("The validation score has largely flattened — more rows alone are unlikely to help much.")

                # =================================================================
                # STEP 12 — RESEARCH INTERPRETATION
                # =================================================================
                st.divider()
                st.write("## 1️⃣2️⃣ Research Interpretation")
                st.write(f"### {res['algo']} — How It Reasons")
                algo_r = res['algo']
                if algo_r == "Decision Tree":
                    render_tree(pipe, res['class_names'])
                else:
                    explain_algo(algo_r)
                if algo_r != "Decision Tree" and res.get('tree_pipe') is not None:
                    with st.expander("🌳 See the Decision Tree from this comparison"):
                        render_tree(res['tree_pipe'], res['class_names'])

                st.write("### Feature Influence")
                st.caption("Permutation importance on the test set: how much the score drops when a clue is shuffled.")
                if res['importance'] is not None:
                    imp_df = res['importance'].sort_values(by='Value').tail(25)
                    fig_imp = px.bar(imp_df, x='Value', y='Feature', orientation='h',
                                     color='Value', color_continuous_scale='Portland')
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Importance could not be computed for this model.")

                st.write("### Suggested Improvements")
                main_score = test_main
                if main_score >= EXCELLENT_SCORE:
                    st.success(f"✅ Excellent: {score_name} {main_score:.2%}" if clf_r else f"✅ Excellent: R² {main_score:.3f}")
                elif main_score >= TARGET_SCORE:
                    st.success(f"✅ Meets the {TARGET_SCORE:.0%} target: {score_name} {main_score:.2%}" if clf_r else
                              f"✅ Meets the {TARGET_SCORE:.2f} target: R² {main_score:.3f}")
                else:
                    st.warning(f"⚠️ Below the {TARGET_SCORE:.0%} target ({score_name} = {main_score:.2%})." if clf_r else
                              f"⚠️ Below the {TARGET_SCORE:.2f} target (R² = {main_score:.3f}).")
                tips = []
                if clf_r and main_score - res['baseline'] < 0.05:
                    tips.append(f"The model barely beats always guessing the most common class ({res['baseline']:.2%}). "
                              "The selected clues probably do not contain enough signal about the target.")
                if gap > 0.10:
                    tips.append(f"Overfitting: train score is {gap:.0%} higher than test. More rows or fewer/cleaner clues would help.")
                if clf_r and res['class_names'] and len(res['class_names']) > 10:
                    tips.append("Many classes make high accuracy much harder; merge similar classes if that makes sense.")
                if main_score > 0.995:
                    tips.append("🔎 Score is almost perfect. Double-check that no clue is a copy or a direct result of the target (data leakage).")
                if not tips:
                    tips.append("The model already looks solid. Consider trying additional feature engineering or "
                               "comparing against the Auto-Compare mode to squeeze out more performance.")
                tips.append("Add more informative columns, fix label noise, or check that the target is really "
                          "predictable from these clues. No algorithm can score higher than the data allows.")
                for t in tips:
                    st.write(f"* {t}")

                # =================================================================
                # STEP 13 — DOWNLOAD RESULTS AND METHODOLOGY
                # =================================================================
                st.divider()
                st.write("## 1️⃣3️⃣ Download Results & Methodology")
                dl1, dl2 = st.columns(2)
                export_df = res['X_test'].copy().reset_index(drop=True)
                if clf_r:
                    export_df["Actual"] = [res["class_names"][i] for i in y_test]
                    export_df["Predicted"] = [res["class_names"][i] for i in preds]
                    export_df["Correct"] = export_df["Actual"] == export_df["Predicted"]
                else:
                    export_df["Actual"] = y_test
                    export_df["Predicted"] = preds
                    export_df["Abs Error"] = np.abs(y_test - preds)
                export_csv = export_df.to_csv(index=False).encode("utf-8")
                with dl1:
                    st.download_button("📥 Download Test Set + Predictions (CSV)", data=export_csv,
                                      file_name=f"{res['algo'].replace(' ', '_').lower()}_predictions.csv",
                                      mime="text/csv")
                report_txt = build_methodology_report(res, "Supervised Learning", target=res['target'])
                with dl2:
                    st.download_button("📄 Download Methodology Report (Markdown)", data=report_txt.encode("utf-8"),
                                      file_name="methodology_report.md", mime="text/markdown")
                st.caption("The CSV is the untouched 20% test set with predictions attached, for row-by-row "
                          "inspection. The methodology report summarizes the split, algorithms compared, "
                          "selection criteria, and final scores.")
            else:
                st.info("Configure and train a model on the left to see Steps 10-13 here.")

    # =========================================================================
    # UNSUPERVISED PATH
    # =========================================================================
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("At least two numeric columns are required for unsupervised discovery.")
        else:
            left, right = st.columns([1, 2])

            with left:
                st.write("#### ⚙️ Discovery Settings")
                unsup_features = st.multiselect(
                    "Select numeric features:", numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))])
                if unsup_features:
                    st.caption("The discovered group descriptions will be based on: " + ", ".join(unsup_features))

                unsup_method = st.selectbox(
                    "Choose method:",
                    ["K-Means Clustering", "Mini-Batch K-Means", "Hierarchical Clustering",
                     "DBSCAN Clustering", "Birch Clustering", "Gaussian Mixture Clustering",
                     "Isolation Forest"])

                scale_features = st.checkbox(
                    "Standardize features", value=True,
                    help="Recommended when columns use different units.")

                n_clusters, eps, min_samples = 3, 0.5, 5
                if unsup_method in ["K-Means Clustering", "Mini-Batch K-Means", "Hierarchical Clustering",
                                    "Birch Clustering", "Gaussian Mixture Clustering"]:
                    n_clusters = st.slider("Number of groups:", min_value=2, max_value=10, value=3)
                if unsup_method == "DBSCAN Clustering":
                    eps = st.slider("Neighborhood radius:", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                    min_samples = st.slider("Minimum neighbors:", min_value=2, max_value=20, value=5)

                run_discovery = st.button("🚀 Discover Hidden Structure", type="primary")

            with right:
                if len(unsup_features) < 2:
                    st.info("Select at least two numeric features.")
                elif run_discovery:
                    try:
                        discovery = run_unsupervised_analysis(
                            df=df, features=unsup_features, method=unsup_method,
                            n_clusters=n_clusters, eps=eps, min_samples=min_samples,
                            scale_features=scale_features, random_state=SEED)

                        result_df = discovery["result"]
                        metrics = discovery["metrics"]
                        st.session_state.unsup_results = {
                            "result_df": result_df, "metrics": metrics, "method": unsup_method,
                            "features": unsup_features, "scale_features": scale_features,
                        }
                    except Exception as error:
                        st.error(f"Unsupervised analysis failed: {error}")
                        st.session_state.unsup_results = None

                unsup_state = st.session_state.get("unsup_results")
                if unsup_state:
                    result_df = unsup_state["result_df"]
                    metrics = unsup_state["metrics"]

                    # =============================================================
                    # STEP 10 — ALGORITHM COMPARISON (elbow chart for K-Means)
                    # =============================================================
                    st.write("## 🔟 Algorithm & Configuration Comparison")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Rows analyzed", f"{metrics['rows']:,}")
                    m2.metric("Features used", metrics["features"])
                    if unsup_method == "Isolation Forest":
                        m3.metric("Anomalies", f"{metrics['anomalies']:,}")
                        anomaly_rate = metrics["anomalies"] / metrics["rows"]
                        m4.metric("Anomaly rate", f"{anomaly_rate:.2%}")
                    else:
                        m3.metric("Groups found", metrics["groups"])
                        m4.metric("Silhouette", "N/A" if metrics["silhouette"] is None
                                 else f"{metrics['silhouette']:.3f}")

                    if unsup_method == "K-Means Clustering":
                        max_k = min(10, max(2, len(result_df) - 1))
                        k_values = list(range(2, max_k + 1))
                        elbow_rows = []
                        clean_data = result_df[unsup_features].replace([np.inf, -np.inf], np.nan)
                        clean_data = clean_data.fillna(clean_data.median(numeric_only=True))
                        elbow_X = StandardScaler().fit_transform(clean_data) if scale_features else clean_data.to_numpy()
                        for k in k_values:
                            temp_model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
                            temp_labels = temp_model.fit_predict(elbow_X)
                            temp_silhouette = silhouette_score(elbow_X, temp_labels)
                            elbow_rows.append({"Groups": k, "Inertia": temp_model.inertia_,
                                              "Silhouette": temp_silhouette})
                        elbow_df = pd.DataFrame(elbow_rows)
                        e1, e2 = st.columns(2)
                        with e1:
                            fig_elbow = px.line(elbow_df, x="Groups", y="Inertia", markers=True,
                                               title="Elbow Method — fewer groups vs. tighter fit", template="plotly_white")
                            st.plotly_chart(fig_elbow, use_container_width=True)
                        with e2:
                            fig_silhouette = px.line(elbow_df, x="Groups", y="Silhouette", markers=True,
                                                    title="Silhouette by Group Count", template="plotly_white")
                            st.plotly_chart(fig_silhouette, use_container_width=True)
                        st.info("The elbow is the point where adding more groups produces only a small "
                               "reduction in inertia. Higher silhouette values generally indicate "
                               "better-separated groups. This is your comparison across configurations.")
                    else:
                        st.info(f"You chose **{unsup_method}**, which evaluates one configuration at a time "
                               "rather than comparing many — its result is shown above.")

                    # =============================================================
                    # STEP 11 — MODEL VALIDATION (cluster quality)
                    # =============================================================
                    st.divider()
                    st.write("## 1️⃣1️⃣ Model Validation")
                    if unsup_method != "Isolation Forest":
                        with st.expander("📏 Clustering Quality Metrics", expanded=True):
                            metric_data = {
                                "Metric": ["Silhouette score", "Davies-Bouldin score", "Calinski-Harabasz score"],
                                "Value": [metrics["silhouette"], metrics["davies_bouldin"], metrics["calinski_harabasz"]],
                            }
                            st.dataframe(pd.DataFrame(metric_data), use_container_width=True, hide_index=True)
                            st.markdown("""
                            - **Silhouette:** Higher is generally better (closer to 1 = well-separated groups).
                            - **Davies-Bouldin:** Lower is generally better.
                            - **Calinski-Harabasz:** Higher is generally better.
                            """)
                        st.write("**Feature Distributions by Group**")
                        selected_box_feature = st.selectbox("Choose a feature:", unsup_features, key="unsup_box_feature")
                        fig_box = px.box(result_df, x="Group Label", y=selected_box_feature, color="Group Label",
                                        points=False, title=f"{selected_box_feature} by discovered profile",
                                        template="plotly_white")
                        fig_box.update_layout(xaxis_title="Discovered group profile", yaxis_title=selected_box_feature)
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.info("Isolation Forest doesn't produce cluster-quality metrics — its output is a "
                               "normal/anomaly flag per row, validated instead by the anomaly rate above.")

                    # =============================================================
                    # STEP 12 — RESEARCH INTERPRETATION
                    # =============================================================
                    st.divider()
                    st.write("## 1️⃣2️⃣ Research Interpretation")
                    if unsup_method != "Isolation Forest":
                        cluster_labels = metrics.get("cluster_labels", {})
                        if cluster_labels:
                            st.write("**What each group means:**")
                            for group_id, description in cluster_labels.items():
                                st.write(f"- **{description}**")
                            st.caption("These labels are generated from the selected features — they describe "
                                     "how each group's average feature values differ from the overall dataset average.")

                        color_column, title = "Group Label", f"{unsup_method} by Feature Profile (PCA Projection)"
                        st.write("**Group Profiles**")
                        profile = result_df.groupby("Group Label")[unsup_features].mean().round(3)
                        st.dataframe(profile, use_container_width=True)
                        profile_z = metrics.get("cluster_profile_z")
                        if profile_z is not None:
                            fig_profile = px.imshow(
                                profile_z.copy(), text_auto=".2f", aspect="auto",
                                color_continuous_scale="RdBu_r", title="Relative Group Profile",
                                labels={"x": "Selected Features", "y": "Discovered Groups", "color": "Relative level"})
                            st.plotly_chart(fig_profile, use_container_width=True)
                    else:
                        color_column, title = "Status", "Normal Rows and Anomalies (PCA Projection)"
                        st.write(f"**{metrics['anomalies']:,} rows** ({metrics['anomalies']/metrics['rows']:.1%}) "
                               "were flagged as anomalies — unusual combinations of the selected features.")

                    fig_map = px.scatter(
                        result_df, x="PCA 1", y="PCA 2", color=color_column,
                        hover_data=unsup_features + ["Row"], title=title, template="plotly_white")
                    fig_map.update_traces(marker={"size": 8, "opacity": 0.75})
                    st.plotly_chart(fig_map, use_container_width=True)
                    st.caption("Each point represents one row. Points near each other have similar feature patterns.")

                    # =============================================================
                    # STEP 13 — DOWNLOAD RESULTS AND METHODOLOGY
                    # =============================================================
                    st.divider()
                    st.write("## 1️⃣3️⃣ Download Results & Methodology")
                    ul1, ul2 = st.columns(2)
                    download_data = result_df.to_csv(index=False).encode("utf-8")
                    with ul1:
                        st.download_button(label="📥 Download Discovery Results (CSV)", data=download_data,
                                          file_name="unsupervised_discovery_results.csv", mime="text/csv")
                    report_txt = build_methodology_report(
                        {}, "Unsupervised Learning", target=None,
                        mode_extra={"method": unsup_method, "features": unsup_features,
                                   "scale_features": scale_features, "metrics": metrics})
                    with ul2:
                        st.download_button("📄 Download Methodology Report (Markdown)", data=report_txt.encode("utf-8"),
                                          file_name="methodology_report.md", mime="text/markdown")
                else:
                    st.info("Configure and run discovery on the left to see Steps 10-13 here.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | Educational ML Laboratory</p>", unsafe_allow_html=True)
