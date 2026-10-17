import json
import os
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

# ML Imports
from sklearn.base import BaseEstimator, RegressorMixin, clone
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
    GridSearchCV, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score,
    learning_curve, train_test_split,
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

# Time-series statistical models (Holt-Winters, ARIMA, SARIMA need `statsmodels`).
# If it is missing, the app installs it automatically the first time it runs.
# The rest of the app still works if the install is blocked.
SM_INSTALL_ERROR = None


def _import_statsmodels():
    global ExponentialSmoothing, ARIMA, SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX


try:
    _import_statsmodels()
    HAS_SM = True
except ImportError:
    HAS_SM = False
    try:
        import importlib
        import subprocess
        import sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "statsmodels"],
            stdout=subprocess.DEVNULL)
        importlib.invalidate_caches()
        _import_statsmodels()
        HAS_SM = True
    except Exception as _err:           # no internet, no permission, etc.
        SM_INSTALL_ERROR = str(_err)[:200]

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20            # default hold-out share (time-series TimeSeriesSplit does not use it)
MAX_CATEGORIES = 30         # categorical columns above this are treated as ID/free-text
MIN_CLASS_ROWS = 10         # classes rarer than this cannot be split/validated reliably
TUNE_ROWS = 6000            # rows used for hyperparameter search (keeps big CSVs fast)
CAP_ROWS = {"SVM": 25000, "KNN": 100000}   # final-fit caps for algorithms that scale badly
TARGET_SCORE = 0.80         # quality gate shown to the user
EXCELLENT_SCORE = min(0.95, TARGET_SCORE + 0.10)
AUTO = "🏆 Auto-Compare All (Recommended)"

# Validation strategy the user can pick for SUPERVISED and UNSUPERVISED analysis
SPLIT_80_20 = "80 / 20 split (train / test)"
SPLIT_70_30 = "70 / 30 split (train / test)"
SPLIT_CV = "Cross-validation (no fixed test set)"
SPLIT_CHOICES = {SPLIT_80_20: 0.20, SPLIT_70_30: 0.30, SPLIT_CV: None}
CV_ROWS_WARN = 20000        # cross-validation gets slow above this many rows

# Unsupervised: row caps for methods that scale badly (random sample is used above the cap)
UNSUP_CAP_ROWS = {"Hierarchical Clustering": 10000, "DBSCAN Clustering": 50000,
                  "Gaussian Mixture Clustering": 100000, "Birch Clustering": 200000}
SILHOUETTE_SAMPLE = 10000   # silhouette is O(n^2); score on a sample for big data

# Time-series constants
MAX_TS_ROWS = 20000         # keep only the most recent points (keeps lag models fast)
MIN_TS_ROWS = 40            # below this a forecast is not trustworthy
TS_FREQS = {                # label -> pandas frequency alias
    "Auto-detect": None, "Hourly": "h", "Daily": "D", "Weekly": "W",
    "Monthly": "MS", "Quarterly": "QS", "Yearly": "YS",
}


def recommend_split(n_rows):
    """Rule-of-thumb suggestion for the validation strategy; the user can always override it."""
    if n_rows < 1000:
        return SPLIT_CV, (f"💡 {n_rows:,} rows is small. A 20-30% test set would be only a few "
                          "hundred rows and a noisy score, so cross-validation (every row gets "
                          "tested) is recommended.")
    if n_rows < 10000:
        return SPLIT_70_30, (f"💡 {n_rows:,} rows is medium-sized. A larger 30% test set gives a "
                             "steadier score, especially for rarer classes.")
    return SPLIT_80_20, (f"💡 {n_rows:,} rows is large. Even 20% is a big, reliable test set, "
                         "so keeping more rows for training is the better trade.")


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


def _to_datetime_quiet(values, dayfirst=False):
    """pd.to_datetime that never raises and hides pandas' format-inference warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(values, errors="coerce", dayfirst=dayfirst)


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
            parsed_sample = _to_datetime_quiet(sample)
        else:
            parsed_sample = _to_datetime_quiet(sample.astype(str), dayfirst=dayfirst)

        parse_rate = parsed_sample.notna().mean() if len(sample) else 0.0

        looks_like_date_text = (
            is_text
            and len(sample) > 0
            and sample.astype(str).str.match(date_pattern).mean() >= parse_threshold
        )

        # Detect by column name or by recognizable date-formatted values.
        if not (
            already_datetime
            or name_hint.search(str(col))
            or (looks_like_date_text and parse_rate >= parse_threshold)
        ):
            continue

        parsed = _to_datetime_quiet(df[col], dayfirst=dayfirst)

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
        if res.get("split_mode") == "cv":
            split_lines = [
                f"- {res['split_note']}",
                f"- Rows: {res['n_train']:,}, evaluated with {res['cv_folds']}-fold nested cross-validation",
            ]
            sel_note = ("Selection and scoring both used nested cross-validation; tuning was repeated "
                        "inside each training fold.")
            eval_head = "## Final Evaluation (out-of-fold predictions from nested cross-validation)"
        else:
            split_lines = [
                f"- {res['split_note']}",
                f"- Training rows: {res['n_train']:,} ({res['train_pct']}%)",
                f"- Test rows: {res['n_test']:,} ({res['test_pct']}%, held out and untouched until final evaluation)",
            ]
            sel_note = ("Selection was based on cross-validation on the training data only "
                        f"({res['scoring']}); the test set was never used to choose the winner.")
            eval_head = "## Final Evaluation (test set, used once)"
        lines += [
            f"**Target variable:** {target}",
            f"**Task type:** {res['task']}",
            f"**Features (clues) used:** {', '.join(res['features'])}",
            "",
            "## Data Split",
        ] + split_lines + [
            "",
            "## Algorithm Selection",
            f"- Algorithm(s) considered: {len(res['leaderboard'])}",
            f"- Selected algorithm: {res['algo']} — {settings_txt}",
            f"- {sel_note}",
            f"- Cross-validation score: {res['cv_mean']:.3f} (± {res['cv_std']:.3f})",
            "",
            eval_head,
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
        if m.get("sampled"):
            lines.append(f"- Note: a random sample of {m['rows']:,} rows was analyzed (method is too slow on the full data).")
        for key, label in [("groups", "Groups found"), ("anomalies", "Anomalies found"),
                           ("silhouette", "Silhouette score"), ("davies_bouldin", "Davies-Bouldin score"),
                           ("calinski_harabasz", "Calinski-Harabasz score")]:
            if key in m and m[key] is not None:
                lines.append(f"- {label}: {m[key]}")
        val = m.get("validation")
        if val is not None and len(val):
            if m.get("validation_mode") == "cv":
                lines += ["", "## Validation",
                          f"- {len(val)}-fold cross-validation (fit on training folds, scored on the held-out fold)"]
            else:
                pct = m.get("validation_test_pct", 20)
                lines += ["", "## Validation",
                          f"- Hold-out: fit on {100 - pct}% of rows, scored on the {pct}% held-out rows"]
            for col in val.columns:
                lines.append(f"- {col.replace('_', ' ').capitalize()} (mean): {val[col].mean():.3f}")
    lines += ["", "---", "Generated by the Machine Learning Intuition Lab."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 3b. METRIC-CALCULATION EXPLAINERS
# ----------------------------------------------------------------------------
def regression_calc_details(y_test, preds):
    """Break R²/MAE/MSE/RMSE down into the raw sums behind them, plus a small
    sample of rows, so the arithmetic can be shown to the user step by step."""
    y_test = np.asarray(y_test, dtype=float)
    preds = np.asarray(preds, dtype=float)
    errors = y_test - preds
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2
    n = len(y_test)
    mean_actual = float(y_test.mean()) if n else float("nan")
    sum_abs = float(abs_errors.sum())
    sum_sq = float(sq_errors.sum())
    ss_res = sum_sq
    ss_tot = float(((y_test - mean_actual) ** 2).sum())
    mae = sum_abs / n if n else float("nan")
    mse = sum_sq / n if n else float("nan")
    rmse = np.sqrt(mse) if n else float("nan")
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    k = min(5, n)
    sample = pd.DataFrame({
        "Actual": y_test[:k], "Predicted": preds[:k],
        "Error": errors[:k], "|Error|": abs_errors[:k], "Error²": sq_errors[:k],
    })
    return {"n": n, "sum_abs": sum_abs, "sum_sq": sum_sq, "mean_actual": mean_actual,
            "ss_res": ss_res, "ss_tot": ss_tot, "mae": mae, "mse": mse, "rmse": rmse,
            "r2": r2, "sample": sample}


def classification_calc_details(y_test, preds, class_names):
    """Break Accuracy/Precision/Recall/F1 down into the raw counts behind them
    (correct/incorrect, and TP/FP/FN/TN for a binary target) for display."""
    y_test = np.asarray(y_test)
    preds = np.asarray(preds)
    n = len(y_test)
    correct = int((y_test == preds).sum())
    details = {"n": n, "correct": correct, "incorrect": n - correct, "binary": False}
    if class_names and len(class_names) == 2:
        pos = 1  # the second class, matching how LabelEncoder orders classes elsewhere in the app
        tp = int(((y_test == pos) & (preds == pos)).sum())
        fp = int(((y_test != pos) & (preds == pos)).sum())
        fn = int(((y_test == pos) & (preds != pos)).sum())
        tn = int(((y_test != pos) & (preds != pos)).sum())
        details.update({"binary": True, "pos_label": class_names[pos],
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return details


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


def train_and_compare(X, y, clf, algos, tune, manual_depth, on_progress,
                      test_size=0.20, cv_folds=5):
    """test_size = 0.20 / 0.30  -> stratified hold-out split (leak-free pipelines, CV tuning on the
                                   TRAIN part only, one honest evaluation on the untouched TEST part).
    test_size = None            -> nested cross-validation: every row is tested once by a model that
                                   never saw it, and tuning is repeated inside each training fold."""
    cv_only = test_size is None

    # ---- 1. split ----------------------------------------------------------------
    if cv_only:
        X_tr, y_tr = X, y
        X_te, y_te = X, y            # predictions below are out-of-fold
        split_note = (f"No fixed test set: {cv_folds}-fold cross-validation. Every row is tested "
                      "once by a model that never saw it, and tuning is repeated inside each "
                      "training fold (nested CV).")
    else:
        if clf:
            strat = y
            split_note = "Stratified by class: every class keeps the same share in train and test."
        else:
            strat = pd.qcut(pd.Series(y).rank(method="first"), q=10, labels=False).values
            split_note = "Stratified by target deciles: train and test cover the same value range."
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=SEED, stratify=strat)

    # ---- 2. imbalance handling & scoring rule ------------------------------------
    balanced, ratio, counts = False, 1.0, None
    if clf:
        counts = np.bincount(y_tr)
        counts = counts[counts > 0]
        ratio = counts.max() / counts.min()
        balanced = ratio > 1.5
        scoring = "balanced_accuracy" if balanced else "accuracy"
        score_fn = balanced_accuracy_score if balanced else accuracy_score
    else:
        scoring = "r2"
        score_fn = r2_score

    # ---- 3. tuning subset + inner CV ---------------------------------------------
    X_t, y_t = subsample(X_tr, y_tr, TUNE_ROWS, clf)
    if clf:
        n_splits = int(max(2, min(5, np.bincount(y_t)[np.bincount(y_t) > 0].min())))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # outer CV folds (only used in cross-validation mode)
    n_outer = int(max(2, min(cv_folds, counts.min()))) if clf else int(max(2, cv_folds))
    if cv_only:
        outer_cv = (StratifiedKFold(n_outer, shuffle=True, random_state=SEED + 1) if clf
                    else KFold(n_outer, shuffle=True, random_state=SEED + 1))

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
        pipe.fit(Xf, yf)                      # final model (fit on all training rows)

        if cv_only:
            # nested CV: tuning is redone inside every outer training fold
            fresh_est, _ = build_candidates(clf, balanced)[name]
            fresh = Pipeline([("prep", clone(pre)), ("model", fresh_est)])
            if name == "Decision Tree" and not (tune and grid):
                fresh.set_params(model__max_depth=manual_depth)
            inner_est = (GridSearchCV(fresh, grid, cv=cv, scoring=scoring, n_jobs=2,
                                      refit=True, error_score=np.nan)
                         if tune and grid else fresh)
            oof = np.zeros(len(y_tr), dtype=y_tr.dtype)
            fold_scores = []
            for tr_i, te_i in outer_cv.split(X_tr, y_tr):
                m = clone(inner_est).fit(X_tr.iloc[tr_i], y_tr[tr_i])
                p = m.predict(X_tr.iloc[te_i])
                oof[te_i] = p
                fold_scores.append(score_fn(y_tr[te_i], p))
            preds = oof
            cv_mean, cv_std = float(np.mean(fold_scores)), float(np.std(fold_scores))
        else:
            preds = pipe.predict(X_te)

        tr_preds = pipe.predict(X_tr_probe)
        if clf:
            test_score = score_fn(y_te, preds)
            train_score = score_fn(y_tr_probe, tr_preds)
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
            "Best settings": ", ".join(f"{k_.split('__')[-1]}={v}" for k_, v in best_params.items()) or "defaults",
        })

        # winner is chosen on CROSS-VALIDATION (never on a held-out test set)
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
            train_sizes=np.linspace(0.2, 1.0, 5), shuffle=True, random_state=SEED)
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
        # validation-strategy info (used by the display + report)
        "split_mode": "cv" if cv_only else "holdout",
        "cv_folds": n_outer if cv_only else None,
        "train_pct": 100 if cv_only else round((1 - test_size) * 100),
        "test_pct": 0 if cv_only else round(test_size * 100),
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
    elif algo_r in ["Neural Network", "MLPClassifier", "MLPRegressor"]:
        with st.expander("🔍 The 'Layered Weights' Logic", expanded=True):
            st.write("Passes the clues through layers of weighted connections, learning curved "
                     "(non-linear) patterns that straight-line models miss.")


def render_algorithm_visualization(algo_name, pipe, X_test, y_test, res):
    """Educational algorithm visualization (each one wrapped so a failure never breaks the app)."""
    try:
        model = pipe.named_steps["model"]
        class_names = res.get("class_names")
        is_class_task = is_clf(res["task"])

        # ---- Linear family: coefficient bar chart ---------------------------------
        if algo_name in ("Linear/Logistic Regression", "Linear Regression",
                         "Ridge Regression", "Lasso Regression", "Elastic Net"):
            if hasattr(model, "coef_"):
                coef = np.asarray(model.coef_)
                note = "Feature Weights"
                if coef.ndim > 1 and coef.shape[0] > 1:
                    coef = np.abs(coef).mean(axis=0)          # multi-class: average strength
                    note = "Average Absolute Feature Weight (all classes)"
                else:
                    coef = coef.ravel()
                features = feature_names_of(pipe)
                coef_df = pd.DataFrame({"Feature": features[:len(coef)], "Coefficient": coef})
                coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values().index).tail(25)
                fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                             title=note, color="Coefficient", color_continuous_scale="RdBu")
                st.plotly_chart(fig, use_container_width=True)

        # ---- LDA -----------------------------------------------------------------
        elif algo_name == "Linear Discriminant Analysis":
            X_prepared = pipe.named_steps["prep"].transform(X_test)
            X_lda = model.transform(X_prepared)
            plot_df = pd.DataFrame({"LDA1": X_lda[:, 0],
                                    "Class": [class_names[i] for i in y_test]})
            if X_lda.shape[1] > 1:
                plot_df["LDA2"] = X_lda[:, 1]
                fig = px.scatter(plot_df, x="LDA1", y="LDA2", color="Class", title="LDA Class Separation")
            else:
                fig = px.strip(plot_df, x="LDA1", color="Class", title="LDA Class Separation")
            st.plotly_chart(fig, use_container_width=True)

        # ---- Naive Bayes ---------------------------------------------------------
        elif algo_name == "Naive Bayes":
            if hasattr(model, "theta_"):
                features = feature_names_of(pipe)
                fig = go.Figure()
                for idx, cls in enumerate(class_names):
                    fig.add_trace(go.Bar(x=features[:len(model.theta_[idx])],
                                         y=model.theta_[idx], name=str(cls)))
                fig.update_layout(barmode="group", title="Mean Feature Values Per Class (standardized)")
                st.plotly_chart(fig, use_container_width=True)

        # ---- KNN -----------------------------------------------------------------
        elif algo_name == "KNN":
            X_vis = pipe.named_steps["prep"].transform(X_test)
            coords = PCA(n_components=2, random_state=SEED).fit_transform(X_vis)
            plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1]})
            if is_class_task and class_names:
                plot_df["Class"] = [class_names[i] for i in y_test]
                fig = px.scatter(plot_df, x="PC1", y="PC2", color="Class", title="KNN Neighbor Space")
            else:
                plot_df["Actual"] = np.asarray(y_test)
                fig = px.scatter(plot_df, x="PC1", y="PC2", color="Actual", title="KNN Neighbor Space")
            st.plotly_chart(fig, use_container_width=True)

        # ---- SVM -----------------------------------------------------------------
        elif algo_name == "SVM":
            inner = getattr(model, "regressor_", model)       # regression wraps SVR
            if hasattr(inner, "support_"):
                st.metric("Support Vectors", len(inner.support_))
            X_vis = pipe.named_steps["prep"].transform(X_test)
            coords = PCA(n_components=2, random_state=SEED).fit_transform(X_vis)
            fig = px.scatter(x=coords[:, 0], y=coords[:, 1], title="SVM Feature Space Projection",
                             labels={"x": "PC1", "y": "PC2"})
            st.plotly_chart(fig, use_container_width=True)

        # ---- Random Forest / Extra Trees ----------------------------------------
        elif algo_name in ("Random Forest", "Extra Trees"):
            depths = [t.tree_.max_depth for t in model.estimators_]
            fig = px.histogram(x=depths, nbins=20, title=f"{algo_name} — Tree Depth Distribution",
                               labels={"x": "Tree depth"})
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Trees Built", len(model.estimators_))

        # ---- Gradient Boosting ---------------------------------------------------
        elif algo_name == "Gradient Boosting":
            X_prep = pipe.named_steps["prep"].transform(X_test)
            scores = []
            for pred in model.staged_predict(X_prep):
                scores.append(accuracy_score(y_test, pred) if is_class_task else r2_score(y_test, pred))
            fig = px.line(y=scores, markers=True, title="Boosting Improvement by Stage",
                          labels={"x": "Boosting stage", "y": "Accuracy" if is_class_task else "R²"})
            st.plotly_chart(fig, use_container_width=True)

        # ---- HistGradient Boosting ----------------------------------------------
        elif algo_name == "HistGradient Boosting":
            numeric_cols = X_test.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                fig = px.histogram(X_test, x=numeric_cols[0], nbins=40,
                                   title=f"Histogram Binning Example ({numeric_cols[0]})")
                st.plotly_chart(fig, use_container_width=True)

        # ---- AdaBoost ------------------------------------------------------------
        elif algo_name == "AdaBoost":
            if hasattr(model, "estimator_weights_"):
                fig = px.line(y=model.estimator_weights_, markers=True,
                              title="Estimator Weight Through Boosting",
                              labels={"x": "Estimator", "y": "Weight"})
                st.plotly_chart(fig, use_container_width=True)

        # ---- Neural Network ------------------------------------------------------
        elif algo_name == "Neural Network":
            if hasattr(model, "loss_curve_"):
                fig = px.line(y=model.loss_curve_, title="Neural Network Learning Curve",
                              labels={"x": "Iteration", "y": "Loss"})
                st.plotly_chart(fig, use_container_width=True)

            hidden = model.hidden_layer_sizes
            hidden = [hidden] if isinstance(hidden, int) else list(hidden)
            n_out = len(class_names) if (is_class_task and class_names and len(class_names) > 2) else 1
            layers = [len(feature_names_of(pipe))] + hidden + [n_out]
            show_cap = 20                                    # keep the drawing readable
            nodes = []
            for layer_idx, count in enumerate(layers):
                shown = min(count, show_cap)
                for node in range(shown):
                    nodes.append({"Layer": layer_idx, "Node": node - shown / 2})
            vis_df = pd.DataFrame(nodes)
            fig = px.scatter(vis_df, x="Layer", y="Node", title="Neural Network Architecture",
                             labels={"Layer": "Layer (0 = inputs, last = output)", "Node": ""})
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Layer sizes: " + " → ".join(str(c) for c in layers)
                       + " (at most 20 nodes per layer are drawn).")

    except Exception as e:
        st.warning(f"Visualization unavailable: {e}")


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


def make_unsup_model(method, n_clusters, eps, min_samples, random_state):
    """One place that builds the unsupervised model, so validation and the final fit match."""
    if method == "K-Means Clustering":
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    if method == "Mini-Batch K-Means":
        return MiniBatchKMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    if method == "Hierarchical Clustering":
        return AgglomerativeClustering(n_clusters=n_clusters)
    if method == "Birch Clustering":
        return Birch(n_clusters=n_clusters)
    if method == "Gaussian Mixture Clustering":
        return GaussianMixture(n_components=n_clusters, random_state=random_state)
    if method == "DBSCAN Clustering":
        return DBSCAN(eps=eps, min_samples=min_samples)
    if method == "Isolation Forest":
        return IsolationForest(random_state=random_state, contamination="auto")
    raise ValueError(f"Unknown method: {method}")


def assign_groups(model, X_train, labels_train, X_new):
    """Put unseen rows into the groups found on the training rows."""
    if hasattr(model, "predict"):                  # KMeans, GMM, Birch, IsolationForest...
        return model.predict(X_new)
    # Hierarchical and DBSCAN cannot predict: use the nearest training rows instead
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, labels_train)
    return knn.predict(X_new)


def cluster_quality(X, labels, random_state):
    """Silhouette / Davies-Bouldin / Calinski-Harabasz (DBSCAN noise ignored)."""
    out = {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    try:
        mask = labels != -1
        if mask.sum() >= 2 and 2 <= len(set(labels[mask])) < mask.sum():
            sil_n = SILHOUETTE_SAMPLE if mask.sum() > SILHOUETTE_SAMPLE else None
            out["silhouette"] = float(silhouette_score(
                X[mask], labels[mask], sample_size=sil_n, random_state=random_state))
            out["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
            out["calinski_harabasz"] = float(calinski_harabasz_score(X[mask], labels[mask]))
    except Exception:
        pass
    return out


def validate_unsupervised(data, method, n_clusters, eps, min_samples, scale_features,
                          random_state, test_size=0.20, cv_folds=5):
    """Does the discovered structure hold up on rows the model did not fit?
    test_size = 0.20 / 0.30 -> one hold-out split.  test_size = None -> k-fold cross-validation.
    The scaler is fitted on the training part only, so held-out rows never leak in."""
    X_raw = data.to_numpy()
    n = len(X_raw)
    if n < 20:
        raise ValueError("Need at least 20 rows to validate.")
    if test_size is None:
        k = int(max(2, min(cv_folds, n // 10)))
        splits = list(KFold(k, shuffle=True, random_state=random_state).split(X_raw))
    else:
        tr, te = train_test_split(np.arange(n), test_size=test_size, random_state=random_state)
        splits = [(tr, te)]

    rows = []
    for tr, te in splits:
        sc = StandardScaler().fit(X_raw[tr]) if scale_features else None
        X_tr = sc.transform(X_raw[tr]) if sc else X_raw[tr]
        X_te = sc.transform(X_raw[te]) if sc else X_raw[te]

        model = make_unsup_model(method, n_clusters, eps, min_samples, random_state)
        lab_tr = model.fit_predict(X_tr)
        lab_te = assign_groups(model, X_tr, lab_tr, X_te)

        if method == "Isolation Forest":
            rows.append({"train_anomaly_rate": float((lab_tr == -1).mean()),
                         "test_anomaly_rate": float((lab_te == -1).mean())})
        else:
            q_tr = cluster_quality(X_tr, lab_tr, random_state)
            q_te = cluster_quality(X_te, lab_te, random_state)
            rows.append({"groups_found": len(set(lab_tr) - {-1}),
                         "train_silhouette": q_tr["silhouette"],
                         "test_silhouette": q_te["silhouette"],
                         "test_davies_bouldin": q_te["davies_bouldin"],
                         "test_calinski_harabasz": q_te["calinski_harabasz"]})
    return pd.DataFrame(rows)


def run_unsupervised_analysis(df, features, method, n_clusters, eps, min_samples,
                              scale_features, random_state, test_size=0.20, cv_folds=5):
    """Clean the chosen numeric columns, run the selected unsupervised method,
    and package a PCA projection plus quality metrics for display. The chosen
    validation strategy (hold-out or cross-validation) is run alongside."""
    data = df[features].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.median(numeric_only=True))

    # Some methods are O(n^2) in memory/time; analyze a random sample above the cap.
    sampled = False
    cap = UNSUP_CAP_ROWS.get(method)
    if cap is not None and len(data) > cap:
        data = data.sample(cap, random_state=random_state).sort_index()
        sampled = True

    X = StandardScaler().fit_transform(data) if scale_features else data.to_numpy()
    n_rows = len(data)

    model = make_unsup_model(method, n_clusters, eps, min_samples, random_state)
    labels = model.fit_predict(X)

    # 2D PCA projection purely for visualization
    n_comp = min(2, X.shape[1])
    coords = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

    result_df = data.copy()
    result_df["Row"] = data.index.to_numpy()          # original row number in the dataset
    result_df["PCA 1"] = coords[:, 0]
    result_df["PCA 2"] = coords[:, 1]

    metrics = {"rows": n_rows, "features": len(features), "sampled": sampled}

    # ---- validation on rows the model did not fit ------------------------------------
    try:
        metrics["validation"] = validate_unsupervised(
            data, method, n_clusters, eps, min_samples, scale_features,
            random_state, test_size=test_size, cv_folds=cv_folds)
        metrics["validation_mode"] = "cv" if test_size is None else "holdout"
        metrics["validation_test_pct"] = 0 if test_size is None else round(test_size * 100)
    except Exception as e:
        metrics["validation"] = None
        metrics["validation_error"] = str(e)[:200]

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
            sil_n = SILHOUETTE_SAMPLE if mask.sum() > SILHOUETTE_SAMPLE else None
            metrics["silhouette"] = float(silhouette_score(
                X[mask], labels[mask], sample_size=sil_n, random_state=random_state))
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
# 5b. TIME SERIES HELPERS  (validated with TimeSeriesSplit — separate from the
#     80/20 · 70/30 · cross-validation choice used by supervised/unsupervised)
# ----------------------------------------------------------------------------
def default_season(freq):
    """Typical repeating cycle length for a frequency."""
    f = str(freq)
    if f.lower().startswith("h"):
        return 24
    if f.startswith("D"):
        return 7
    if f.startswith("W"):
        return 52
    if f.startswith("M"):
        return 12
    if f.startswith("Q"):
        return 4
    return 1


def prepare_series(df, date_col, value_col, freq_label):
    """Turn two columns into a clean, evenly spaced, gap-free time series.
    Rows sharing a period are averaged; empty periods are interpolated."""
    d = df[[date_col, value_col]].copy()
    d[date_col] = _to_datetime_quiet(d[date_col])
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col)
    if d.empty:
        raise ValueError("No valid dates found in the chosen date column.")

    freq = TS_FREQS.get(freq_label)
    if freq is None:
        idx = pd.DatetimeIndex(d[date_col].unique())
        freq = pd.infer_freq(idx) if len(idx) >= 3 else None
        if freq is None:                       # irregular -> use the typical gap
            gap = pd.Series(idx).diff().median()
            if pd.isna(gap) or gap <= pd.Timedelta(hours=1):
                freq = "h"
            elif gap <= pd.Timedelta(days=1):
                freq = "D"
            elif gap <= pd.Timedelta(days=7):
                freq = "W"
            elif gap <= pd.Timedelta(days=31):
                freq = "MS"
            else:
                freq = "QS"

    s = d.set_index(date_col)[value_col].resample(freq).mean()
    n_filled = int(s.isna().sum())
    if s.notna().sum() == 0:
        raise ValueError("The value column has no numeric data.")
    s = s.interpolate(limit_direction="both").tail(MAX_TS_ROWS)
    s.name = value_col
    return s, freq, n_filled


def ts_metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    nz = y != 0
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAPE %": float(np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100) if nz.any() else np.nan,
        "R²": float(r2_score(y, p)),
    }


class BinnedClassifierRegressor(BaseEstimator, RegressorMixin):
    """Lets classifiers (LDA, Naive Bayes) forecast a number: the target is cut
    into quantile bins, the classifier predicts bin probabilities, and the
    forecast is the probability-weighted average of the bin means."""

    def __init__(self, classifier=None, n_bins=10):
        self.classifier = classifier
        self.n_bins = n_bins

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        n_bins = int(max(2, min(self.n_bins, len(y) // 20)))
        bins = pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates="drop").values
        self.model_ = clone(self.classifier).fit(X, bins)
        self.bin_means_ = np.array([y[bins == c].mean() for c in self.model_.classes_])
        return self

    def predict(self, X):
        return self.model_.predict_proba(X) @ self.bin_means_


def ts_ml_models():
    """Machine-learning forecasters that learn from lagged values."""
    scaled = lambda m: Pipeline([("sc", StandardScaler()), ("m", m)])
    scaled_y = lambda m: TransformedTargetRegressor(regressor=scaled(m), transformer=StandardScaler())
    return {
        "Linear/Logistic Regression (lags)": scaled(LinearRegression()),
        "Linear Discriminant Analysis (lags)": scaled(BinnedClassifierRegressor(LinearDiscriminantAnalysis())),
        "Decision Tree (lags)": DecisionTreeRegressor(max_depth=7, min_samples_leaf=5, random_state=SEED),
        "Naive Bayes (lags)": scaled(BinnedClassifierRegressor(GaussianNB())),
        "KNN (lags)": scaled(KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "SVM (lags)": scaled_y(SVR(C=5, epsilon=0.1)),
        "Random Forest (lags)": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
        "Extra Trees (lags)": ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
        "Gradient Boosting (lags)": GradientBoostingRegressor(random_state=SEED),
        "HistGradient Boosting (lags)": HistGradientBoostingRegressor(random_state=SEED),
        "AdaBoost (lags)": AdaBoostRegressor(n_estimators=100, random_state=SEED),
        "Neural Network (lags)": scaled_y(
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                         early_stopping=True, random_state=SEED)),
        "Ridge (lags)": scaled(Ridge(alpha=1.0)),
    }


def available_ts_models(season):
    names = ["Naive (last value)", "Seasonal Naive", "Moving Average"]
    if HAS_SM:
        names += ["Holt-Winters", "ARIMA"]
        if 2 <= season <= 12:
            names.append("SARIMA")
    return names + list(ts_ml_models().keys())


def lag_row(window, ts, n_lags):
    """Features for ONE future point, using only the previous n_lags values."""
    row = {f"lag_{i}": float(window[-i]) for i in range(1, n_lags + 1)}
    row["roll_mean"] = float(np.mean(window[-n_lags:]))
    row["month"] = ts.month
    row["dayofweek"] = ts.dayofweek
    row["hour"] = ts.hour
    return row


def build_lag_matrix(series, n_lags):
    vals, idx = series.values, series.index
    rows = [lag_row(vals[i - n_lags:i], idx[i], n_lags) for i in range(n_lags, len(vals))]
    return pd.DataFrame(rows), vals[n_lags:]


def ts_forecast(name, train, steps, season, n_lags, freq):
    """Fit `name` on `train` and forecast the next `steps` points."""
    vals = train.values.astype(float)

    if name == "Naive (last value)":
        return np.repeat(vals[-1], steps)

    if name == "Seasonal Naive":
        s = season if (season > 1 and len(vals) >= season) else 1
        last = vals[-s:]
        return np.array([last[i % s] for i in range(steps)])

    if name == "Moving Average":
        w = max(2, min(season if season > 1 else 7, len(vals)))
        return np.repeat(vals[-w:].mean(), steps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if name == "Holt-Winters":
            seasonal = season > 1 and len(vals) >= 2 * season
            model = ExponentialSmoothing(
                train, trend="add",
                seasonal="add" if seasonal else None,
                seasonal_periods=season if seasonal else None).fit()
            return np.asarray(model.forecast(steps))

        if name == "ARIMA":
            return np.asarray(ARIMA(train, order=(2, 1, 2)).fit().forecast(steps))

        if name == "SARIMA":
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, season),
                            enforce_stationarity=False, enforce_invertibility=False)
            return np.asarray(model.fit(disp=False).forecast(steps))

        if name in ts_ml_models():
            if len(train) < n_lags + 10:
                raise ValueError("Series too short for this many lags.")
            X, y = build_lag_matrix(train, n_lags)
            model = clone(ts_ml_models()[name]).fit(X, y)
            history = list(vals[-n_lags:])
            future_idx = pd.date_range(train.index[-1], periods=steps + 1, freq=freq)[1:]
            out = []
            for ts in future_idx:               # recursive multi-step forecast
                pred = float(model.predict(pd.DataFrame([lag_row(history, ts, n_lags)]))[0])
                out.append(pred)
                history = history[1:] + [pred]
            return np.array(out)

    raise ValueError(f"Unknown time-series model: {name}")


def ts_folds(n_points, n_splits, n_lags):
    """Expanding-window folds from TimeSeriesSplit: each fold trains on everything
    BEFORE its test block, so the model never sees the future. Folds whose
    training part is too short for the lag models are dropped."""
    n_splits = int(max(2, min(n_splits, n_points // 10)))
    tss = TimeSeriesSplit(n_splits=n_splits)
    return [(tr, te) for tr, te in tss.split(np.arange(n_points))
            if len(tr) >= n_lags + 10 and len(te) >= 2]


def run_time_series(series, model_names, season, n_lags, horizon, freq, on_progress,
                    test_size=None, n_splits=5):
    """Chronological validation with TimeSeriesSplit (expanding window, never shuffled).
    test_size=None (default) -> the score is the average over the folds, and the plots stitch
                                each fold's out-of-sample forecast together.
    test_size=0.20/0.30      -> optional final untouched test slice; the winner is picked with
                                TimeSeriesSplit on the training part."""
    cv_only = test_size is None
    n = len(series)
    if cv_only:
        train, test = series, None
    else:
        n_test = max(int(round(n * test_size)), 2)
        train, test = series.iloc[:-n_test], series.iloc[-n_test:]

    folds = ts_folds(len(train), n_splits, n_lags)
    if not folds:
        raise ValueError("Not enough history for this many folds/lags. Reduce the folds or lags, "
                         "or use more data.")

    rows, test_preds, skipped = [], {}, []
    for i, name in enumerate(model_names):
        on_progress(i / len(model_names), f"Training {name} ({i + 1}/{len(model_names)})...")
        t0 = time.time()
        try:
            fold_m, fold_p = [], []
            for tr, te in folds:
                p = ts_forecast(name, train.iloc[tr], len(te), season, n_lags, freq)
                fold_m.append(ts_metrics(train.iloc[te].values, p))
                fold_p.append(p)
            if not cv_only:
                p_test = ts_forecast(name, train, len(test), season, n_lags, freq)
        except Exception as e:
            skipped.append((name, str(e)[:120]))
            continue

        fold_rmse = [m["RMSE"] for m in fold_m]
        if cv_only:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = {k: float(np.nanmean([f[k] for f in fold_m])) for k in fold_m[0]}
            test_preds[name] = np.concatenate(fold_p)
        else:
            m = ts_metrics(test.values, p_test)
            test_preds[name] = p_test

        rows.append({
            "Algorithm": name,
            "Validation RMSE": float(np.mean(fold_rmse)),
            "Validation ± std": float(np.std(fold_rmse)),
            "Test RMSE": m["RMSE"], "Test MAE": m["MAE"],
            "Test MAPE %": m["MAPE %"], "Test R²": m["R²"],
            "Time (s)": time.time() - t0,
        })
    on_progress(1.0, "Done!")

    if not rows:
        raise ValueError("No algorithm could be fitted on this series.")

    board = pd.DataFrame(rows).sort_values("Validation RMSE").reset_index(drop=True)
    winner = board.iloc[0]["Algorithm"]       # chosen on TimeSeriesSplit folds

    # baseline: "repeat the last value"
    if cv_only:
        base = [np.sqrt(mean_squared_error(
                    train.iloc[te].values,
                    ts_forecast("Naive (last value)", train.iloc[tr], len(te), season, n_lags, freq)))
                for tr, te in folds]
        baseline_rmse = float(np.mean(base))
        te_idx = np.concatenate([te for _, te in folds])
        test_series = series.iloc[te_idx]
        train_series = series.iloc[:folds[0][1][0]]     # data before the first test fold
    else:
        baseline_pred = ts_forecast("Naive (last value)", train, len(test), season, n_lags, freq)
        baseline_rmse = float(np.sqrt(mean_squared_error(test.values, baseline_pred)))
        test_series, train_series = test, train

    future = ts_forecast(winner, series, horizon, season, n_lags, freq)
    future_idx = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
    win_row = board.iloc[0]

    return {
        "series": series, "train": train_series, "test": test_series, "leaderboard": board,
        "winner": winner, "test_pred": pd.Series(test_preds[winner], index=test_series.index),
        "forecast": pd.Series(future, index=future_idx),
        "band": 1.96 * float(win_row["Test RMSE"]),
        "baseline_rmse": baseline_rmse, "season": season, "n_lags": n_lags,
        "freq": freq, "n_train": len(train_series), "n_test": len(test_series), "skipped": skipped,
        "horizon": horizon,
        "split_mode": "cv" if cv_only else "holdout",
        "n_splits": len(folds),
        "train_pct": 100 if cv_only else round((1 - test_size) * 100),
        "test_pct": 0 if cv_only else round(test_size * 100),
    }


def build_ts_report(res, value_col, date_col, n_filled):
    b = res["leaderboard"].iloc[0]
    if res["split_mode"] == "cv":
        split_lines = [
            f"- TimeSeriesSplit with {res['n_splits']} expanding-window folds (chronological, never shuffled)",
            f"- Points tested across the folds: {res['n_test']:,}"]
        sel = ("- The winner was chosen on the average error across the folds. There is no separate "
               "test set, so the scores below are averages over the folds.")
        eval_head = "## Final Evaluation (average over TimeSeriesSplit folds)"
    else:
        split_lines = [
            f"- Training points: {res['n_train']:,} (earliest {res['train_pct']}%)",
            f"- Test points: {res['n_test']:,} (latest {res['test_pct']}%, untouched until final scoring)"]
        sel = (f"- The winner was chosen with {res['n_splits']}-fold TimeSeriesSplit on the training part, "
               "not on the test set.")
        eval_head = "## Final Evaluation (test set, used once)"
    lines = [
        "# Methodology Report — Time Series Forecasting", "",
        f"**Value forecasted:** {value_col}",
        f"**Date column:** {date_col}",
        f"**Frequency:** {res['freq']}  |  **Seasonal period:** {res['season']}  |  **Lags (ML models):** {res['n_lags']}",
        f"**Gaps filled by interpolation:** {n_filled:,}", "",
        "## Data Split (chronological — never shuffled)"] + split_lines + [
        "",
        "## Algorithm Selection",
        f"- Algorithms compared: {len(res['leaderboard'])}",
        sel,
        f"- Selected algorithm: {res['winner']}", "",
        eval_head,
        f"- RMSE: {b['Test RMSE']:.3f}",
        f"- MAE: {b['Test MAE']:.3f}",
        f"- MAPE: {b['Test MAPE %']:.2f}%",
        f"- R²: {b['Test R²']:.3f}",
        f"- Baseline (repeat last value) RMSE: {res['baseline_rmse']:.3f}", "",
        f"## Forecast\n- Next {res['horizon']} periods forecasted by refitting {res['winner']} on all data.",
        "- The shaded band is approximate (±1.96 × test RMSE).", "",
        "---", "Generated by the Machine Learning Intuition Lab."]
    return "\n".join(lines)


def render_time_series_section(df_raw, raw_date_cols, total_missing):
    """The whole Time Series workspace (Steps 10-13 for forecasting)."""
    st.info("⏱️ **Time series** data has an order in time, so the app never shuffles it. "
            "It validates with **TimeSeriesSplit** (each fold trains on the past and forecasts the "
            "next block), then forecasts the future.")

    date_options = list(raw_date_cols) or list(df_raw.columns)
    num_options = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not num_options:
        st.warning("A time series needs at least one numeric column to forecast.")
        return

    left, right = st.columns([1, 2])
    with left:
        st.write("#### ⚙️ Forecast Settings")
        if not raw_date_cols:
            st.caption("No date column was auto-detected — pick one manually.")
        date_col = st.selectbox("Date / time column:", date_options, key="ts_date")
        value_options = [c for c in num_options if c != date_col]
        if not value_options:
            st.warning("No numeric column available to forecast besides the date column.")
            return
        value_col = st.selectbox("Value to forecast:", value_options, key="ts_value")
        freq_label = st.selectbox("Data frequency:", list(TS_FREQS.keys()), key="ts_freq")

        try:
            series, freq, n_filled = prepare_series(df_raw, date_col, value_col, freq_label)
        except Exception as e:
            st.error(f"⚠️ {e}")
            return

        st.caption(f"✅ {len(series):,} points · frequency `{freq}` · "
                   f"{series.index.min():%Y-%m-%d} → {series.index.max():%Y-%m-%d}"
                   + (f" · {n_filled:,} empty periods interpolated" if n_filled else ""))
        if len(series) < MIN_TS_ROWS:
            st.error(f"Need at least {MIN_TS_ROWS} time points; this series has {len(series)}.")
            return

        season = st.number_input("Seasonal period (points per cycle):", 1, 400,
                                 default_season(freq), key=f"ts_season_{freq}",
                                 help="Daily data: 7 · Monthly: 12 · Hourly: 24 · Quarterly: 4")
        n_lags = st.number_input("Lags for ML models:", 2, 100,
                                 int(min(max(season, 7), 30)), key=f"ts_lags_{freq}",
                                 help="How many past values each ML model looks at.")
        horizon = st.slider("Forecast horizon (periods ahead):", 1, 365, int(min(max(season, 7), 30)))

        options = available_ts_models(int(season))
        chosen = st.multiselect("Algorithms to compare:", options, default=options)
        if not HAS_SM:
            st.caption("ℹ️ Holt-Winters, ARIMA and SARIMA are unavailable because `statsmodels` could not be "
                       "installed automatically. Run `pip install statsmodels` in the app's environment "
                       "and restart." + (f" (Details: {SM_INSTALL_ERROR})" if SM_INSTALL_ERROR else ""))

        st.write("**Validation (TimeSeriesSplit)**")
        ts_n_splits = st.slider("TimeSeriesSplit folds:", 2, 10, 5, key="ts_n_splits",
                                help="Each fold trains on everything before it and forecasts the next block.")
        if len(series) > 5000:
            st.warning("⚠️ Long series with many folds will be slow: every algorithm is refitted on each fold.")

        run = st.button("🚀 Train & Forecast", type="primary", key="ts_run")
        if run:
            if not chosen:
                st.error("Pick at least one algorithm.")
            else:
                try:
                    prog = st.progress(0.0, text="Starting...")
                    res = run_time_series(series, chosen, int(season), int(n_lags), int(horizon), freq,
                                          lambda p, msg: prog.progress(min(p, 1.0), text=msg),
                                          n_splits=int(ts_n_splits))
                    prog.empty()
                    res.update({"value_col": value_col, "date_col": date_col, "n_filled": n_filled})
                    st.session_state.ts_results = res
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

    with right:
        res = st.session_state.get("ts_results")
        if not res:
            st.info("Configure and run a forecast on the left to see Steps 10-13 here.")
            return

        board, winner = res["leaderboard"], res["winner"]
        b = board.iloc[0]

        # ---------------- STEP 10 ----------------
        st.write(f"## 🔟 Algorithm Comparison — {res['value_col']}")
        d1, d2, d3, d4 = st.columns(4)
        if res["split_mode"] == "cv":
            d1.metric("Points before 1st fold", f"{res['n_train']:,}")
            d2.metric("Points tested (all folds)", f"{res['n_test']:,}")
        else:
            d1.metric(f"Training points ({res['train_pct']}%)", f"{res['n_train']:,}")
            d2.metric(f"Test points ({res['test_pct']}%)", f"{res['n_test']:,}")
        d3.metric("Algorithms compared", len(board))
        d4.metric("Gaps interpolated", f"{res['n_filled']:,}")
        st.success(f"🏆 **Best algorithm: {winner}** — lowest average error across "
                   f"{res['n_splits']} TimeSeriesSplit folds.")
        if len(board) > 1:
            fig_cmp = px.bar(board, x="Algorithm", y="Validation RMSE", color="Test RMSE",
                             hover_data=["Validation ± std", "Test MAE", "Test MAPE %", "Test R²", "Time (s)"],
                             title="Validation RMSE by Algorithm (lower is better)",
                             color_continuous_scale="RdYlBu_r", template="plotly_white")
            fig_cmp.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_cmp, use_container_width=True)
        with st.expander("🏁 See the full algorithm comparison table"):
            st.dataframe(board.style.format({c: "{:.3f}" for c in board.columns if c != "Algorithm"}),
                         use_container_width=True, hide_index=True)
        for name, why in res["skipped"]:
            st.caption(f"⚠️ Skipped **{name}**: {why}")

        # ---------------- STEP 11 ----------------
        st.divider()
        st.write("## 1️⃣1️⃣ Model Validation")
        if res["split_mode"] == "cv":
            st.caption(f"Scores are averaged over {res['n_splits']} expanding-window folds. The chart stitches "
                       "together each fold's out-of-sample forecast.")
        else:
            st.caption(f"The winner is scored once on the most recent {res['test_pct']}% of the timeline, "
                       "which it never saw.")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("RMSE", f"{b['Test RMSE']:.3f}")
        v2.metric("MAE", f"{b['Test MAE']:.3f}")
        v3.metric("MAPE", "N/A" if pd.isna(b['Test MAPE %']) else f"{b['Test MAPE %']:.1f}%")
        v4.metric("R²", f"{b['Test R²']:.3f}")
        with st.expander("📖 What do these scores mean?"):
            st.markdown("""
            * **RMSE / MAE:** Typical forecast error in the same units as your data (lower is better).
            * **MAPE:** Average error as a percentage of the real value.
            * **R²:** 1.0 is perfect; 0 means no better than predicting the average; negative is worse than that.
            * **Baseline:** "Tomorrow equals today". A real model must beat this.
            """)

        gain = 1 - float(b["Test RMSE"]) / res["baseline_rmse"] if res["baseline_rmse"] else 0
        if gain > 0.05:
            st.success(f"✅ Beats the 'repeat the last value' baseline by {gain:.0%} (baseline RMSE {res['baseline_rmse']:.3f}).")
        else:
            st.warning(f"⚠️ Barely beats the 'repeat the last value' baseline (RMSE {res['baseline_rmse']:.3f}). "
                       "The series may be mostly random, or the seasonal period may be wrong.")

        fig_test = go.Figure()
        fig_test.add_scatter(x=res["train"].index[-3 * res["n_test"]:], y=res["train"].values[-3 * res["n_test"]:],
                             name="Training (recent)", line=dict(color="#9ca3af"))
        fig_test.add_scatter(x=res["test"].index, y=res["test"].values, name="Actual (tested)",
                             line=dict(color="#2563eb"))
        fig_test.add_scatter(x=res["test_pred"].index, y=res["test_pred"].values, name=f"{winner} prediction",
                             line=dict(color="#dc2626", dash="dash"))
        fig_test.update_layout(title="Actual vs Predicted on the Tested Period", template="plotly_white")
        st.plotly_chart(fig_test, use_container_width=True)

        resid = res["test"].values - res["test_pred"].values
        fig_res = px.scatter(x=res["test"].index, y=resid, template="plotly_white",
                             labels={"x": "Date", "y": "Error (Actual − Predicted)"}, title="Forecast Errors Over Time")
        fig_res.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_res, use_container_width=True)
        st.info("Errors should bounce randomly around zero. A steady drift or wave means the model is missing a trend or season.")

        # ---------------- STEP 12 ----------------
        st.divider()
        st.write("## 1️⃣2️⃣ Research Interpretation & Forecast")
        hist = res["series"].tail(max(res["horizon"] * 4, 100))
        fc = res["forecast"]
        fig_fc = go.Figure()
        fig_fc.add_scatter(x=hist.index, y=hist.values, name="History", line=dict(color="#2563eb"))
        fig_fc.add_scatter(x=fc.index, y=fc.values + res["band"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig_fc.add_scatter(x=fc.index, y=fc.values - res["band"], fill="tonexty", line=dict(width=0),
                           fillcolor="rgba(220,38,38,0.15)", name="Approx. 95% range")
        fig_fc.add_scatter(x=fc.index, y=fc.values, name="Forecast", line=dict(color="#dc2626"))
        fig_fc.update_layout(title=f"{res['horizon']}-period Forecast ({winner})", template="plotly_white")
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption("The shaded range is approximate (±1.96 × test RMSE). Forecasts get less reliable the further ahead they go.")

        trend = float(fc.iloc[-1] - res["series"].iloc[-1])
        st.write(f"- Last observed value: **{res['series'].iloc[-1]:,.2f}** → forecast at the end of the horizon: "
                 f"**{fc.iloc[-1]:,.2f}** ({'up' if trend >= 0 else 'down'} {abs(trend):,.2f}).")
        if "(lags)" in winner:
            st.write("- This is a **lag-based machine-learning model**: it learns how the last "
                     f"{res['n_lags']} values plus the calendar (month, weekday, hour) predict the next value, "
                     "then feeds its own predictions back in to look further ahead.")
        elif winner in ("Holt-Winters", "ARIMA", "SARIMA"):
            st.write("- This is a **classical statistical model**: it separates the series into level, trend "
                     "and repeating season (or past values and past errors) and extends them forward.")
        else:
            st.write("- A simple rule-based model won, which usually means the series has little structure beyond "
                     "its most recent level or its repeating season.")

        # ---------------- STEP 13 ----------------
        st.divider()
        st.write("## 1️⃣3️⃣ Download Results & Methodology")
        dl1, dl2 = st.columns(2)
        out = pd.DataFrame({"Date": fc.index, "Forecast": fc.values,
                            "Lower": fc.values - res["band"], "Upper": fc.values + res["band"]})
        with dl1:
            st.download_button("📥 Download Forecast (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                               file_name="forecast.csv", mime="text/csv")
        with dl2:
            st.download_button("📄 Download Methodology Report (Markdown)",
                               data=build_ts_report(res, res["value_col"], res["date_col"], res["n_filled"]).encode("utf-8"),
                               file_name="methodology_report.md", mime="text/markdown")


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
              "9. Supervised/Unsupervised/Time series · 10. Comparison · 11. Validation · "
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
                    if len(upper):
                        pair = upper.abs().sort_values(ascending=False).index[0]
                        st.success(f"**Insight:** Strongest relationship found between **{pair[0]}** and **{pair[1]}** ({corr_matrix.loc[pair]:.2f}).")

    # =========================================================================
    # STEP 9 — SUPERVISED, UNSUPERVISED OR TIME SERIES ANALYSIS
    # =========================================================================
    st.divider()
    st.header("9️⃣ Supervised, Unsupervised or Time Series Analysis")
    analysis_type = st.radio(
        "Choose your analysis approach:",
        ["Supervised — predict the target from Step 7",
         "Unsupervised — discover structure, no target needed",
         "Time Series — forecast a value over time"],
        horizontal=True)
    is_supervised = analysis_type.startswith("Supervised")
    is_timeseries = analysis_type.startswith("Time Series")

    # =========================================================================
    # TIME SERIES PATH  (validated with TimeSeriesSplit)
    # =========================================================================
    if is_timeseries:
        render_time_series_section(df_raw, raw_date_cols, total_missing)

    # =========================================================================
    # SUPERVISED PATH
    # =========================================================================
    elif is_supervised:
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
                              if c != target and c not in raw_date_cols]
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
                        # df_model (not df_raw): it also contains the derived date features
                        mi = rank_predictors(df_model, target, usable, clf)
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
                st.caption("Every algorithm is tuned with cross-validation on the training data, "
                          "then scored on rows it never trained on (see the validation strategy below). "
                          "The winner is picked by cross-validation, not by the test set.")
            else:
                tune = st.checkbox("Auto-tune hyperparameters (recommended)", value=True)
                if algo == "Decision Tree" and not tune:
                    depth = st.number_input("Select Max Tree Depth:", 1, 10, 5)

            # ---- validation strategy: 80/20, 70/30 or cross-validation ---------------
            st.write("**Validation strategy**")
            n_usable = int(df_model[target].notna().sum())
            rec_choice, rec_msg = recommend_split(n_usable)
            split_choices = list(SPLIT_CHOICES.keys())
            split_choice = st.selectbox("How should the data be split?", split_choices,
                                        index=split_choices.index(rec_choice), key="sup_split")
            st.caption(rec_msg)
            test_size_choice = SPLIT_CHOICES[split_choice]
            cv_folds = 5
            if test_size_choice is None:
                cv_folds = st.slider("Number of folds:", 3, 10, 5, key="sup_folds")
                if n_usable > CV_ROWS_WARN:
                    st.warning(f"⚠️ Nested cross-validation retrains every algorithm {cv_folds} times "
                               f"(plus tuning). With {n_usable:,} rows this will be slow, so a hold-out "
                               "split is usually the better choice here.")

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
                            lambda p, msg: prog.progress(min(p, 1.0), text=msg),
                            test_size=test_size_choice, cv_folds=cv_folds)
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
                if res["split_mode"] == "cv":
                    d1.metric("Rows (all used)", f"{res['n_train']:,}")
                    d2.metric("CV folds", f"{res['cv_folds']}")
                else:
                    d1.metric(f"Training rows ({res['train_pct']}%)", f"{res['n_train']:,}")
                    d2.metric(f"Test rows ({res['test_pct']}%)", f"{res['n_test']:,}")
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
                        "chosen from cross-validation; a separate test set is never used to pick it.")
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
                # Overfit gap of the WINNING algorithm (looked up by name, not by row position)
                winner_row = lb[lb["Algorithm"] == res['algo']]
                gap = float(winner_row.iloc[0]["Overfit gap"]) if len(winner_row) else 0.0

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Cross-validation score", f"{res['cv_mean']:.2%}" if clf_r else f"{res['cv_mean']:.3f}",
                          f"± {res['cv_std']:.3f}", delta_color="off")
                test_label = (f"Out-of-fold score ({score_name})" if res["split_mode"] == "cv"
                              else f"Test score ({score_name})")
                sc2.metric(test_label, f"{test_main:.2%}" if clf_r else f"{test_main:.3f}")
                sc3.metric("Train − test gap", f"{gap:.3f}")
                if res["split_mode"] == "cv":
                    st.caption("Nested cross-validation: every row was predicted by a model that never saw it, "
                              "and tuning was repeated inside each training fold. The train − test gap compares "
                              "the final model's score on its own training rows with that out-of-fold score.")
                else:
                    st.caption("The cross-validation score picked the winner on training data only; the test score is "
                              f"this model's performance on the {res['test_pct']}% of rows it never saw during "
                              "training or tuning.")

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

                    calc = classification_calc_details(y_test, preds, res['class_names'])
                    with st.expander("🧮 See the actual computation"):
                        st.write(f"Test set size: n = {calc['n']:,} rows")
                        st.markdown(
                            f"- **Accuracy** = correct / total = {calc['correct']:,} / {calc['n']:,} = **{acc:.2%}**"
                        )
                        if calc["binary"]:
                            st.write(f"Using **{calc['pos_label']}** as the positive class:")
                            cm_calc_df = pd.DataFrame({
                                "": ["Actual positive", "Actual negative"],
                                "Predicted positive": [calc['tp'], calc['fp']],
                                "Predicted negative": [calc['fn'], calc['tn']],
                            })
                            st.dataframe(cm_calc_df, use_container_width=True, hide_index=True)
                            st.markdown(f"""
- **Precision** = TP / (TP + FP) = {calc['tp']} / ({calc['tp']} + {calc['fp']}) = **{prec:.2%}**
- **Recall** = TP / (TP + FN) = {calc['tp']} / ({calc['tp']} + {calc['fn']}) = **{rec:.2%}**
- **F1** = 2 × (Precision × Recall) / (Precision + Recall) = 2 × ({prec:.3f} × {rec:.3f}) / ({prec:.3f} + {rec:.3f}) = **{f1:.2%}**
""")
                        else:
                            st.caption("With more than 2 classes, Precision/Recall/F1 above are a support-weighted "
                                      "average across all classes — see the class-by-class table below for each "
                                      "class's own numbers.")

                    report = classification_report(y_test, preds, labels=list(range(len(res['class_names']))),
                                                    target_names=res['class_names'],
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

                    reg_calc = regression_calc_details(y_test, preds)
                    with st.expander("🧮 See the actual computation"):
                        st.write(f"Test set size: n = {reg_calc['n']:,} rows")
                        st.dataframe(reg_calc['sample'].style.format("{:.3f}"),
                                    use_container_width=True, hide_index=True)
                        st.caption("First 5 test rows shown as an example; the metrics above use all rows.")
                        st.markdown(f"""
- **MAE** = Σ|actual − predicted| / n = {reg_calc['sum_abs']:.3f} / {reg_calc['n']:,} = **{reg_calc['mae']:.3f}**
- **MSE** = Σ(actual − predicted)² / n = {reg_calc['sum_sq']:.3f} / {reg_calc['n']:,} = **{reg_calc['mse']:.3f}**
- **RMSE** = √MSE = √{reg_calc['mse']:.3f} = **{reg_calc['rmse']:.3f}**
- **R²** = 1 − SS_res / SS_tot = 1 − {reg_calc['ss_res']:.3f} / {reg_calc['ss_tot']:.3f} = **{reg_calc['r2']:.3f}**
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
                render_algorithm_visualization(algo_r, pipe, res["X_test"], y_test, res)
                if algo_r == "Decision Tree":
                    render_tree(pipe, res['class_names'])
                else:
                    explain_algo(algo_r)
                if algo_r != "Decision Tree" and res.get('tree_pipe') is not None:
                    with st.expander("🌳 See the Decision Tree from this comparison"):
                        render_tree(res['tree_pipe'], res['class_names'])

                st.write("### Feature Influence")
                if res["split_mode"] == "cv":
                    st.caption("Permutation importance: how much the score drops when a clue is shuffled. "
                              "In cross-validation mode this is computed on all rows with the final model, "
                              "so treat it as indicative.")
                else:
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
                if res["split_mode"] == "cv":
                    st.caption("The CSV holds every row with its out-of-fold prediction (each row was predicted by "
                              "a model that never saw it). The methodology report summarizes the split, "
                              "algorithms compared, selection criteria, and final scores.")
                else:
                    st.caption(f"The CSV is the untouched {res['test_pct']}% test set with predictions attached, for "
                              "row-by-row inspection. The methodology report summarizes the split, algorithms "
                              "compared, selection criteria, and final scores.")
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

                if unsup_method in UNSUP_CAP_ROWS:
                    st.caption(f"ℹ️ For speed, {unsup_method} analyzes a random sample of at most "
                              f"{UNSUP_CAP_ROWS[unsup_method]:,} rows.")

                # ---- validation strategy: 80/20, 70/30 or cross-validation -----------
                st.write("**Validation strategy**")
                un_choices = list(SPLIT_CHOICES.keys())
                un_rec, un_msg = recommend_split(len(df))
                unsup_split = st.selectbox("How should the data be split?", un_choices,
                                           index=un_choices.index(un_rec), key="unsup_split")
                st.caption(un_msg)
                unsup_test_size = SPLIT_CHOICES[unsup_split]
                unsup_folds = 5
                if unsup_test_size is None:
                    unsup_folds = st.slider("Number of folds:", 3, 10, 5, key="unsup_folds")
                    if len(df) > CV_ROWS_WARN:
                        st.warning("⚠️ Cross-validation refits the method once per fold, so large files "
                                   "will be slow. A hold-out split is usually better here.")

                run_discovery = st.button("🚀 Discover Hidden Structure", type="primary")

            with right:
                if len(unsup_features) < 2:
                    st.info("Select at least two numeric features.")
                elif run_discovery:
                    try:
                        discovery = run_unsupervised_analysis(
                            df=df, features=unsup_features, method=unsup_method,
                            n_clusters=n_clusters, eps=eps, min_samples=min_samples,
                            scale_features=scale_features, random_state=SEED,
                            test_size=unsup_test_size, cv_folds=unsup_folds)

                        st.session_state.unsup_results = {
                            "result_df": discovery["result"], "metrics": discovery["metrics"],
                            "method": unsup_method, "features": list(unsup_features),
                            "scale_features": scale_features,
                        }
                    except Exception as error:
                        st.error(f"Unsupervised analysis failed: {error}")
                        st.session_state.unsup_results = None

                unsup_state = st.session_state.get("unsup_results")
                if unsup_state:
                    # Always use what was STORED with the results, never the live widgets
                    # (otherwise changing a dropdown before re-running breaks the display).
                    result_df = unsup_state["result_df"]
                    metrics = unsup_state["metrics"]
                    u_method = unsup_state["method"]
                    u_features = unsup_state["features"]
                    u_scale = unsup_state["scale_features"]

                    # =============================================================
                    # STEP 10 — ALGORITHM COMPARISON (elbow chart for K-Means)
                    # =============================================================
                    st.write("## 🔟 Algorithm & Configuration Comparison")
                    st.caption(f"Showing results for: **{u_method}** on {len(u_features)} features.")
                    if metrics.get("sampled"):
                        st.info(f"A random sample of {metrics['rows']:,} rows was analyzed because this method "
                               "is too slow on the full dataset.")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Rows analyzed", f"{metrics['rows']:,}")
                    m2.metric("Features used", metrics["features"])
                    if u_method == "Isolation Forest":
                        m3.metric("Anomalies", f"{metrics['anomalies']:,}")
                        anomaly_rate = metrics["anomalies"] / metrics["rows"]
                        m4.metric("Anomaly rate", f"{anomaly_rate:.2%}")
                    else:
                        m3.metric("Groups found", metrics["groups"])
                        m4.metric("Silhouette", "N/A" if metrics["silhouette"] is None
                                 else f"{metrics['silhouette']:.3f}")

                    if u_method == "K-Means Clustering":
                        max_k = min(10, max(2, len(result_df) - 1))
                        k_values = list(range(2, max_k + 1))
                        elbow_rows = []
                        clean_data = result_df[u_features].replace([np.inf, -np.inf], np.nan)
                        clean_data = clean_data.fillna(clean_data.median(numeric_only=True))
                        elbow_X = StandardScaler().fit_transform(clean_data) if u_scale else clean_data.to_numpy()
                        elbow_sil_n = SILHOUETTE_SAMPLE if len(elbow_X) > SILHOUETTE_SAMPLE else None
                        for k in k_values:
                            temp_model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
                            temp_labels = temp_model.fit_predict(elbow_X)
                            temp_silhouette = silhouette_score(elbow_X, temp_labels,
                                                               sample_size=elbow_sil_n, random_state=SEED)
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
                        st.info(f"You chose **{u_method}**, which evaluates one configuration at a time "
                               "rather than comparing many — its result is shown above.")

                    # =============================================================
                    # STEP 11 — MODEL VALIDATION (cluster quality + hold-out / CV check)
                    # =============================================================
                    st.divider()
                    st.write("## 1️⃣1️⃣ Model Validation")
                    if u_method != "Isolation Forest":
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
                        selected_box_feature = st.selectbox("Choose a feature:", u_features, key="unsup_box_feature")
                        fig_box = px.box(result_df, x="Group Label", y=selected_box_feature, color="Group Label",
                                        points=False, title=f"{selected_box_feature} by discovered profile",
                                        template="plotly_white")
                        fig_box.update_layout(xaxis_title="Discovered group profile", yaxis_title=selected_box_feature)
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.info("Isolation Forest doesn't produce cluster-quality metrics — its output is a "
                               "normal/anomaly flag per row, validated instead by the anomaly rate above.")

                    st.write("**Does the structure hold up on rows the model did not fit?**")
                    val = metrics.get("validation")
                    if val is None:
                        st.info(f"Validation was not available: {metrics.get('validation_error', 'unknown reason')}")
                    else:
                        if metrics.get("validation_mode") == "cv":
                            st.caption(f"{len(val)}-fold cross-validation: each fold fits on the other folds, "
                                       "then scores the held-out fold (mean across folds shown).")
                        else:
                            pct = metrics["validation_test_pct"]
                            st.caption(f"Fit on {100 - pct}% of the rows, then scored on the {pct}% held-out rows "
                                       "(new rows join the nearest discovered group).")

                        if u_method == "Isolation Forest":
                            a1, a2 = st.columns(2)
                            a1.metric("Anomaly rate — fit rows", f"{val['train_anomaly_rate'].mean():.2%}")
                            a2.metric("Anomaly rate — held-out rows", f"{val['test_anomaly_rate'].mean():.2%}")
                            st.caption("Similar rates mean the detector flags unusual rows consistently. "
                                       "A big difference means its notion of 'normal' does not carry over to new rows.")
                        else:
                            tr_sil = val["train_silhouette"].mean()
                            te_sil = val["test_silhouette"].mean()
                            a1, a2, a3 = st.columns(3)
                            a1.metric("Silhouette — fit rows", "N/A" if pd.isna(tr_sil) else f"{tr_sil:.3f}")
                            a2.metric("Silhouette — held-out rows", "N/A" if pd.isna(te_sil) else f"{te_sil:.3f}")
                            te_db = val["test_davies_bouldin"].mean()
                            a3.metric("Davies-Bouldin — held-out", "N/A" if pd.isna(te_db) else f"{te_db:.3f}")
                            if not (pd.isna(tr_sil) or pd.isna(te_sil)):
                                if tr_sil - te_sil > 0.10:
                                    st.warning("⚠️ Groups are noticeably less clear on held-out rows — the structure may be "
                                               "specific to the rows it was fitted on. Try fewer groups or different features.")
                                else:
                                    st.success("✅ Group quality on held-out rows is close to the fit rows, so the structure generalizes.")
                            if metrics.get("validation_mode") == "cv" and len(val) > 1 and not pd.isna(te_sil):
                                st.caption(f"Held-out silhouette varies by ±{val['test_silhouette'].std(ddof=0):.3f} across folds "
                                           "— a small value means the groups are stable.")
                        with st.expander("See per-split results"):
                            st.dataframe(val, use_container_width=True, hide_index=True)

                    # =============================================================
                    # STEP 12 — RESEARCH INTERPRETATION
                    # =============================================================
                    st.divider()
                    st.write("## 1️⃣2️⃣ Research Interpretation")
                    if u_method != "Isolation Forest":
                        cluster_labels = metrics.get("cluster_labels", {})
                        if cluster_labels:
                            st.write("**What each group means:**")
                            for group_id, description in cluster_labels.items():
                                st.write(f"- **{description}**")
                            st.caption("These labels are generated from the selected features — they describe "
                                     "how each group's average feature values differ from the overall dataset average.")

                        color_column, title = "Group Label", f"{u_method} by Feature Profile (PCA Projection)"
                        st.write("**Group Profiles**")
                        group_profile = result_df.groupby("Group Label")[u_features].mean().round(3)
                        st.dataframe(group_profile, use_container_width=True)
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
                        hover_data=u_features + ["Row"], title=title, template="plotly_white")
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
                        mode_extra={"method": u_method, "features": u_features,
                                   "scale_features": u_scale, "metrics": metrics})
                    with ul2:
                        st.download_button("📄 Download Methodology Report (Markdown)", data=report_txt.encode("utf-8"),
                                          file_name="methodology_report.md", mime="text/markdown")
                else:
                    st.info("Configure and run discovery on the left to see Steps 10-13 here.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© timothymarkbale2026 | Educational ML Laboratory</p>", unsafe_allow_html=True)
