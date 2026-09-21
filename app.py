import os
import json
import psutil
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, r2_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.inspection import permutation_importance

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Machine Learning Intuition Lab",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Machine Learning Intuition Lab")
st.caption("Automatic preprocessing, stratified 80/20 splitting, cross-validation, and model comparison")

# -----------------------------
# Utility functions
# -----------------------------
def get_process_ram_mb():
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return 0.0


def make_one_hot_encoder():
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data(show_spinner=False)
def load_and_clean_csv(file, max_rows):
    df = pd.read_csv(file, nrows=max_rows, low_memory=False)
    missing_before = int(df.isna().sum().sum())

    # Reduce memory where possible.
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df, missing_before


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder())
    ])

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipe, categorical_features))

    if not transformers:
        raise ValueError("No usable predictor columns were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def classification_models(preprocessor):
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=3,
                class_weight="balanced", random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=250, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
        "Extra Trees": Pipeline([
            ("preprocessor", preprocessor),
            ("model", ExtraTreesClassifier(
                n_estimators=250, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05,
                max_depth=3, random_state=42
            ))
        ])
    }


def regression_models(preprocessor):
    return {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(
                max_depth=8, min_samples_leaf=3, random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=250, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            ))
        ]),
        "Extra Trees": Pipeline([
            ("preprocessor", preprocessor),
            ("model", ExtraTreesRegressor(
                n_estimators=250, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.05,
                max_depth=3, random_state=42
            ))
        ])
    }


def prepare_target(df, target, task):
    data = df.copy()

    if task == "Classification":
        data[target] = data[target].astype("string").fillna("Unknown").astype(str)
        y = data[target]
        X = data.drop(columns=[target])
        return X, y

    # For regression, invalid target values cannot be used.
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])
    y = data[target].astype(float)
    X = data.drop(columns=[target])
    return X, y


def evaluate_classification(X, y):
    if y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    if y.value_counts().min() < 2:
        raise ValueError("Every class needs at least two rows for stratified splitting.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    smallest_class = int(y_train.value_counts().min())
    if smallest_class >= 5:
        n_splits = 5
    elif smallest_class >= 3:
        n_splits = 3
    else:
        n_splits = 2

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = classification_models(build_preprocessor(X_train))
    rows = []
    fitted = {}

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted"
    }

    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=-1, error_score="raise"
        )
        model.fit(X_train, y_train)
        fitted[name] = model
        row = {
            "Algorithm": name,
            "CV Accuracy": scores["test_accuracy"].mean(),
            "CV Balanced Accuracy": scores["test_balanced_accuracy"].mean(),
            "CV Precision": scores["test_precision"].mean(),
            "CV Recall": scores["test_recall"].mean(),
            "CV F1": scores["test_f1"].mean()
        }
        row["Selection Score"] = 0.5 * row["CV Balanced Accuracy"] + 0.5 * row["CV F1"]
        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values(
        "Selection Score", ascending=False
    ).reset_index(drop=True)

    best_name = leaderboard.iloc[0]["Algorithm"]
    best_model = fitted[best_name]
    predictions = best_model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0)
    }

    return {
        "task": "Classification",
        "best_name": best_name,
        "best_model": best_model,
        "leaderboard": leaderboard,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
        "classes": sorted(y.unique().tolist())
    }


def evaluate_regression(X, y):
    if len(X) < 10:
        raise ValueError("Regression requires at least 10 valid rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = regression_models(build_preprocessor(X_train))
    rows = []
    fitted = {}

    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error"
    }

    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=-1, error_score="raise"
        )
        model.fit(X_train, y_train)
        fitted[name] = model
        rows.append({
            "Algorithm": name,
            "CV R²": scores["test_r2"].mean(),
            "CV MAE": -scores["test_mae"].mean(),
            "CV RMSE": -scores["test_rmse"].mean()
        })

    leaderboard = pd.DataFrame(rows).sort_values(
        "CV R²", ascending=False
    ).reset_index(drop=True)

    best_name = leaderboard.iloc[0]["Algorithm"]
    best_model = fitted[best_name]
    predictions = best_model.predict(X_test)

    metrics = {
        "R²": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions))
    }

    return {
        "task": "Regression",
        "best_name": best_name,
        "best_model": best_model,
        "leaderboard": leaderboard,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics
    }


def show_model_results(result):
    task = result["task"]
    st.subheader(f"Best algorithm: {result['best_name']}")

    if task == "Classification":
        st.info(
            "The model was selected using cross-validation on the training set. "
            "The final metrics below come from the untouched 20% test set."
        )
    else:
        st.info(
            "Regression does not use classification accuracy. R², MAE, and RMSE "
            "measure how close the numeric predictions are to the real values."
        )

    st.write("### Algorithm comparison")
    leaderboard = result["leaderboard"].copy()
    format_cols = [c for c in leaderboard.columns if c != "Algorithm"]
    st.dataframe(
        leaderboard.style.format({c: "{:.3f}" for c in format_cols}),
        use_container_width=True,
        hide_index=True
    )

    st.write("### Final 20% test-set performance")
    metric_cols = st.columns(len(result["metrics"]))
    for col, (name, value) in zip(metric_cols, result["metrics"].items()):
        if task == "Classification":
            col.metric(name, f"{value:.2%}")
        else:
            col.metric(name, f"{value:.4f}")

    if task == "Classification":
        score = result["metrics"]["Balanced Accuracy"]
        if score >= 0.95:
            st.success("Excellent test performance. Still check for leakage and validate on future data.")
        elif score >= 0.90:
            st.success("Strong test performance, but it is not a guarantee for future data.")
        elif score >= 0.70:
            st.warning("Moderate performance. More informative predictors or better labels may be needed.")
        else:
            st.error("Low performance. The available predictors may not contain enough signal.")

        cm = confusion_matrix(result["y_test"], result["predictions"], labels=result["classes"])
        fig, ax = plt.subplots(figsize=(8, 6))
        show_labels = len(result["classes"]) <= 20
        sns.heatmap(
            cm, annot=show_labels, fmt="d", cmap="Purples", ax=ax,
            xticklabels=result["classes"] if show_labels else False,
            yticklabels=result["classes"] if show_labels else False
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Classification report"):
            report = classification_report(
                result["y_test"], result["predictions"], zero_division=0
            )
            st.code(report)

    else:
        score = result["metrics"]["R²"]
        if score >= 0.90:
            st.success("The model explains at least 90% of test-set target variation.")
        elif score >= 0.70:
            st.warning("The model has moderate explanatory power.")
        else:
            st.error("The model explains less than 70% of the test-set variation.")

        actual = result["y_test"]
        predicted = result["predictions"]
        fig = px.scatter(
            x=actual, y=predicted,
            labels={"x": "Actual value", "y": "Predicted value"},
            title="Actual versus Predicted Values"
        )
        low = min(actual.min(), predicted.min())
        high = max(actual.max(), predicted.max())
        fig.add_shape(
            type="line", x0=low, y0=low, x1=high, y1=high,
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig, use_container_width=True)

    # Permutation importance is calculated on the untouched test set.
    st.write("### Predictor influence")
    try:
        scoring = "balanced_accuracy" if task == "Classification" else "r2"
        perm = permutation_importance(
            result["best_model"], result["X_test"], result["y_test"],
            n_repeats=5, random_state=42, scoring=scoring, n_jobs=-1
        )
        imp_df = pd.DataFrame({
            "Feature": result["X_test"].columns,
            "Importance": perm.importances_mean
        }).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            title="Permutation Importance on Test Data",
            color="Importance", color_continuous_scale="Portland"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as exc:
        st.warning(f"Feature importance could not be calculated: {exc}")


# -----------------------------
# Sidebar and upload
# -----------------------------
with st.sidebar:
    st.header("System status")
    ram = get_process_ram_mb()
    st.write(f"Python process RAM: {ram:.1f} MB")
    max_rows = st.slider("Maximum rows to load", 1_000, 1_000_000, 500_000, step=1_000)
    st.caption("The test set is never artificially balanced. Class weighting is applied only inside training models.")

uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    df, missing_before = load_and_clean_csv(uploaded_file, max_rows)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

if df.empty:
    st.error("The CSV file contains no rows.")
    st.stop()

st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")
st.write(f"Missing values detected: **{missing_before:,}**. Missing values are imputed inside each model pipeline.")

st.download_button(
    "Download the uploaded data",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="uploaded_dataset.csv",
    mime="text/csv"
)

with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)

mode = st.radio(
    "Workspace",
    ["Automatic Machine Learning", "Exploratory Analysis"],
    horizontal=True
)

if mode == "Exploratory Analysis":
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_columns) >= 2:
        st.subheader("Correlation heatmap")
        corr = df[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=len(numeric_columns) <= 15, fmt=".2f", cmap="RdBu", center=0, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("At least two numeric columns are needed for a correlation heatmap.")

    st.subheader("Column summary")
    summary = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    st.subheader("Automatic model selection")

    target = st.selectbox("Target column to predict", df.columns)
    task = st.radio(
        "Prediction task",
        ["Classification", "Regression"],
        horizontal=True,
        help="Classification predicts groups or labels. Regression predicts numeric values."
    )

    default_features = [c for c in df.columns if c != target]
    features = st.multiselect(
        "Predictor columns",
        options=default_features,
        default=default_features,
        help="The target column is automatically excluded from the predictors."
    )

    if task == "Classification":
        counts = df[target].astype("string").fillna("Unknown").value_counts()
        st.write("Target class distribution")
        st.dataframe(counts.rename("Rows").to_frame(), use_container_width=False)
        if len(counts) > 20:
            st.warning("This target has more than 20 classes. Classification may be difficult; regression may be more appropriate if the values are numeric.")

    if st.button("🚀 Compare algorithms and select the best", type="primary"):
        if not features:
            st.error("Select at least one predictor column.")
        else:
            try:
                X = df[features].copy()
                X = X.replace([np.inf, -np.inf], np.nan)
                X, y = prepare_target(df[[*features, target]], target, task)

                with st.spinner("Splitting data, cross-validating models, and testing the winner..."):
                    if task == "Classification":
                        result = evaluate_classification(X, y)
                    else:
                        result = evaluate_regression(X, y)

                st.session_state["ml_result"] = result
                st.session_state["ml_target"] = target
                st.success("Model comparison completed.")

            except Exception as exc:
                st.error(f"Training failed: {exc}")

    if "ml_result" in st.session_state:
        show_model_results(st.session_state["ml_result"])

st.divider()
st.caption("Educational ML Laboratory | Model scores are estimates, not guarantees of future performance.")
