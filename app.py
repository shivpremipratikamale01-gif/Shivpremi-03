"""
Streamlit ML Prediction Studio
Upload a CSV, explore it, train a classification or regression model,
evaluate it, and make predictions on new rows.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


st.set_page_config(
    page_title="ML Prediction Studio",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ML Prediction Studio")
st.write(
    "Upload your own CSV dataset, explore the data, train a machine-learning "
    "model, review its performance, and predict outcomes for new records."
)

st.sidebar.header("About this app")
st.sidebar.info(
    "This app supports tabular CSV datasets with one selected target column. "
    "It automatically prepares numeric and categorical features."
)
st.sidebar.caption("Built with Python, Streamlit and scikit-learn.")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("👆 Upload a CSV file to get started.")
    st.markdown(
        """
        ### How to use
        1. Upload a CSV dataset.
        2. Choose the column you want to predict.
        3. Select a model and train it.
        4. Review evaluation results.
        5. Upload new records to generate predictions.

        **Dataset tip:** Your CSV should contain rows as records and columns
        as features. Include the answer/outcome column that you want the model
        to learn as the target.
        """
    )
    st.stop()

try:
    uploaded_file.seek(0)
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if data.empty:
    st.error("The uploaded CSV has no data rows.")
    st.stop()

# Clean column labels and remove fully empty rows/columns.
data.columns = [str(col).strip() for col in data.columns]
data = data.dropna(axis=0, how="all").dropna(axis=1, how="all")

if data.shape[1] < 2:
    st.error("The dataset needs at least two columns: features and a target.")
    st.stop()

st.subheader("1. Dataset overview")
m1, m2, m3 = st.columns(3)
m1.metric("Rows", f"{data.shape[0]:,}")
m2.metric("Columns", f"{data.shape[1]:,}")
m3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")

with st.expander("Preview dataset", expanded=True):
    st.dataframe(data.head(100), use_container_width=True)

with st.expander("Summary statistics"):
    st.dataframe(data.describe(include="all").transpose(), use_container_width=True)

left, right = st.columns(2)
with left:
    st.markdown("**Missing values by column**")
    missing = data.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values found.")
    else:
        st.bar_chart(missing)
with right:
    st.markdown("**Numeric feature distributions**")
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        selected_numeric = st.selectbox("Choose a numeric column", numeric_columns)
        fig, ax = plt.subplots()
        data[selected_numeric].dropna().hist(ax=ax, bins=25)
        ax.set_xlabel(selected_numeric)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numeric columns available for a histogram.")

st.subheader("2. Configure your machine-learning model")
target = st.selectbox("Select the target column (what should the model predict?)", data.columns)
X = data.drop(columns=[target]).copy()
y = data[target].copy()

# Remove rows with missing target values; features are imputed inside the pipeline.
valid_target = y.notna()
X = X.loc[valid_target].reset_index(drop=True)
y = y.loc[valid_target].reset_index(drop=True)

if len(X) < 5:
    st.error("At least 5 rows with a non-missing target are recommended.")
    st.stop()

# Drop columns with no usable values after target filtering.
empty_features = [c for c in X.columns if X[c].isna().all()]
if empty_features:
    X = X.drop(columns=empty_features)
    st.warning(f"Removed completely empty feature columns: {', '.join(empty_features)}")

if X.shape[1] == 0:
    st.error("No usable feature columns remain after selecting the target.")
    st.stop()

# Treat target as classification if it is non-numeric or has relatively few unique values.
is_classification = (
    not pd.api.types.is_numeric_dtype(y)
    or y.nunique() <= min(20, max(2, int(len(y) * 0.05)))
)
task = "Classification" if is_classification else "Regression"
st.write(f"Detected task: **{task}**")

if is_classification and y.nunique() < 2:
    st.error("Classification requires at least two different target classes.")
    st.stop()

if is_classification:
    model_name = st.selectbox(
        "Choose a classification model",
        ["Random Forest", "Decision Tree", "Logistic Regression"],
    )
    model_options: dict[str, Any] = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }
else:
    model_name = st.selectbox(
        "Choose a regression model",
        ["Random Forest", "Decision Tree", "Linear Regression"],
    )
    model_options = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
    }

test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)
random_state = 42

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

transformers = []
if numeric_features:
    transformers.append(("numeric", numeric_pipeline, numeric_features))
if categorical_features:
    transformers.append(("categorical", categorical_pipeline, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers)
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model_options[model_name]),
    ]
)

if st.button("🚀 Train and evaluate model", type="primary"):
    try:
        stratify = None
        if is_classification:
            counts = y.value_counts()
            # Stratify only when every class can appear in both splits.
            if counts.min() >= 2 and int(round(len(y) * test_size / 100)) >= y.nunique():
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=random_state,
            stratify=stratify,
        )

        model_pipeline.fit(X_train, y_train)
        predictions = model_pipeline.predict(X_test)

        st.session_state["trained_pipeline"] = model_pipeline
        st.session_state["feature_columns"] = X.columns.tolist()
        st.session_state["target_column"] = target
        st.session_state["task_type"] = task
        st.session_state["model_name"] = model_name
        st.session_state["test_score"] = (
            accuracy_score(y_test, predictions)
            if is_classification
            else r2_score(y_test, predictions)
        )

        st.subheader("3. Model evaluation")
        if is_classification:
            a, b = st.columns(2)
            a.metric("Accuracy", f"{accuracy_score(y_test, predictions):.2%}")
            b.metric("Test records", f"{len(y_test):,}")
            st.markdown("**Classification report**")
            report = classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(
                y_test, predictions, ax=ax, xticks_rotation=45, colorbar=False
            )
            ax.set_title("Confusion matrix")
            st.pyplot(fig)
            plt.close(fig)
        else:
            rmse = mean_squared_error(y_test, predictions) ** 0.5
            a, b, c = st.columns(3)
            a.metric("MAE", f"{mean_absolute_error(y_test, predictions):.4f}")
            b.metric("RMSE", f"{rmse:.4f}")
            c.metric("R² score", f"{r2_score(y_test, predictions):.4f}")
            comparison = pd.DataFrame(
                {"Actual": y_test.values, "Predicted": predictions}
            )
            st.markdown("**Actual vs predicted (test data)**")
            st.dataframe(comparison.head(100), use_container_width=True)

        st.success(f"{model_name} training completed successfully.")
    except Exception as exc:
        st.error(f"Training failed: {exc}")
        st.info(
            "Check that your target column is suitable, the CSV has enough rows, "
            "and the feature columns contain usable values."
        )

# Prediction section uses the model stored after successful training.
if "trained_pipeline" in st.session_state:
    st.subheader("4. Make predictions on new data")
    st.caption(
        "Upload a CSV containing the same feature columns used during training. "
        "The target column should not be included."
    )
    prediction_file = st.file_uploader(
        "Upload new records for prediction", type=["csv"], key="prediction_upload"
    )
    if prediction_file is not None:
        try:
            new_data = pd.read_csv(prediction_file)
            required = st.session_state["feature_columns"]
            missing_cols = [c for c in required if c not in new_data.columns]
            if missing_cols:
                st.error(f"Missing required feature columns: {', '.join(missing_cols)}")
            else:
                new_data = new_data[required]
                new_predictions = st.session_state["trained_pipeline"].predict(new_data)
                result = new_data.copy()
                result["Prediction"] = new_predictions
                st.dataframe(result, use_container_width=True)

                csv_bytes = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions as CSV",
                    data=csv_bytes,
                    file_name="ml_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Could not generate predictions: {exc}")

st.divider()
st.caption(
    "Note: Model performance depends on dataset quality, target selection, "
    "sample size and the train/test split. This app is an educational baseline, "
    "not a guarantee of real-world predictive accuracy."
)
