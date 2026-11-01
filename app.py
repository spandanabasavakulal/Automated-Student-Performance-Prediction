from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Paths and project configuration
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student-mat.csv"
MODEL_DIR = ROOT / "models"

MODEL_FILES = {
    "Random Forest": MODEL_DIR / "random_forest_model.pkl",
    "Tuned Logistic Regression": MODEL_DIR / "tuned_logistic_model.pkl",
}

RANDOM_STATE = 42
AT_RISK_CLASS = 0
PERFORMING_WELL_CLASS = 1
GRADE_THRESHOLD = 10


st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)


# ---------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def make_target(data: pd.DataFrame) -> pd.Series:
    """Notebook mapping: 0 = At Risk; 1 = Performing Well."""
    return pd.Series(
        np.where(data["G3"] < GRADE_THRESHOLD, AT_RISK_CLASS, PERFORMING_WELL_CLASS),
        index=data.index,
        name="performance",
    )


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    return {
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "at_risk_precision": precision_score(
            y_test, predictions, pos_label=AT_RISK_CLASS, zero_division=0
        ),
        "at_risk_recall": recall_score(
            y_test, predictions, pos_label=AT_RISK_CLASS, zero_division=0
        ),
        "at_risk_f1": f1_score(
            y_test, predictions, pos_label=AT_RISK_CLASS, zero_division=0
        ),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "report": classification_report(
            y_test,
            predictions,
            labels=[AT_RISK_CLASS, PERFORMING_WELL_CLASS],
            target_names=["At Risk", "Performing Well"],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[AT_RISK_CLASS, PERFORMING_WELL_CLASS],
        ),
    }


def get_random_forest_feature_importance(pipeline, original_columns):
    """Return feature importance, grouping one-hot columns by source feature."""
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(classifier, "feature_importances_"):
        return None

    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = preprocessor.transformers_[1][2]
    encoder = preprocessor.named_transformers_["cat"]

    encoded_categorical = (
        encoder.get_feature_names_out(categorical_features).tolist()
        if categorical_features
        else []
    )
    transformed_names = list(numeric_features) + encoded_categorical

    importances = pd.DataFrame(
        {
            "Transformed Feature": transformed_names,
            "Importance": classifier.feature_importances_,
        }
    )

    def original_name(transformed_name):
        if transformed_name in original_columns:
            return transformed_name
        # OneHotEncoder feature names are formatted as "<column>_<category>".
        matches = [
            column
            for column in categorical_features
            if transformed_name.startswith(f"{column}_")
        ]
        return max(matches, key=len) if matches else transformed_name

    importances["Feature"] = importances["Transformed Feature"].map(original_name)
    grouped = (
        importances.groupby("Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
    )
    return grouped


# ---------------------------------------------------------------------
# Header and data validation
# ---------------------------------------------------------------------
st.title("🎓 Automated Student Performance Prediction")
st.caption(
    "Jupyter-trained machine learning pipelines for early identification "
    "of students who may need additional academic support."
)

if not DATA_PATH.exists():
    st.error(f"Mathematics dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Unable to load the Mathematics dataset: {error}")
    st.stop()

required_columns = {"G1", "G2", "G3"}
if not required_columns.issubset(df.columns):
    st.error("The dataset must contain G1, G2, and G3 grade columns.")
    st.stop()

if df.duplicated().any():
    st.warning(
        f"The dataset contains {int(df.duplicated().sum())} exact duplicate rows. "
        "The notebook split is reproduced for consistency; inspect duplicates "
        "before making research claims."
    )

# The notebook trained on Mathematics only, with G1/G2/G3 excluded.
X = df.drop(columns=["G1", "G2", "G3"])
y = make_target(df)

# Reproduce the notebook's exact held-out split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

class_counts = y.value_counts().reindex(
    [AT_RISK_CLASS, PERFORMING_WELL_CLASS], fill_value=0
)


# ---------------------------------------------------------------------
# Sidebar and dataset overview
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Model settings")
    st.selectbox("Dataset", ["Mathematics"], index=0, disabled=True)
    st.caption("The saved Jupyter models were trained on the Mathematics dataset.")
    st.selectbox(
        "At-risk definition",
        ["G3 < 10"],
        index=0,
        disabled=True,
        help="The notebook models were trained with G3 < 10 as the at-risk definition.",
    )
    model_name = st.selectbox(
        "Trained model",
        list(MODEL_FILES.keys()),
        help="Choose one of the two pipelines saved from the Jupyter notebook.",
    )

st.subheader("Dataset overview")
overview = st.columns(4)
overview[0].metric("Total students", f"{len(df):,}")
overview[1].metric("Input features", f"{X.shape[1]}")
overview[2].metric("Training samples", f"{len(X_train):,}")
overview[3].metric("At-risk students", f"{int(class_counts.loc[0]):,}")

distribution_col, grade_col = st.columns(2)

with distribution_col:
    st.markdown("#### Student performance distribution")
    distribution = pd.DataFrame(
        {
            "Category": ["At Risk", "Performing Well"],
            "Students": [
                int(class_counts.loc[AT_RISK_CLASS]),
                int(class_counts.loc[PERFORMING_WELL_CLASS]),
            ],
        }
    ).set_index("Category")
    st.bar_chart(distribution)

with grade_col:
    st.markdown("#### Final grade distribution")
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.histplot(df["G3"], bins=20, kde=True, ax=axis)
    axis.set_xlabel("Final grade (G3)")
    axis.set_ylabel("Number of students")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

st.info(
    "**Class mapping used by the Jupyter notebook:** class 0 = At Risk "
    "(G3 < 10); class 1 = Performing Well (G3 ≥ 10). "
    "The app reads probabilities and predictions using this mapping."
)

st.markdown(
    "The model uses the 30 non-grade student attributes. G1, G2, and G3 are "
    "excluded from prediction inputs to avoid target leakage. Mathematics and "
    "Portuguese are not mixed: the saved pipelines were trained on Mathematics."
)


# ---------------------------------------------------------------------
# Load saved model and evaluate on the same held-out test split
# ---------------------------------------------------------------------
model_path = MODEL_FILES[model_name]
if not model_path.exists():
    st.error(
        f"Saved model not found: {model_path}. "
        "Check that the .pkl file is inside the project's models folder."
    )
    st.stop()

try:
    pipeline = load_model(str(model_path))
except Exception as error:
    st.error(f"Could not load {model_name}: {error}")
    st.stop()

expected_features = X.columns.tolist()
if not hasattr(pipeline, "named_steps"):
    st.error("The selected file is not a scikit-learn Pipeline.")
    st.stop()

# The notebook pipeline uses these exact step names.
if not {"preprocessor", "classifier"}.issubset(pipeline.named_steps.keys()):
    st.error(
        "The saved pipeline structure does not match the notebook. "
        "Expected steps named 'preprocessor' and 'classifier'."
    )
    st.stop()

classifier_classes = list(pipeline.named_steps["classifier"].classes_)
if AT_RISK_CLASS not in classifier_classes or PERFORMING_WELL_CLASS not in classifier_classes:
    st.error(f"Unexpected model classes: {classifier_classes}")
    st.stop()

try:
    test_results = evaluate_model(pipeline, X_test, y_test)
except Exception as error:
    st.error(
        "The saved model could not process the Mathematics test data. "
        f"Check the feature schema and scikit-learn versions. Details: {error}"
    )
    st.stop()

st.subheader(f"{model_name} – Held-out test performance")
metric_columns = st.columns(5)
metric_columns[0].metric("Accuracy", f"{test_results['accuracy']:.1%}")
metric_columns[1].metric("At-risk precision", f"{test_results['at_risk_precision']:.1%}")
metric_columns[2].metric("At-risk recall", f"{test_results['at_risk_recall']:.1%}")
metric_columns[3].metric("At-risk F1", f"{test_results['at_risk_f1']:.3f}")
metric_columns[4].metric("Macro F1", f"{test_results['macro_f1']:.3f}")

if test_results["at_risk_recall"] < 0.50:
    st.warning(
        "At-risk recall is below 50% on this test split. The model misses more "
        "than half of the students who are actually at risk."
    )

report_df = pd.DataFrame(test_results["report"]).transpose()
st.markdown("#### Classification report")
st.dataframe(
    report_df.style.format(
        "{:.3f}", subset=report_df.select_dtypes(include="number").columns
    ),
    use_container_width=True,
)

st.markdown("#### Confusion matrix")
matrix = test_results["confusion_matrix"]
figure, axis = plt.subplots(figsize=(6, 5))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["At Risk", "Performing Well"],
    yticklabels=["At Risk", "Performing Well"],
    ax=axis,
)
axis.set_xlabel("Predicted label")
axis.set_ylabel("True label")
axis.set_title(f"{model_name} – Confusion Matrix")
figure.tight_layout()
st.pyplot(figure)
plt.close(figure)

st.caption(
    "Confusion matrix order is [At Risk, Performing Well]. "
    "The top-left cell counts correctly identified at-risk students; "
    "the top-right cell counts at-risk students missed by the model."
)


# ---------------------------------------------------------------------
# Notebook model comparison and cross-validation results
# ---------------------------------------------------------------------
st.subheader("Notebook model comparison")

comparison_path = MODEL_DIR / "model_comparison_results.csv"
if comparison_path.exists():
    comparison_df = pd.read_csv(comparison_path)
    st.markdown("#### Held-out test comparison – all seven models")
    st.dataframe(
        comparison_df.style.format(
            {
                column: "{:.3f}"
                for column in comparison_df.select_dtypes(include="number").columns
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    if {"Model", "Accuracy"}.issubset(comparison_df.columns):
        figure, axis = plt.subplots(figsize=(10, 5))
        sns.barplot(data=comparison_df, x="Model", y="Accuracy", ax=axis)
        axis.set_title("Accuracy Comparison of Machine Learning Models")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
else:
    st.warning("model_comparison_results.csv was not found in the models folder.")

cv_path = MODEL_DIR / "cross_validation_results.csv"
at_risk_cv_path = MODEL_DIR / "at_risk_cv_results.csv"

if cv_path.exists() and at_risk_cv_path.exists():
    cv_df = pd.read_csv(cv_path)
    at_risk_cv_df = pd.read_csv(at_risk_cv_path)

    # CV precision/recall/F1 in cross_validation_results.csv used class 1.
    # Replace those columns with the explicitly calculated class-0 metrics.
    if "Model" in cv_df.columns and "Model" in at_risk_cv_df.columns:
        cv_display = cv_df.merge(
            at_risk_cv_df,
            on="Model",
            how="left",
            suffixes=("_class1", ""),
        )
        st.markdown("#### Cross-validation comparison")
        st.caption(
            "At-risk precision, recall, and F1 are taken from the separate "
            "class-0 cross-validation results. This corrects the notebook's "
            "default positive-class convention."
        )
        st.dataframe(
            cv_display.style.format(
                {
                    column: "{:.3f}"
                    for column in cv_display.select_dtypes(include="number").columns
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.warning(
        "One or both cross-validation CSV files are missing. "
        "The held-out test evaluation above is still available."
    )


# ---------------------------------------------------------------------
# Feature importance – Random Forest
# ---------------------------------------------------------------------
st.subheader("Feature importance")

rf_path = MODEL_FILES["Random Forest"]
if rf_path.exists():
    try:
        rf_pipeline = load_model(str(rf_path))
        importance_df = get_random_forest_feature_importance(
            rf_pipeline, expected_features
        )
        if importance_df is not None:
            top_features = importance_df.head(15)
            figure, axis = plt.subplots(figsize=(9, 6))
            sns.barplot(
                data=top_features,
                x="Importance",
                y="Feature",
                ax=axis,
            )
            axis.set_title("Top 15 Features – Random Forest")
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
            st.caption(
                "Random Forest impurity-based feature importance. "
                "These scores describe model reliance, not causal effects."
            )
        else:
            st.info("The saved Random Forest model does not expose feature_importances_.")
    except Exception as error:
        st.warning(f"Could not calculate feature importance: {error}")
else:
    st.info("Save random_forest_model.pkl to display feature importance.")


# ---------------------------------------------------------------------
# Individual student prediction
# ---------------------------------------------------------------------
st.subheader("Predict for an individual student")
st.write(
    "Enter student attributes below. The selected Jupyter-trained pipeline "
    "will preprocess the inputs and predict the student's performance category."
)

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

with st.form("student_prediction_form"):
    input_values = {}
    numeric_column, categorical_column = st.columns(2)

    with numeric_column:
        st.markdown("##### Numerical attributes")
        for feature in numeric_features:
            minimum = float(X[feature].min())
            maximum = float(X[feature].max())
            default = float(X[feature].median())
            if pd.api.types.is_integer_dtype(X[feature]):
                input_values[feature] = st.number_input(
                    feature,
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(round(default)),
                    step=1,
                )
            else:
                input_values[feature] = st.number_input(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                )

    with categorical_column:
        st.markdown("##### Categorical attributes")
        for feature in categorical_features:
            options = sorted(X[feature].dropna().unique().tolist())
            input_values[feature] = st.selectbox(feature, options)

    submitted = st.form_submit_button("Predict student category", type="primary")

if submitted:
    prediction_input = pd.DataFrame([input_values], columns=expected_features)

    try:
        predicted_class = int(pipeline.predict(prediction_input)[0])
        probabilities = pipeline.predict_proba(prediction_input)[0]
        class_to_probability = dict(zip(classifier_classes, probabilities))

        at_risk_probability = float(class_to_probability[AT_RISK_CLASS])
        performing_probability = float(class_to_probability[PERFORMING_WELL_CLASS])

        if predicted_class == AT_RISK_CLASS:
            st.warning("Prediction: **At Risk**")
        else:
            st.success("Prediction: **Performing Well**")

        probability_columns = st.columns(2)
        probability_columns[0].metric(
            "At-risk probability", f"{at_risk_probability:.1%}"
        )
        probability_columns[1].metric(
            "Performing-well probability", f"{performing_probability:.1%}"
        )

        st.caption(
            "Class mapping: 0 = At Risk; 1 = Performing Well. "
            "Probabilities are model estimates and may not be calibrated."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")


st.divider()
st.caption(
    "This dashboard uses the models trained in model_training.ipynb. "
    "The notebook's class mapping is preserved: 0 = At Risk and "
    "1 = Performing Well. Predictions are decision-support estimates, "
    "not definitive judgments about students."
)
