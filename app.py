from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student-mat.csv"
MODEL_PATH = ROOT / "models" / "gradient_boosting_model.pkl"

RANDOM_STATE = 42
AT_RISK = 0
PERFORMING_WELL = 1
GRADE_THRESHOLD = 10

st.set_page_config(
    page_title="Automated Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def load_data(path):
    return pd.read_csv(path, sep=";")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def make_target(data):
    return pd.Series(
        np.where(data["G3"] < GRADE_THRESHOLD, AT_RISK, PERFORMING_WELL),
        index=data.index,
        name="performance",
    )


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return {
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "at_risk_precision": precision_score(
            y_test, predictions, pos_label=AT_RISK, zero_division=0
        ),
        "at_risk_recall": recall_score(
            y_test, predictions, pos_label=AT_RISK, zero_division=0
        ),
        "at_risk_f1": f1_score(
            y_test, predictions, pos_label=AT_RISK, zero_division=0
        ),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "report": classification_report(
            y_test, predictions,
            labels=[AT_RISK, PERFORMING_WELL],
            target_names=["At Risk", "Performing Well"],
            zero_division=0, output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[AT_RISK, PERFORMING_WELL]
        ),
    }


def get_feature_importance(pipeline):
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    if not hasattr(classifier, "feature_importances_"):
        return None

    names = preprocessor.get_feature_names_out()
    values = classifier.feature_importances_
    return (
        pd.DataFrame({"Feature": names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


st.title("🎓 Automated Student Performance Prediction")
st.caption(
    "Gradient Boosting model for early identification of students "
    "who may need additional academic support."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

if not {"G1", "G2", "G3"}.issubset(df.columns):
    st.error("Dataset must contain G1, G2, and G3 columns.")
    st.stop()

# Include G1 and G2; exclude only G3 from model inputs.
X = df.drop(columns=["G3"])
y = make_target(df)

# Reproduce notebook's stratified 80:20 test split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

class_counts = y.value_counts().reindex([AT_RISK, PERFORMING_WELL], fill_value=0)

with st.sidebar:
    st.header("Model configuration")
    st.selectbox("Dataset", ["UCI Student Performance — Mathematics"], disabled=True)
    st.selectbox("At-risk definition", ["Final grade G3 < 10"], disabled=True)
    st.info("Selected model: Gradient Boosting")

st.subheader("Dataset overview")
cols = st.columns(4)
cols[0].metric("Total students", f"{len(df):,}")
cols[1].metric("Input features", f"{X.shape[1]}")
cols[2].metric("Training samples", f"{len(X_train):,}")
cols[3].metric("At-risk students", f"{int(class_counts.loc[AT_RISK]):,}")

dist_col, grade_col = st.columns(2)
with dist_col:
    st.markdown("#### Student performance distribution")
    distribution = pd.DataFrame({
        "Category": ["At Risk", "Performing Well"],
        "Students": [
            int(class_counts.loc[AT_RISK]),
            int(class_counts.loc[PERFORMING_WELL]),
        ],
    }).set_index("Category")
    st.bar_chart(distribution)

with grade_col:
    st.markdown("#### Final grade distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["G3"], bins=20, kde=True, ax=ax)
    ax.set_xlabel("Final grade (G3)")
    ax.set_ylabel("Number of students")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.info("Class mapping: 0 = At Risk (G3 < 10); 1 = Performing Well (G3 >= 10).")
st.write(
    "The model uses 32 inputs, including G1 and G2. G3 is used only to "
    "derive the target and is not an input. This predicts final performance "
    "from previous grades and student attributes."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH}. Save "
        "gradient_boosting_model.pkl in the models folder."
    )
    st.stop()

try:
    pipeline = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load Gradient Boosting model: {error}")
    st.stop()

if not hasattr(pipeline, "named_steps") or not {
    "preprocessor", "classifier"
}.issubset(pipeline.named_steps):
    st.error("Expected a Pipeline with 'preprocessor' and 'classifier' steps.")
    st.stop()

expected_features = X.columns.tolist()
classifier_classes = list(pipeline.named_steps["classifier"].classes_)

try:
    saved_features = list(pipeline.named_steps["preprocessor"].feature_names_in_)
except AttributeError:
    saved_features = expected_features

if saved_features != expected_features:
    st.error(
        "Saved model's input features do not match the dataset. Retrain it "
        "with X = df.drop(columns=['G3']) so G1 and G2 are included."
    )
    st.stop()

try:
    test_results = evaluate_model(pipeline, X_test, y_test)
except Exception as error:
    st.error(f"Model evaluation failed. Check schema/library versions: {error}")
    st.stop()

st.subheader("Gradient Boosting — Held-out test performance")
metric_cols = st.columns(5)
metric_cols[0].metric("Accuracy", f"{test_results['accuracy']:.2%}")
metric_cols[1].metric("Balanced accuracy", f"{test_results['balanced_accuracy']:.2%}")
metric_cols[2].metric("At-risk precision", f"{test_results['at_risk_precision']:.2%}")
metric_cols[3].metric("At-risk recall", f"{test_results['at_risk_recall']:.2%}")
metric_cols[4].metric("Macro F1", f"{test_results['macro_f1']:.3f}")

st.markdown("#### Classification report")
report_df = pd.DataFrame(test_results["report"]).transpose()
st.dataframe(
    report_df.style.format(
        "{:.3f}", subset=report_df.select_dtypes(include="number").columns
    ),
    use_container_width=True,
)

st.markdown("#### Confusion matrix")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    test_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
    xticklabels=["At Risk", "Performing Well"],
    yticklabels=["At Risk", "Performing Well"], ax=ax,
)
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
ax.set_title("Gradient Boosting — Confusion Matrix")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Feature importance")
importance_df = get_feature_importance(pipeline)
if importance_df is not None:
    top = importance_df.head(15).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["Feature"], top["Importance"])
    ax.set_xlabel("Feature importance")
    ax.set_title("Top 15 Features — Gradient Boosting")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("View importance values"):
        st.dataframe(importance_df, use_container_width=True, hide_index=True)
    st.caption(
        "Importance scores describe model reliance, not causation. One-hot "
        "encoded categories are displayed as separate transformed features."
    )
else:
    st.info("This classifier does not expose feature_importances_.")

st.subheader("Predict for an individual student")
st.write("Enter the student's information, including earlier grades G1 and G2.")

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

with st.form("student_prediction_form"):
    input_values = {}
    num_col, cat_col = st.columns(2)

    with num_col:
        st.markdown("##### Numerical attributes")
        for feature in numeric_features:
            minimum = float(X[feature].min())
            maximum = float(X[feature].max())
            default = float(X[feature].median())
            if pd.api.types.is_integer_dtype(X[feature]):
                input_values[feature] = st.number_input(
                    feature, min_value=int(minimum), max_value=int(maximum),
                    value=int(round(default)), step=1,
                )
            else:
                input_values[feature] = st.number_input(
                    feature, min_value=minimum, max_value=maximum, value=default
                )

    with cat_col:
        st.markdown("##### Categorical attributes")
        for feature in categorical_features:
            options = sorted(X[feature].dropna().unique().tolist())
            if not options:
                st.error(f"No valid options for {feature}.")
                st.stop()
            input_values[feature] = st.selectbox(feature, options)

    submitted = st.form_submit_button("Predict student category", type="primary")

if submitted:
    prediction_input = pd.DataFrame([input_values], columns=expected_features)
    try:
        predicted_class = int(pipeline.predict(prediction_input)[0])

        if predicted_class == AT_RISK:
            st.warning("Prediction: **At Risk**")
        else:
            st.success("Prediction: **Performing Well**")

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(prediction_input)[0]
            probability_map = dict(zip(classifier_classes, probabilities))
            prob_cols = st.columns(2)
            prob_cols[0].metric("At-risk probability", f"{probability_map[AT_RISK]:.1%}")
            prob_cols[1].metric(
                "Performing-well probability",
                f"{probability_map[PERFORMING_WELL]:.1%}",
            )
            st.caption("Probabilities are model estimates and may not be calibrated.")
    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption(
    "Gradient Boosting pipeline trained in the notebook. Class 0 = At Risk; "
    "class 1 = Performing Well. Predictions are decision-support estimates, "
    "not definitive judgments about students."
)
