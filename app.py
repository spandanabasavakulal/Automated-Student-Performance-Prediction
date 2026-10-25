from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "Mathematics": ROOT / "data" / "student-mat.csv",
    "Portuguese": ROOT / "data" / "student-por.csv",
}
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
LOW_RECALL_WARNING_THRESHOLD = 0.50
CURRENT_DECISION_THRESHOLD = 0.50
PREVIOUS_HIGH_RECALL_THRESHOLD = 0.2859112295921685
DECISION_THRESHOLDS = (
    PREVIOUS_HIGH_RECALL_THRESHOLD,
    0.35,
    0.40,
    0.45,
    0.50,
)

st.set_page_config(
    page_title="Student Performance | AutoML",
    page_icon="🎓",
    layout="wide",
)


@st.cache_data
def load_data(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(dataset_path, sep=";")


def make_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        # Dense encoded output is compatible with HistGradientBoosting.
        sparse_threshold=0,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def model_search_spaces():
    return {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            {"model__C": [0.05, 0.2, 1.0, 5.0, 20.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "model__max_depth": [None, 3, 5, 8, 12],
                "model__min_samples_leaf": [2, 5, 10, 20],
                "model__criterion": ["gini", "entropy"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
            ),
            {
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [None, 5, 10, 18],
                "model__min_samples_leaf": [2, 5, 10],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
            ),
            {
                "model__n_estimators": [150, 250, 400],
                "model__max_depth": [None, 5, 10, 18],
                "model__min_samples_leaf": [2, 5, 10],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        "SVM": (
            SVC(
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [75, 125, 200],
                "model__learning_rate": [0.03, 0.07, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__min_samples_leaf": [5, 10, 20],
            },
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            {
                "model__max_iter": [75, 125, 200],
                "model__learning_rate": [0.03, 0.07, 0.1],
                "model__max_leaf_nodes": [7, 15, 31],
                "model__l2_regularization": [0.0, 1.0, 5.0],
                "model__min_samples_leaf": [10, 20, 30],
            },
        ),
    }


def score_predictions(y_true, predictions) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "At-risk precision": precision_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "At-risk recall": recall_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "At-risk F1": f1_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "Macro F1": f1_score(y_true, predictions, average="macro"),
        "Balanced accuracy": balanced_accuracy_score(y_true, predictions),
    }


def threshold_sweep(y_true, probabilities, model_name: str) -> list[dict]:
    rows = []
    for decision_threshold in DECISION_THRESHOLDS:
        predictions = (probabilities >= decision_threshold).astype(int)
        rows.append(
            {
                "Model": model_name,
                "Probability threshold": decision_threshold,
                **score_predictions(y_true, predictions),
            }
        )
    return rows


def current_reference_estimator():
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        l2_regularization=1.0,
        learning_rate=0.1,
        max_iter=125,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        random_state=RANDOM_STATE,
    )


def dummy_search_baseline():
    return DummyClassifier(strategy="most_frequent")


def model_path(dataset_name: str, threshold: int) -> Path:
    safe_name = re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_")
    return MODEL_DIR / f"{safe_name}_risk_below_{threshold}_macro_f1_sweep.joblib"


st.title("🎓 Student Performance Prediction")
st.caption(
    "An interactive, leakage-aware AutoML dashboard for early identification "
    "of students who may need additional support."
)

with st.sidebar:
    st.header("Analysis settings")
    dataset_name = st.selectbox("Student dataset", list(DATASETS))
    threshold = st.slider(
        "At-risk grade threshold (G3)",
        min_value=1,
        max_value=20,
        value=10,
        help="A student is At Risk when final grade G3 is below this value.",
    )
    st.info(
        f"Definition: G3 < {threshold} is **At Risk**; "
        f"G3 ≥ {threshold} is **Performing Well**. "
        "G1, G2, and G3 are excluded from prediction inputs."
    )
    st.caption(
        "OOF cutoff sweep: previous high-recall cutoff 0.286, plus 0.35, "
        "0.40, 0.45, and current cutoff 0.50."
    )

try:
    df = load_data(str(DATASETS[dataset_name]))
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Unable to load the {dataset_name} dataset: {error}")
    st.stop()

if "G3" not in df.columns:
    st.error("The selected dataset must contain the final-grade column G3.")
    st.stop()

excluded_columns = ["G1", "G2", "G3"]
missing_grades = [column for column in excluded_columns if column not in df]
if missing_grades:
    st.error(f"Dataset is missing expected grade columns: {', '.join(missing_grades)}")
    st.stop()

exact_duplicate_rows = int(df.duplicated().sum())
if exact_duplicate_rows:
    st.error(
        f"Found {exact_duplicate_rows} exact duplicate records. Resolve duplicates "
        "before a row-wise split so duplicate records cannot cross train/test."
    )
    st.stop()

y = (df["G3"] < threshold).astype(int).rename("At Risk")
X = df.drop(columns=[*excluded_columns])
class_counts = y.value_counts().reindex([1, 0], fill_value=0)

st.subheader("Dataset overview")
overview = st.columns(4)
overview[0].metric("Total students", f"{len(df):,}")
overview[1].metric("Input features", X.shape[1])
overview[2].metric("Training samples", f"{int(len(df) * 0.8):,}")
overview[3].metric("At-risk students", f"{class_counts.loc[1]:,}")

distribution_col, definition_col = st.columns([1, 1])
with distribution_col:
    st.markdown("#### Class distribution")
    distribution = pd.DataFrame(
        {
            "Category": ["At Risk", "Performing Well"],
            "Students": [int(class_counts.loc[1]), int(class_counts.loc[0])],
        }
    ).set_index("Category")
    st.bar_chart(distribution, color="#5B8FF9")
with definition_col:
    st.markdown("#### Prediction setup")
    st.write(
        f"The target is derived from **G3**: grades below **{threshold}** are "
        "At Risk; all other grades are Performing Well. The default threshold "
        "of 10 reflects the dataset's pass-mark convention. Grade columns "
        "G1/G2/G3 are excluded, so the model uses the other student attributes "
        "for an early prediction."
    )
    st.write(
        "Preprocessing (median imputation and scaling for numeric fields; "
        "most-frequent imputation and one-hot encoding for categorical fields) "
        "is fitted inside each training fold to prevent leakage."
    )
    st.write(
        "Mathematics and Portuguese are modeled separately. Their records are "
        "not combined because students can appear in both subject files."
    )
    st.caption(
        f"Exact duplicate rows in this subject file: {exact_duplicate_rows}. "
        "The supplied data does not include a student ID; row-level duplicate "
        "checks cannot prove identity across files."
    )

split_counts = class_counts
if (split_counts == 0).any() or int(split_counts.min()) < 3:
    st.warning(
        "This threshold leaves too few examples in one class for a stratified "
        "train/test split and cross-validation. Choose a threshold with at "
        "least three examples in each class."
    )
    st.stop()

training_size = int(len(df) * 0.8)
minimum_training_class_count = int(
    (split_counts * training_size / len(df)).min()
)
cv_folds = min(5, minimum_training_class_count)
if cv_folds < 2:
    st.warning("There are too few training examples per class to run cross-validation.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)
if not X_train.index.intersection(X_test.index).empty:
    st.error("Train/test row overlap detected; evaluation was stopped.")
    st.stop()

run_key = (dataset_name, threshold)
if st.session_state.get("run_key") != run_key:
    st.session_state.pop("evaluation", None)
    st.session_state["run_key"] = run_key

st.subheader("AutoML model comparison")
st.write(
    "Seven classifiers are tuned for macro F1 using randomized search and "
    f"stratified {cv_folds}-fold cross-validation on the training set. "
    "For each tuned model, the previous 0.286 cutoff and current cutoff "
    "0.50 (plus 0.35/0.40/0.45) "
    "are compared using training-only out-of-fold predictions. The model and "
    "cutoff with the highest OOF macro F1 are selected; balanced accuracy "
    "breaks ties. The held-out test set is not used for selection."
)

if st.button("Tune models and evaluate best model", type="primary"):
    scorer = {
        "accuracy": "accuracy",
        "precision_at_risk": make_scorer(
            precision_score, pos_label=1, zero_division=0
        ),
        "recall_at_risk": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f1_at_risk": make_scorer(f1_score, pos_label=1, zero_division=0),
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    searches = {}
    candidate_pipelines = {}
    cv_rows = []
    threshold_rows = []
    progress = st.progress(0, text="Tuning candidate models...")

    for index, (name, (estimator, parameters)) in enumerate(model_search_spaces().items()):
        search = RandomizedSearchCV(
            estimator=make_pipeline(X_train, estimator),
            param_distributions=parameters,
            n_iter=5,
            scoring=scorer,
            refit="macro_f1",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        searches[name] = search
        candidate_pipelines[name] = search.best_estimator_
        best_index = search.best_index_
        cv_rows.append(
            {
                "Model": name,
                "CV accuracy": search.cv_results_["mean_test_accuracy"][best_index],
                "CV precision (At Risk)": search.cv_results_[
                    "mean_test_precision_at_risk"
                ][best_index],
                "CV recall (At Risk)": search.cv_results_[
                    "mean_test_recall_at_risk"
                ][best_index],
                "CV F1 (At Risk)": search.cv_results_["mean_test_f1_at_risk"][
                    best_index
                ],
                "CV macro F1": search.cv_results_["mean_test_macro_f1"][
                    best_index
                ],
                "CV balanced accuracy": search.cv_results_[
                    "mean_test_balanced_accuracy"
                ][best_index],
                "CV macro F1 std": search.cv_results_["std_test_macro_f1"][
                    best_index
                ],
            }
        )
        oof_probabilities = cross_val_predict(
            search.best_estimator_,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )
        positive_index = list(
            search.best_estimator_.named_steps["model"].classes_
        ).index(1)
        threshold_rows.extend(
            threshold_sweep(y_train, oof_probabilities[:, positive_index], name)
        )
        progress.progress(
            (index + 1) / (len(model_search_spaces()) + 2),
            text=f"Tuned {name} and generated training OOF probabilities",
        )

    dummy_pipeline = make_pipeline(X_train, dummy_search_baseline())
    dummy_pipeline.fit(X_train, y_train)
    dummy_positive_index = list(dummy_pipeline.named_steps["model"].classes_).index(1)
    dummy_cv = cross_validate(
        dummy_pipeline,
        X_train,
        y_train,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    dummy_oof = cross_val_predict(
        dummy_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )
    dummy_cv_row = {"Model": "Dummy (most frequent)"}
    for metric_name, output_name in (
        ("accuracy", "CV accuracy"),
        ("precision_at_risk", "CV precision (At Risk)"),
        ("recall_at_risk", "CV recall (At Risk)"),
        ("f1_at_risk", "CV F1 (At Risk)"),
        ("macro_f1", "CV macro F1"),
        ("balanced_accuracy", "CV balanced accuracy"),
    ):
        dummy_cv_row[output_name] = float(
            np.mean(dummy_cv[f"test_{metric_name}"])
        )
    dummy_cv_row["CV macro F1 std"] = float(np.std(dummy_cv["test_macro_f1"]))
    cv_rows.append(dummy_cv_row)
    threshold_rows.extend(
        threshold_sweep(
            y_train,
            dummy_oof[:, dummy_positive_index],
            "Dummy (most frequent)",
        )
    )
    candidate_pipelines["Dummy (most frequent)"] = dummy_pipeline

    reference_pipeline = make_pipeline(X_train, current_reference_estimator())
    reference_oof = cross_val_predict(
        reference_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )
    reference_positive_index = 1
    threshold_rows.extend(
        threshold_sweep(
            y_train,
            reference_oof[:, reference_positive_index],
            "Current HistGradientBoosting reference",
        )
    )
    reference_pipeline.fit(X_train, y_train)

    threshold_results = pd.DataFrame(threshold_rows).sort_values(
        [
            "Macro F1",
            "Balanced accuracy",
            "At-risk F1",
            "At-risk precision",
            "Probability threshold",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    selected_configuration = threshold_results.iloc[0]
    best_name = selected_configuration["Model"]
    decision_threshold = float(selected_configuration["Probability threshold"])
    best_pipeline = (
        reference_pipeline
        if best_name == "Current HistGradientBoosting reference"
        else candidate_pipelines[best_name]
    )
    positive_class_index = list(best_pipeline.named_steps["model"].classes_).index(1)

    # Model and threshold selection are complete before evaluating the test set.
    test_probabilities = best_pipeline.predict_proba(X_test)[:, positive_class_index]
    test_predictions = (test_probabilities >= decision_threshold).astype(int)
    metrics = score_predictions(y_test, test_predictions)
    report = classification_report(
        y_test,
        test_predictions,
        labels=[1, 0],
        target_names=["At Risk", "Performing Well"],
        zero_division=0,
        output_dict=True,
    )
    reference_test_probabilities = reference_pipeline.predict_proba(X_test)[
        :, reference_positive_index
    ]
    reference_test_predictions = (
        reference_test_probabilities >= CURRENT_DECISION_THRESHOLD
    ).astype(int)
    reference_metrics = score_predictions(y_test, reference_test_predictions)
    reference_report = classification_report(
        y_test,
        reference_test_predictions,
        labels=[1, 0],
        target_names=["At Risk", "Performing Well"],
        zero_division=0,
        output_dict=True,
    )
    previous_threshold_predictions = (
        reference_test_probabilities >= PREVIOUS_HIGH_RECALL_THRESHOLD
    ).astype(int)
    previous_threshold_metrics = score_predictions(
        y_test, previous_threshold_predictions
    )
    previous_threshold_report = classification_report(
        y_test,
        previous_threshold_predictions,
        labels=[1, 0],
        target_names=["At Risk", "Performing Well"],
        zero_division=0,
        output_dict=True,
    )
    cv_results = pd.DataFrame(cv_rows).sort_values(
        ["CV macro F1", "CV balanced accuracy"], ascending=False
    ).reset_index(drop=True)
    evaluation = {
        "best_name": best_name,
        "decision_threshold": decision_threshold,
        "reference_metrics": reference_metrics,
        "reference_report": reference_report,
        "reference_predictions": reference_test_predictions,
        "previous_threshold_metrics": previous_threshold_metrics,
        "previous_threshold_report": previous_threshold_report,
        "previous_threshold_predictions": previous_threshold_predictions,
        "metrics": metrics,
        "cv_results": cv_results,
        "threshold_results": threshold_results,
        "predictions": test_predictions,
        "report": report,
        "y_test": y_test,
    }
    st.session_state["evaluation"] = evaluation

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": best_pipeline,
        "dataset": dataset_name,
        "threshold": threshold,
        "decision_threshold": decision_threshold,
        "input_features": X.columns.tolist(),
        "selected_model": best_name,
    }
    joblib.dump(artifact, model_path(dataset_name, threshold))
    progress.progress(1.0, text="Evaluation complete")
    progress.empty()

evaluation = st.session_state.get("evaluation")
if evaluation:
    st.markdown("#### Cross-validated model comparison")
    st.dataframe(
        evaluation["cv_results"].style.format(
            {
                column: "{:.3f}"
                for column in evaluation["cv_results"].columns
                if column != "Model"
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.bar_chart(
        evaluation["cv_results"].set_index("Model")[
            [
                "CV macro F1",
                "CV balanced accuracy",
                "CV precision (At Risk)",
                "CV recall (At Risk)",
                "CV F1 (At Risk)",
            ]
        ]
    )
    st.markdown("#### Training-only out-of-fold threshold sweep")
    st.caption(
        "The previous high-recall cutoff (0.286), current cutoff (0.50), "
        "and 0.35/0.40/0.45 were compared "
        "using the same stratified training folds. Rows rank by macro F1, "
        "then balanced accuracy."
    )
    st.dataframe(
        evaluation["threshold_results"].style.format(
            {
                "Probability threshold": "{:.3f}",
                "Accuracy": "{:.3f}",
                "At-risk precision": "{:.3f}",
                "At-risk recall": "{:.3f}",
                "At-risk F1": "{:.3f}",
                "Macro F1": "{:.3f}",
                "Balanced accuracy": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.success(
        f"Selected from training OOF results: **{evaluation['best_name']}** "
        f"at probability cutoff **{evaluation['decision_threshold']:.3f}**. "
        "Selection maximized macro F1, with balanced accuracy as tie-breaker. "
        f"Saved pipeline: `{model_path(dataset_name, threshold).relative_to(ROOT)}`."
    )

    st.markdown("#### Held-out test comparison")
    baseline_name = "Current HistGradientBoosting (cutoff 0.50)"
    previous_threshold_name = "Previous high-recall cutoff (0.286)"
    selected_name = (
        f"Selected: {evaluation['best_name']} "
        f"(cutoff {evaluation['decision_threshold']:.3f})"
    )
    comparison = pd.DataFrame(
        {
            baseline_name: evaluation["reference_metrics"],
            previous_threshold_name: evaluation["previous_threshold_metrics"],
            selected_name: evaluation["metrics"],
        }
    ).transpose()
    st.dataframe(comparison.style.format("{:.3f}"), width="stretch")
    if evaluation["metrics"]["At-risk recall"] < LOW_RECALL_WARNING_THRESHOLD:
        st.warning(
            "At Risk recall remains below 50% on the held-out test set. Many "
            "at-risk students may not be flagged."
        )
    if evaluation["metrics"]["Macro F1"] <= evaluation["reference_metrics"]["Macro F1"]:
        st.info(
            "The selected configuration did not improve held-out macro F1 over "
            "the current reference on this split. Selection still used "
            "training-only out-of-fold results."
        )
    st.caption(
        "The current model is refit on training data and evaluated at 0.50. "
        "The previous high-recall cutoff is also shown for context. Neither "
        "test result was used for selection."
    )

    st.markdown("##### Classification reports")
    report_columns = st.columns(3)
    for column, report_name, report_values in (
        (report_columns[0], baseline_name, evaluation["reference_report"]),
        (
            report_columns[1],
            previous_threshold_name,
            evaluation["previous_threshold_report"],
        ),
        (report_columns[2], selected_name, evaluation["report"]),
    ):
        column.markdown(f"**{report_name}**")
        report_df = pd.DataFrame(report_values).transpose()
        column.dataframe(
            report_df.style.format(
                "{:.3f}", subset=report_df.select_dtypes("number").columns
            ),
            width="stretch",
        )

    st.markdown("##### Confusion matrices (held-out test set)")
    matrix_columns = st.columns(3)
    matrix_data = (
        (
            baseline_name,
            confusion_matrix(
                evaluation["y_test"],
                evaluation["reference_predictions"],
                labels=[1, 0],
            ),
        ),
        (
            previous_threshold_name,
            confusion_matrix(
                evaluation["y_test"],
                evaluation["previous_threshold_predictions"],
                labels=[1, 0],
            ),
        ),
        (
            selected_name,
            confusion_matrix(
                evaluation["y_test"], evaluation["predictions"], labels=[1, 0]
            ),
        ),
    )
    for column, (title, matrix) in zip(matrix_columns, matrix_data):
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["At Risk", "Performing Well"],
            yticklabels=["At Risk", "Performing Well"],
            xlabel="Predicted class",
            ylabel="True class",
            title=title,
        )
        for row in range(2):
            for matrix_column in range(2):
                axis.text(
                    matrix_column,
                    row,
                    str(matrix[row, matrix_column]),
                    ha="center",
                    va="center",
                )
        figure.tight_layout()
        column.pyplot(figure)
        plt.close(figure)

    reference_matrix = confusion_matrix(
        evaluation["y_test"], evaluation["reference_predictions"], labels=[1, 0]
    )
    selected_matrix = confusion_matrix(
        evaluation["y_test"], evaluation["predictions"], labels=[1, 0]
    )
    previous_matrix = confusion_matrix(
        evaluation["y_test"],
        evaluation["previous_threshold_predictions"],
        labels=[1, 0],
    )
    error_columns = st.columns(3)
    error_columns[0].metric(
        "Current reference false positives",
        int(reference_matrix[1, 0]),
        help="Performing Well students incorrectly flagged as At Risk.",
    )
    error_columns[1].metric(
        "Previous cutoff false positives",
        int(previous_matrix[1, 0]),
        help="Performing Well students incorrectly flagged as At Risk.",
    )
    error_columns[2].metric(
        "Selected model false positives",
        int(selected_matrix[1, 0]),
        help="Performing Well students incorrectly flagged as At Risk.",
    )
    false_negative_columns = st.columns(3)
    false_negative_columns[0].metric(
        "Current reference missed At Risk students",
        int(reference_matrix[0, 1]),
        help="At Risk students incorrectly predicted as Performing Well.",
    )
    false_negative_columns[1].metric(
        "Previous cutoff missed At Risk students",
        int(previous_matrix[0, 1]),
        help="At Risk students incorrectly predicted as Performing Well.",
    )
    false_negative_columns[2].metric(
        "Selected model missed At Risk students",
        int(selected_matrix[0, 1]),
        help="At Risk students incorrectly predicted as Performing Well.",
    )
    st.caption(
        "False positives may lead to unnecessary follow-up; false negatives "
        "may leave a student without timely support. Their relative cost is a "
        "policy decision, not something determined by this dataset."
    )

    selected_pipeline = joblib.load(model_path(dataset_name, threshold))["pipeline"]
    fitted_model = selected_pipeline.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        encoded_names = selected_pipeline.named_steps[
            "preprocessing"
        ].get_feature_names_out()
        importances = pd.Series(
            fitted_model.feature_importances_, index=encoded_names
        )
        original_feature_names = [
            next(
                (
                    feature
                    for feature in X.columns
                    if encoded_name == f"numeric__{feature}"
                    or encoded_name.startswith(f"categorical__{feature}_")
                ),
                encoded_name,
            )
            for encoded_name in encoded_names
        ]
        grouped_importances = (
            importances.groupby(original_feature_names)
            .sum()
            .sort_values(ascending=False)
            .head(15)
        )
        st.markdown("##### Top feature importance")
        st.caption(
            "Native tree-based importance from the selected training model; "
            "one-hot encoded categories are grouped under their original field."
        )
        st.bar_chart(grouped_importances, x_label="Importance")
    else:
        selected_positive_index = list(fitted_model.classes_).index(1)
        importance = permutation_importance(
            selected_pipeline,
            X_train,
            y_train,
            scoring=lambda estimator, features, labels: f1_score(
                labels,
                estimator.predict_proba(features)[:, selected_positive_index]
                >= evaluation["decision_threshold"],
                pos_label=1,
                zero_division=0,
            ),
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        permutation_scores = pd.Series(
            importance.importances_mean, index=X_train.columns
        ).sort_values(ascending=False).head(15)
        st.markdown("##### Training-set permutation importance")
        st.caption(
            "Mean drop in At Risk F1 after shuffling each input column, using "
            "the selected probability cutoff. Descriptive only; computed on "
            "training data and not used for model selection."
        )
        st.bar_chart(permutation_scores, x_label="Decrease in At Risk F1")

elif model_path(dataset_name, threshold).exists():
    st.info(
        "A previously saved model exists for these settings. Its pipeline will "
        "be available in the prediction form below. Run the comparison above "
        "to see this session's evaluation results."
    )
else:
    st.info("Run the model comparison to tune, evaluate, and save a pipeline.")

st.subheader("Predict for a student")
saved_model_path = model_path(dataset_name, threshold)
if not saved_model_path.exists():
    st.info(
        "Train and save a model with the selected dataset and threshold before "
        "making predictions."
    )
else:
    try:
        saved_artifact = joblib.load(saved_model_path)
    except (OSError, EOFError, ValueError, KeyError) as error:
        st.error(f"Unable to load the saved model: {error}")
        st.stop()

    if (
        saved_artifact.get("dataset") != dataset_name
        or saved_artifact.get("threshold") != threshold
        or saved_artifact.get("input_features") != X.columns.tolist()
        or "decision_threshold" not in saved_artifact
    ):
        st.warning(
            "The saved model metadata does not match the selected data. "
            "Run model comparison again to create a compatible model."
        )
    else:
        st.caption(
            f"Using the saved **{saved_artifact['selected_model']}** pipeline. "
            "Predictions reflect associations in the selected dataset and "
            "should support, not replace, educator judgment."
        )
        with st.form("student_prediction_form"):
            st.write("Enter the student's available attributes:")
            input_values = {}
            numeric_features = X.select_dtypes(include="number").columns.tolist()
            categorical_features = X.select_dtypes(exclude="number").columns.tolist()
            numeric_fields, categorical_fields = st.columns(2)

            with numeric_fields:
                for feature in numeric_features:
                    minimum = int(X[feature].min())
                    maximum = int(X[feature].max())
                    default = int(X[feature].median())
                    input_values[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=1,
                    )
            with categorical_fields:
                for feature in categorical_features:
                    options = sorted(X[feature].dropna().unique().tolist())
                    input_values[feature] = st.selectbox(feature, options)

            submitted = st.form_submit_button("Predict student category")

        if submitted:
            prediction_input = pd.DataFrame([input_values], columns=X.columns)
            pipeline = saved_artifact["pipeline"]
            if list(prediction_input.columns) != saved_artifact["input_features"]:
                st.error("Prediction inputs do not match the saved model's feature schema.")
                st.stop()
            probabilities = pipeline.predict_proba(prediction_input)[0]
            class_indices = list(pipeline.named_steps["model"].classes_)
            positive_class_index = class_indices.index(1)
            at_risk_probability = probabilities[positive_class_index]
            decision_threshold = saved_artifact["decision_threshold"]
            prediction = int(at_risk_probability >= decision_threshold)
            if prediction == 1:
                st.warning("Prediction: **At Risk**")
            else:
                st.success("Prediction: **Performing Well**")
            probability_by_class = dict(
                zip(pipeline.named_steps["model"].classes_, probabilities)
            )
            probability_columns = st.columns(2)
            probability_columns[0].metric(
                "At Risk probability", f"{probability_by_class[1]:.1%}"
            )
            probability_columns[1].metric(
                "Performing Well probability", f"{probability_by_class[0]:.1%}"
            )
            st.caption(
                f"Prediction cutoff: {decision_threshold:.3f}. "
                "Probabilities are model estimates and are not guaranteed "
                "to be calibrated."
            )

st.divider()
st.caption(
    "Model and cutoff selection use training-only stratified CV and OOF "
    "predictions. The held-out test set is reserved for final comparison. "
    "Predictions are decision-support estimates, not definitive judgments "
    "about students. Saved joblib files should only be loaded from this "
    "project's trusted models folder."
)
