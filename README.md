# Automated Student Performance Prediction Using Machine Learning and AutoML

An end-to-end machine learning project that analyzes student-related
attributes and predicts whether a student is **At Risk** or **Performing
Well**. The project uses the UCI Student Performance dataset, a Jupyter
Notebook training workflow, scikit-learn pipelines, model evaluation,
and a Streamlit dashboard.

> **Current implementation note:** The Jupyter Notebook workflow
> described here was built and evaluated using the **Mathematics
> (`student-mat.csv`) dataset**. The saved pipelines are trained for the
> target definition `G3 < 10`. The current Streamlit integration
> therefore uses the Mathematics dataset and this fixed target
> definition. Do not select the Portuguese dataset or change the grade
> threshold unless you train and save compatible models for those
> settings.

------------------------------------------------------------------------

## Table of Contents

1.  [Project Overview](#1-project-overview)
2.  [Problem Statement](#2-problem-statement)
3.  [Objectives](#3-objectives)
4.  [Key Features](#4-key-features)
5.  [Technology Stack](#5-technology-stack)
6.  [Dataset Description](#6-dataset-description)
7.  [Target Variable and Class
    Mapping](#7-target-variable-and-class-mapping)
8.  [Machine Learning Methodology](#8-machine-learning-methodology)
9.  [Models Implemented](#9-models-implemented)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Observed Experimental Results](#11-observed-experimental-results)
12. [Feature Importance](#12-feature-importance)
13. [Project Structure](#13-project-structure)
14. [Installation and Setup](#14-installation-and-setup)
15. [Running the Jupyter Notebook](#15-running-the-jupyter-notebook)
16. [Running the Streamlit
    Application](#16-running-the-streamlit-application)
17. [How to Use the Dashboard](#17-how-to-use-the-dashboard)
18. [Saved Models and Generated
    Outputs](#18-saved-models-and-generated-outputs)
19. [Reproducibility and Data-Leakage
    Controls](#19-reproducibility-and-data-leakage-controls)
20. [Limitations](#20-limitations)
21. [Future Enhancements](#21-future-enhancements)
22. [Troubleshooting](#22-troubleshooting)
23. [Ethical Use and Disclaimer](#23-ethical-use-and-disclaimer)
24. [References](#24-references)

------------------------------------------------------------------------

## 1. Project Overview

Student performance is influenced by several academic, personal, family,
and social factors. Identifying students who may be at academic risk
early can help educators decide where additional attention or support
may be useful.

This project develops a machine learning-based student performance
prediction system. It uses student attributes from the UCI Student
Performance dataset to classify students into two categories:

-   **At Risk:** The student's final Mathematics grade (`G3`) is below
    10.
-   **Performing Well:** The student's final Mathematics grade (`G3`) is
    10 or above.

The training workflow is implemented in a Jupyter Notebook. Multiple
classification algorithms are trained and evaluated using a consistent
train/test split. The notebook also generates comparison tables,
evaluation graphs, a confusion matrix, and feature-importance analysis.
Trained preprocessing-and-model pipelines are saved as `.pkl` files.

A Streamlit application loads these saved pipelines and provides an
interactive dashboard. Users can review dataset information, compare
recorded model results, inspect evaluation metrics, and enter student
attributes to obtain a model prediction.

### Project workflow

``` text
UCI Student Performance Dataset
              |
              v
       Data Inspection
              |
              v
    Exploratory Data Analysis
              |
              v
  Target Creation (G3 < 10)
              |
              v
 Exclude G1, G2 and G3 from Inputs
              |
              v
   Stratified Train/Test Split
              |
              v
      Data Preprocessing
              |
              v
 Train and Compare ML Classifiers
              |
              v
 Cross-Validation and Model Tuning
              |
              v
       Model Evaluation
              |
              v
  Save Trained Pipelines and Results
              |
              v
       Streamlit Dashboard
              |
              v
  Individual Student Prediction
```

------------------------------------------------------------------------

## 2. Problem Statement

Educational institutions may need to identify students who could benefit
from additional academic support. Reviewing student information manually
can be time-consuming, and a student's academic outcome may be
associated with multiple factors.

The aim of this project is to develop a machine learning-based system
that analyzes available student attributes and predicts a binary
academic-performance category. The system compares multiple
classification algorithms and presents evaluation results through an
interactive dashboard.

The prediction is intended to support academic review. It is not
intended to replace educators' judgment or make consequential decisions
about students automatically.

------------------------------------------------------------------------

## 3. Objectives

The main objectives are:

1.  Analyze student-related data using the UCI Student Performance
    dataset.
2.  Perform exploratory data analysis to understand feature
    distributions and the final-grade distribution.
3.  Transform the final grade into a binary classification target.
4.  Exclude earlier and final grade columns from model inputs to support
    an early-prediction setup.
5.  Preprocess numerical and categorical attributes using scikit-learn.
6.  Train and compare multiple machine learning classification
    algorithms.
7.  Evaluate models using accuracy, precision, recall, F1-score, macro
    F1, and confusion matrices.
8.  use stratified cross-validation to assess model performance across
    multiple training folds.
9.  Apply hyperparameter tuning to a candidate model.
10. Identify influential input attributes using model feature
    importance.
11. Save trained model pipelines and evaluation outputs.
12. Build a Streamlit dashboard for model evaluation and individual
    student prediction.

------------------------------------------------------------------------

## 4. Key Features

### 4.1 Dataset overview

The dashboard displays the number of student records, number of input
attributes, training sample count, and number of at-risk students. It
also visualizes the performance-class distribution and final-grade
distribution.

### 4.2 Multiple-model experimentation

The notebook trains and compares seven classifiers:

-   Logistic Regression
-   Decision Tree
-   Random Forest
-   Extra Trees
-   Support Vector Machine (SVM)
-   Gradient Boosting
-   HistGradientBoosting

### 4.3 Data preprocessing

Numerical features are standardized using `StandardScaler`. Categorical
features are transformed using `OneHotEncoder` with
`handle_unknown="ignore"`.

The preprocessing transformer is placed inside a scikit-learn
`Pipeline`, so it is fitted on training data rather than on the complete
dataset before splitting.

### 4.4 Model evaluation

The notebook evaluates the classifiers on the held-out test set and
produces a comparison table. It also calculates cross-validation metrics
on the training set.

### 4.5 Hyperparameter tuning

`GridSearchCV` is used to test Logistic Regression configurations using
five-fold stratified cross-validation and macro F1 as the search
objective.

### 4.6 Visualizations

The notebook creates:

-   Final-grade distribution
-   Student performance class distribution
-   Model accuracy comparison
-   At-risk precision, recall, and F1 comparison
-   Random Forest confusion matrix
-   Random Forest feature-importance graph

### 4.7 Model persistence

The trained pipelines are stored with `joblib`. The complete pipeline
includes the fitted preprocessing transformer and classifier.

### 4.8 Interactive prediction

The Streamlit dashboard accepts student attributes and uses a saved
pipeline to predict the student's category and display class
probabilities.

------------------------------------------------------------------------

## 5. Technology Stack

  -----------------------------------------------------------------------
  Technology                          Purpose
  ----------------------------------- -----------------------------------
  Python                              Main programming language

  Jupyter Notebook                    Data analysis, model training, and
                                      experimentation

  Visual Studio Code                  Development environment

  pandas                              Dataset loading and tabular data
                                      manipulation

  NumPy                               Numerical operations and target
                                      creation

  scikit-learn                        Preprocessing, classification,
                                      model selection, and evaluation

  Matplotlib                          Data visualization

  Seaborn                             Statistical plots and
                                      confusion-matrix visualization

  joblib                              Saving and loading trained model
                                      pipelines

  Streamlit                           Interactive web dashboard

  Git and GitHub                      Version control and project hosting
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 6. Dataset Description

The project uses the **Student Performance** dataset from the UCI
Machine Learning Repository.

The dataset contains student information related to demographic
background, family circumstances, study habits, school-related factors,
social activities, health, absences, and academic grades.

The dataset provides two subject files:

  File                Subject         Records in the downloaded project copy
  ------------------- ------------- ----------------------------------------
  `student-mat.csv`   Mathematics                                        395
  `student-por.csv`   Portuguese                                         649

The current notebook and saved model pipelines use `student-mat.csv`.

### 6.1 Input attributes

The Mathematics dataset includes attributes such as:

-   `school`: Student's school.
-   `sex`: Student's sex as recorded in the dataset.
-   `age`: Student's age.
-   `address`: Type of home address.
-   `famsize`: Family size category.
-   `Pstatus`: Parent cohabitation status.
-   `Medu`: Mother's education.
-   `Fedu`: Father's education.
-   `Mjob`: Mother's job.
-   `Fjob`: Father's job.
-   `reason`: Reason for choosing the school.
-   `guardian`: Student's guardian.
-   `traveltime`: Travel time to school.
-   `studytime`: Weekly study-time category.
-   `failures`: Number of past class failures.
-   `schoolsup`: Extra educational support.
-   `famsup`: Family educational support.
-   `paid`: Extra paid classes.
-   `activities`: Extra-curricular activities.
-   `nursery`: Attended nursery school.
-   `higher`: Wants to pursue higher education.
-   `internet`: Internet access at home.
-   `romantic`: Romantic relationship status.
-   `famrel`: Quality of family relationships.
-   `freetime`: Free time after school.
-   `goout`: Going out with friends.
-   `Dalc`: Workday alcohol-consumption category in the source dataset.
-   `Walc`: Weekend alcohol-consumption category in the source dataset.
-   `health`: Current health status.
-   `absences`: Number of school absences.

The exact feature list is obtained from the CSV file by the notebook.
The model input columns are all columns except `G1`, `G2`, and `G3`.

### 6.2 Dataset files

Place the downloaded CSV files in the `data/` directory:

``` text
data/
├── student-mat.csv
└── student-por.csv
```

The CSV files use semicolon (`;`) separators. The notebook and
application therefore load them using:

``` python
pd.read_csv("student-mat.csv", sep=";")
```

### 6.3 Dataset source

UCI Machine Learning Repository -- Student Performance:

https://archive.ics.uci.edu/dataset/320/student+performance

Dataset citation and original publication details should be checked
against the repository record when preparing the final academic paper.

------------------------------------------------------------------------

## 7. Target Variable and Class Mapping

The target variable is created from the final Mathematics grade, `G3`.

The project uses a threshold of 10:

``` python
df["performance"] = np.where(df["G3"] < 10, 0, 1)
```

### Class definitions

    Encoded label Category          Condition
  --------------- ----------------- ------------
                0 At Risk           `G3 < 10`
                1 Performing Well   `G3 >= 10`

**Important:** The class mapping must remain consistent throughout
training, evaluation, model saving, and Streamlit prediction.

The Jupyter notebook uses `0` for At Risk and `1` for Performing Well.
The integrated Streamlit application must interpret predictions and
probability columns using the same mapping.

For example:

``` python
class_to_probability = dict(zip(model.classes_, probabilities))

at_risk_probability = class_to_probability[0]
performing_well_probability = class_to_probability[1]
```

Do not assume that the second probability column always represents At
Risk. Use the classifier's `classes_` attribute to locate the correct
class.

### Why are G1, G2, and G3 excluded from input features?

`G3` defines the target. Including it as an input would directly reveal
the outcome and cause target leakage.

`G1` and `G2` are earlier period grades. They are also excluded to make
the prediction use non-grade student attributes rather than earlier
grade outcomes.

The notebook uses:

``` python
X = df.drop(columns=["G1", "G2", "G3", "performance"])
y = df["performance"]
```

This results in 30 input features for the Mathematics dataset.

------------------------------------------------------------------------

## 8. Machine Learning Methodology

### 8.1 Data loading

The Mathematics CSV is loaded into a pandas DataFrame. The notebook
inspects the first records, dataset dimensions, data types, missing
values, and descriptive statistics.

### 8.2 Exploratory data analysis

Exploratory analysis is performed to understand the data before
training.

The notebook examines:

-   Final-grade frequency distribution.
-   Final-grade descriptive statistics.
-   At-risk and performing-well class counts.
-   Class imbalance in the binary target.

### 8.3 Feature selection

The target is derived from `G3`. The columns `G1`, `G2`, and `G3` are
excluded from the input matrix.

The remaining 30 attributes are used as model inputs.

### 8.4 Train/test split

The notebook uses an 80:20 train/test split:

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

-   80% of records are used for training.
-   20% are reserved for final test evaluation.
-   `random_state=42` makes the split reproducible.
-   `stratify=y` preserves class proportions as closely as possible in
    both subsets.

For the 395-record Mathematics dataset, this produces 316 training
records and 79 test records.

### 8.5 Preprocessing

The notebook identifies numerical and categorical columns separately.

Numerical columns:

``` python
StandardScaler()
```

Categorical columns:

``` python
OneHotEncoder(handle_unknown="ignore")
```

The transformations are combined using `ColumnTransformer`:

``` python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)
```

The preprocessor is then included in a model pipeline. This ensures that
the transformations are fitted when the pipeline is trained.

### 8.6 Model training

Seven classification algorithms are trained using the same training and
test partitions and preprocessing configuration.

The notebook uses class weighting for several models to account for
differences in class frequency.

### 8.7 Model comparison

Each trained model predicts the held-out test records. Accuracy and
class-specific metrics are calculated and stored in a pandas DataFrame.

The project pays particular attention to the At Risk class because
failing to identify an at-risk student may mean that the student does
not receive timely support.

### 8.8 Cross-validation

Five-fold `StratifiedKFold` cross-validation is applied to the training
data.

The training data is divided into five folds. Each fold is used for
validation once, while the remaining folds are used for fitting.

The notebook first calculates general classification metrics. It then
explicitly calculates precision, recall, and F1-score for class `0` (At
Risk).

This distinction is important because scikit-learn's default binary
precision, recall, and F1 scoring treats class `1` as the positive class
unless the positive label is specified.

### 8.9 Hyperparameter tuning

The notebook uses `GridSearchCV` for Logistic Regression.

The search tests:

``` python
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__class_weight": [
        "balanced",
        {0: 1, 1: 1},
        {0: 2, 1: 1},
        {0: 3, 1: 1}
    ]
}
```

The search uses five-fold stratified cross-validation and macro F1
scoring.

Macro F1 gives equal weight to the F1-score of each class, rather than
allowing the more frequent class to dominate the metric.

### 8.10 Final test evaluation

The held-out test set is used to assess how the trained pipelines
perform on records not used for fitting.

The test set should not be used to select hyperparameters, choose a
probability threshold, or repeatedly optimize model configurations.

### 8.11 Saving the trained pipelines

The notebook saves the trained pipelines using `joblib`.

A saved pipeline contains both preprocessing and the classifier. This is
important because a model trained on scaled and one-hot-encoded data
must receive the same transformations during prediction.

------------------------------------------------------------------------

## 9. Models Implemented

  -----------------------------------------------------------------------
  Model                               Description
  ----------------------------------- -----------------------------------
  Logistic Regression                 Linear classification model that
                                      estimates class probabilities.

  Decision Tree                       Tree-based model that learns
                                      decision rules from feature values.

  Random Forest                       Ensemble of decision trees trained
                                      using randomized samples and
                                      feature selection.

  Extra Trees                         Tree ensemble that introduces
                                      additional randomization when
                                      selecting splits.

  Support Vector Machine              Classifier that identifies a
                                      separating boundary between
                                      classes.

  Gradient Boosting                   Sequential ensemble method in which
                                      new trees attempt to correct
                                      previous errors.

  HistGradientBoosting                Histogram-based gradient boosting
                                      implementation.
  -----------------------------------------------------------------------

These algorithms provide different modeling approaches. Their
performance is compared using the same dataset split and evaluation
metrics.

------------------------------------------------------------------------

## 10. Evaluation Metrics

### 10.1 Accuracy

Accuracy measures the fraction of all predictions that are correct.

\[ Accuracy = `\frac{TP + TN}{TP + TN + FP + FN}`{=tex} \]

Accuracy alone may be misleading when class frequencies differ.

### 10.2 Precision for At Risk

Precision measures how many students predicted as At Risk actually
belong to the At Risk class.

\[ Precision = `\frac{TP}{TP + FP}`{=tex} \]

Higher precision means fewer performing-well students are incorrectly
flagged.

### 10.3 Recall for At Risk

Recall measures how many actual at-risk students are identified by the
model.

\[ Recall = `\frac{TP}{TP + FN}`{=tex} \]

This is an important metric for this project because low recall means
the model misses students who may need additional support.

### 10.4 F1-score

F1-score is the harmonic mean of precision and recall.

\[ F1 = 2
`\times `{=tex}`\frac{Precision \times Recall}{Precision + Recall}`{=tex}
\]

### 10.5 Macro F1

Macro F1 calculates the F1-score separately for each class and takes
their unweighted average.

It gives both classes equal importance.

### 10.6 Balanced accuracy

Balanced accuracy is the average of recall obtained on each class. It is
useful when the class distribution is uneven.

### 10.7 Confusion matrix

For this project, the confusion matrix uses the label order:

``` python
labels=[0, 1]
```

Therefore, its rows and columns are:

  -----------------------------------------------------------------------
                          Predicted At Risk       Predicted Performing
                                                  Well
  ----------------------- ----------------------- -----------------------
  Actual At Risk          Correctly identified    At-risk students missed
                          at-risk students        by the model

  Actual Performing Well  Performing-well         Correctly identified
                          students incorrectly    performing-well
                          flagged                 students
  -----------------------------------------------------------------------

Always verify the label order before interpreting a confusion matrix.

------------------------------------------------------------------------

## 11. Observed Experimental Results

The following values are the results recorded during the notebook
experiment for the Mathematics dataset, using the 80:20 split with
`random_state=42`.

### 11.1 Held-out test comparison

  Model                    Accuracy   At-risk precision   At-risk recall   At-risk F1
  ---------------------- ---------- ------------------- ---------------- ------------
  Logistic Regression         0.595               0.400            0.462        0.429
  Decision Tree               0.595               0.417            0.577        0.484
  Random Forest               0.671               0.500            0.423        0.458
  Extra Trees                 0.671               0.500            0.231        0.316
  SVM                         0.633               0.440            0.423        0.431
  Gradient Boosting           0.646               0.438            0.269        0.333
  HistGradientBoosting        0.620               0.400            0.308        0.348

### 11.2 Interpretation

-   Random Forest and Extra Trees achieved the highest test accuracy in
    this experiment, approximately 67.1%.
-   Decision Tree achieved the highest at-risk recall on this particular
    test split, approximately 57.7%.
-   Decision Tree also achieved the highest at-risk F1-score on this
    test split, approximately 0.484.
-   Extra Trees and Gradient Boosting had relatively low at-risk recall
    on this split.
-   The highest-accuracy model is not automatically the best model for
    identifying at-risk students.

### 11.3 Cross-validation results

The notebook recorded the following five-fold cross-validation results
for class `0` (At Risk):

  Model                    At-risk precision   At-risk recall   At-risk F1
  ---------------------- ------------------- ---------------- ------------
  Logistic Regression                  0.512            0.597        0.549
  Decision Tree                        0.393            0.328        0.348
  Random Forest                        0.561            0.404        0.467
  Extra Trees                          0.455            0.230        0.303
  SVM                                  0.469            0.395        0.425
  Gradient Boosting                    0.568            0.309        0.395
  HistGradientBoosting                 0.528            0.328        0.399

Logistic Regression recorded the highest mean at-risk recall and
F1-score in this cross-validation experiment.

### 11.4 Tuned Logistic Regression

The notebook's GridSearchCV experiment tested 20 combinations over five
folds, resulting in 100 fits.

Recorded best cross-validation macro F1:

``` text
0.658
```

The tuned Logistic Regression model was then evaluated on the held-out
test set:

  Metric                Original Logistic Regression   Tuned Logistic Regression
  ------------------- ------------------------------ ---------------------------
  Accuracy                                     0.595                       0.582
  At-risk precision                            0.400                       0.387
  At-risk recall                               0.462                       0.462
  At-risk F1                                   0.429                       0.421

Tuning did not improve the held-out test results in this experiment.
This is a valid result and should be reported honestly.

### 11.5 Random Forest confusion matrix

The recorded Random Forest confusion matrix was:

\[
```{=tex}
\begin{bmatrix}
11 & 15\\
11 & 42
\end{bmatrix}
```
\]

Using the order `[At Risk, Performing Well]`:

-   11 at-risk students were correctly identified.
-   15 at-risk students were incorrectly predicted as performing well.
-   11 performing-well students were incorrectly predicted as at risk.
-   42 performing-well students were correctly identified.

This corresponds to an at-risk recall of:

\[ `\frac{11}{11+15}`{=tex} `\approx 0.423`{=tex} \]

The result shows why at-risk recall must be considered alongside
accuracy.

> These are experimental results from one particular split and
> cross-validation run. They should not be presented as guaranteed
> performance on new institutions or student populations. Re-run the
> notebook if the data, preprocessing, software versions, random seed,
> or model configuration changes.

------------------------------------------------------------------------

## 12. Feature Importance

The notebook extracts Random Forest feature-importance values and
visualizes the top 15 attributes.

The recorded leading attributes were:

1.  `failures`
2.  `absences`
3.  `goout`
4.  `age`
5.  `health`
6.  `freetime`
7.  `Walc`
8.  `Medu`
9.  `Fedu`
10. `famrel`
11. `studytime`
12. `traveltime`
13. `Dalc`
14. `schoolsup_yes`
15. `guardian_father`

### Interpretation

-   `failures` received the highest feature-importance score in the
    trained Random Forest.
-   `absences` was another highly ranked attribute.
-   Social and lifestyle-related attributes also contributed to the
    model's predictions.
-   Family and study-related attributes appeared among the leading
    features.

Feature importance describes how the trained model uses attributes. It
does **not** establish that an attribute causes a particular academic
outcome.

The ranking may change when the dataset split, model parameters, or
training data changes.

------------------------------------------------------------------------

## 13. Project Structure

The project directory is organized as follows:

``` text
Automated Student Performance Prediction/
│
├── data/
│   ├── student-mat.csv
│   └── student-por.csv
│
├── models/
│   ├── random_forest_model.pkl
│   ├── tuned_logistic_model.pkl
│   ├── model_comparison_results.csv
│   ├── cross_validation_results.csv
│   ├── at_risk_cv_results.csv
│   ├── accuracy_comparison.png
│   ├── at_risk_performance_comparison.png
│   ├── random_forest_confusion_matrix.png
│   └── feature_importance.png
│
├── notebooks/
│   └── model_training.ipynb
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
```

The `venv/` directory is a local Python virtual environment. It should
not be committed to GitHub.

The exact contents of `models/` may include additional files created
during previous experiments. Keep the model files required by the
current application.

------------------------------------------------------------------------

## 14. Installation and Setup

### 14.1 Prerequisites

Install the following:

-   Python
-   Visual Studio Code
-   Python extension for VS Code
-   Jupyter extension for VS Code
-   Git (optional, for version control)

The project has been developed and tested in a Windows environment.
Commands below use PowerShell syntax.

### 14.2 Clone the repository

If the project is hosted on GitHub:

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

``` bash
cd "Automated Student Performance Prediction"
```

Replace the placeholder with the actual repository URL.

### 14.3 Create a virtual environment

``` bash
python -m venv venv
```

### 14.4 Activate the environment

In Windows PowerShell:

``` powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the appropriate execution-policy
setting for your machine or activate the environment through VS Code's
Python interpreter selector.

### 14.5 Install dependencies

If `requirements.txt` is present:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If dependencies need to be installed individually:

``` bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib jupyter ipykernel
```

### 14.6 Select the VS Code interpreter

1.  Open the project folder in VS Code.
2.  Open the Command Palette using `Ctrl + Shift + P`.
3.  Search for `Python: Select Interpreter`.
4.  Select the interpreter inside the project's `venv` directory.
5.  Open the notebook and select the same environment as its kernel.

The notebook kernel and Streamlit terminal should use the same virtual
environment.

------------------------------------------------------------------------

## 15. Running the Jupyter Notebook

Open the project in VS Code.

1.  Open `notebooks/model_training.ipynb`.
2.  Select the project's virtual environment as the kernel.
3.  Run the notebook cells in order.

The notebook workflow includes:

1.  Importing libraries.
2.  Loading the Mathematics dataset.
3.  Inspecting data types and missing values.
4.  Exploring the final-grade distribution.
5.  Creating the binary target.
6.  Selecting input features.
7.  Splitting training and test data.
8.  Identifying numerical and categorical columns.
9.  Creating the preprocessing transformer.
10. Training Logistic Regression.
11. Training Decision Tree.
12. Training Random Forest.
13. Training Extra Trees.
14. Training SVM.
15. Training Gradient Boosting.
16. Training HistGradientBoosting.
17. Evaluating all trained models.
18. Creating the model comparison table.
19. Generating model-comparison graphs.
20. Performing stratified cross-validation.
21. Calculating cross-validation metrics for the At Risk class.
22. Tuning Logistic Regression using GridSearchCV.
23. Evaluating the tuned model on the held-out test set.
24. Generating the confusion matrix.
25. Analyzing feature importance.
26. Saving trained pipelines.
27. Saving evaluation CSV files and graphs.
28. Loading saved models and verifying predictions.

### Notebook execution

Run a cell using:

``` text
Shift + Enter
```

Execute the notebook from top to bottom so that variables and trained
model objects are created in the correct order.

### Expected outputs

After successful execution, the `models/` directory should contain the
saved pipelines, CSV comparison results, and generated plots.

If you change the training code, rerun the affected cells and regenerate
the model artifacts before using the updated application.

------------------------------------------------------------------------

## 16. Running the Streamlit Application

The integrated application loads the saved Jupyter-trained pipelines.

From the project root directory, activate the virtual environment and
run:

``` bash
streamlit run app.py
```

Streamlit normally opens the application in a browser.

If the browser does not open automatically, use the local URL printed in
the terminal, commonly:

``` text
http://localhost:8501
```

To stop the server, press:

``` text
Ctrl + C
```

### Required files

Before launching the application, verify that these files exist:

``` text
data/student-mat.csv
models/random_forest_model.pkl
models/tuned_logistic_model.pkl
models/model_comparison_results.csv
models/cross_validation_results.csv
models/at_risk_cv_results.csv
```

The dashboard can display some information even if optional result CSV
files are missing, but the corresponding comparison sections will not be
available.

------------------------------------------------------------------------

## 17. How to Use the Dashboard

### 17.1 Open the application

Run:

``` bash
streamlit run app.py
```

### 17.2 Review the dataset overview

The dashboard displays:

-   Total number of students.
-   Number of model input features.
-   Training sample count.
-   Number of students classified as at risk.
-   Student performance-class distribution.
-   Final-grade distribution.

### 17.3 Select a trained model

The sidebar provides the saved model choices:

-   Random Forest
-   Tuned Logistic Regression

Both models were trained using the Mathematics dataset and the target
definition `G3 < 10`.

### 17.4 Review model performance

The application evaluates the selected saved model using the same
stratified 80:20 split and random state used in the notebook.

The dashboard displays:

-   Accuracy.
-   At-risk precision.
-   At-risk recall.
-   At-risk F1-score.
-   Macro F1.
-   Classification report.
-   Confusion matrix.

The application also displays the saved seven-model comparison and
cross-validation results.

### 17.5 Review feature importance

The dashboard uses the saved Random Forest pipeline to calculate and
display the top feature-importance values.

These values describe the model's learned patterns; they should not be
interpreted as causal effects.

### 17.6 Enter student details

In the individual prediction section:

1.  Enter the student's numerical attributes.
2.  Select categorical attribute values.
3.  Submit the form.
4.  Review the predicted performance category.
5.  Review the estimated probabilities for both classes.

The output uses the notebook's class mapping:

-   `0` = At Risk.
-   `1` = Performing Well.

The prediction is based on the selected saved model and the input
attributes. It does not use G1, G2, or G3 as prediction inputs.

------------------------------------------------------------------------

## 18. Saved Models and Generated Outputs

### 18.1 Saved model pipelines

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `random_forest_model.pkl`           Trained Random Forest pipeline,
                                      including preprocessing.

  `tuned_logistic_model.pkl`          Tuned Logistic Regression pipeline,
                                      including preprocessing.
  -----------------------------------------------------------------------

The `.pkl` files are generated using `joblib`.

Load a saved pipeline with:

``` python
import joblib

model = joblib.load("models/random_forest_model.pkl")
```

Generate a prediction using a DataFrame containing the same input
columns:

``` python
prediction = model.predict(student_input)
```

Generate class probabilities:

``` python
probabilities = model.predict_proba(student_input)
```

Use `model.named_steps["classifier"].classes_` to identify the
probability-column order.

**Security note:** Only load trusted model files. Loading an untrusted
pickle/joblib file can execute arbitrary code.

### 18.2 Saved result files

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `model_comparison_results.csv`      Held-out test comparison for the
                                      seven models.

  `cross_validation_results.csv`      Cross-validation results recorded
                                      by the notebook.

  `at_risk_cv_results.csv`            Cross-validation precision, recall,
                                      and F1 explicitly calculated for
                                      class 0.
  -----------------------------------------------------------------------

### 18.3 Saved graph files

  --------------------------------------------------------------------------
  File                                   Description
  -------------------------------------- -----------------------------------
  `accuracy_comparison.png`              Accuracy comparison across seven
                                         classifiers.

  `at_risk_performance_comparison.png`   At-risk precision, recall, and F1
                                         comparison.

  `random_forest_confusion_matrix.png`   Random Forest confusion matrix.

  `feature_importance.png`               Top 15 Random Forest
                                         feature-importance values.
  --------------------------------------------------------------------------

These images can be used as figures in the project report or IEEE paper
after checking figure numbering, captions, resolution, and formatting.

------------------------------------------------------------------------

## 19. Reproducibility and Data-Leakage Controls

The following practices are important for the validity of the
experiment.

### 19.1 Fixed random state

The notebook uses `random_state=42` for the train/test split and
relevant randomized algorithms.

This helps reproduce the same experiment when the dataset and software
environment are unchanged.

### 19.2 Stratified split

Stratification helps preserve the class distribution in training and
test partitions.

### 19.3 Exclusion of grade columns

`G1`, `G2`, and `G3` are excluded from the model inputs. This avoids
using the target grade directly or indirectly through earlier grade
columns.

### 19.4 Preprocessing inside pipelines

Scaling and one-hot encoding are fitted as part of the model pipeline.
During cross-validation, each fold fits preprocessing using only that
fold's training portion.

### 19.5 Separate training and test data

The test set is reserved for final evaluation. It should not be
repeatedly used to choose hyperparameters or thresholds.

### 19.6 Class-specific metrics

The project explicitly evaluates class `0` as At Risk. This prevents
confusion caused by scikit-learn's default positive-label convention.

### 19.7 Subject datasets are not silently combined

The Mathematics and Portuguese files are treated as separate subject
datasets. Their records should not be concatenated without a clear
strategy for possible overlap between students and for subject-specific
target definitions.

### 19.8 Version consistency

A saved scikit-learn pipeline may not be compatible with a substantially
different scikit-learn version. Keep the training and deployment
environments consistent. Record package versions when preparing a
reproducible release.

------------------------------------------------------------------------

## 20. Limitations

1.  **Small dataset:** The Mathematics dataset contains 395 records.
    Results from a small dataset can vary noticeably across different
    train/test splits.
2.  **Single subject:** The current trained pipelines use Mathematics
    data only.
3.  **Binary target:** The project reduces final grades to two
    categories and does not predict the exact numerical grade.
4.  **Threshold dependence:** The current At Risk definition uses
    `G3 < 10`. Different thresholds will change the class distribution
    and evaluation results.
5.  **Model performance:** The recorded test accuracy values are
    moderate. The models do not identify every at-risk student.
6.  **False negatives:** A student who is actually at risk may be
    predicted as performing well.
7.  **Feature importance:** Feature-importance values are
    model-dependent and do not demonstrate causality.
8.  **Probability calibration:** Predicted probabilities are model
    estimates and may not be calibrated.
9.  **Generalization:** Performance on this dataset does not guarantee
    similar results at another school, institution, or student
    population.
10. **Human review:** Predictions should not be used as the sole basis
    for academic interventions or student decisions.

------------------------------------------------------------------------

## 21. Future Enhancements

Potential improvements include:

-   Evaluate additional datasets from different institutions.
-   Train and validate a separate model for Portuguese performance.
-   Perform repeated stratified cross-validation to estimate performance
    variability.
-   Compare additional algorithms and carefully justified ensemble
    methods.
-   Explore threshold selection based on an explicitly defined cost of
    false negatives and false positives.
-   Evaluate probability calibration.
-   Use permutation importance or SHAP for more detailed model
    interpretation.
-   Add model-version and experiment tracking.
-   Add downloadable prediction reports.
-   Add student-level explanations with appropriate safeguards.
-   Improve the dashboard's accessibility and responsive layout.
-   Deploy the Streamlit application using Streamlit Community Cloud or
    another hosting platform.
-   Monitor model performance when new, appropriately governed data
    becomes available.

Any new model or threshold should be validated on data that was not used
during model selection.

------------------------------------------------------------------------

## 22. Troubleshooting

### Problem: `ModuleNotFoundError`

**Cause:** A required package is not installed in the active Python
environment.

**Solution:**

``` bash
pip install -r requirements.txt
```

Check that VS Code and the terminal use the project's virtual
environment.

### Problem: `streamlit` is not recognized

**Solution:**

``` bash
python -m pip install streamlit
python -m streamlit run app.py
```

### Problem: Saved model file not found

**Cause:** The `.pkl` file is missing or stored in a different
directory.

**Solution:** Verify that the model files are inside the root-level
`models/` directory:

``` text
project/
├── app.py
└── models/
    ├── random_forest_model.pkl
    └── tuned_logistic_model.pkl
```

### Problem: Model loading error

**Possible causes:**

-   The model file is incomplete.
-   The wrong file was loaded.
-   The training and application environments use incompatible package
    versions.
-   The model file is not trusted or is corrupted.

**Solution:** Confirm the file path and use the same scikit-learn and
joblib versions used during training. If necessary, rerun the notebook's
training and saving cells.

### Problem: Feature-name or feature-count mismatch

**Cause:** The input DataFrame does not have the same columns as the
training data.

**Solution:** Keep the same Mathematics dataset and exclude `G1`, `G2`,
and `G3` from prediction inputs. Preserve the original feature names and
column order.

### Problem: Predictions show the wrong category

**Cause:** The encoded class labels may have been interpreted
incorrectly.

**Solution:** Use the notebook's mapping:

``` python
0 = At Risk
1 = Performing Well
```

Use the estimator's `classes_` attribute to map predicted probabilities
to their corresponding labels.

### Problem: Notebook kernel is not available

**Solution:**

1.  Install the VS Code Python and Jupyter extensions.
2.  Activate the project virtual environment.
3.  Install `ipykernel`:

``` bash
pip install ipykernel
```

4.  Select the project's `venv` interpreter as the notebook kernel.

### Problem: Dashboard does not open

**Solution:**

``` bash
python -m streamlit run app.py
```

Check the terminal for error messages and use the local URL printed by
Streamlit.

------------------------------------------------------------------------

## 23. Ethical Use and Disclaimer

This project is an academic machine learning prototype.

Its predictions are estimates based on patterns in the dataset. They are
not definitive assessments of a student's ability, motivation,
circumstances, or future.

Student information can be sensitive. Any real-world use should follow
applicable privacy, consent, data-protection, fairness, and
institutional policies.

The system should be used as a decision-support tool for educators, not
as an automated mechanism for punishment, exclusion, grading, or other
high-impact decisions.

A human should review relevant context before any intervention is made.

------------------------------------------------------------------------

## 24. References

1.  UCI Machine Learning Repository, **Student Performance Dataset**.\
    https://archive.ics.uci.edu/dataset/320/student+performance

2.  scikit-learn documentation, **Pipeline**.\
    https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

3.  scikit-learn documentation, **ColumnTransformer**.\
    https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html

4.  scikit-learn documentation, **OneHotEncoder**.\
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

5.  scikit-learn documentation, **StandardScaler**.\
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

6.  scikit-learn documentation, **StratifiedKFold**.\
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

7.  scikit-learn documentation, **GridSearchCV**.\
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

8.  Streamlit documentation.\
    https://docs.streamlit.io/

9.  Joblib documentation, **Persistence**.\
    https://joblib.readthedocs.io/en/latest/persistence.html

------------------------------------------------------------------------

## Project Status

**Implemented in the current project workflow:**

-   Mathematics dataset loading and exploration.
-   Binary target creation.
-   Feature selection excluding grade columns.
-   Train/test split.
-   Numerical scaling and categorical encoding.
-   Training and comparison of seven classifiers.
-   Five-fold stratified cross-validation.
-   At-risk-specific cross-validation metrics.
-   Logistic Regression hyperparameter tuning.
-   Held-out test evaluation.
-   Confusion matrix and feature-importance analysis.
-   Saving trained pipelines and evaluation outputs.
-   Streamlit integration with saved Jupyter pipelines and corrected
    class mapping.

**Still requiring verification before a final release:**

-   Run the integrated Streamlit app in the target environment.
-   Test both saved model choices using actual student inputs.
-   Verify all displayed metrics against the notebook outputs.
-   Confirm the dependency versions and deployment environment.
-   Test the hosted deployment, if one is created.

------------------------------------------------------------------------

**Developed as an academic project in Automated Student Performance
Prediction using Machine Learning and AutoML.**
