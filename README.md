# Student Performance Prediction with AutoML

A Streamlit dashboard for comparing classifiers that flag students as **At Risk**
or **Performing Well** using the UCI Student Performance mathematics or
Portuguese dataset.

## Run locally on Windows

Open PowerShell in this project folder and activate your existing virtual
environment, then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If you need a new environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Using the dashboard

1. Choose Mathematics or Portuguese and set the final-grade threshold.
2. The target is **At Risk** when `G3 < threshold`; it is **Performing Well**
   otherwise. The default threshold of 10 follows the dataset's pass-mark
   convention.
3. Click **Tune models and evaluate best model**. Seven classifiers are tuned
   with randomized search using stratified folds from the training split only.
   Hyperparameters are selected by CV macro F1. A most-frequent
   `DummyClassifier` is included as a no-skill reference.
4. For each model, including the current HistGradientBoosting reference and
   Dummy baseline, the app compares cutoffs 0.286, 0.35, 0.40, 0.45, and 0.50
   using out-of-fold probabilities generated only from training data. It
   selects the model/cutoff with the highest OOF macro F1, using balanced
   accuracy as a tie-breaker. The selected model is evaluated once on the
   held-out stratified test set and saved under `models/`. Test results do not
   influence model, hyperparameter, or cutoff selection.
5. Use the student form to request predictions for the selected dataset and
   threshold. Change either setting to use or train a separate model.

`G1`, `G2`, and `G3` are excluded from all input features to support prediction
before term grades are available. Numeric and categorical preprocessing is
part of the scikit-learn pipeline and is fitted within each cross-validation
fold. Class weighting is used for classifiers that support it. Mathematics
and Portuguese records are analyzed separately; they are not concatenated,
avoiding duplicate-student overlap and subject-specific distribution issues.
The dashboard compares the current HistGradientBoosting configuration at 0.50,
the previous high-recall cutoff of 0.286, and the OOF-selected configuration
on the same held-out test split, with metrics, classification reports, and
confusion matrices.
At Risk precision, recall, and F1 treat At Risk as the positive class; macro
F1 and balanced accuracy are also reported. The dashboard warns if held-out
At Risk recall is below 50%.

The G3 threshold defines the label; the probability cutoff determines when the
trained model predicts At Risk. They are separate settings. The OOF threshold
sweep is used for selection, not as an independent final performance estimate.
Use the held-out test results for that estimate. The current reference model
is refit on the current training split and evaluated at cutoff 0.50; the
previous high-recall cutoff is also measured for context. This makes the
comparison reproducible without loading an old serialized artifact.

## Reproduced Mathematics results

With `G3 < 10`, `random_state=42`, and the stratified 80/20 split, selection
returned HistGradientBoosting at 0.50, matching the current configuration.
There was no measured change between the current and selected held-out model.
These are measurements from this run, not expected guarantees:

| Held-out metric | Current HistGradientBoosting (0.50) | Previous cutoff (0.286) | OOF-selected HistGradientBoosting (0.50) |
| --- | ---: | ---: | ---: |
| Accuracy | 0.646 | 0.595 | 0.646 |
| At-risk precision | 0.469 | 0.435 | 0.469 |
| At-risk recall | 0.577 | 0.769 | 0.577 |
| At-risk F1 | 0.517 | 0.556 | 0.517 |
| Macro F1 | 0.619 | 0.592 | 0.619 |
| Balanced accuracy | 0.628 | 0.639 | 0.628 |

Confusion matrices use actual rows and predicted columns in the order
At Risk, Performing Well:

- Current and OOF-selected configuration: `[[15, 11], [17, 36]]`
- Previous high-recall cutoff: `[[20, 6], [26, 27]]`

For the fixed current HistGradientBoosting model, training OOF macro F1 was
0.560 at 0.286, 0.584 at 0.35, 0.594 at 0.40, 0.612 at 0.45, and 0.633 at
0.50. At-risk recall correspondingly fell from 0.721 to 0.529. The in-sample
OOF sweep favored 0.50, but the independent held-out model comparison did not
produce a different selected model or cutoff. Consider repeated/nested
stratified CV and external school data to assess selection stability.

The randomized-search CV comparison (best hyperparameters per classifier)
ranked HistGradientBoosting first at macro F1 0.630 and balanced accuracy
0.636. Gradient Boosting scored 0.613/0.612, Logistic Regression
0.612/0.622, Decision Tree 0.600/0.614, Random Forest 0.599/0.602,
Extra Trees 0.591/0.595, SVM 0.587/0.595, and the most-frequent Dummy
baseline 0.402/0.500 (macro F1/balanced accuracy).

The mathematics dataset has 395 records (130 At Risk, 265 Performing Well)
and 30 inputs after excluding grade columns; the inspected file had no missing
values or exact duplicate rows. The Portuguese file has 649 records (100 At
Risk, 549 Performing Well). 39 records have matching non-grade attribute
profiles across the subject files; these matches are not confirmed identities,
but illustrate why the subjects are kept separate rather than concatenated.
The files do not provide a student ID, so exact duplicate checks cannot prove
that distinct rows never represent the same person.

The stratified split preserves original row indices and the app checks that
train and test indices are disjoint. Since the supplied files have no student
identifier, identity cannot be verified beyond the available record-level
checks. No feature selector is added: with 30 input columns and only 316
training examples, selection may add variance, and the current comparison does
not establish that it would improve held-out performance. Any future selector
should be placed inside the pipeline so it is fitted only within CV folds.

Performance is limited by the small cohort, the moderate class imbalance, and
the early-prediction requirement to exclude G1/G2, which are strong grade
signals but unavailable at the intended prediction time. The remaining
demographic, family, and study-habit features may not fully explain individual
outcomes. There are no missing values to impute in the supplied files, but
fold-fitted imputation remains in the pipeline for robustness. These
observational models learn associations rather than causes and may not
generalize to other schools or populations. Use them only as decision support,
not definitive judgments about individual students.

Saved files use Python `joblib` serialization. Only load model files that you
created or otherwise trust; a joblib artifact can execute code when loaded.
The `models/` directory is generated/used by the app and should be writable.

## Deploy to Streamlit Community Cloud

1. Push this project, including `app.py`, `requirements.txt`, and both CSV files
   in `data/`, to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository, select
   `app.py` as the main file, and deploy.
3. The app installs dependencies from `requirements.txt`. No secrets or
   external services are required.

Models created in Community Cloud are stored in the app's writable filesystem,
which may be reset when the app restarts or redeploys. Run training again if
the selected model file is no longer present.
