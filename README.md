# 😴 Slumbr – Sleep Quality Prediction for Better Health

A beginner-friendly, MCA-level machine learning project that predicts a
person's **Sleep Quality Category** (Poor / Average / Good) from health and
lifestyle data, wrapped in an interactive Streamlit dashboard.

> **This is an educational project, not a medical diagnosis system.**
> Predictions and lifestyle tips are for learning purposes only.

---

## 1. Project Title
**Slumbr – Sleep Quality Prediction for Better Health**

---

## 2. Abstract
Slumbr uses the **Sleep Health and Lifestyle** dataset to train a
machine learning model that classifies a person's sleep quality into three
categories — Poor, Average, or Good — based on demographic, lifestyle, and
health measurements. The project compares a Logistic Regression baseline
against a Random Forest Classifier, evaluates both with standard
classification metrics and 5-fold stratified cross-validation, and exposes
the final Random Forest model through a Streamlit web application with
prediction, dataset exploration, model performance, feature importance, and
session-based prediction history sections.

---

## 3. Problem Statement
Poor sleep quality is linked to numerous health and productivity issues, yet
people often don't have an easy way to see how their day-to-day habits
(stress, activity, sleep duration, etc.) relate to how well they sleep. This
project builds a simple, transparent classification model that estimates
sleep quality from lifestyle and health inputs, and explains *why* it made
that prediction (via feature importance) — as a learning exercise in
applied classification, not as a clinical tool.

---

## 4. Objectives
- Load and explore a real-world tabular health/lifestyle dataset.
- Engineer a 3-class target variable from a numeric sleep-quality score.
- Build a leakage-free preprocessing pipeline using `ColumnTransformer`.
- Train and compare a Logistic Regression baseline and a Random Forest
  Classifier.
- Evaluate models with accuracy, precision, recall, F1-score, a confusion
  matrix, and stratified cross-validation.
- Interpret the Random Forest model using feature importance.
- Deploy the trained model in an interactive Streamlit application.

---

## 5. Dataset Description
- **Source:** Sleep Health and Lifestyle dataset (as published on Kaggle,
  associated with `kaggle.com/code/cerenadiyamn/sleep-health`).
- **Size:** 374 rows × 13 columns (no duplicate rows).
- **Missing values:** Only the `Sleep Disorder` column has missing values
  (219 of 374 rows — meaning "no diagnosed sleep disorder"). This column is
  **not used** as a model input (see Section 9), so no imputation is
  required for the model itself.

---

## 6. Dataset Features (original columns)
| Column | Description |
|---|---|
| Person ID | Row identifier |
| Gender | Male / Female |
| Age | Age in years |
| Occupation | 11 job categories (e.g. Doctor, Nurse, Engineer) |
| Sleep Duration | Average hours of sleep per day |
| Quality of Sleep | Subjective rating, 1–10 (source of our target) |
| Physical Activity Level | Average minutes of daily physical activity |
| Stress Level | Subjective rating, 1–10 |
| BMI Category | Normal / Normal Weight / Overweight / Obese |
| Blood Pressure | "systolic/diastolic" string, e.g. "126/83" |
| Heart Rate | Resting heart rate (bpm) |
| Daily Steps | Average daily step count |
| Sleep Disorder | None / Insomnia / Sleep Apnea (many missing = none reported) |

**Data quality note:** the raw data contains both `"Normal"` and
`"Normal Weight"` as separate BMI Category values, which likely refer to
the same underlying category but are kept as-is (per the actual dataset)
rather than invented or merged, since this project's rule is to use the
real column values without modification. A future improvement could
consolidate these two labels.

---

## 7. Target Variable
`Sleep Quality Category`, engineered from the numeric `Quality of Sleep`
column (1–10):

| Quality of Sleep | Category |
|---|---|
| 1 – 4 | Poor |
| 5 – 7 | Average |
| 8 – 10 | Good |

**Actual class distribution in this dataset:**

| Category | Count |
|---|---|
| Average | 189 |
| Good | 180 |
| Poor | 5 |

`Poor` is a small minority class here because the raw `Quality of Sleep`
scores in this dataset never go below 4. This is a real characteristic of
the data (see Limitations) — it is not something the code manufactures.

---

## 8. Data Preprocessing
Implemented in `train.py`, in this order:

1. Load `data/sleep_health.csv`.
2. Inspect shape, dtypes, missing values, uniques, and statistics.
3. Drop `Person ID` (a row identifier with no predictive value).
4. Create `Sleep Quality Category` from `Quality of Sleep`.
5. Drop `Quality of Sleep` (it *is* the source of the target — keeping it
   as a feature would leak the answer directly to the model).
6. Split `Blood Pressure` into `Systolic Blood Pressure` and
   `Diastolic Blood Pressure`; drop the original column (see Section 9).
7. Drop `Sleep Disorder` (excluded from inputs by design, see Section 9).
   No missing values remain in the modeling data after this.
8. Separate `X` (features) and `y` (target).
9. Identify numerical vs. categorical features.
10. Build a `ColumnTransformer`: `StandardScaler` for numerical features,
    `OneHotEncoder(handle_unknown="ignore")` for categorical features.
11. `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
12. Fit the preprocessor **only on the training data**, then transform both
    the training and test sets — this avoids data leakage.

The fitted preprocessor is saved (`models/preprocessor.pkl`) and reused
unchanged by the Streamlit app for every prediction, so training-time and
prediction-time preprocessing are always identical.

---

## 9. Feature Engineering
- **Blood Pressure split:** `"126/83"` → `Systolic Blood Pressure = 126`,
  `Diastolic Blood Pressure = 83`. Blood pressure is naturally two related
  but distinct numbers; splitting it lets the model use each independently
  instead of trying to parse a string.
- **Excluded from inputs:** `Person ID`, `Quality of Sleep`,
  `Sleep Quality Category` (target leakage — see viva Q13), and
  `Sleep Disorder` (the goal is to predict sleep quality from lifestyle and
  health information, not from an existing diagnosis label).
- **Final input features (11):** Gender, Age, Occupation, Sleep Duration,
  Physical Activity Level, Stress Level, BMI Category,
  Systolic Blood Pressure, Diastolic Blood Pressure, Heart Rate, Daily Steps.

---

## 10. Random Forest Methodology
- **Decision tree:** a flowchart-like model that repeatedly splits the data
  on feature thresholds (e.g. "Stress Level > 6?") until it reaches a
  prediction. Simple but prone to overfitting on its own.
- **Random Forest:** trains many decision trees (`n_estimators=100` here),
  each on a random subset of the training data and a random subset of
  features at each split, then combines their predictions.
- **Voting:** for classification, each tree "votes" for a class, and the
  forest predicts the class with the most votes (`predict_proba` reports
  the vote proportions as class probabilities).
- **Why Random Forest here:** it handles a mix of numerical and
  one-hot-encoded categorical features well after preprocessing, is
  fairly robust to a small/noisy dataset, needs little hyperparameter
  tuning, and — unlike a single decision tree — averaging many trees
  reduces overfitting and variance.

---

## 11. Logistic Regression Baseline
A `LogisticRegression(max_iter=1000, random_state=42)` model trained on the
same preprocessed features and the same train/test split, used purely as a
simple baseline to check whether the more complex Random Forest is actually
worth it.

---

## 12. Model Evaluation

Metrics use `average="weighted"` because this is a multiclass problem.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 96.0% | 94.8% | 96.0% | 95.3% |
| Random Forest | 100.0% | 100.0% | 100.0% | 100.0% |

*(Exact numbers are also written to `visualizations/model_comparison.csv`
and printed every time you run `train.py` — re-running will reproduce
these same numbers because `random_state=42` is fixed everywhere.)*

**Random Forest selected as the final model** — it matched or outperformed
Logistic Regression on every metric on this dataset and test split. No
results were adjusted or cherry-picked to favor it.

Random Forest confusion matrix classes appear as **Poor / Average / Good**
(see `visualizations/confusion_matrix.png`).

**Why both models score so highly:** the `Quality of Sleep` scores in this
particular dataset map quite cleanly onto the Poor/Average/Good buckets,
and the dataset is small, clean, and has strong signal in a few features
(especially Stress Level and Sleep Duration). This is a property of this
specific dataset, not a guarantee for other populations — see Limitations.

---

## 13. Cross-Validation
Because the dataset is relatively small (374 rows) and the "Poor" class has
very few examples, a single train/test split can be sensitive to which
rows happen to land in the test set. **5-fold Stratified Cross-Validation**
was therefore also run on the Random Forest model, using only the training
data (the held-out test set is never touched during cross-validation):

- Individual fold accuracies: `[1.0, 0.983, 1.0, 1.0, 0.983]`
- Mean cross-validation accuracy: **99.3%**
- Standard deviation: **0.0082**

The low standard deviation across folds suggests the strong test-set result
above isn't just a lucky split.

---

## 14. Feature Importance
Computed from the trained Random Forest's `feature_importances_`, with
one-hot-encoded column names correctly mapped back to readable labels. Top
features (see `visualizations/feature_importance.png` for the full chart):

1. Sleep Duration
2. Stress Level
3. Heart Rate
4. Age
5. Physical Activity Level

**Interpretation caveat:** feature importance shows how useful a feature
was to the model's predictions — it does **not** prove that a feature
directly *causes* better or worse sleep quality.

---

## 15. Streamlit Application
`app.py` launches **Slumbr** as a personal daily check-in and weekly
tracker (not a data-science demo — the model's internal accuracy/precision/
confusion-matrix/feature-importance details are intentionally left out of
the app; they live in this README and in the console output of `train.py`
instead). It has three tabs:

1. **Check-In** — enter today's details (date, demographics, sleep
   duration, activity, stress, blood pressure, heart rate, steps — each
   with a short explanation of what it means, e.g. what systolic/diastolic
   blood pressure are). Click "Log today and check my sleep quality" to
   get today's category (Poor / Average / Excellent) and points. If the
   result is Average or Poor, a suggested schedule and diet appear, with a
   note that it's best to also confirm with a local doctor.
2. **This Week** — a week-at-a-glance strip (Monday-Sunday) showing points
   for each day you've logged, plus total and average points for the week.
   It updates automatically as you log more days. If your most recent
   check-in was Average or Poor, the schedule/diet guidance is repeated
   here as a standing reminder.
3. **History** — every check-in you've ever logged, with an option to
   clear all stored data.

**Points system:** Excellent = 100, Average = 65, Poor = 30.

**Persistence:** entries are saved locally to `logs/sleep_log.json` (a
plain JSON file, not a database) so your week builds up correctly across
multiple days and app restarts — logging the same date twice simply
overwrites that day's entry.

Predictions always go through the **saved preprocessing pipeline**
(`models/preprocessor.pkl`) before being passed to the **saved Random
Forest model** (`models/random_forest_model.pkl`), so training and
prediction use identical transformations.

---

## 16. Project Structure
```
Slumbr/
│
├── data/
│   └── sleep_health.csv          # the real dataset
│
├── models/                       # created by train.py
│   ├── random_forest_model.pkl
│   ├── preprocessor.pkl
│   ├── metadata.pkl              # UI ranges/options + evaluation results
│   └── dataset_overview.csv
│
├── visualizations/                # created by train.py (reference charts, not shown in-app)
│   ├── model_comparison.png / .csv
│   ├── confusion_matrix.png
│   ├── feature_importance.png / .csv
│   └── ... dataset overview charts
│
├── logs/
│   └── sleep_log.json            # created by app.py as you log daily check-ins
│
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 17. Installation (Windows)
```bat
:: 1. (Optional but recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate

:: 2. Install dependencies
pip install -r requirements.txt
```

---

## 18. Training Instructions
```bat
python train.py
```
This prints every inspection/preprocessing/evaluation step to the console
and saves the model, preprocessor, and charts described above. Re-run any
time you change the code or data — it always overwrites the saved files.

---

## 19. Running the Streamlit App
```bat
streamlit run app.py
```
This opens the dashboard in your browser (usually `http://localhost:8501`).
**Run `python train.py` at least once before this step**, since the app
loads the saved model/preprocessor files.

---

## 20. Screenshots
*(Add screenshots of each tab here after running the app locally, e.g.
`screenshots/prediction_tab.png`, `screenshots/dataset_overview.png`, etc.)*

---

## 21. Limitations
- The dataset is relatively small (374 rows) and comes from a single
  source population, so results may not generalize to other populations.
- The `Poor` sleep-quality class has only 5 examples in the entire
  dataset — evaluation metrics for this class are based on very few
  samples and should be interpreted cautiously.
- The target (`Quality of Sleep`) is itself a subjective 1–10 rating in
  the original data, not an objective clinical measurement.
- The Poor/Average/Good thresholds (1–4 / 5–7 / 8–10) are **project-defined
  categories for this assignment**, not clinical or medical standards.
- Very high accuracy on this dataset does **not** guarantee similar
  performance on new, real-world data.
- This is an educational tool. Predictions and suggestions must never be
  treated as medical advice or a diagnosis.

---

## 22. Future Enhancements
- Merge the `"Normal"` / `"Normal Weight"` BMI categories after confirming
  with a domain expert that they mean the same thing.
- Collect more data, especially more `Poor`-quality-sleep examples, to
  make that class's evaluation more reliable.
- Try additional models (e.g. Gradient Boosting, SVM) and proper
  hyperparameter tuning.
- Add a "what-if" feature so users can see how changing one input (e.g.
  reducing stress by 2 points) shifts the prediction.
- Persist prediction history to a lightweight file/database if the app
  needs to remember predictions across sessions.

---

## 23. Viva Explanation

**1. What is the objective of this project?**
To predict a person's sleep quality category (Poor/Average/Good) from
lifestyle and health data using a Random Forest classifier, packaged in an
interactive app.

**2. What dataset did you use?**
The Sleep Health and Lifestyle dataset (374 rows, 13 columns) covering
demographics, lifestyle habits, and health measurements.

**3. What is the target variable?**
`Sleep Quality Category` — a 3-class label (Poor/Average/Good) derived from
the original numeric `Quality of Sleep` (1–10) column.

**4. Why did you convert Quality of Sleep into categories?**
To turn a numeric rating into a simpler, more interpretable 3-class
classification problem, which is easier to explain and act on than a raw
1–10 score, and matches the assignment's goal of building a classifier.

**5. Why is this a classification problem?**
Because the target has a small number of discrete labels (Poor, Average,
Good) rather than a continuous numeric value — the goal is to assign each
person to one of these categories.

**6. Why did you choose Random Forest?**
It handles a mix of numeric and categorical (one-hot-encoded) features
well, needs minimal tuning, resists overfitting better than a single
decision tree by averaging many trees, and provides feature importance for
interpretability.

**7. What is a decision tree?**
A model that splits data step-by-step based on feature thresholds (like a
flowchart of yes/no questions) until it reaches a predicted class.

**8. How does Random Forest work?**
It builds many decision trees, each trained on a random subset of rows and
a random subset of features, then combines their outputs — for
classification, by majority vote.

**9. Why use multiple trees?**
A single tree can overfit the training data and be sensitive to small
changes in it. Averaging many different trees reduces this variance and
usually improves generalization to new data.

**10. What is train-test split?**
Splitting the dataset into a portion used to train the model (80% here)
and a separate portion held out to evaluate it (20% here), so performance
is measured on data the model never saw during training.

**11. Why use stratification?**
`stratify=y` keeps the same class proportions in both the training and
test sets as in the full dataset — important here because the `Poor`
class is rare, and a plain random split could easily leave it out of the
test set entirely.

**12. What is data leakage?**
When information that wouldn't be available at prediction time (or that
directly reveals the answer) is accidentally included as a feature,
making the model look better than it really is. Fitting preprocessing on
the full dataset instead of only the training set is also a form of
leakage.

**13. Why can't Quality of Sleep be used as an input?**
Because `Sleep Quality Category` is directly derived from it — including
it as a feature would let the model "cheat" by essentially looking up the
answer instead of learning genuine patterns. This is a textbook case of
target leakage.

**14. Why did you remove Person ID?**
It's just a row identifier with no real relationship to sleep quality;
including it could let the model memorize specific rows instead of
learning generalizable patterns.

**15. Why did you split Blood Pressure?**
The raw column is a text string like `"126/83"` combining two related but
distinct numeric measurements (systolic and diastolic pressure). Splitting
it into two numeric columns lets the model use each value properly instead
of treating the whole string as an opaque category.

**16. Why do we encode categorical variables?**
Machine learning models like Random Forest and Logistic Regression (as
implemented in scikit-learn) require numeric input. One-hot encoding
converts categories (e.g. Gender, Occupation) into numeric 0/1 columns
without implying a false numeric order between categories.

**17. What is feature importance?**
A score from the Random Forest indicating how much each feature
contributed to the model's predictions overall — higher means the model
relied on it more, not that it necessarily causes the outcome.

**18. What is a confusion matrix?**
A table comparing actual vs. predicted classes, showing how many
predictions were correct and, for the incorrect ones, which classes were
confused with which.

**19. What are precision, recall, and F1-score?**
- **Precision:** of everything the model predicted as class X, what
  fraction was actually class X.
- **Recall:** of everything that was actually class X, what fraction did
  the model correctly identify.
- **F1-score:** the harmonic mean of precision and recall, balancing both.
`average="weighted"` combines the per-class scores weighted by how many
true examples each class has, since this is a multiclass problem.

**20. Why compare Random Forest with Logistic Regression?**
To have a simple, well-understood baseline. If Random Forest didn't clearly
outperform Logistic Regression, the added complexity of Random Forest
wouldn't be justified.

**21. Why use cross-validation?**
The dataset is small, so a single train/test split can give a
score that's partly due to luck in how the split happened. Averaging
performance across 5 different stratified splits gives a more trustworthy
estimate of how the model generalizes.

**22. What are the limitations?**
Small dataset, a very small `Poor` class, a subjective original rating,
project-defined (not clinical) thresholds, and no guarantee the model
generalizes beyond this specific dataset. See Section 21 above for the
full list.

**23. What could be improved in the future?**
More data (especially more `Poor` examples), trying additional models and
proper hyperparameter tuning, resolving the duplicate BMI category labels,
and adding a "what-if" exploration feature. See Section 22 above.

---

## Disclaimer
Slumbr is a student/educational machine learning project. It is **not**
a certified medical device, does not diagnose sleep disorders, and should
never be used as a substitute for professional medical advice.
