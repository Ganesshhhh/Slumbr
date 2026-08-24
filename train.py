"""
train.py
========
SleepSense - Sleep Quality Prediction
Trains a Random Forest classifier (with a Logistic Regression baseline) to
predict a person's Sleep Quality Category (Poor / Average / Good) from
health and lifestyle data.

Run:
    python train.py

This script will:
  1. Load and inspect data/sleep_health.csv
  2. Engineer the target variable (Sleep Quality Category)
  3. Build a preprocessing pipeline (ColumnTransformer)
  4. Train a Logistic Regression baseline and a Random Forest model
  5. Evaluate both models (accuracy, precision, recall, F1, confusion matrix)
  6. Run 5-fold Stratified Cross-Validation on the Random Forest model
  7. Plot and save feature importance + evaluation charts
  8. Save the fitted preprocessor and the trained Random Forest model with joblib
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI needed when running from the command line
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join("data", "sleep_health.csv")
MODELS_DIR = "models"
VIZ_DIR = "visualizations"
RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# STEP 1: LOAD & INSPECT THE RAW DATASET
# ---------------------------------------------------------------------------
section("STEP 1: LOAD & INSPECT DATASET")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{df.head()}")

print("\nUnique values of categorical columns:")
categorical_preview_cols = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]
for col in categorical_preview_cols:
    print(f"  {col} ({df[col].nunique()} unique): {sorted(df[col].dropna().unique().tolist())}")

print(f"\nBasic statistics:\n{df.describe(include='all')}")

# NOTE: "Sleep Disorder" is missing (NaN) for people who have no sleep
# disorder. That is expected -- it is not used as a model input in this
# project anyway (see Step 4), so it is left as-is here.

# ---------------------------------------------------------------------------
# STEP 2: REMOVE PERSON ID
# ---------------------------------------------------------------------------
section("STEP 2: REMOVE PERSON ID")
df = df.drop(columns=["Person ID"])
print("Removed 'Person ID' (a row identifier that carries no predictive signal "
      "and would risk the model memorizing a row instead of learning patterns).")

# ---------------------------------------------------------------------------
# STEP 3 & 5: CREATE TARGET VARIABLE FROM "Quality of Sleep"
# ---------------------------------------------------------------------------
section("STEP 3: CREATE TARGET VARIABLE (Sleep Quality Category)")


def bucket_quality(score):
    if score <= 4:
        return "Poor"
    elif score <= 7:
        return "Average"
    else:
        return "Good"


df["Sleep Quality Category"] = df["Quality of Sleep"].apply(bucket_quality)

print("Class distribution after bucketing 'Quality of Sleep' into categories:")
print(df["Sleep Quality Category"].value_counts())
print(
    "\nNote: 'Poor' (score 1-4) is a small minority class in this dataset "
    "(the raw scores only go as low as 4). This class imbalance is a real "
    "property of the data, not a bug -- see the README for how this affects "
    "evaluation and its listed limitations."
)

# Remove "Quality of Sleep" now that the target has been derived from it --
# keeping it as a feature would leak the answer directly into the model.
df = df.drop(columns=["Quality of Sleep"])
print("\nRemoved 'Quality of Sleep' from the feature set to avoid target leakage.")

# ---------------------------------------------------------------------------
# STEP 5: SPLIT BLOOD PRESSURE INTO SYSTOLIC / DIASTOLIC
# ---------------------------------------------------------------------------
section("STEP 4: SPLIT BLOOD PRESSURE INTO SYSTOLIC / DIASTOLIC")

bp_split = df["Blood Pressure"].str.split("/", expand=True)
df["Systolic Blood Pressure"] = bp_split[0].astype(int)
df["Diastolic Blood Pressure"] = bp_split[1].astype(int)
df = df.drop(columns=["Blood Pressure"])

print("Example: '126/83' -> Systolic Blood Pressure = 126, Diastolic Blood Pressure = 83")
print(df[["Systolic Blood Pressure", "Diastolic Blood Pressure"]].describe())

# ---------------------------------------------------------------------------
# STEP 6: HANDLE MISSING VALUES / DROP UNUSED COLUMN
# ---------------------------------------------------------------------------
section("STEP 5: HANDLE MISSING VALUES")

# "Sleep Disorder" is excluded from the initial model (see project spec --
# we are predicting sleep quality from lifestyle/health data, not from an
# existing sleep-disorder diagnosis). Dropping it also removes the only
# column with missing values, so no imputation is required for this model.
df_model = df.drop(columns=["Sleep Disorder"])
missing_after = df_model.isnull().sum().sum()
print(f"'Sleep Disorder' dropped (excluded from inputs by design; see README). "
      f"Missing values remaining in the modeling frame: {missing_after}")

# ---------------------------------------------------------------------------
# STEP 7-9: FEATURES / TARGET, NUMERIC vs CATEGORICAL
# ---------------------------------------------------------------------------
section("STEP 6: SEPARATE FEATURES (X) AND TARGET (y)")

TARGET = "Sleep Quality Category"
FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart Rate",
    "Daily Steps",
]

X = df_model[FEATURE_COLUMNS].copy()
y = df_model[TARGET].copy()

CATEGORICAL_FEATURES = ["Gender", "Occupation", "BMI Category"]
NUMERICAL_FEATURES = [
    "Age",
    "Sleep Duration",
    "Physical Activity Level",
    "Stress Level",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Heart Rate",
    "Daily Steps",
]

print(f"Feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
print(f"Categorical features: {CATEGORICAL_FEATURES}")
print(f"Numerical features: {NUMERICAL_FEATURES}")
print(f"Target: '{TARGET}' -> classes: {sorted(y.unique())}")

# ---------------------------------------------------------------------------
# STEP 10-11: BUILD PREPROCESSING PIPELINE (ColumnTransformer)
# ---------------------------------------------------------------------------
section("STEP 7: BUILD PREPROCESSING PIPELINE")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

print(
    "Using a single ColumnTransformer for all preprocessing:\n"
    "  - Numerical features -> StandardScaler (zero mean / unit variance)\n"
    "  - Categorical features -> OneHotEncoder (handle_unknown='ignore' so the\n"
    "    app never crashes on an unseen category typed in the UI)\n"
    "This same fitted object is reused for training AND for every prediction\n"
    "made in the Streamlit app, which guarantees identical preprocessing and\n"
    "prevents data leakage."
)

# ---------------------------------------------------------------------------
# STEP 12: TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
section("STEP 8: TRAIN / TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows")
print("Training class distribution:\n", y_train.value_counts())
print("Test class distribution:\n", y_test.value_counts())

# Fit the preprocessor ONLY on training data, then transform both sets.
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ---------------------------------------------------------------------------
# STEP 13: TRAIN LOGISTIC REGRESSION BASELINE
# ---------------------------------------------------------------------------
section("STEP 9: TRAIN LOGISTIC REGRESSION (BASELINE)")

log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_processed, y_train)
y_pred_lr = log_reg.predict(X_test_processed)

lr_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_lr),
    "Precision": precision_score(y_test, y_pred_lr, average="weighted", zero_division=0),
    "Recall": recall_score(y_test, y_pred_lr, average="weighted", zero_division=0),
    "F1 Score": f1_score(y_test, y_pred_lr, average="weighted", zero_division=0),
}
print("Logistic Regression test metrics:", lr_metrics)

# ---------------------------------------------------------------------------
# STEP 14: TRAIN RANDOM FOREST (MAIN MODEL)
# ---------------------------------------------------------------------------
section("STEP 10: TRAIN RANDOM FOREST CLASSIFIER")

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
    max_features="sqrt",
)
rf_model.fit(X_train_processed, y_train)
y_pred_rf = rf_model.predict(X_test_processed)

rf_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf, average="weighted", zero_division=0),
    "Recall": recall_score(y_test, y_pred_rf, average="weighted", zero_division=0),
    "F1 Score": f1_score(y_test, y_pred_rf, average="weighted", zero_division=0),
}
print("Random Forest test metrics:", rf_metrics)

print("\nRandom Forest classification report:\n")
print(classification_report(y_test, y_pred_rf, zero_division=0))

# ---------------------------------------------------------------------------
# STEP 15: MODEL COMPARISON TABLE
# ---------------------------------------------------------------------------
section("STEP 11: MODEL COMPARISON")

comparison_df = pd.DataFrame(
    [
        {"Model": "Logistic Regression", **lr_metrics},
        {"Model": "Random Forest", **rf_metrics},
    ]
)
print(comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(VIZ_DIR, "model_comparison.csv"), index=False)

# Bar chart: Random Forest vs Logistic Regression
metrics_for_chart = ["Accuracy", "Precision", "Recall", "F1 Score"]
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(metrics_for_chart))
width = 0.35
ax.bar(x_pos - width / 2, [lr_metrics[m] for m in metrics_for_chart], width, label="Logistic Regression")
ax.bar(x_pos + width / 2, [rf_metrics[m] for m in metrics_for_chart], width, label="Random Forest")
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_for_chart)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Logistic Regression vs Random Forest")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "model_comparison.png"), dpi=150)
plt.close()
print(f"\nSaved model comparison chart to {VIZ_DIR}/model_comparison.png")

# ---------------------------------------------------------------------------
# STEP 16: 5-FOLD STRATIFIED CROSS-VALIDATION (Random Forest)
# ---------------------------------------------------------------------------
section("STEP 12: 5-FOLD STRATIFIED CROSS-VALIDATION (RANDOM FOREST)")

print(
    "The dataset is relatively small (374 rows) and one class ('Poor') has\n"
    "very few examples, so a single train/test split can be sensitive to which\n"
    "rows happen to land in the test set. Stratified 5-fold cross-validation\n"
    "trains and evaluates the model 5 times on different train/validation\n"
    "splits of the TRAINING data (the held-out test set above is never touched\n"
    "here), which gives a more reliable estimate of how the model generalizes."
)

# Cross-validation is run on the training data only, using a fresh pipeline
# so the preprocessing is correctly refit within each fold (no leakage).
cv_pipeline = Pipeline(
    steps=[
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERICAL_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ]
        )),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ]
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=skf, scoring="accuracy")

print(f"Individual fold accuracy scores: {np.round(cv_scores, 4)}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# ---------------------------------------------------------------------------
# STEP 17: CONFUSION MATRIX
# ---------------------------------------------------------------------------
section("STEP 13: CONFUSION MATRIX (RANDOM FOREST)")

CLASS_ORDER = ["Poor", "Average", "Good"]
cm = confusion_matrix(y_test, y_pred_rf, labels=CLASS_ORDER)
print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
disp.plot(ax=ax, cmap="Blues", colorbar=True)
ax.set_title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print(f"\nSaved confusion matrix chart to {VIZ_DIR}/confusion_matrix.png")

# ---------------------------------------------------------------------------
# STEP 18: FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
section("STEP 14: FEATURE IMPORTANCE (RANDOM FOREST)")

# Recover human-readable feature names after the ColumnTransformer, so the
# one-hot-encoded columns (e.g. "Gender_Male") are labelled correctly.
ohe = preprocessor.named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
all_feature_names = NUMERICAL_FEATURES + cat_feature_names

importances = rf_model.feature_importances_
importance_df = pd.DataFrame(
    {"Feature": all_feature_names, "Importance": importances}
).sort_values("Importance", ascending=False)

print(importance_df.to_string(index=False))
print(
    "\nNote: feature importance shows how useful each feature was for the "
    "model's predictions on this dataset. It does NOT prove that a feature "
    "directly causes better or worse sleep quality."
)

fig, ax = plt.subplots(figsize=(9, 7))
plot_df = importance_df.sort_values("Importance")
ax.barh(plot_df["Feature"], plot_df["Importance"], color="#4C72B0")
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "feature_importance.png"), dpi=150)
plt.close()
print(f"\nSaved feature importance chart to {VIZ_DIR}/feature_importance.png")

importance_df.to_csv(os.path.join(VIZ_DIR, "feature_importance.csv"), index=False)

# ---------------------------------------------------------------------------
# SAVE MODEL ARTIFACTS BEFORE OPTIONAL VISUALIZATIONS
joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest_model.pkl"))
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
metadata = {
    "feature_columns": FEATURE_COLUMNS, "categorical_features": CATEGORICAL_FEATURES,
    "numerical_features": NUMERICAL_FEATURES, "class_order": CLASS_ORDER,
    "gender_options": sorted(df_model["Gender"].unique().tolist()),
    "occupation_options": sorted(df_model["Occupation"].unique().tolist()),
    "bmi_options": sorted(df_model["BMI Category"].unique().tolist()),
    "ranges": {col: {"min": float(df_model[col].min()), "max": float(df_model[col].max()), "mean": float(df_model[col].mean())} for col in NUMERICAL_FEATURES},
    "lr_metrics": lr_metrics, "rf_metrics": rf_metrics, "cv_scores": cv_scores.tolist(),
    "cv_mean": float(cv_scores.mean()), "cv_std": float(cv_scores.std()),
}
joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))
df_model.to_csv(os.path.join(MODELS_DIR, "dataset_overview.csv"), index=False)

# ---------------------------------------------------------------------------
# STEP 19: DATASET OVERVIEW VISUALIZATIONS (for the app's Dataset page)
# ---------------------------------------------------------------------------
section("STEP 15: SAVING DATASET OVERVIEW VISUALIZATIONS")

sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(6, 5))
sns.countplot(data=df_model, x="Sleep Quality Category", order=CLASS_ORDER, ax=ax)
ax.set_title("Sleep Quality Category Distribution")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "sleep_quality_distribution.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.histplot(data=df_model, x="Sleep Duration", kde=True, ax=ax, color="#55A868")
ax.set_title("Sleep Duration Distribution")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "sleep_duration_distribution.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.histplot(data=df_model, x="Stress Level", kde=True, ax=ax, color="#C44E52")
ax.set_title("Stress Level Distribution")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "stress_level_distribution.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.histplot(data=df_model, x="Physical Activity Level", kde=True, ax=ax, color="#8172B2")
ax.set_title("Physical Activity Level Distribution")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "physical_activity_distribution.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=df_model, x="Sleep Quality Category", y="Sleep Duration", order=CLASS_ORDER, ax=ax)
ax.set_title("Sleep Duration vs Sleep Quality")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "sleep_duration_vs_quality.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=df_model, x="Sleep Quality Category", y="Stress Level", order=CLASS_ORDER, ax=ax)
ax.set_title("Stress Level vs Sleep Quality")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "stress_vs_quality.png"), dpi=150)
plt.close()

print(f"Saved 6 dataset overview charts to {VIZ_DIR}/")

# ---------------------------------------------------------------------------
# STEP 20: SAVE MODEL, PREPROCESSOR, AND SUPPORTING METADATA
# ---------------------------------------------------------------------------
section("STEP 16: SAVE MODEL, PREPROCESSOR, AND METADATA")

joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest_model.pkl"))
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))

# Metadata the Streamlit app needs to build correct input widgets
# (dropdown options, sensible numeric ranges) directly from the real data --
# this avoids hard-coding arbitrary ranges in app.py.
metadata = {
    "feature_columns": FEATURE_COLUMNS,
    "categorical_features": CATEGORICAL_FEATURES,
    "numerical_features": NUMERICAL_FEATURES,
    "class_order": CLASS_ORDER,
    "gender_options": sorted(df_model["Gender"].unique().tolist()),
    "occupation_options": sorted(df_model["Occupation"].unique().tolist()),
    "bmi_options": sorted(df_model["BMI Category"].unique().tolist()),
    "ranges": {
        col: {"min": float(df_model[col].min()), "max": float(df_model[col].max()),
              "mean": float(df_model[col].mean())}
        for col in NUMERICAL_FEATURES
    },
    "lr_metrics": lr_metrics,
    "rf_metrics": rf_metrics,
    "cv_scores": cv_scores.tolist(),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
}
joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.pkl"))

# Save the processed dataframe (post target-engineering, pre train/test-split)
# so the Streamlit app's "Dataset Overview" tab can load it directly.
df_model.to_csv(os.path.join(MODELS_DIR, "dataset_overview.csv"), index=False)

print("Saved:")
print(f"  - {MODELS_DIR}/random_forest_model.pkl")
print(f"  - {MODELS_DIR}/preprocessor.pkl")
print(f"  - {MODELS_DIR}/metadata.pkl (UI ranges/options + evaluation results)")
print(f"  - {MODELS_DIR}/dataset_overview.csv (for the app's Dataset Overview tab)")

section("TRAINING COMPLETE")
print("Run 'streamlit run app.py' to launch the Slumbr application.")
