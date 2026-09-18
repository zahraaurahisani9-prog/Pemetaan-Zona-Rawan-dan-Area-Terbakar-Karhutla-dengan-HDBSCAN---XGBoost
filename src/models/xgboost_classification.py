# =============================================================================
# SECTION 1 — IMPORT LIBRARIES
# =============================================================================

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

warnings.filterwarnings("ignore")

print("=" * 70)
print("XGBoost Classification Pipeline")
print("=" * 70)

# =============================================================================
# SECTION 2 — CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Project Directory
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -----------------------------------------------------------------------------
# Input Dataset
# -----------------------------------------------------------------------------

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "classification_dataset.csv"
)

# -----------------------------------------------------------------------------
# Output Directories
# -----------------------------------------------------------------------------

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

MODEL_DIR = OUTPUT_ROOT / "models"
VISUALIZATION_DIR = OUTPUT_ROOT / "visualization"
METADATA_DIR = OUTPUT_ROOT / "metadata"

# Create output directories if they do not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Output Files
# -----------------------------------------------------------------------------

MODEL_PATH = MODEL_DIR / "xgboost_burn_classification.pkl"

METRICS_PATH = METADATA_DIR / "evaluation_metrics.csv"
FEATURE_IMPORTANCE_PATH = METADATA_DIR / "feature_importance.csv"
PREDICTION_PATH = METADATA_DIR / "test_prediction.csv"

CONFUSION_MATRIX_FIG = VISUALIZATION_DIR / "confusion_matrix.png"
ROC_CURVE_FIG = VISUALIZATION_DIR / "roc_curve.png"
FEATURE_IMPORTANCE_FIG = VISUALIZATION_DIR / "feature_importance.png"

EDA_DIR = VISUALIZATION_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------------------------------------------------------------
# Feature Configuration
# -----------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "Mean_B2",
    "Mean_B3",
    "Mean_B4",
    "Mean_B11",
    "Mean_NBR",
]

TARGET_COLUMN = "Burn_Label"

ID_COLUMNS = [
    "grid_id",
    "latitude",
    "longitude",
]

# -----------------------------------------------------------------------------
# Data Split Configuration
# -----------------------------------------------------------------------------

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# -----------------------------------------------------------------------------
# XGBoost Configuration (Baseline)
# -----------------------------------------------------------------------------

XGB_BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

print("Configuration loaded successfully.")

# =============================================================================
# SECTION 3 — LOGGING
# =============================================================================

# Create log directory
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "xgboost_classification.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("XGBoost Burn Area Classification Pipeline")
logger.info("=" * 80)

logger.info("Project Root      : %s", PROJECT_ROOT)
logger.info("Dataset           : %s", DATASET_PATH)
logger.info("Model Directory   : %s", MODEL_DIR)
logger.info("Visualization Dir : %s", VISUALIZATION_DIR)
logger.info("Metadata Dir      : %s", METADATA_DIR)

logger.info("Features          : %s", ", ".join(FEATURE_COLUMNS))
logger.info("Target            : %s", TARGET_COLUMN)

logger.info("Train Split       : %.0f%%", TRAIN_SIZE * 100)
logger.info("Validation Split  : %.0f%%", VALIDATION_SIZE * 100)
logger.info("Test Split        : %.0f%%", TEST_SIZE * 100)

logger.info("Random State      : %d", RANDOM_STATE)

logger.info("=" * 80)

# =============================================================================
# SECTION 4 — LOAD DATASET
# =============================================================================

logger.info("=" * 80)
logger.info("SECTION 4 - LOAD DATASET")
logger.info("=" * 80)

# -----------------------------------------------------------------------------
# Check dataset existence
# -----------------------------------------------------------------------------

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

logger.info("Dataset found.")

# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

logger.info("Dataset loaded successfully.")

# -----------------------------------------------------------------------------
# Validate required columns
# -----------------------------------------------------------------------------

required_columns = ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

logger.info("All required columns are available.")

# -----------------------------------------------------------------------------
# Display dataset information
# -----------------------------------------------------------------------------

logger.info("Number of rows    : %d", df.shape[0])
logger.info("Number of columns : %d", df.shape[1])

logger.info("Feature columns:")
for feature in FEATURE_COLUMNS:
    logger.info("   • %s", feature)

logger.info("Target column : %s", TARGET_COLUMN)

logger.info("Dataset Preview (First 5 Rows):")

logger.info(
    "\n%s",
    df.head().to_string(index=False)
)

logger.info("=" * 80)

# =============================================================================
# SECTION 5 — DATA QUALITY CHECK
# =============================================================================

logger.info("=" * 80)
logger.info("SECTION 5 - DATA QUALITY CHECK")
logger.info("=" * 80)

# -----------------------------------------------------------------------------
# Missing Values
# -----------------------------------------------------------------------------

logger.info("Checking missing values...")

missing_summary = df.isnull().sum()

if missing_summary.sum() == 0:
    logger.info("No missing values found.")
else:
    logger.warning("Missing values detected:")

    for column, count in missing_summary.items():
        if count > 0:
            logger.warning(
                "%-15s : %d missing values",
                column,
                count
            )

    raise ValueError("Dataset contains missing values.")

# -----------------------------------------------------------------------------
# Duplicate Records
# -----------------------------------------------------------------------------

logger.info("Checking duplicate records...")

duplicate_count = df.duplicated().sum()

logger.info("Duplicate rows : %d", duplicate_count)

if duplicate_count > 0:
    raise ValueError(
        f"Dataset contains {duplicate_count} duplicate rows."
    )

# -----------------------------------------------------------------------------
# Infinite Values
# -----------------------------------------------------------------------------

logger.info("Checking infinite values...")

numeric_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

inf_count = np.isinf(df[numeric_columns]).sum().sum()

logger.info("Infinite values : %d", inf_count)

if inf_count > 0:
    raise ValueError(
        "Dataset contains infinite values."
    )

# -----------------------------------------------------------------------------
# Data Types
# -----------------------------------------------------------------------------

logger.info("Checking data types...")

logger.info("\n%s", df.dtypes.to_string())

# -----------------------------------------------------------------------------
# Descriptive Statistics
# -----------------------------------------------------------------------------

logger.info("Generating descriptive statistics...")

statistics = df[FEATURE_COLUMNS].describe().round(4)

logger.info("\n%s", statistics.to_string())

# -----------------------------------------------------------------------------
# Burn Label Distribution
# -----------------------------------------------------------------------------

logger.info("Checking target distribution...")

label_distribution = (
    df[TARGET_COLUMN]
    .value_counts()
    .sort_index()
)

for label, count in label_distribution.items():
    percentage = (count / len(df)) * 100

    logger.info(
        "Burn_Label = %s : %d (%.2f%%)",
        label,
        count,
        percentage
    )

# -----------------------------------------------------------------------------
# Final Summary
# -----------------------------------------------------------------------------

logger.info("Data quality check completed successfully.")

logger.info("=" * 80)

valid_labels = {0, 1}
actual_labels = set(df[TARGET_COLUMN].unique())

if actual_labels != valid_labels:
    raise ValueError(
        f"Burn_Label must contain only {valid_labels}, "
        f"but found {actual_labels}."
    )

# =============================================================================
# SECTION 6 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

logger.info("=" * 80)
logger.info("SECTION 6 - EXPLORATORY DATA ANALYSIS")
logger.info("=" * 80)

# -----------------------------------------------------------------------------
# Class Distribution
# -----------------------------------------------------------------------------

logger.info("Creating class distribution plot...")

class_counts = (
    df[TARGET_COLUMN]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(6, 5))

class_counts.plot(
    kind="bar"
)

plt.title("Burn Label Distribution")
plt.xlabel("Burn Label")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    EDA_DIR / "class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------------------------------------------
# Feature Histograms
# -----------------------------------------------------------------------------

logger.info("Creating feature histograms...")

df[FEATURE_COLUMNS].hist(
    figsize=(14, 8),
    bins=30
)

plt.tight_layout()

plt.savefig(
    EDA_DIR / "histogram_features.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------------------------------------------
# Feature Boxplots
# -----------------------------------------------------------------------------

logger.info("Creating feature boxplots...")

plt.figure(figsize=(10, 6))

df[FEATURE_COLUMNS].boxplot()

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    EDA_DIR / "boxplot_features.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------------------------------------------
# Correlation Matrix
# -----------------------------------------------------------------------------

logger.info("Creating correlation matrix...")

correlation_matrix = (
    df[FEATURE_COLUMNS]
    .corr(method="pearson")
)

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(correlation_matrix)

ax.set_xticks(np.arange(len(FEATURE_COLUMNS)))
ax.set_yticks(np.arange(len(FEATURE_COLUMNS)))

ax.set_xticklabels(FEATURE_COLUMNS)
ax.set_yticklabels(FEATURE_COLUMNS)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right"
)

for i in range(len(FEATURE_COLUMNS)):
    for j in range(len(FEATURE_COLUMNS)):
        ax.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=9
        )

fig.colorbar(im)

plt.tight_layout()

plt.savefig(
    EDA_DIR / "correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

logger.info("EDA completed successfully.")
logger.info("=" * 80)

# =============================================================================
# SECTION 7 — DATA PREPARATION
# =============================================================================

logger.info("=" * 80)
logger.info("SECTION 7 - DATA PREPARATION")
logger.info("=" * 80)

# -----------------------------------------------------------------------------
# Select Features and Target
# -----------------------------------------------------------------------------

logger.info("Selecting feature columns...")

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

metadata = df[ID_COLUMNS].copy()

logger.info("Feature matrix shape : %s", X.shape)
logger.info("Target vector shape  : %s", y.shape)

# -----------------------------------------------------------------------------
# Convert Data Types
# -----------------------------------------------------------------------------

logger.info("Converting feature columns to numeric...")

for column in FEATURE_COLUMNS:
    X[column] = pd.to_numeric(X[column], errors="raise")

logger.info("Converting target column to integer...")

y = y.astype(int)

# -----------------------------------------------------------------------------
# Final Validation
# -----------------------------------------------------------------------------

logger.info("Validating prepared dataset...")

if X.isnull().sum().sum() != 0:
    raise ValueError("Feature matrix contains missing values.")

if y.isnull().sum() != 0:
    raise ValueError("Target vector contains missing values.")

if X.empty:
    raise ValueError("Feature matrix is empty.")

if y.empty:
    raise ValueError("Target vector is empty.")

if len(X) != len(y):
    raise ValueError("Mismatch between feature matrix and target vector.")

logger.info("Feature columns used:")

for feature in FEATURE_COLUMNS:
    logger.info("   • %s", feature)

logger.info("Target column : %s", TARGET_COLUMN)

logger.info("Prepared dataset summary")

logger.info("Samples  : %d", len(X))
logger.info("Features : %d", X.shape[1])

logger.info("=" * 80)

# =============================================================================
# SECTION 8 — TRAIN / VALIDATION / TEST SPLIT
# =============================================================================

logger.info("=" * 80)
logger.info("SECTION 8 - TRAIN / VALIDATION / TEST SPLIT")
logger.info("=" * 80)

# -----------------------------------------------------------------------------
# First Split (Train : Temp)
# -----------------------------------------------------------------------------

logger.info("Splitting dataset into training and temporary sets...")

(
    X_train,
    X_temp,
    y_train,
    y_temp,
    metadata_train,
    metadata_temp,
) = train_test_split(
    X,
    y,
    metadata,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# -----------------------------------------------------------------------------
# Second Split (Validation : Test)
# -----------------------------------------------------------------------------

logger.info("Splitting temporary set into validation and testing sets...")

validation_ratio = VALIDATION_SIZE / (VALIDATION_SIZE + TEST_SIZE)

(
    X_validation,
    X_test,
    y_validation,
    y_test,
    metadata_validation,
    metadata_test,
) = train_test_split(
    X_temp,
    y_temp,
    metadata_temp,
    train_size=validation_ratio,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

# -----------------------------------------------------------------------------
# Dataset Summary
# -----------------------------------------------------------------------------

logger.info("Dataset splitting completed successfully.")

logger.info(
    "Training Set   : %5d samples (%.2f%%)",
    len(X_train),
    len(X_train) / len(df) * 100,
)

logger.info(
    "Validation Set : %5d samples (%.2f%%)",
    len(X_validation),
    len(df) and len(X_validation) / len(df) * 100,
)

logger.info(
    "Testing Set    : %5d samples (%.2f%%)",
    len(X_test),
    len(X_test) / len(df) * 100,
)

# -----------------------------------------------------------------------------
# Burn Label Distribution
# -----------------------------------------------------------------------------

logger.info("-" * 80)
logger.info("Burn Label Distribution")

for name, target in [
    ("Training", y_train),
    ("Validation", y_validation),
    ("Testing", y_test),
]:

    distribution = (
        target
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    logger.info("%s Set", name)

    for label, percentage in distribution.items():
        logger.info(
            "   Burn_Label = %d : %.2f%%",
            label,
            percentage,
        )

logger.info("-" * 80)

# -----------------------------------------------------------------------------
# Shape Summary
# -----------------------------------------------------------------------------

logger.info("Feature Matrix Shapes")

logger.info("X_train      : %s", X_train.shape)
logger.info("X_validation : %s", X_validation.shape)
logger.info("X_test       : %s", X_test.shape)

logger.info("Target Shapes")

logger.info("y_train      : %s", y_train.shape)
logger.info("y_validation : %s", y_validation.shape)
logger.info("y_test       : %s", y_test.shape)

logger.info("=" * 80)

# =============================================================================
# SECTION 9 — BASELINE XGBOOST MODEL
# =============================================================================

logger.info("=" * 80)
logger.info("SECTION 9 - BASELINE XGBOOST MODEL")
logger.info("=" * 80)

# -----------------------------------------------------------------------------
# Calculate Class Weight
# -----------------------------------------------------------------------------

logger.info("Calculating class distribution from training set...")

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

logger.info("Negative samples : %d", negative_count)
logger.info("Positive samples : %d", positive_count)
logger.info("scale_pos_weight : %.4f", scale_pos_weight)

# -----------------------------------------------------------------------------
# Build Baseline Model
# -----------------------------------------------------------------------------

logger.info("Initializing XGBoost baseline model...")

baseline_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
)

# -----------------------------------------------------------------------------
# Train Model
# -----------------------------------------------------------------------------

logger.info("Training XGBoost model...")

baseline_model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_train, y_train),
        (X_validation, y_validation),
    ],
    verbose=False,
)

logger.info("Training completed successfully.")

# -----------------------------------------------------------------------------
# Best Iteration
# -----------------------------------------------------------------------------

logger.info("Best Iteration : %d", baseline_model.best_iteration)
logger.info("Best Score     : %.6f", baseline_model.best_score)

logger.info("=" * 80)

# =============================================================================
# SECTION 10 — HYPERPARAMETER TUNING
# =============================================================================

logger.info("=" * 80)
logger.info("SECTION 10 - HYPERPARAMETER TUNING")
logger.info("=" * 80)

# -----------------------------------------------------------------------------
# Hyperparameter Search Space
# -----------------------------------------------------------------------------

logger.info("Preparing hyperparameter search space...")

param_distributions = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0.0, 0.1, 0.3, 0.5],
}

# -----------------------------------------------------------------------------
# Randomized Search
# -----------------------------------------------------------------------------

logger.info("Running RandomizedSearchCV...")

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    ),
    param_distributions=param_distributions,
    n_iter=30,
    scoring="f1",
    cv=5,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
)

random_search.fit(
    X_train,
    y_train,
)

# -----------------------------------------------------------------------------
# Best Model
# -----------------------------------------------------------------------------

best_model = random_search.best_estimator_

logger.info("Hyperparameter tuning completed.")

logger.info("Best Cross Validation F1 Score : %.4f",
            random_search.best_score_)

logger.info("Best Parameters:")

for parameter, value in random_search.best_params_.items():

    logger.info(
        "   %-20s : %s",
        parameter,
        value
    )

logger.info("=" * 80)

# =============================================================================
# SECTION 11 — FINAL MODEL EVALUATION
# =============================================================================

logger.info("=" * 80)
logger.info("SECTION 11 - FINAL MODEL EVALUATION")
logger.info("=" * 80)

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------

logger.info("Generating predictions on the test set...")

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

logger.info("Prediction completed successfully.")

# -----------------------------------------------------------------------------
# Evaluation Metrics
# -----------------------------------------------------------------------------

logger.info("Calculating evaluation metrics...")

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC",
    ],
    "Value": [
        accuracy,
        balanced_accuracy,
        precision,
        recall,
        f1,
        roc_auc,
    ],
})

logger.info("")
logger.info("Evaluation Metrics")
logger.info("-" * 40)

for _, row in metrics.iterrows():
    logger.info(
        "%-20s : %.4f",
        row["Metric"],
        row["Value"],
    )

logger.info("-" * 40)

# -----------------------------------------------------------------------------
# Classification Report
# -----------------------------------------------------------------------------

logger.info("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    digits=4,
)

logger.info("\n%s", report)

# -----------------------------------------------------------------------------
# Confusion Matrix
# -----------------------------------------------------------------------------

logger.info("Generating confusion matrix...")

cm = confusion_matrix(
    y_test,
    y_pred,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(
    ax=ax,
    colorbar=False,
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FIG,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# -----------------------------------------------------------------------------
# ROC Curve
# -----------------------------------------------------------------------------

logger.info("Generating ROC Curve...")

fig, ax = plt.subplots(figsize=(6, 6))

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    ax=ax,
)

plt.tight_layout()

plt.savefig(
    ROC_CURVE_FIG,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# -----------------------------------------------------------------------------
# Save Evaluation Metrics
# -----------------------------------------------------------------------------

metrics.to_csv(
    METRICS_PATH,
    index=False,
)

logger.info("Evaluation metrics saved.")

logger.info("=" * 80)

# =============================================================================
# SECTION 12 — MODEL INTERPRETATION & SAVE RESULTS
# =============================================================================

logger.info("=" * 80)
logger.info("SECTION 12 - MODEL INTERPRETATION & SAVE RESULTS")
logger.info("=" * 80)

# -----------------------------------------------------------------------------
# Feature Importance
# -----------------------------------------------------------------------------

logger.info("Calculating feature importance...")

feature_importance = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Importance": best_model.feature_importances_,
})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False,
    )
    .reset_index(drop=True)
)

logger.info("\n%s", feature_importance.to_string(index=False))

# Save Feature Importance CSV
feature_importance.to_csv(
    FEATURE_IMPORTANCE_PATH,
    index=False,
)

# -----------------------------------------------------------------------------
# Feature Importance Plot
# -----------------------------------------------------------------------------

logger.info("Creating feature importance plot...")

fig, ax = plt.subplots(figsize=(8, 5))

ax.barh(
    feature_importance["Feature"],
    feature_importance["Importance"],
)

ax.invert_yaxis()

ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
ax.set_title("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig(
    FEATURE_IMPORTANCE_FIG,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# -----------------------------------------------------------------------------
# Prediction Result
# -----------------------------------------------------------------------------

logger.info("Saving prediction results...")

prediction_result = metadata_test.copy()

prediction_result["Actual_Label"] = y_test.values
prediction_result["Predicted_Label"] = y_pred
prediction_result["Prediction_Probability"] = y_prob

prediction_result.to_csv(
    PREDICTION_PATH,
    index=False,
)

logger.info("Prediction results saved.")

# -----------------------------------------------------------------------------
# Save Model
# -----------------------------------------------------------------------------

logger.info("Saving trained model...")

joblib.dump(
    best_model,
    MODEL_PATH,
)

logger.info("Model saved successfully.")

# -----------------------------------------------------------------------------
# Final Summary
# -----------------------------------------------------------------------------

logger.info("=" * 80)
logger.info("PIPELINE COMPLETED SUCCESSFULLY")
logger.info("=" * 80)

logger.info("Model")
logger.info("  %s", MODEL_PATH)

logger.info("Evaluation Metrics")
logger.info("  %s", METRICS_PATH)

logger.info("Prediction")
logger.info("  %s", PREDICTION_PATH)

logger.info("Feature Importance")
logger.info("  %s", FEATURE_IMPORTANCE_PATH)

logger.info("Confusion Matrix")
logger.info("  %s", CONFUSION_MATRIX_FIG)

logger.info("ROC Curve")
logger.info("  %s", ROC_CURVE_FIG)

logger.info("Feature Importance Figure")
logger.info("  %s", FEATURE_IMPORTANCE_FIG)

logger.info("=" * 80)