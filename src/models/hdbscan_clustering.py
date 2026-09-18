"""
HDBSCAN Clustering Pipeline for VIIRS Hotspot Analysis.

Title
-----
Analisis Zona Rawan dan Area Terbakar Kebakaran Hutan dan Lahan
Menggunakan HDBSCAN dan XGBoost Berbasis Data Hotspot VIIRS dan
Citra Sentinel-2 di Kabupaten Ketapang Tahun 2025–2026.

Description
-----------
This module implements the HDBSCAN clustering pipeline used in the
research workflow. The clustering process is performed exclusively
using VIIRS hotspot features and follows the finalized research
methodology.

Pipeline
--------
Section 1  : Configuration
Section 2  : Load Dataset
Section 3  : Exploratory Data Analysis
Section 4  : Feature Preparation
Section 5  : Hyperparameter Optimization
Section 6  : Final HDBSCAN Training
Section 7  : Model Evaluation
Section 8  : Cluster Profiling
Section 9  : Cluster Interpretation
Section 10 : Visualization
Section 11 : Export Results
Section 12 : Cleanup
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from sklearn.preprocessing import RobustScaler

import numpy as np
import pandas as pd

from dataclasses import dataclass
from itertools import product

import hdbscan
from hdbscan.validity import validity_index

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib
import os

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Project Metadata
# -----------------------------------------------------------------------------

PROJECT_NAME: str = "VIIRS Hotspot HDBSCAN Clustering"

RANDOM_SEED: int = 42

# -----------------------------------------------------------------------------
# Directory Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"

INPUT_DIR: Path = DATA_DIR / "processed"

OUTPUT_DIR: Path = DATA_DIR / "results" / "hdbscan"

FIGURE_DIR: Path = OUTPUT_DIR / "figures"

MODEL_DIR: Path = OUTPUT_DIR / "model"

# -----------------------------------------------------------------------------
# Input / Output Files
# -----------------------------------------------------------------------------

INPUT_DIR: Path = DATA_DIR / "features"

INPUT_CSV: Path = INPUT_DIR / "viirs_clustering_features.csv"

CLUSTER_RESULT_CSV: Path = OUTPUT_DIR / "viirs_cluster_results.csv"

CLUSTER_PROFILE_CSV: Path = OUTPUT_DIR / "cluster_profile.csv"

BEST_MODEL_FILE: Path = MODEL_DIR / "best_hdbscan_model.pkl"

CLUSTER_MAP_FIGURE = FIGURE_DIR / "cluster_map.png"
HOTSPOT_PROFILE_FIGURE = FIGURE_DIR / "hotspot_activity_profile.png"
FRP_PROFILE_FIGURE = FIGURE_DIR / "frp_profile.png"

# -----------------------------------------------------------------------------
# Export Files
# -----------------------------------------------------------------------------

GRID_SEARCH_RESULT_CSV: Path = OUTPUT_DIR / "grid_search_results.csv"

EVALUATION_SUMMARY_CSV: Path = OUTPUT_DIR / "evaluation_summary.csv"

CLUSTER_PROFILE_CSV: Path = OUTPUT_DIR / "cluster_profile.csv"

CLUSTER_INTERPRETATION_CSV: Path = OUTPUT_DIR / "cluster_interpretation.csv"

CLUSTERED_DATASET_CSV: Path = OUTPUT_DIR / "clustered_dataset.csv"

FINAL_MODEL_FILE: Path = MODEL_DIR / "final_hdbscan_model.pkl"

# -----------------------------------------------------------------------------
# Dataset Columns
# -----------------------------------------------------------------------------

ID_COLUMN: str = "pixel_id"

LATITUDE_COLUMN: str = "latitude"

LONGITUDE_COLUMN: str = "longitude"

FEATURE_COLUMNS: list[str] = [
    "Hotspot_Frequency",
    "FRP_Max",
]

METADATA_COLUMNS: list[str] = [
    ID_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
]

CONFIDENCE_COLUMN: str = "Confidence_Score"

# -----------------------------------------------------------------------------
# HDBSCAN Fixed Parameters
# -----------------------------------------------------------------------------

HDBSCAN_METRIC: str = "euclidean"

CLUSTER_SELECTION_METHOD: str = "eom"

ALLOW_SINGLE_CLUSTER: bool = False

PREDICTION_DATA: bool = True

# -----------------------------------------------------------------------------
# Hyperparameter Search Space
# -----------------------------------------------------------------------------

MIN_CLUSTER_SIZE_RANGE: list[int] = [
    25,
    50,
    75,
    100,
    150,
    200,
    300,
]

MIN_SAMPLES_RANGE: list[int] = [
    5,
    10,
    15,
    20,
    25,
    30,
]

# -----------------------------------------------------------------------------
# Evaluation Configuration
# -----------------------------------------------------------------------------

PRIMARY_METRIC: str = "DBCV"

SECONDARY_METRIC: str = "Silhouette"

NOISE_LABEL: int = -1

# -----------------------------------------------------------------------------
# Grid Search Configuration
# -----------------------------------------------------------------------------

DBCV_TOLERANCE: float = 0.005

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

FIGURE_DPI: int = 300

FIGURE_FORMAT: str = "png"

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

LOGGER_NAME: str = "HDBSCAN_PIPELINE"

logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(handler)

logger.propagate = False

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

logger.info("Configuration initialized successfully.")
logger.info("Project           : %s", PROJECT_NAME)
logger.info("Input dataset     : %s", INPUT_CSV)
logger.info("Output directory  : %s", OUTPUT_DIR)
logger.info("Feature columns   : %s", FEATURE_COLUMNS)
logger.info("Random seed       : %d", RANDOM_SEED)

# =============================================================================
# SECTION 2 — LOAD DATASET
# =============================================================================

REQUIRED_COLUMNS: list[str] = [
    ID_COLUMN,
    LATITUDE_COLUMN,
    LONGITUDE_COLUMN,
    *FEATURE_COLUMNS,
    CONFIDENCE_COLUMN,
]


def load_dataset(csv_path: Path = INPUT_CSV) -> pd.DataFrame:
    """
    Load and validate the VIIRS hotspot dataset.

    This function is responsible for reading the extracted CSV file and
    performing structural validation before the dataset enters the analysis
    pipeline.

    Validation performed
    --------------------
    1. Input file exists.
    2. Dataset is not empty.
    3. Required columns are present.

    Parameters
    ----------
    csv_path : Path, default=INPUT_CSV
        Path to the extracted VIIRS feature CSV.

    Returns
    -------
    pandas.DataFrame
        Validated dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file cannot be found.

    ValueError
        If the dataset is empty or required columns are missing.
    """

    logger.info("=" * 80)
    logger.info("SECTION 2 — LOAD DATASET")
    logger.info("=" * 80)

    logger.info("Reading dataset from:")
    logger.info("%s", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Input dataset is empty.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns:\n"
            f"{missing_columns}"
        )

    logger.info("Dataset loaded successfully.")

    logger.info("Number of observations : %d", len(df))
    logger.info("Number of variables    : %d", df.shape[1])

    logger.info("Feature columns:")

    for feature in FEATURE_COLUMNS:
        logger.info("  • %s", feature)

    logger.info(
        "Coordinate columns : %s, %s",
        LATITUDE_COLUMN,
        LONGITUDE_COLUMN,
    )

    logger.info("Dataset validation passed.")

    return df

# =============================================================================
# SECTION 3 — EXPLORATORY DATA ANALYSIS
# =============================================================================

def exploratory_data_analysis(df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis (EDA) on the VIIRS hotspot dataset.

    This function provides descriptive statistics and quality assessment
    of the dataset before feature preprocessing. The dataset is not modified.

    Analysis performed
    ------------------
    1. Dataset dimensions.
    2. Data types.
    3. Missing values.
    4. Duplicate observations.
    5. Descriptive statistics.
    6. Feature range.
    7. Coordinate range.

    Parameters
    ----------
    df : pandas.DataFrame
        Validated VIIRS hotspot dataset.

    Returns
    -------
    None
    """

    logger.info("=" * 80)
    logger.info("SECTION 3 — EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Dataset Overview
    # -------------------------------------------------------------------------

    logger.info("Dataset overview")
    logger.info("Observations : %d", len(df))
    logger.info("Variables    : %d", df.shape[1])

    # -------------------------------------------------------------------------
    # Data Types
    # -------------------------------------------------------------------------

    logger.info("-" * 80)
    logger.info("Column data types")

    for column, dtype in df.dtypes.items():
        logger.info("%-20s : %s", column, dtype)

    # -------------------------------------------------------------------------
    # Missing Values
    # -------------------------------------------------------------------------

    logger.info("-" * 80)
    logger.info("Missing value assessment")

    missing = df.isna().sum()

    total_missing = int(missing.sum())

    if total_missing == 0:
        logger.info("No missing values detected.")
    else:
        logger.warning("Missing values detected.")

        for column, value in missing.items():
            if value > 0:
                logger.warning(
                    "%-20s : %d",
                    column,
                    value,
                )

    # -------------------------------------------------------------------------
    # Duplicate Rows
    # -------------------------------------------------------------------------

    logger.info("-" * 80)
    logger.info("Duplicate assessment")

    duplicate_rows = int(df.duplicated().sum())

    logger.info(
        "Duplicate rows : %d",
        duplicate_rows,
    )

    # -------------------------------------------------------------------------
    # Descriptive Statistics
    # -------------------------------------------------------------------------

    logger.info("-" * 80)
    logger.info("Descriptive statistics")

    statistics = df[FEATURE_COLUMNS].describe()

    for feature in FEATURE_COLUMNS:

        logger.info("")
        logger.info(feature)

        logger.info(
            "Count : %.0f",
            statistics.loc["count", feature],
        )

        logger.info(
            "Mean  : %.4f",
            statistics.loc["mean", feature],
        )

        logger.info(
            "Std   : %.4f",
            statistics.loc["std", feature],
        )

        logger.info(
            "Min   : %.4f",
            statistics.loc["min", feature],
        )

        logger.info(
            "25%%   : %.4f",
            statistics.loc["25%", feature],
        )

        logger.info(
            "50%%   : %.4f",
            statistics.loc["50%", feature],
        )

        logger.info(
            "75%%   : %.4f",
            statistics.loc["75%", feature],
        )

        logger.info(
            "Max   : %.4f",
            statistics.loc["max", feature],
        )

    # -------------------------------------------------------------------------
    # Feature Range
    # -------------------------------------------------------------------------

    logger.info("-" * 80)
    logger.info("Feature ranges")

    for feature in FEATURE_COLUMNS:

        logger.info(
            "%s : %.4f → %.4f",
            feature,
            df[feature].min(),
            df[feature].max(),
        )

    # -------------------------------------------------------------------------
    # Coordinate Range
    # -------------------------------------------------------------------------

    logger.info("-" * 80)
    logger.info("Coordinate ranges")

    logger.info(
        "Latitude  : %.6f → %.6f",
        df[LATITUDE_COLUMN].min(),
        df[LATITUDE_COLUMN].max(),
    )

    logger.info(
        "Longitude : %.6f → %.6f",
        df[LONGITUDE_COLUMN].min(),
        df[LONGITUDE_COLUMN].max(),
    )

    logger.info("-" * 80)
    logger.info("EDA completed successfully.")

# =============================================================================
# SECTION 4 — FEATURE PREPARATION
# =============================================================================

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, RobustScaler]:
    """
    Prepare the feature matrix for HDBSCAN clustering.

    This function selects the finalized clustering features and applies
    RobustScaler according to the finalized research methodology.

    Selected Features
    -----------------
    - Hotspot_Frequency
    - FRP_Max

    Parameters
    ----------
    df : pandas.DataFrame
        Validated VIIRS hotspot dataset.

    Returns
    -------
    tuple[pandas.DataFrame, RobustScaler]
        A tuple containing:

        - Robust-scaled feature matrix.
        - Fitted RobustScaler instance.
    """

    logger.info("=" * 80)
    logger.info("SECTION 4 — FEATURE PREPARATION")
    logger.info("=" * 80)

    logger.info("Selecting clustering features...")

    X = df.loc[:, FEATURE_COLUMNS].copy()

    logger.info("Selected features : %s", FEATURE_COLUMNS)

    logger.info("Feature matrix shape : %s", X.shape)

    logger.info("Applying RobustScaler...")

    scaler = RobustScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=FEATURE_COLUMNS,
        index=df.index,
    )

    logger.info("RobustScaler successfully fitted.")

    logger.info("Scaled feature matrix shape : %s", X_scaled.shape)

    logger.info("Scaler statistics:")

    for feature, center, scale in zip(
        FEATURE_COLUMNS,
        scaler.center_,
        scaler.scale_,
    ):
        logger.info(
            "%-20s | Median = %10.4f | IQR = %10.4f",
            feature,
            center,
            scale,
        )

    logger.info("Scaled feature summary:")

    summary = X_scaled.describe()

    for feature in FEATURE_COLUMNS:

        logger.info(
            "%-20s | Mean = %8.4f | Std = %8.4f | Min = %8.4f | Max = %8.4f",
            feature,
            summary.loc["mean", feature],
            summary.loc["std", feature],
            summary.loc["min", feature],
            summary.loc["max", feature],
        )

    logger.info("Feature preparation completed successfully.")

    return X_scaled, scaler

# =============================================================================
# SECTION 5A — GRID SEARCH RESULT
# =============================================================================

@dataclass(slots=True)
class GridSearchResult:
    """
    Store the evaluation result of one hyperparameter combination.
    """

    min_cluster_size: int
    min_samples: int

    dbcv: float
    silhouette: float

    n_clusters: int
    n_noise: int
    noise_ratio: float

    is_valid: bool

    @property
    def has_noise(self) -> bool:
        return self.n_noise > 0

    @property
    def has_multiple_clusters(self) -> bool:
        return self.n_clusters >= 2

    # ------------------------------------------------------------------
    # Model Status
    # ------------------------------------------------------------------

    is_valid: bool

    @property
    def has_noise(self) -> bool:
        """
        Return True if the clustering result contains noise observations.
        """
        return self.n_noise > 0

    @property
    def has_multiple_clusters(self) -> bool:
        """
        Return True if at least two valid clusters are produced.
        """
        return self.n_clusters >= 2

# =============================================================================
# FINAL MODEL RESULT
# =============================================================================

@dataclass(slots=True)
class BestModelResult:
    """
    Store the final trained HDBSCAN model.
    """

    model: hdbscan.HDBSCAN

    result: GridSearchResult

    labels: np.ndarray

    @property
    def probabilities(self) -> np.ndarray:
        """
        Return membership probabilities estimated by HDBSCAN.
        """

        return self.model.probabilities_

    @property
    def outlier_scores(self) -> np.ndarray:
        """
        Return HDBSCAN outlier scores.
        """

        return self.model.outlier_scores_
    @property
    def dbcv(self) -> float:
        return self.result.dbcv


    @property
    def silhouette(self) -> float:
        return self.result.silhouette


    @property
    def n_clusters(self) -> int:
        return self.result.n_clusters


    @property
    def n_noise(self) -> int:
        return self.result.n_noise


    @property
    def noise_ratio(self) -> float:
        return self.result.noise_ratio

def count_clusters(labels: np.ndarray) -> int:
    """
    Count the number of valid clusters.

    Noise points (-1) are excluded.

    Parameters
    ----------
    labels : numpy.ndarray
        Cluster labels produced by HDBSCAN.

    Returns
    -------
    int
        Number of valid clusters.
    """

    unique_labels = np.unique(labels)

    return int(np.sum(unique_labels != NOISE_LABEL))

def count_noise(labels: np.ndarray) -> int:
    """
    Count the number of noise observations.

    Parameters
    ----------
    labels : numpy.ndarray

    Returns
    -------
    int
    """

    return int(np.sum(labels == NOISE_LABEL))

def calculate_noise_ratio(labels: np.ndarray) -> float:
    """
    Calculate the proportion of noise observations.

    Parameters
    ----------
    labels : numpy.ndarray

    Returns
    -------
    float
        Noise ratio in the range [0, 1].
    """

    if len(labels) == 0:
        raise ValueError(...)

    return count_noise(labels) / len(labels)

def is_valid_clustering(labels: np.ndarray) -> bool:
    """
    Determine whether a clustering result is valid for evaluation.

    A valid clustering must contain at least two non-noise clusters.

    Parameters
    ----------
    labels : numpy.ndarray

    Returns
    -------
    bool
    """

    return count_clusters(labels) >= 2

def log_grid_search_header() -> None:
    """
    Log the Grid Search section header.
    """

    logger.info("=" * 80)
    logger.info("SECTION 5 — HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)

    logger.info(
        "Grid Search combinations : %d",
        len(MIN_CLUSTER_SIZE_RANGE) * len(MIN_SAMPLES_RANGE),
    )

    logger.info(
        "DBCV tolerance : %.3f",
        DBCV_TOLERANCE,
    )

# =============================================================================
# SECTION 5B — CLUSTER EVALUATION
# =============================================================================

def calculate_dbcv(
    X: pd.DataFrame,
    labels: np.ndarray,
) -> float:
    """
    Calculate the Density-Based Clustering Validation (DBCV) score.

    Parameters
    ----------
    X : pandas.DataFrame
        Robust-scaled feature matrix.

    labels : numpy.ndarray
        Cluster labels produced by HDBSCAN.

    Returns
    -------
    float
        DBCV score. Returns NaN if the clustering result is not valid.
    """

    if not is_valid_clustering(labels):
        return float("nan")

    try:
        return float(
            validity_index(
                X.to_numpy(),
                labels,
                metric=HDBSCAN_METRIC,
            )
        )

    except Exception as exc:

        logger.warning(
            "Failed to calculate DBCV: %s",
            exc,
        )

        return float("nan")
    
def calculate_silhouette(
    X: pd.DataFrame,
    labels: np.ndarray,
) -> float:
    """
    Calculate the Silhouette Score.

    Noise observations are excluded from the calculation.

    Parameters
    ----------
    X : pandas.DataFrame
        Robust-scaled feature matrix.

    labels : numpy.ndarray
        Cluster labels.

    Returns
    -------
    float
        Silhouette Score. Returns NaN if the clustering result
        is not valid.
    """

    if not is_valid_clustering(labels):
        return float("nan")

    mask = labels != NOISE_LABEL

    X_valid = X.loc[mask]

    labels_valid = labels[mask]

    if len(np.unique(labels_valid)) < 2:
        return float("nan")

    try:

        return float(
            silhouette_score(
                X_valid,
                labels_valid,
                metric="euclidean",
            )
        )

    except Exception as exc:

        logger.warning(
            "Failed to calculate Silhouette Score: %s",
            exc,
        )

        return float("nan")

def evaluate_clustering(
    X: pd.DataFrame,
    labels: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> GridSearchResult:
    """
    Evaluate a single HDBSCAN clustering result.

    Parameters
    ----------
    X : pandas.DataFrame
        Robust-scaled feature matrix.

    labels : numpy.ndarray
        Cluster labels.

    min_cluster_size : int
        HDBSCAN min_cluster_size.

    min_samples : int
        HDBSCAN min_samples.

    Returns
    -------
    GridSearchResult
        Evaluation summary for one hyperparameter combination.
    """

    n_clusters = count_clusters(labels)

    n_noise = count_noise(labels)

    noise_ratio = calculate_noise_ratio(labels)

    valid = is_valid_clustering(labels)

    dbcv = calculate_dbcv(
        X,
        labels,
    )

    silhouette = calculate_silhouette(
        X,
        labels,
    )

    logger.info(
        (
            "Evaluation | "
            "min_cluster_size=%d | "
            "min_samples=%d | "
            "clusters=%d | "
            "noise=%d (%.2f%%) | "
            "DBCV=%.4f | "
            "Silhouette=%.4f"
        ),
        min_cluster_size,
        min_samples,
        n_clusters,
        n_noise,
        noise_ratio * 100,
        dbcv,
        silhouette,
    )

    return GridSearchResult(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        dbcv=dbcv,
        silhouette=silhouette,
        n_clusters=n_clusters,
        n_noise=n_noise,
        noise_ratio=noise_ratio,
        is_valid=valid,
    )

# =============================================================================
# SECTION 5C — GRID SEARCH
# =============================================================================

def perform_grid_search(
    X: pd.DataFrame,
) -> list[GridSearchResult]:
    """
    Perform exhaustive Grid Search for HDBSCAN hyperparameter optimization.

    Every combination of min_cluster_size and min_samples is evaluated.

    Parameters
    ----------
    X : pandas.DataFrame
        Robust-scaled feature matrix.

    Returns
    -------
    list[GridSearchResult]
        Evaluation results for all hyperparameter combinations.
    """

    log_grid_search_header()

    results: list[GridSearchResult] = []

    parameter_grid = list(
        product(
            MIN_CLUSTER_SIZE_RANGE,
            MIN_SAMPLES_RANGE,
        )
    )

    logger.info(
        "Evaluating %d hyperparameter combinations...",
        len(parameter_grid),
    )

    for index, (min_cluster_size, min_samples) in enumerate(
        parameter_grid,
        start=1,
    ):

        logger.info(
            "[%d/%d] "
            "min_cluster_size=%d | min_samples=%d",
            index,
            len(parameter_grid),
            min_cluster_size,
            min_samples,
        )

        try:

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=HDBSCAN_METRIC,
                cluster_selection_method=CLUSTER_SELECTION_METHOD,
                allow_single_cluster=ALLOW_SINGLE_CLUSTER,
                prediction_data=PREDICTION_DATA,
            )

            labels = clusterer.fit_predict(X)

            result = evaluate_clustering(
                X=X,
                labels=labels,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )

            results.append(result)

        except Exception as exc:

            logger.exception(
                (
                    "Grid Search failed for "
                    "min_cluster_size=%d, "
                    "min_samples=%d"
                ),
                min_cluster_size,
                min_samples,
            )

            results.append(
                GridSearchResult(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    dbcv=float("nan"),
                    silhouette=float("nan"),
                    n_clusters=0,
                    n_noise=0,
                    noise_ratio=0.0,
                    is_valid=False,
                )
            )

    logger.info(
        "Grid Search completed successfully."
    )

    logger.info(
        "Total evaluated models : %d",
        len(results),
    )

    valid_models = sum(
        result.is_valid
        for result in results
    )

    logger.info(
        "Valid models           : %d",
        valid_models,
    )

    logger.info(
        "Invalid models         : %d",
        len(results) - valid_models,
    )

    return results

# =============================================================================
# SECTION 5D — MODEL SELECTION
# =============================================================================

def select_best_model(
    results: list[GridSearchResult],
) -> tuple[GridSearchResult, pd.DataFrame]:
    """
    Select the best hyperparameter combination from Grid Search.

    Model selection strategy
    ------------------------
    1. Highest DBCV.
    2. Models within DBCV_TOLERANCE from the best DBCV become candidates.
    3. Highest Silhouette Score among the candidates.
    4. If still tied, preserve Grid Search order.

    Parameters
    ----------
    results : list[GridSearchResult]
        Grid Search evaluation results.

    Returns
    -------
    tuple[GridSearchResult, pandas.DataFrame]
        Best result and Grid Search summary table.
    """

    logger.info("=" * 80)
    logger.info("SECTION 5D — MODEL SELECTION")
    logger.info("=" * 80)

    records = [
        {
            "min_cluster_size": r.min_cluster_size,
            "min_samples": r.min_samples,
            "dbcv": r.dbcv,
            "silhouette": r.silhouette,
            "n_clusters": r.n_clusters,
            "n_noise": r.n_noise,
            "noise_ratio": r.noise_ratio,
            "is_valid": r.is_valid,
        }
        for r in results
    ]

    summary = pd.DataFrame(records)

    valid_summary = summary.loc[
        summary["is_valid"]
    ].copy()

    if valid_summary.empty:
        raise RuntimeError(
            "Grid Search did not produce any valid clustering result."
        )

    best_dbcv = valid_summary["dbcv"].max()

    logger.info("Best DBCV found : %.6f", best_dbcv)

    candidate_mask = (
        valid_summary["dbcv"]
        >= best_dbcv - DBCV_TOLERANCE
    )

    candidates = valid_summary.loc[candidate_mask].copy()

    logger.info(
        "Candidate models within tolerance : %d",
        len(candidates),
    )

    candidates = candidates.sort_values(
        by="silhouette",
        ascending=False,
        kind="stable",
    )

    best_row = candidates.iloc[0]

    best_result = next(
        result
        for result in results
        if (
            result.min_cluster_size
            == best_row["min_cluster_size"]
            and result.min_samples
            == best_row["min_samples"]
        )
    )

    logger.info("-" * 80)

    logger.info("Best Hyperparameters")

    logger.info(
        "min_cluster_size : %d",
        best_result.min_cluster_size,
    )

    logger.info(
        "min_samples      : %d",
        best_result.min_samples,
    )

    logger.info(
        "DBCV             : %.6f",
        best_result.dbcv,
    )

    logger.info(
        "Silhouette       : %.6f",
        best_result.silhouette,
    )

    logger.info(
        "Clusters         : %d",
        best_result.n_clusters,
    )

    logger.info(
        "Noise            : %d (%.2f%%)",
        best_result.n_noise,
        best_result.noise_ratio * 100,
    )

    logger.info("=" * 80)
    logger.info("Hyperparameter optimization completed.")
    logger.info("=" * 80)

    return best_result, summary

# =============================================================================
# SECTION 6 — FINAL HDBSCAN TRAINING
# =============================================================================

def train_final_model(
    X: pd.DataFrame,
    best_result: GridSearchResult,
) -> BestModelResult:
    """
    Train the final HDBSCAN model using the best hyperparameters
    obtained from Grid Search.

    Parameters
    ----------
    X : pandas.DataFrame
        Robust-scaled feature matrix.

    best_result : GridSearchResult
        Best hyperparameter combination selected during Grid Search.

    Returns
    -------
    BestModelResult
        Final trained HDBSCAN model together with its clustering labels
        and evaluation summary.
    """

    logger.info("=" * 80)
    logger.info("SECTION 6 — FINAL HDBSCAN TRAINING")
    logger.info("=" * 80)

    logger.info("Training final HDBSCAN model...")

    logger.info(
        "Best parameters | min_cluster_size=%d | min_samples=%d",
        best_result.min_cluster_size,
        best_result.min_samples,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=best_result.min_cluster_size,
        min_samples=best_result.min_samples,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        allow_single_cluster=ALLOW_SINGLE_CLUSTER,
        prediction_data=PREDICTION_DATA,
    )

    labels = clusterer.fit_predict(X)

    logger.info("Final model trained successfully.")

    # -------------------------------------------------------------------------
    # Consistency Check
    # -------------------------------------------------------------------------

    n_clusters = count_clusters(labels)
    n_noise = count_noise(labels)
    noise_ratio = calculate_noise_ratio(labels)

    logger.info("Final clustering summary")

    logger.info("Clusters      : %d", n_clusters)

    logger.info(
        "Noise         : %d (%.2f%%)",
        n_noise,
        noise_ratio * 100,
    )

    logger.info(
        "Observations  : %d",
        len(labels),
    )

    logger.info("Training completed successfully.")

    return BestModelResult(
        model=clusterer,
        result=best_result,
        labels=labels,
    )

# =============================================================================
# SECTION 7 — MODEL EVALUATION
# =============================================================================

def evaluate_final_model(
    best_model: BestModelResult,
) -> pd.DataFrame:
    """
    Summarize the evaluation results of the final HDBSCAN model.

    This function reports the evaluation metrics and clustering statistics
    obtained from the final trained model. No retraining or metric
    recalculation is performed.

    Parameters
    ----------
    best_model : BestModelResult
        Final trained HDBSCAN model.

    Returns
    -------
    pandas.DataFrame
        Evaluation summary containing the final model configuration,
        clustering statistics, and evaluation metrics.
    """

    logger.info("=" * 80)
    logger.info("SECTION 7 — MODEL EVALUATION")
    logger.info("=" * 80)

    logger.info("Final HDBSCAN Model Evaluation")

    logger.info(
        "min_cluster_size : %d",
        best_model.result.min_cluster_size,
    )

    logger.info(
        "min_samples      : %d",
        best_model.result.min_samples,
    )

    logger.info(
        "Number of clusters : %d",
        best_model.n_clusters,
    )

    logger.info(
        "Noise observations : %d",
        best_model.n_noise,
    )

    logger.info(
        "Noise ratio        : %.2f%%",
        best_model.noise_ratio * 100,
    )

    logger.info(
        "DBCV Score         : %.6f",
        best_model.dbcv,
    )

    logger.info(
        "Silhouette Score   : %.6f",
        best_model.silhouette,
    )

    logger.info(
        "Total observations : %d",
        len(best_model.labels),
    )

    evaluation_summary = pd.DataFrame(
        [
            {
                "min_cluster_size": best_model.result.min_cluster_size,
                "min_samples": best_model.result.min_samples,
                "n_clusters": best_model.n_clusters,
                "n_noise": best_model.n_noise,
                "noise_ratio": best_model.noise_ratio,
                "dbcv": best_model.dbcv,
                "silhouette": best_model.silhouette,
                "total_observations": len(best_model.labels),
            }
        ]
    )

    logger.info("Model evaluation completed successfully.")

    return evaluation_summary

# =============================================================================
# SECTION 8 — CLUSTER PROFILING
# =============================================================================

def profile_clusters(
    df: pd.DataFrame,
    best_model: BestModelResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate descriptive statistics for each HDBSCAN cluster.

    This function appends the final cluster labels to the original dataset
    and computes descriptive statistics for each cluster.

    Parameters
    ----------
    df : pandas.DataFrame
        Original VIIRS hotspot dataset.

    best_model : BestModelResult
        Final trained HDBSCAN model.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        A tuple containing:

        - Dataset with cluster labels.
        - Cluster profiling table.
    """

    logger.info("=" * 80)
    logger.info("SECTION 8 — CLUSTER PROFILING")
    logger.info("=" * 80)

    logger.info("Attaching cluster labels to dataset...")

    clustered_df = df.copy()

    clustered_df["Cluster"] = best_model.labels

    logger.info("Cluster labels successfully added.")

    # -------------------------------------------------------------------------
    # Dataset Statistics
    # -------------------------------------------------------------------------

    total_observations = len(clustered_df)

    logger.info(
        "Total observations : %d",
        total_observations,
    )

    # -------------------------------------------------------------------------
    # Cluster Profiling
    # -------------------------------------------------------------------------

    logger.info("Computing cluster statistics...")

    cluster_profile = (
        clustered_df
        .groupby("Cluster", dropna=False)
        .agg(
            Pixel_Count=("pixel_id", "count"),
            Mean_Hotspot_Frequency=("Hotspot_Frequency", "mean"),
            Median_Hotspot_Frequency=("Hotspot_Frequency", "median"),
            Std_Hotspot_Frequency=("Hotspot_Frequency", "std"),
            Min_Hotspot_Frequency=("Hotspot_Frequency", "min"),
            Max_Hotspot_Frequency=("Hotspot_Frequency", "max"),
            Mean_FRP_Max=("FRP_Max", "mean"),
            Median_FRP_Max=("FRP_Max", "median"),
            Std_FRP_Max=("FRP_Max", "std"),
            Min_FRP_Max=("FRP_Max", "min"),
            Max_FRP_Max=("FRP_Max", "max"),
        )
        .reset_index()
    )

    cluster_profile["Percentage"] = (
        cluster_profile["Pixel_Count"]
        / total_observations
        * 100
    )

    cluster_profile = cluster_profile[
        [
            "Cluster",
            "Pixel_Count",
            "Percentage",
            "Mean_Hotspot_Frequency",
            "Median_Hotspot_Frequency",
            "Std_Hotspot_Frequency",
            "Min_Hotspot_Frequency",
            "Max_Hotspot_Frequency",
            "Mean_FRP_Max",
            "Median_FRP_Max",
            "Std_FRP_Max",
            "Min_FRP_Max",
            "Max_FRP_Max",
        ]
    ]

    logger.info(
        "Cluster profiling completed."
    )

    logger.info(
        "Number of profiled clusters (including noise): %d",
        len(cluster_profile),
    )

    for _, row in cluster_profile.iterrows():

        logger.info(
            (
                "Cluster %s | "
                "Pixels=%d | "
                "Percentage=%.2f%% | "
                "Mean Hotspot=%.2f | "
                "Mean FRP=%.2f"
            ),
            row["Cluster"],
            row["Pixel_Count"],
            row["Percentage"],
            row["Mean_Hotspot_Frequency"],
            row["Mean_FRP_Max"],
        )

    logger.info("Section 8 completed successfully.")

    return clustered_df, cluster_profile

# =============================================================================
# SECTION 9 — CLUSTER INTERPRETATION
# =============================================================================

def interpret_clusters(
    cluster_profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare scientific interpretation for each HDBSCAN cluster.

    This function transforms the descriptive cluster profile into an
    interpretable wildfire activity ranking. The interpretation is based on
    multiple hotspot activity indicators rather than a single variable.

    Workflow
    --------
    1. Copy cluster profile.
    2. Separate valid clusters and noise observations.
    3. Prepare data for multi-criteria ranking.
    4. Assign activity rank.
    5. Assign wildfire risk level.
    6. Generate interpretation for each cluster.

    Parameters
    ----------
    cluster_profile : pandas.DataFrame
        Cluster profiling table generated in Section 8.

    Returns
    -------
    pandas.DataFrame
        Cluster interpretation table.
    """

    logger.info("=" * 80)
    logger.info("SECTION 9 — CLUSTER INTERPRETATION")
    logger.info("=" * 80)

    logger.info("Preparing cluster interpretation...")

    # -------------------------------------------------------------------------
    # Create Working Copy
    # -------------------------------------------------------------------------

    interpretation = cluster_profile.copy()

    logger.info(
        "Total profiled clusters (including noise): %d",
        len(interpretation),
    )

    # -------------------------------------------------------------------------
    # Separate Valid Clusters and Noise
    # -------------------------------------------------------------------------

    logger.info("Separating valid clusters and noise observations...")

    valid_clusters = (
        interpretation.loc[
            interpretation["Cluster"] != NOISE_LABEL
        ]
        .copy()
        .reset_index(drop=True)
    )

    noise_cluster = (
        interpretation.loc[
            interpretation["Cluster"] == NOISE_LABEL
        ]
        .copy()
        .reset_index(drop=True)
    )

    logger.info(
        "Valid clusters : %d",
        len(valid_clusters),
    )

    logger.info(
        "Noise clusters : %d",
        len(noise_cluster),
    )

    # -------------------------------------------------------------------------
    # Empty Validation
    # -------------------------------------------------------------------------

    if valid_clusters.empty:

        raise ValueError(
            "No valid clusters available for interpretation."
        )

    logger.info("Cluster interpretation data prepared successfully.")

    # -------------------------------------------------------------------------
    # Multi-Criteria Ranking
    # -------------------------------------------------------------------------
    logger.info(
        "Performing multi-criteria cluster ranking..."
    )

    # -------------------------------------------------------------------------
    # Sort Clusters by Scientific Priority
    # -------------------------------------------------------------------------

    valid_clusters = (
        valid_clusters
        .sort_values(
            by=[
                "Mean_Hotspot_Frequency",
                "Mean_FRP_Max",
                "Pixel_Count",
                "Percentage",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Clusters successfully sorted using multi-criteria ranking."
    )

    # -------------------------------------------------------------------------
    # Assign Activity Rank
    # -------------------------------------------------------------------------

    valid_clusters["Activity_Rank"] = (
        np.arange(1, len(valid_clusters) + 1)
    )

    logger.info(
        "Activity ranks assigned."
    )

    # -------------------------------------------------------------------------
    # Document Ranking Basis
    # -------------------------------------------------------------------------

    ranking_basis = []

    for _, row in valid_clusters.iterrows():

        basis = (
            "Ranked by "
            "Hotspot Frequency "
            f"({row['Mean_Hotspot_Frequency']:.2f}), "
            "FRP "
            f"({row['Mean_FRP_Max']:.2f}), "
            "Pixel Count "
            f"({int(row['Pixel_Count'])}, "
            f"{row['Percentage']:.2f}% of dataset)."
        )

        ranking_basis.append(basis)

    valid_clusters["Ranking_Basis"] = ranking_basis

    logger.info(
        "Ranking basis successfully generated."
    )

    # -------------------------------------------------------------------------
    # Preview Ranking
    # -------------------------------------------------------------------------

    logger.info("Top ranked clusters:")

    preview = min(5, len(valid_clusters))

    for _, row in valid_clusters.head(preview).iterrows():

        logger.info(
            (
                "Rank %d | "
                "Cluster %d | "
                "Mean Hotspot = %.2f | "
                "Mean FRP = %.2f | "
                "Pixels = %d | "
                "Percentage = %.2f%%"
            ),
            row["Activity_Rank"],
            row["Cluster"],
            row["Mean_Hotspot_Frequency"],
            row["Mean_FRP_Max"],
            row["Pixel_Count"],
            row["Percentage"],
        )

    logger.info(
        "Multi-criteria ranking completed successfully."
    )

    # -------------------------------------------------------------------------
    # Risk Level Classification
    # -------------------------------------------------------------------------
    logger.info(
        "Assigning wildfire risk levels..."
    )

    # -------------------------------------------------------------------------
    # Determine Number of Valid Clusters
    # -------------------------------------------------------------------------

    total_clusters = len(valid_clusters)

    logger.info(
        "Valid clusters available : %d",
        total_clusters,
    )

    # -------------------------------------------------------------------------
    # Dynamic Risk Labels
    # -------------------------------------------------------------------------

    risk_labels = [
        "Very High",
        "High",
        "Moderate",
        "Low",
        "Very Low",
    ]

    if total_clusters == 1:

        assigned_levels = [
            "Very High"
        ]

    elif total_clusters == 2:

        assigned_levels = [
            "High",
            "Low",
        ]

    elif total_clusters == 3:

        assigned_levels = [
            "High",
            "Moderate",
            "Low",
        ]

    elif total_clusters == 4:

        assigned_levels = [
            "Very High",
            "High",
            "Moderate",
            "Low",
        ]

    elif total_clusters == 5:

        assigned_levels = risk_labels

    else:

        # -------------------------------------------------------------
        # Dynamic Equal Split
        # -------------------------------------------------------------

        indices = np.linspace(
            0,
            len(risk_labels) - 1,
            total_clusters,
        )

        assigned_levels = [
            risk_labels[
                round(index)
            ]
            for index in indices
        ]

    valid_clusters["Risk_Level"] = assigned_levels

    logger.info(
        "Risk levels successfully assigned."
    )

    # -------------------------------------------------------------------------
    # Preview
    # -------------------------------------------------------------------------

    logger.info(
        "Assigned wildfire risk levels:"
    )

    for _, row in valid_clusters.iterrows():

        logger.info(
            (
                "Rank %d | "
                "Cluster %d | "
                "%s"
            ),
            row["Activity_Rank"],
            row["Cluster"],
            row["Risk_Level"],
        )

    logger.info(
        "Risk level classification completed successfully."
    )

    # -------------------------------------------------------------------------
    # Cluster Interpretation
    # -------------------------------------------------------------------------
    logger.info(
        "Generating scientific cluster interpretation..."
    )

    # -------------------------------------------------------------------------
    # Generate Interpretation
    # -------------------------------------------------------------------------

    interpretations = []

    for _, row in valid_clusters.iterrows():

        hotspot = row["Mean_Hotspot_Frequency"]
        frp = row["Mean_FRP_Max"]
        pixels = int(row["Pixel_Count"])
        percentage = row["Percentage"]
        risk = row["Risk_Level"]

        if risk == "Very High":

            text = (
                "Very high wildfire activity. "
                f"This cluster ranks #{int(row['Activity_Rank'])} "
                "with the highest hotspot frequency "
                f"({hotspot:.2f}) "
                "and strong fire radiative power "
                f"({frp:.2f}). "
                f"The cluster contains {pixels:,} pixels "
                f"({percentage:.2f}% of all observations). "
                "This area should be considered the highest priority "
                "for wildfire prevention, monitoring, and mitigation."
            )

        elif risk == "High":

            text = (
                "High wildfire activity. "
                f"This cluster ranks #{int(row['Activity_Rank'])} "
                "with high hotspot occurrence "
                f"({hotspot:.2f}) "
                "and elevated fire radiative power "
                f"({frp:.2f}). "
                f"The cluster represents {pixels:,} pixels "
                f"({percentage:.2f}% of the dataset). "
                "Preventive monitoring is strongly recommended."
            )

        elif risk == "Moderate":

            text = (
                "Moderate wildfire activity. "
                f"This cluster ranks #{int(row['Activity_Rank'])}. "
                "Hotspot occurrence "
                f"({hotspot:.2f}) "
                "and fire radiative power "
                f"({frp:.2f}) "
                "indicate moderate wildfire activity. "
                f"The cluster contains {pixels:,} pixels "
                f"({percentage:.2f}% of observations). "
                "Routine monitoring should be maintained."
            )

        elif risk == "Low":

            text = (
                "Low wildfire activity. "
                f"This cluster ranks #{int(row['Activity_Rank'])}. "
                "Both hotspot frequency "
                f"({hotspot:.2f}) "
                "and fire radiative power "
                f"({frp:.2f}) "
                "are relatively low. "
                f"The cluster represents {pixels:,} pixels "
                f"({percentage:.2f}% of observations). "
                "Periodic observation is considered sufficient."
            )

        else:

            text = (
                "Very low wildfire activity. "
                f"This cluster ranks #{int(row['Activity_Rank'])}. "
                "It has the lowest hotspot activity "
                "among identified clusters. "
                f"The cluster contains {pixels:,} pixels "
                f"({percentage:.2f}% of observations). "
                "Only basic monitoring is recommended."
            )

        interpretations.append(text)

    valid_clusters["Interpretation"] = interpretations

    logger.info(
        "Scientific interpretation successfully generated."
    )

    # -------------------------------------------------------------------------
    # Noise Cluster Interpretation
    # -------------------------------------------------------------------------

    if not noise_cluster.empty:

        noise_cluster["Activity_Rank"] = np.nan

        noise_cluster["Risk_Level"] = "Noise"

        noise_cluster["Ranking_Basis"] = (
            "Excluded from ranking because HDBSCAN "
            "classified these observations as noise."
        )

        noise_cluster["Interpretation"] = (
            "Noise observations identified by HDBSCAN. "
            "These pixels do not satisfy the minimum density "
            "required to belong to any wildfire activity cluster."
        )

    logger.info(
        "Noise interpretation completed."
    )

    # -------------------------------------------------------------------------
    # Final Assembly
    # -------------------------------------------------------------------------
    logger.info(
        "Preparing final interpretation table..."
    )

    # -------------------------------------------------------------------------
    # Short Interpretation
    # -------------------------------------------------------------------------

    short_mapping = {
        "Very High": "Highest Wildfire Priority",
        "High": "High Wildfire Activity",
        "Moderate": "Moderate Wildfire Activity",
        "Low": "Low Wildfire Activity",
        "Very Low": "Very Low Wildfire Activity",
        "Noise": "Noise Observation",
    }

    valid_clusters["Interpretation_Short"] = (
        valid_clusters["Risk_Level"]
        .map(short_mapping)
    )

    if not noise_cluster.empty:

        noise_cluster["Interpretation_Short"] = (
            "Noise Observation"
        )

    # -------------------------------------------------------------------------
    # Merge Valid Clusters and Noise
    # -------------------------------------------------------------------------

    interpretation = pd.concat(
        [
            valid_clusters,
            noise_cluster,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Column Arrangement
    # -------------------------------------------------------------------------

    interpretation = interpretation[
        [
            "Cluster",
            "Activity_Rank",
            "Risk_Level",
            "Pixel_Count",
            "Percentage",
            "Mean_Hotspot_Frequency",
            "Median_Hotspot_Frequency",
            "Std_Hotspot_Frequency",
            "Min_Hotspot_Frequency",
            "Max_Hotspot_Frequency",
            "Mean_FRP_Max",
            "Median_FRP_Max",
            "Std_FRP_Max",
            "Min_FRP_Max",
            "Max_FRP_Max",
            "Ranking_Basis",
            "Interpretation_Short",
            "Interpretation",
        ]
    ]

    # -------------------------------------------------------------------------
    # Final Logging
    # -------------------------------------------------------------------------

    logger.info("-" * 80)

    logger.info(
        "Final Cluster Interpretation Summary"
    )

    for _, row in interpretation.iterrows():

        if row["Cluster"] == NOISE_LABEL:

            logger.info(
                "Noise | Pixels=%d",
                row["Pixel_Count"],
            )

        else:

            logger.info(
                (
                    "Rank %d | "
                    "Cluster %d | "
                    "%s | "
                    "Pixels=%d | "
                    "Mean Hotspot=%.2f | "
                    "Mean FRP=%.2f"
                ),
                int(row["Activity_Rank"]),
                int(row["Cluster"]),
                row["Risk_Level"],
                int(row["Pixel_Count"]),
                row["Mean_Hotspot_Frequency"],
                row["Mean_FRP_Max"],
            )

    logger.info("-" * 80)

    logger.info(
        "Scientific cluster interpretation completed successfully."
    )

    logger.info(
        "Interpreted clusters : %d",
        len(valid_clusters),
    )

    logger.info(
        "Noise clusters : %d",
        len(noise_cluster),
    )

    logger.info("=" * 80)

    return interpretation

# =============================================================================
# SECTION 10 — VISUALIZATION
# =============================================================================

def create_visualizations(
    clustered_df: pd.DataFrame,
    interpretation: pd.DataFrame,
) -> dict[str, Path]:
    """
    Create all visualization outputs for the HDBSCAN clustering pipeline.

    Three figures are generated:

    1. Cluster map.
    2. Mean hotspot frequency profile.
    3. Mean FRP profile.

    Parameters
    ----------
    clustered_df : pandas.DataFrame
        Dataset containing cluster labels.

    interpretation : pandas.DataFrame
        Cluster interpretation table.

    Returns
    -------
    dict[str, pathlib.Path]
        Dictionary containing generated figure paths.
    """

    logger.info("=" * 80)
    logger.info("SECTION 10 — VISUALIZATION")
    logger.info("=" * 80)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure_paths = {}

    # -------------------------------------------------------------------------
    # Figure 1 : Cluster Map
    # -------------------------------------------------------------------------

    logger.info("Creating cluster map...")

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        clustered_df[LONGITUDE_COLUMN],
        clustered_df[LATITUDE_COLUMN],
        c=clustered_df["Cluster"],
        s=8,
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("HDBSCAN Cluster Map")
    plt.colorbar(scatter, label="Cluster")

    plt.tight_layout()

    plt.savefig(
        CLUSTER_MAP_FIGURE,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close()

    figure_paths["cluster_map"] = CLUSTER_MAP_FIGURE

    # -------------------------------------------------------------------------
    # Figure 2 : Mean Hotspot Frequency
    # -------------------------------------------------------------------------

    logger.info("Creating hotspot activity profile...")

    profile = interpretation[
        interpretation["Cluster"] != NOISE_LABEL
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        profile["Cluster"].astype(str),
        profile["Mean_Hotspot_Frequency"],
    )

    plt.xlabel("Cluster")
    plt.ylabel("Mean Hotspot Frequency")
    plt.title("Mean Hotspot Frequency by Cluster")

    plt.tight_layout()

    plt.savefig(
        HOTSPOT_PROFILE_FIGURE,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close()

    figure_paths["hotspot_profile"] = HOTSPOT_PROFILE_FIGURE

    # -------------------------------------------------------------------------
    # Figure 3 : Mean FRP
    # -------------------------------------------------------------------------

    logger.info("Creating FRP profile...")

    plt.figure(figsize=(8, 5))

    plt.bar(
        profile["Cluster"].astype(str),
        profile["Mean_FRP_Max"],
    )

    plt.xlabel("Cluster")
    plt.ylabel("Mean FRP")
    plt.title("Mean FRP by Cluster")

    plt.tight_layout()

    plt.savefig(
        FRP_PROFILE_FIGURE,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close()

    figure_paths["frp_profile"] = FRP_PROFILE_FIGURE

    logger.info("Visualization completed successfully.")

    return figure_paths

# =============================================================================
# SECTION 11 — EXPORT RESULTS
# =============================================================================

def export_results(
    best_model: BestModelResult,
    grid_search_summary: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
    clustered_df: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    interpretation: pd.DataFrame,
    figure_paths: dict[str, Path],
) -> None:
    """
    Export all outputs generated by the HDBSCAN clustering pipeline.

    Parameters
    ----------
    best_model : BestModelResult
        Final trained HDBSCAN model.

    grid_search_summary : pandas.DataFrame
        Grid Search summary table.

    evaluation_summary : pandas.DataFrame
        Final model evaluation summary.

    clustered_df : pandas.DataFrame
        Original dataset with cluster labels.

    cluster_profile : pandas.DataFrame
        Cluster profiling table.

    interpretation : pandas.DataFrame
        Cluster interpretation table.

    figure_paths : dict[str, pathlib.Path]
        Generated visualization files.

    Returns
    -------
    None
    """

    logger.info("=" * 80)
    logger.info("SECTION 11 — EXPORT RESULTS")
    logger.info("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting CSV files...")

    grid_search_summary.to_csv(
        GRID_SEARCH_RESULT_CSV,
        index=False,
    )

    evaluation_summary.to_csv(
        EVALUATION_SUMMARY_CSV,
        index=False,
    )

    clustered_df.to_csv(
        CLUSTERED_DATASET_CSV,
        index=False,
    )

    cluster_profile.to_csv(
        CLUSTER_PROFILE_CSV,
        index=False,
    )

    interpretation.to_csv(
        CLUSTER_INTERPRETATION_CSV,
        index=False,
    )

    logger.info("CSV export completed.")

    logger.info("Saving final HDBSCAN model...")

    joblib.dump(
        best_model.model,
        FINAL_MODEL_FILE,
    )

    logger.info("Model saved successfully.")

    logger.info("Generated outputs:")

    exported_files = {
        "Grid Search Summary": GRID_SEARCH_RESULT_CSV,
        "Evaluation Summary": EVALUATION_SUMMARY_CSV,
        "Clustered Dataset": CLUSTERED_DATASET_CSV,
        "Cluster Profile": CLUSTER_PROFILE_CSV,
        "Cluster Interpretation": CLUSTER_INTERPRETATION_CSV,
        "Final Model": FINAL_MODEL_FILE,
    }

    exported_files.update(
        {
            "Cluster Map": figure_paths["cluster_map"],
            "Hotspot Activity Profile": figure_paths["hotspot_profile"],
            "FRP Profile": figure_paths["frp_profile"],
        }
    )

    for name, path in exported_files.items():
        logger.info("%-28s : %s", name, path)

    logger.info("=" * 80)
    logger.info("All pipeline outputs exported successfully.")
    logger.info("=" * 80)

# =============================================================================
# SECTION 12 — CLEANUP
# =============================================================================

def finalize_pipeline(
    exported_files: dict[str, Path],
) -> None:
    """
    Finalize the HDBSCAN clustering pipeline.

    This function performs final validation of exported outputs,
    closes remaining visualization resources, and logs a summary
    of the completed pipeline.

    Parameters
    ----------
    exported_files : dict[str, pathlib.Path]
        Dictionary containing exported pipeline outputs.

    Returns
    -------
    None
    """

    logger.info("=" * 80)
    logger.info("SECTION 12 — CLEANUP")
    logger.info("=" * 80)

    logger.info("Validating exported files...")

    missing_files = []

    for name, path in exported_files.items():

        if path.exists():

            logger.info(
                "[OK] %-28s : %s",
                name,
                path,
            )

        else:

            logger.error(
                "[MISSING] %-22s : %s",
                name,
                path,
            )

            missing_files.append(path)

    plt.close("all")

    logger.info("Closed all matplotlib figures.")

    if missing_files:

        raise FileNotFoundError(
            "One or more exported files were not found."
        )

    logger.info("-" * 80)

    logger.info("Pipeline completed successfully.")

    logger.info(
        "Total exported files : %d",
        len(exported_files),
    )

    logger.info(
        "Output directory : %s",
        OUTPUT_DIR,
    )

    logger.info("=" * 80)
    logger.info("HDBSCAN clustering pipeline finished.")
    logger.info("=" * 80)

def main():

    df = load_dataset()

    exploratory_data_analysis(df)

    X_scaled, scaler = prepare_features(df)

    grid_results = perform_grid_search(X_scaled)

    best_result, grid_summary = select_best_model(grid_results)

    best_model = train_final_model(
        X_scaled,
        best_result,
    )

    evaluation_summary = evaluate_final_model(best_model)

    clustered_df, cluster_profile = profile_clusters(
        df,
        best_model,
    )

    interpretation = interpret_clusters(cluster_profile)

    figure_paths = create_visualizations(
        clustered_df,
        interpretation,
    )

    export_results(
        best_model,
        grid_summary,
        evaluation_summary,
        clustered_df,
        cluster_profile,
        interpretation,
        figure_paths,
    )

    exported_files = {
        "Grid Search Summary": GRID_SEARCH_RESULT_CSV,
        "Evaluation Summary": EVALUATION_SUMMARY_CSV,
        "Clustered Dataset": CLUSTERED_DATASET_CSV,
        "Cluster Profile": CLUSTER_PROFILE_CSV,
        "Cluster Interpretation": CLUSTER_INTERPRETATION_CSV,
        "Final Model": FINAL_MODEL_FILE,
        "Cluster Map": figure_paths["cluster_map"],
        "Hotspot Activity Profile": figure_paths["hotspot_profile"],
        "FRP Profile": figure_paths["frp_profile"],
    }

    finalize_pipeline(exported_files)


if __name__ == "__main__":
    main()