"""
===============================================================================
Project     : Forest and Land Fire Analysis using HDBSCAN and XGBoost
Script      : extract_viirs_pixels.py
Author      : Zahra Aura Hisani
Description :
    Extract pixel-based hotspot features from VIIRS raster into a CSV dataset
    for HDBSCAN clustering analysis.

Input
-----
- viirs_hotspot_stats.tif

Output
------
- viirs_clustering_features.csv

Each valid pixel represents one observation.

Output Columns
--------------
- pixel_id
- latitude
- longitude
- Hotspot_Frequency
- Confidence_Score
- FRP_Max

===============================================================================
"""

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "features"

INPUT_RASTER = RAW_DIR / "viirs_hotspot_stats.tif"

OUTPUT_CSV = FEATURE_DIR / "viirs_clustering_features.csv"

# =============================================================================
# RASTER BAND INDEX
# =============================================================================
# Raster band order (1-based indexing)
#
# Band 1 : Hotspot_Frequency
# Band 2 : Confidence_Score
# Band 3 : FRP_Max
# =============================================================================

BAND_HOTSPOT = 1
BAND_CONFIDENCE = 2
BAND_FRP = 3


# =============================================================================
# OUTPUT COLUMN NAMES
# =============================================================================

OUTPUT_COLUMNS = [
    "pixel_id",
    "latitude",
    "longitude",
    "Hotspot_Frequency",
    "Confidence_Score",
    "FRP_Max",
]


# =============================================================================
# NUMERICAL SETTINGS
# =============================================================================

FLOAT_PRECISION = 6

# Nilai minimum hotspot agar pixel dianggap memiliki observasi.
# Karena raster berasal dari hasil agregasi VIIRS, nilai 0 berarti
# tidak terdapat hotspot pada pixel tersebut.
MIN_HOTSPOT_VALUE = 0.0


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================================
# STARTUP
# =============================================================================

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

logger.info("=" * 70)
logger.info("VIIRS Pixel Feature Extraction")
logger.info("=" * 70)

logger.info("Input Raster : %s", INPUT_RASTER)
logger.info("Output CSV   : %s", OUTPUT_CSV)
logger.info("=" * 70)

# =============================================================================
# SECTION 2 — LOAD AND VALIDATE VIIRS RASTER
# =============================================================================

logger.info("Loading VIIRS raster...")

try:
    src = rasterio.open(INPUT_RASTER)

except Exception as e:
    logger.exception("Unable to open raster.")
    raise RuntimeError(f"Failed to open raster:\n{INPUT_RASTER}") from e


# =============================================================================
# VALIDATE BAND COUNT
# =============================================================================

EXPECTED_BAND_COUNT = 3

if src.count != EXPECTED_BAND_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_BAND_COUNT} raster bands "
        f"but found {src.count}."
    )


# =============================================================================
# READ RASTER BANDS
# =============================================================================

hotspot = src.read(BAND_HOTSPOT)
confidence = src.read(BAND_CONFIDENCE)
frp = src.read(BAND_FRP)


# =============================================================================
# STORE RASTER METADATA
# =============================================================================

transform = src.transform
crs = src.crs
width = src.width
height = src.height
bounds = src.bounds
nodata = src.nodata
dtype = hotspot.dtype


# =============================================================================
# BASIC VALIDATION
# =============================================================================

if crs is None:
    raise ValueError("Raster CRS is missing.")

if transform is None:
    raise ValueError("Raster transform is missing.")

if width <= 0 or height <= 0:
    raise ValueError("Invalid raster dimension.")


# =============================================================================
# CHECK BAND SHAPES
# =============================================================================

if hotspot.shape != confidence.shape:
    raise ValueError("Hotspot and Confidence dimensions do not match.")

if hotspot.shape != frp.shape:
    raise ValueError("Hotspot and FRP dimensions do not match.")


# =============================================================================
# LOG RASTER INFORMATION
# =============================================================================

logger.info("Raster successfully loaded.")
logger.info("CRS              : %s", crs)
logger.info("Raster Size      : %d x %d pixels", width, height)
logger.info("Number of Bands  : %d", src.count)
logger.info("Pixel Size       : %.3f x %.3f", transform.a, abs(transform.e))
logger.info("Data Type        : %s", dtype)
logger.info("NoData Value     : %s", nodata)
logger.info(
    "Bounds           : (%.6f, %.6f) - (%.6f, %.6f)",
    bounds.left,
    bounds.bottom,
    bounds.right,
    bounds.top,
)

logger.info("Band validation passed.")

# =============================================================================
# SECTION 3 — PIXEL FEATURE EXTRACTION
# =============================================================================

logger.info("Extracting valid hotspot pixels...")


# =============================================================================
# CREATE PIXEL INDEX
# =============================================================================

rows, cols = np.indices((height, width))


# =============================================================================
# FLATTEN ALL RASTER BANDS
# =============================================================================

rows = rows.ravel()
cols = cols.ravel()

hotspot_values = hotspot.ravel()
confidence_values = confidence.ravel()
frp_values = frp.ravel()


# =============================================================================
# CREATE VALID PIXEL MASK
# =============================================================================

valid_mask = hotspot_values > MIN_HOTSPOT_VALUE

if nodata is not None:
    valid_mask &= hotspot_values != nodata


# =============================================================================
# FILTER VALID PIXELS
# =============================================================================

rows = rows[valid_mask]
cols = cols[valid_mask]

hotspot_values = hotspot_values[valid_mask]
confidence_values = confidence_values[valid_mask]
frp_values = frp_values[valid_mask]


# =============================================================================
# CONVERT PIXEL TO GEOGRAPHIC COORDINATES
# =============================================================================

longitude, latitude = xy(
    transform,
    rows,
    cols,
    offset="center"
)

longitude = np.asarray(longitude, dtype=np.float64)
latitude = np.asarray(latitude, dtype=np.float64)


# =============================================================================
# GENERATE PIXEL ID
# =============================================================================

pixel_id = np.arange(
    1,
    hotspot_values.size + 1,
    dtype=np.int64
)


# =============================================================================
# BUILD FEATURE RECORDS
# =============================================================================

records = np.column_stack(
    (
        pixel_id,
        np.round(latitude, FLOAT_PRECISION),
        np.round(longitude, FLOAT_PRECISION),
        hotspot_values.astype(np.float32),
        confidence_values.astype(np.float32),
        frp_values.astype(np.float32),
    )
)


logger.info("Pixel extraction completed.")
logger.info("Valid hotspot pixels : %d", records.shape[0])

# =============================================================================
# SECTION 4 — DATA VALIDATION
# =============================================================================

logger.info("Validating extracted hotspot features...")


# =============================================================================
# VALIDATE PIXEL COUNT
# =============================================================================

total_pixels = height * width
valid_pixels = hotspot_values.size

if valid_pixels == 0:
    raise RuntimeError(
        "No valid hotspot pixels were extracted."
    )


# =============================================================================
# CHECK MISSING VALUES
# =============================================================================

if np.isnan(hotspot_values).any():
    raise ValueError("Hotspot_Frequency contains NaN values.")

if np.isnan(confidence_values).any():
    raise ValueError("Confidence_Score contains NaN values.")

if np.isnan(frp_values).any():
    raise ValueError("FRP_Max contains NaN values.")


# =============================================================================
# VALIDATE CONFIDENCE SCORE
# =============================================================================

valid_confidence = np.isin(
    confidence_values,
    [0.0, 0.5, 1.0]
)

if np.any(confidence_values < 0) or np.any(confidence_values > 1):
    raise ValueError(
        "Confidence_Score must be within the range [0, 1]."
    )


# =============================================================================
# VALIDATE HOTSPOT VALUES
# =============================================================================

if np.any(hotspot_values <= 0):
    raise ValueError(
        "Hotspot_Frequency must be greater than zero."
    )


# =============================================================================
# VALIDATE FRP VALUES
# =============================================================================

if np.any(frp_values < 0):
    raise ValueError(
        "FRP_Max contains negative values."
    )


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

logger.info("=" * 70)
logger.info("Extraction Summary")
logger.info("=" * 70)

logger.info("Total Pixels        : %d", total_pixels)
logger.info("Valid Pixels        : %d", valid_pixels)
logger.info(
    "Retained Pixels     : %.2f%%",
    (valid_pixels / total_pixels) * 100
)

logger.info("-" * 70)

logger.info(
    "Hotspot Frequency : min=%.2f | max=%.2f | mean=%.2f",
    hotspot_values.min(),
    hotspot_values.max(),
    hotspot_values.mean(),
)

logger.info(
    "Confidence Score  : min=%.2f | max=%.2f | mean=%.2f",
    confidence_values.min(),
    confidence_values.max(),
    confidence_values.mean(),
)

logger.info(
    "FRP Max           : min=%.2f | max=%.2f | mean=%.2f",
    frp_values.min(),
    frp_values.max(),
    frp_values.mean(),
)

logger.info("-" * 70)

unique_confidence, confidence_count = np.unique(
    confidence_values,
    return_counts=True
)

logger.info("Confidence Score Distribution")

for value, count in zip(unique_confidence, confidence_count):
    logger.info(
        "  %.1f : %d pixels",
        value,
        count,
    )

logger.info("=" * 70)

logger.info("Validation passed.")

# =============================================================================
# SECTION 5 — EXPORT CLUSTERING FEATURES
# =============================================================================

logger.info("Creating output dataset...")


# =============================================================================
# CREATE OUTPUT DATAFRAME
# =============================================================================

df = pd.DataFrame(
    {
        "pixel_id": pixel_id,
        "latitude": np.round(latitude, FLOAT_PRECISION),
        "longitude": np.round(longitude, FLOAT_PRECISION),
        "Hotspot_Frequency": hotspot_values.astype(np.float32),
        "Confidence_Score": confidence_values.astype(np.float32),
        "FRP_Max": frp_values.astype(np.float32),
    }
)


# =============================================================================
# ENSURE COLUMN ORDER
# =============================================================================

df = df[OUTPUT_COLUMNS]


# =============================================================================
# SORT OUTPUT
# =============================================================================

df = df.sort_values(
    by="pixel_id",
    ascending=True,
    ignore_index=True,
)


# =============================================================================
# EXPORT CSV
# =============================================================================
df["latitude"] = df["latitude"].round(6)
df["longitude"] = df["longitude"].round(6)

df["Hotspot_Frequency"] = (
    df["Hotspot_Frequency"]
    .astype(np.int32)
)

df["Confidence_Score"] = (
    df["Confidence_Score"]
    .round(2)
)

df["FRP_Max"] = (
    df["FRP_Max"]
    .round(2)
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# =============================================================================
# FINAL REPORT
# =============================================================================

logger.info("=" * 70)
logger.info("Export completed successfully.")
logger.info("=" * 70)

logger.info("Output File : %s", OUTPUT_CSV)
logger.info("Total Rows  : %d", len(df))
logger.info("Total Cols  : %d", len(df.columns))

logger.info("Columns:")

for column in df.columns:
    logger.info("  - %s", column)

logger.info("=" * 70)

src.close()

logger.info("Raster closed.")
logger.info("VIIRS feature extraction finished successfully.")
logger.info("=" * 70)

# =============================================================================
# SECTION 6 — CLEANUP AND PROGRAM COMPLETION
# =============================================================================

logger.info("Finalizing extraction process...")


# =============================================================================
# RELEASE RASTER RESOURCE
# =============================================================================

if src is not None:
    src.close()
    logger.info("Raster dataset closed.")


# =============================================================================
# FINAL EXECUTION SUMMARY
# =============================================================================

logger.info("=" * 70)
logger.info("VIIRS PIXEL FEATURE EXTRACTION COMPLETED")
logger.info("=" * 70)

logger.info("Input Raster")
logger.info("  %s", INPUT_RASTER)

logger.info("Output CSV")
logger.info("  %s", OUTPUT_CSV)

logger.info("Records Exported")
logger.info("  %d", len(df))

logger.info("Features")
for column in OUTPUT_COLUMNS:
    logger.info("  - %s", column)

logger.info("=" * 70)
logger.info("Pipeline finished successfully.")
logger.info("=" * 70)

