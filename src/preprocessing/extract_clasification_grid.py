# =============================================================================
# SECTION 1 — CONFIGURATION & INITIALIZATION
# =============================================================================

from __future__ import annotations

import gc
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.features import geometry_mask
from rasterstats import zonal_stats

# =============================================================================
# PROJECT DIRECTORY
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_DIR / "processed"

FEATURE_DIR = DATA_DIR / "features"

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =============================================================================
# INPUT DATA
# =============================================================================

SENTINEL_PATH = PROCESSED_DIR / "sentinel_processed.tif"

BURNED_PATH = PROCESSED_DIR / "burned_area.tif"

GRID_PATH = DATA_DIR / "shapefile" / "grid_500m.gpkg"

# =============================================================================
# OUTPUT DATA
# =============================================================================

OUTPUT_FEATURE = FEATURE_DIR / "classification_dataset_500m.csv"

OUTPUT_SUMMARY = FEATURE_DIR / "classification_summary.json"

# =============================================================================
# GRID CONFIGURATION
# =============================================================================

GRID_SIZE = 500        # meter

MIN_VALID_PIXEL = 10

NODATA_VALUE = -9999

# =============================================================================
# FEATURE NAME
# =============================================================================

FEATURE_COLUMNS = [

    "Mean_B2",

    "Mean_B3",

    "Mean_B4",

    "Mean_B8",

    "Mean_B11",

    "Mean_B12",

    "Mean_NBR",

    "Std_NBR",

]

TARGET_COLUMN = "Burn_Label"

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

    handlers=[

        logging.StreamHandler(sys.stdout)

    ]

)

logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("GRID-BASED CLASSIFICATION FEATURE EXTRACTION")
logger.info("=" * 70)

logger.info(f"Project Root : {PROJECT_ROOT}")
logger.info(f"Sentinel     : {SENTINEL_PATH}")
logger.info(f"Burned Area  : {BURNED_PATH}")
logger.info(f"Grid         : {GRID_PATH}")
logger.info(f"Output       : {OUTPUT_FEATURE}")
logger.info("=" * 70)

# =============================================================================
# SECTION 2 — DATASET AUDIT
# =============================================================================

def audit_input_datasets() -> None:
    """
    Validate all datasets before feature extraction.
    """

    logger.info("=" * 70)
    logger.info("SECTION 2 : DATASET AUDIT")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Check file existence
    # -------------------------------------------------------------------------

    required_files = {

        "Sentinel": SENTINEL_PATH,

        "Burned Area": BURNED_PATH,

        "Grid": GRID_PATH,

    }

    for name, path in required_files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )

        logger.info(f"✓ {name} found")

    # -------------------------------------------------------------------------
    # Sentinel Audit
    # -------------------------------------------------------------------------

    with rasterio.open(SENTINEL_PATH) as src:

        sentinel_crs = src.crs

        logger.info("-" * 70)
        logger.info("Sentinel Raster")
        logger.info("-" * 70)

        logger.info(f"Bands       : {src.count}")
        logger.info(f"Size        : {src.width} x {src.height}")
        logger.info(f"Resolution  : {src.res}")
        logger.info(f"CRS         : {src.crs}")
        logger.info(f"NoData      : {src.nodata}")
        logger.info(f"Bounds      : {src.bounds}")

    # -------------------------------------------------------------------------
    # Burned Area Audit
    # -------------------------------------------------------------------------

    with rasterio.open(BURNED_PATH) as src:

        burned_crs = src.crs

        logger.info("-" * 70)
        logger.info("Burned Area Raster")
        logger.info("-" * 70)

        logger.info(f"Bands       : {src.count}")
        logger.info(f"Size        : {src.width} x {src.height}")
        logger.info(f"Resolution  : {src.res}")
        logger.info(f"CRS         : {src.crs}")
        logger.info(f"NoData      : {src.nodata}")
        logger.info(f"Bounds      : {src.bounds}")

    # -------------------------------------------------------------------------
    # Grid Audit
    # -------------------------------------------------------------------------

    grid = gpd.read_file(GRID_PATH)

    logger.info("-" * 70)
    logger.info("Grid")
    logger.info("-" * 70)

    logger.info(f"Total Grid  : {len(grid):,}")
    logger.info(f"CRS         : {grid.crs}")
    logger.info(f"Bounds      : {grid.total_bounds}")

    invalid = (~grid.is_valid).sum()

    empty = grid.geometry.is_empty.sum()

    logger.info(f"Invalid     : {invalid}")
    logger.info(f"Empty       : {empty}")

    if invalid > 0:

        raise ValueError(
            f"Found {invalid} invalid geometries."
        )

    # -------------------------------------------------------------------------
    # CRS Validation
    # -------------------------------------------------------------------------

    if sentinel_crs != burned_crs:

        raise ValueError(
            "Sentinel CRS and Burned CRS are different."
        )

    if sentinel_crs != grid.crs:

        raise ValueError(
            "Grid CRS and Sentinel CRS are different."
        )

    logger.info("=" * 70)
    logger.info("✓ Dataset audit completed successfully.")
    logger.info("=" * 70)

# =============================================================================
# SECTION 3 — LOAD SPATIAL DATA
# =============================================================================

from shapely.geometry import box


@dataclass
class RasterContext:
    """
    Container for all spatial datasets used during
    feature extraction.
    """

    sentinel: rasterio.io.DatasetReader

    burned: rasterio.io.DatasetReader

    grid: gpd.GeoDataFrame


def initialize_spatial_data() -> RasterContext:
    """
    Load and validate all spatial datasets before
    feature extraction.
    """

    logger.info("=" * 70)
    logger.info("SECTION 3 : LOAD SPATIAL DATA")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Open Sentinel Raster
    # -------------------------------------------------------------------------

    sentinel = rasterio.open(SENTINEL_PATH)

    logger.info("✓ Sentinel raster loaded")

    logger.info(f"Bands      : {sentinel.count}")

    logger.info(
        f"Raster Size: {sentinel.width:,} x {sentinel.height:,}"
    )

    logger.info(f"Resolution : {sentinel.res}")

    logger.info(f"CRS        : {sentinel.crs}")

    logger.info(f"NoData     : {sentinel.nodata}")

    # -------------------------------------------------------------------------
    # Open Burned Area Raster
    # -------------------------------------------------------------------------

    burned = rasterio.open(BURNED_PATH)

    logger.info("✓ Burned Area raster loaded")

    logger.info(f"Bands      : {burned.count}")

    logger.info(
        f"Raster Size: {burned.width:,} x {burned.height:,}"
    )

    logger.info(f"Resolution : {burned.res}")

    logger.info(f"CRS        : {burned.crs}")

    logger.info(f"NoData     : {burned.nodata}")

    # -------------------------------------------------------------------------
    # Load Grid
    # -------------------------------------------------------------------------

    grid = gpd.read_file(GRID_PATH)

    if grid.empty:

        raise ValueError(
            "Grid dataset is empty."
        )

    logger.info("✓ Grid loaded")

    logger.info(f"Grid Count : {len(grid):,}")

    logger.info(f"Grid CRS   : {grid.crs}")

    # -------------------------------------------------------------------------
    # Validate Grid Coverage
    # -------------------------------------------------------------------------

    sentinel_extent = box(

        sentinel.bounds.left,

        sentinel.bounds.bottom,

        sentinel.bounds.right,

        sentinel.bounds.top,

    )

    inside = grid.intersects(
        sentinel_extent
    )

    inside_count = int(inside.sum())

    outside_count = int((~inside).sum())

    logger.info(
        f"Grid Inside Raster : {inside_count:,}"
    )

    logger.info(
        f"Grid Outside Raster: {outside_count:,}"
    )

    if outside_count > 0:

        logger.warning(
            f"{outside_count:,} grid berada di luar "
            "cakupan raster Sentinel."
        )

    # -------------------------------------------------------------------------
    # Validate Geometry
    # -------------------------------------------------------------------------

    invalid_geometry = int((~grid.is_valid).sum())

    if invalid_geometry > 0:

        raise ValueError(

            f"Found {invalid_geometry:,} invalid geometries."

        )

    logger.info("✓ Grid geometry validation passed")

    logger.info("=" * 70)
    logger.info("Spatial data initialized successfully.")
    logger.info("=" * 70)

    return RasterContext(

        sentinel=sentinel,

        burned=burned,

        grid=grid,

    )
    
# =============================================================================
# SECTION 4A — READ RASTER WINDOW
# =============================================================================

def read_raster_window(
    dataset: rasterio.io.DatasetReader,
    geometry,
) -> tuple[np.ma.MaskedArray, np.ndarray]:
    """
    Read all Sentinel bands inside a grid polygon.

    Parameters
    ----------
    dataset : rasterio DatasetReader

    geometry : shapely Polygon

    Returns
    -------
    data : np.ma.MaskedArray

        Shape:
        (6, rows, cols)

    mask : np.ndarray

        Boolean mask.
        True = inside polygon.
    """

    # ----------------------------------------------------------
    # Compute raster window
    # ----------------------------------------------------------

    window = from_bounds(

        *geometry.bounds,

        transform=dataset.transform,

    )

    window = window.round_offsets()

    window = window.round_lengths()

    # ----------------------------------------------------------
    # Read all Sentinel bands
    # ----------------------------------------------------------

    data = dataset.read(

        indexes=[1, 2, 3, 4, 5, 6],

        window=window,

        masked=True,

    )

    # ----------------------------------------------------------
    # Window transform
    # ----------------------------------------------------------

    transform = dataset.window_transform(
        window
    )

    # ----------------------------------------------------------
    # Polygon mask
    # ----------------------------------------------------------

    polygon_mask = geometry_mask(

        [geometry],

        transform=transform,

        invert=True,

        out_shape=data.shape[1:],

    )

    return data, polygon_mask

# =============================================================================
# SECTION 4B — CALCULATE GRID STATISTICS
# =============================================================================

def calculate_grid_statistics(
    data: np.ndarray,
    polygon_mask: np.ndarray,
) -> dict:
    """
    Calculate spectral statistics for one grid.

    Parameters
    ----------
    data : ndarray

        Shape:
        (6, rows, cols)

    polygon_mask : ndarray

        Boolean mask.

    Returns
    -------
    dict

        Spectral features.
    """

    # ----------------------------------------------------------
    # Extract pixels inside polygon
    # ----------------------------------------------------------

    b2 = data[0][polygon_mask]
    b3 = data[1][polygon_mask]
    b4 = data[2][polygon_mask]
    b8 = data[3][polygon_mask]
    b11 = data[4][polygon_mask]
    b12 = data[5][polygon_mask]

    # ----------------------------------------------------------
    # Minimum valid pixels
    # ----------------------------------------------------------

    valid_pixel = np.count_nonzero(
        ~np.isnan(b2)
    )

    if valid_pixel < MIN_VALID_PIXEL:

        return {

            "Mean_B2": np.nan,

            "Mean_B3": np.nan,

            "Mean_B4": np.nan,

            "Mean_B8": np.nan,

            "Mean_B11": np.nan,

            "Mean_B12": np.nan,

            "Mean_NBR": np.nan,

            "Std_NBR": np.nan,

        }

    # ----------------------------------------------------------
    # Mean Reflectance
    # ----------------------------------------------------------

    mean_b2 = np.nanmean(b2)

    mean_b3 = np.nanmean(b3)

    mean_b4 = np.nanmean(b4)

    mean_b8 = np.nanmean(b8)

    mean_b11 = np.nanmean(b11)

    mean_b12 = np.nanmean(b12)

    # ----------------------------------------------------------
    # Calculate NBR
    # ----------------------------------------------------------

    nbr = (b8 - b12) / (b8 + b12)

    nbr = np.where(

        np.isfinite(nbr),

        nbr,

        np.nan,

    )

    mean_nbr = np.nanmean(
        nbr
    )

    std_nbr = np.nanstd(
        nbr
    )

    # ----------------------------------------------------------
    # Return
    # ----------------------------------------------------------

    return {

        "Mean_B2": float(mean_b2),

        "Mean_B3": float(mean_b3),

        "Mean_B4": float(mean_b4),

        "Mean_B8": float(mean_b8),

        "Mean_B11": float(mean_b11),

        "Mean_B12": float(mean_b12),

        "Mean_NBR": float(mean_nbr),

        "Std_NBR": float(std_nbr),

    }

# =============================================================================
# SECTION 4C — EXTRACT SENTINEL FEATURES
# =============================================================================

def extract_sentinel_features(
    context: RasterContext,
) -> gpd.GeoDataFrame:
    """
    Extract Sentinel spectral features
    for every grid polygon.
    """

    logger.info("=" * 70)
    logger.info("SECTION 4 : EXTRACT SENTINEL FEATURES")
    logger.info("=" * 70)

    grid = context.grid.copy()

    # ----------------------------------------------------------
    # Grid ID
    # ----------------------------------------------------------

    grid["grid_id"] = np.arange(
        1,
        len(grid) + 1,
    )

    # ----------------------------------------------------------
    # Centroid
    # ----------------------------------------------------------

    centroid = grid.geometry.centroid

    grid["longitude"] = centroid.x

    grid["latitude"] = centroid.y

    # ----------------------------------------------------------
    # Output Columns
    # ----------------------------------------------------------

    for column in FEATURE_COLUMNS:

        grid[column] = np.nan

    total_grid = len(grid)

    logger.info(
        f"Total Grid : {total_grid:,}"
    )

    start = time.time()

    # ----------------------------------------------------------
    # Feature Extraction
    # ----------------------------------------------------------

    for i, geometry in enumerate(grid.geometry):

        try:

            data, polygon_mask = read_raster_window(

                context.sentinel,

                geometry,

            )

            stats = calculate_grid_statistics(

                data,

                polygon_mask,

            )

            for key, value in stats.items():

                grid.at[i, key] = value

        except Exception as e:

            logger.warning(

                f"Grid {i+1:,} failed : {e}"

            )

        # ------------------------------------------------------
        # Progress
        # ------------------------------------------------------

        if (i + 1) % 2000 == 0:

            elapsed = time.time() - start

            speed = (i + 1) / elapsed

            eta = (

                total_grid - (i + 1)

            ) / speed

            logger.info(

                f"{i+1:,}/{total_grid:,} "

                f"({(i+1)/total_grid*100:.1f}%) | "

                f"{speed:.2f} grid/sec | "

                f"ETA {eta/60:.1f} min"

            )

    elapsed = time.time() - start

    logger.info("-" * 70)

    logger.info(

        f"Finished : "

        f"{elapsed/60:.2f} minutes"

    )

    logger.info(

        f"Average Speed : "

        f"{total_grid/elapsed:.2f} grid/sec"

    )

    logger.info("=" * 70)

    return grid


# =============================================================================
# SECTION 5 — GENERATE BURN LABEL
# =============================================================================

def generate_burn_label(
    feature_grid: gpd.GeoDataFrame,
    context: RasterContext,
) -> gpd.GeoDataFrame:
    """
    Generate binary burn label for every grid.
    """

    logger.info("=" * 70)
    logger.info("SECTION 5 : GENERATE BURN LABEL")
    logger.info("=" * 70)

    burn_stats = zonal_stats(

        vectors=feature_grid.geometry,

        raster=BURNED_PATH,

        band=1,

        stats=["max"],

        nodata=context.burned.nodata,

        geojson_out=False,

    )

    feature_grid["Burn_Label"] = [

        1 if item["max"] == 1 else 0

        for item in burn_stats

    ]

    burned = int(feature_grid["Burn_Label"].sum())

    non_burned = len(feature_grid) - burned

    logger.info(f"Burned Grid     : {burned:,}")

    logger.info(f"Non Burned Grid : {non_burned:,}")

    logger.info("✓ Burn label generated")

    logger.info("=" * 70)

    return feature_grid

# =============================================================================
# SECTION 6 — DATASET QUALITY CONTROL
# =============================================================================

def validate_dataset(
    feature_grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Validate classification dataset before export.
    """

    logger.info("=" * 70)
    logger.info("SECTION 6 : DATASET QUALITY CONTROL")
    logger.info("=" * 70)

    initial_grid = len(feature_grid)

    logger.info(f"Initial Grid : {initial_grid:,}")

    # -------------------------------------------------------------------------
    # Replace Infinite Value
    # -------------------------------------------------------------------------

    feature_grid = feature_grid.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # -------------------------------------------------------------------------
    # Remove Missing Feature
    # -------------------------------------------------------------------------

    required_columns = [

        "Mean_B2",
        "Mean_B3",
        "Mean_B4",
        "Mean_B8",
        "Mean_B11",
        "Mean_B12",
        "Mean_NBR",
        "Std_NBR",
        "Burn_Label",

    ]

    feature_grid = feature_grid.dropna(
        subset=required_columns
    )

    # -------------------------------------------------------------------------
    # Burn Label Validation
    # -------------------------------------------------------------------------

    invalid_label = ~feature_grid["Burn_Label"].isin(
        [0, 1]
    )

    if invalid_label.any():

        raise ValueError(
            "Invalid Burn_Label detected."
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    final_grid = len(feature_grid)

    removed = initial_grid - final_grid

    logger.info(f"Remaining Grid : {final_grid:,}")

    logger.info(f"Removed Grid   : {removed:,}")

    logger.info(
        f"Burned Grid    : "
        f"{feature_grid['Burn_Label'].sum():,}"
    )

    logger.info(
        f"Non Burned     : "
        f"{(feature_grid['Burn_Label']==0).sum():,}"
    )

    logger.info("✓ Dataset validation passed")

    logger.info("=" * 70)

    return feature_grid

# =============================================================================
# SECTION 7 — EXPORT DATASET
# =============================================================================

OUTPUT_CSV = FEATURE_DIR / "classification_dataset.csv"

OUTPUT_GPKG = FEATURE_DIR / "classification_dataset.gpkg"


def export_dataset(
    feature_grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Export classification dataset.
    """

    logger.info("=" * 70)
    logger.info("SECTION 7 : EXPORT DATASET")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Column Order
    # -------------------------------------------------------------------------

    ordered_columns = [

        "grid_id",

        "latitude",

        "longitude",

        "Mean_B2",

        "Mean_B3",

        "Mean_B4",

        "Mean_B8",

        "Mean_B11",

        "Mean_B12",

        "Mean_NBR",

        "Std_NBR",

        "Burn_Label",

        "geometry",

    ]

    feature_grid = feature_grid[ordered_columns]

    # -------------------------------------------------------------------------
    # Export GeoPackage
    # -------------------------------------------------------------------------

    feature_grid.to_file(

        OUTPUT_GPKG,

        driver="GPKG",

    )

    logger.info("✓ GeoPackage exported")

    # -------------------------------------------------------------------------
    # Export CSV
    # -------------------------------------------------------------------------

    dataframe = feature_grid.drop(
        columns="geometry"
    )

    dataframe.to_csv(

        OUTPUT_CSV,

        index=False,

        float_format="%.6f",

    )

    logger.info("✓ CSV exported")

    logger.info(f"Rows    : {len(dataframe):,}")

    logger.info(f"Columns : {len(dataframe.columns)}")

    logger.info("=" * 70)

    return dataframe

# =============================================================================
# SECTION 8 — DATASET SUMMARY
# =============================================================================

import json
from datetime import datetime

OUTPUT_SUMMARY = FEATURE_DIR / "classification_summary.json"


def export_summary(
    feature_grid: gpd.GeoDataFrame,
) -> None:
    """
    Export dataset summary.
    """

    logger.info("=" * 70)
    logger.info("SECTION 8 : DATASET SUMMARY")
    logger.info("=" * 70)

    feature_columns = [

        "Mean_B2",

        "Mean_B3",

        "Mean_B4",

        "Mean_B8",

        "Mean_B11",

        "Mean_B12",

        "Mean_NBR",

        "Std_NBR",

        "Burn_Label",

    ]

    summary = {

        "metadata": {

            "pipeline":

                "Grid-based Burned Area Classification",

            "grid_size":

                "500 meter",

            "created_at":

                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "author":

                "Zahra Aura Hisani",

            "version":

                "1.0",

        },

        "dataset": {

            "total_grid":

                int(len(feature_grid)),

            "burned_grid":

                int(feature_grid["Burn_Label"].sum()),

            "non_burned_grid":

                int(
                    (feature_grid["Burn_Label"] == 0).sum()
                ),

            "burned_percentage":

                round(

                    feature_grid["Burn_Label"].mean() * 100,

                    2,

                ),

            "non_burned_percentage":

                round(

                    100 -

                    feature_grid["Burn_Label"].mean() * 100,

                    2,

                ),

        },

        "features": {}

    }

    for column in feature_columns:

        summary["features"][column] = {

            "min":

                float(feature_grid[column].min()),

            "max":

                float(feature_grid[column].max()),

            "mean":

                float(feature_grid[column].mean()),

            "std":

                float(feature_grid[column].std()),

        }

    with open(

        OUTPUT_SUMMARY,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            summary,

            file,

            indent=4,

        )

    logger.info("✓ Summary JSON exported")

    logger.info(f"Total Grid : {len(feature_grid):,}")

    logger.info(

        f"Burned Grid : "

        f"{feature_grid['Burn_Label'].sum():,}"

    )

    logger.info(

        f"Non Burned : "

        f"{(feature_grid['Burn_Label']==0).sum():,}"

    )

    logger.info("=" * 70)

# =============================================================================
# SECTION 9 — OUTPUT VALIDATION
# =============================================================================

import json


def validate_output() -> None:
    """
    Validate exported dataset.
    """

    logger.info("=" * 70)
    logger.info("SECTION 9 : OUTPUT VALIDATION")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # File existence
    # -------------------------------------------------------------------------

    outputs = [

        OUTPUT_CSV,

        OUTPUT_GPKG,

        OUTPUT_SUMMARY,

    ]

    for path in outputs:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing output:\n{path}"
            )

        logger.info(f"✓ {path.name}")

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    csv = pd.read_csv(
        OUTPUT_CSV
    )

    if csv.empty:

        raise ValueError(
            "CSV dataset is empty."
        )

    logger.info(
        f"CSV Rows    : {len(csv):,}"
    )

    logger.info(
        f"CSV Columns : {len(csv.columns)}"
    )

    # -------------------------------------------------------------------------
    # GeoPackage
    # -------------------------------------------------------------------------

    gpkg = gpd.read_file(
        OUTPUT_GPKG
    )

    logger.info(
        f"GPKG Grid   : {len(gpkg):,}"
    )

    if len(csv) != len(gpkg):

        raise ValueError(
            "CSV and GeoPackage row count mismatch."
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    with open(
        OUTPUT_SUMMARY,
        encoding="utf-8",
    ) as file:

        summary = json.load(file)

    logger.info(
        "Summary JSON loaded successfully."
    )

    # -------------------------------------------------------------------------
    # Dataset Validation
    # -------------------------------------------------------------------------

    expected_columns = [

        "grid_id",

        "latitude",

        "longitude",

        "Mean_B2",

        "Mean_B3",

        "Mean_B4",

        "Mean_B8",

        "Mean_B11",

        "Mean_B12",

        "Mean_NBR",

        "Std_NBR",

        "Burn_Label",

    ]

    missing = [

        col

        for col in expected_columns

        if col not in csv.columns

    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    logger.info(
        "✓ Column validation passed."
    )

    logger.info("=" * 70)
    logger.info("Output validation completed successfully.")
    logger.info("=" * 70)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    audit_input_datasets()

    context = initialize_spatial_data()

    feature_grid = extract_sentinel_features(
        context
    )

    feature_grid = generate_burn_label(
        feature_grid,
        context,
    )

    feature_grid = validate_dataset(
        feature_grid
    )

    export_dataset(
        feature_grid
    )

    export_summary(
        feature_grid
    )

    validate_output()

    context.sentinel.close()

    context.burned.close()

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


if __name__ == "__main__":

    main()