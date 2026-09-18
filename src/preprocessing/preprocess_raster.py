"""
=========================================================
Raster Preprocessing Pipeline
=========================================================

Project
-------
Analisis Zona Rawan dan Area Terbakar Kebakaran Hutan
dan Lahan Menggunakan HDBSCAN dan XGBoost Berbasis
Data Hotspot FIRMS dan Citra Sentinel-2
Kabupaten Ketapang (2025–2026)

Description
-----------
Tahap preprocessing seluruh raster sebelum dilakukan
feature extraction berbasis grid.

Catatan
--------
Sentinel Mosaic merupakan hasil Median Composite
yang dibuat di Google Earth Engine (GEE).
Cloud masking tidak dilakukan lagi pada tahap
preprocessing lokal karena efek awan telah
diminimalkan melalui proses median composite.

Pipeline
--------
1. Validate raster
2. Raster alignment
3. NBR calculation
4. Save processed raster
5. Metadata report

Output
------
data/processed/
    sentinel_processed.tif

outputs/metadata/
    preprocessing_report.txt
=========================================================
"""

# =========================================================
# SECTION 1
# Configuration & Project Constants
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging

import numpy as np
import rasterio
from rasterio.io import DatasetReader

# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================================================
# DATA DIRECTORY
# =========================================================

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

METADATA_DIR = OUTPUT_DIR / "metadata"

# =========================================================
# INPUT RASTER
# =========================================================

SENTINEL_INPUT = (
    PROCESSED_DIR /
    "sentinel_mosaic.tif"
)

VIIRS_INPUT = (
    RAW_DATA_DIR /
    "viirs_hotspot_stats.tif"
)

BURNED_INPUT = (
    RAW_DATA_DIR /
    "burned_area.tif"
)

# =========================================================
# OUTPUT
# =========================================================

SENTINEL_OUTPUT = (
    PROCESSED_DIR /
    "sentinel_processed.tif"
)

PREPROCESS_REPORT = (
    METADATA_DIR /
    "preprocessing_report.txt"
)

# =========================================================
# LOGGER
# =========================================================

LOGGER_NAME = "preprocess_raster"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

logger = logging.getLogger(
    LOGGER_NAME
)

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =========================================================
# GENERAL SETTINGS
# =========================================================

TARGET_CRS = "EPSG:4326"

NODATA_VALUE = -9999.0

OUTPUT_DTYPE = np.float32

# =========================================================
# EXPECTED BAND COUNT
# =========================================================

SENTINEL_BAND_COUNT = 6

VIIRS_BAND_COUNT = 3

BURNED_BAND_COUNT = 1

# =========================================================
# SENTINEL-2
# =========================================================

SENTINEL_BANDS = (
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
)

NBR_NAME = "NBR"

OUTPUT_BAND_NAMES = (
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NBR",
)

# =========================================================
# OUTPUT SETTINGS
# =========================================================

COMPRESS = "LZW"

BIGTIFF = "YES"


# =========================================================
# SECTION 2
# Raster Metadata & Input Validation
# =========================================================

from dataclasses import dataclass


# =========================================================
# Raster Metadata Container
# =========================================================

@dataclass(slots=True)
class RasterMetadata:
    """
    Immutable container for raster metadata.
    """

    name: str

    width: int

    height: int

    count: int

    crs: str

    transform: Any

    resolution: tuple[float, float]

    dtype: str

    nodata: float | None

    descriptions: tuple[str | None, ...]


# =========================================================
# Read Raster Metadata
# =========================================================

def read_metadata(
    dataset: DatasetReader,
) -> RasterMetadata:
    """
    Read raster metadata without loading
    raster values into memory.
    """

    return RasterMetadata(

        name=Path(dataset.name).name,

        width=dataset.width,

        height=dataset.height,

        count=dataset.count,

        crs=str(dataset.crs),

        transform=dataset.transform,

        resolution=dataset.res,

        dtype=dataset.dtypes[0],

        nodata=dataset.nodata,

        descriptions=dataset.descriptions,

    )


# =========================================================
# Open Raster
# =========================================================

def open_raster(
    path: Path,
) -> DatasetReader:
    """
    Safely open a raster file.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Raster not found:\n{path.resolve()}"
        )

    logger.info(
        "Opening raster: %s",
        path.name,
    )

    return rasterio.open(path)


# =========================================================
# Validate Input Raster
# =========================================================

def validate_input(
    dataset: DatasetReader,
    expected_band: int | None = None,
    expected_descriptions: tuple[str, ...] | None = None,
) -> None:
    """
    Validate raster metadata before preprocessing.

    Parameters
    ----------
    dataset : DatasetReader
        Open raster dataset.

    expected_band : int, optional
        Expected number of bands.

    expected_descriptions : tuple[str], optional
        Expected band descriptions.
    """

    metadata = read_metadata(
        dataset,
    )

    logger.info(

        "Raster : %s | Bands : %d | CRS : %s",

        metadata.name,

        metadata.count,

        metadata.crs,

    )

    # -----------------------------------------------------
    # CRS
    # -----------------------------------------------------

    if dataset.crs is None:

        raise ValueError(
            f"{metadata.name} has no CRS."
        )

    # -----------------------------------------------------
    # Band Count
    # -----------------------------------------------------

    if (
        expected_band is not None
        and
        metadata.count != expected_band
    ):

        raise ValueError(

            f"{metadata.name}\n"

            f"Expected Band : {expected_band}\n"

            f"Found         : {metadata.count}"

        )

    # -----------------------------------------------------
    # Band Description
    # -----------------------------------------------------

    if expected_descriptions is not None:

        descriptions = tuple(

            "" if desc is None else desc

            for desc in metadata.descriptions

        )

        if descriptions != expected_descriptions:

            raise ValueError(

                f"{metadata.name}\n"

                f"Band description mismatch.\n\n"

                f"Expected:\n"

                f"{expected_descriptions}\n\n"

                f"Found:\n"

                f"{descriptions}"

            )

    logger.info(
        "Validation passed."
    )

# =========================================================
# SECTION 3
# Raster Alignment Validation
# =========================================================

BOUND_TOLERANCE = 1e-9

RESOLUTION_TOLERANCE = 1e-9


# =========================================================
# CRS Validation
# =========================================================

def validate_crs(
    *datasets: DatasetReader,
) -> None:
    """
    Validate that all rasters use
    the same Coordinate Reference System.
    """

    crs_list = [
        str(ds.crs)
        for ds in datasets
    ]

    if len(set(crs_list)) != 1:

        raise ValueError(
            "CRS mismatch detected:\n"
            f"{crs_list}"
        )

    logger.info(
        "CRS validation passed."
    )


# =========================================================
# Bounds Validation
# =========================================================

def validate_bounds(
    *datasets: DatasetReader,
) -> None:
    """
    Validate raster extent.

    Small floating-point differences
    are ignored.
    """

    reference = datasets[0].bounds

    for ds in datasets[1:]:

        bounds = ds.bounds

        same = (

            np.isclose(
                reference.left,
                bounds.left,
                atol=BOUND_TOLERANCE,
            )

            and

            np.isclose(
                reference.right,
                bounds.right,
                atol=BOUND_TOLERANCE,
            )

            and

            np.isclose(
                reference.top,
                bounds.top,
                atol=BOUND_TOLERANCE,
            )

            and

            np.isclose(
                reference.bottom,
                bounds.bottom,
                atol=BOUND_TOLERANCE,
            )

        )

        if not same:

            logger.warning(
                "%s has different extent.",
                Path(ds.name).name,
            )

    logger.info(
        "Extent validation finished."
    )


# =========================================================
# Resolution Validation
# =========================================================

def validate_resolution(
    *datasets: DatasetReader,
) -> None:
    """
    Report raster resolution and compare
    against the first raster.
    """

    logger.info("-" * 60)

    logger.info(
        "Raster Resolution"
    )

    reference = datasets[0].res

    for ds in datasets:

        logger.info(

            "%-30s : %.8f x %.8f",

            Path(ds.name).name,

            ds.res[0],

            ds.res[1],

        )

        same = (

            np.isclose(
                reference[0],
                ds.res[0],
                atol=RESOLUTION_TOLERANCE,
            )

            and

            np.isclose(
                reference[1],
                ds.res[1],
                atol=RESOLUTION_TOLERANCE,
            )

        )

        if not same:

            logger.warning(
                "%s has different resolution.",
                Path(ds.name).name,
            )

    logger.info("-" * 60)


# =========================================================
# NoData Validation
# =========================================================

def validate_nodata(
    *datasets: DatasetReader,
) -> None:
    """
    Report NoData value
    for every raster.
    """

    logger.info(
        "NoData Values"
    )

    for ds in datasets:

        logger.info(

            "%-30s : %s",

            Path(ds.name).name,

            ds.nodata,

        )


# =========================================================
# Alignment Summary
# =========================================================

def alignment_summary(
    *datasets: DatasetReader,
) -> None:
    """
    Print raster alignment summary.

    This function accepts any number
    of raster datasets.
    """

    if len(datasets) < 2:

        raise ValueError(
            "At least two rasters are required "
            "for alignment validation."
        )

    logger.info("=" * 70)

    logger.info(
        "ALIGNMENT SUMMARY"
    )

    logger.info("=" * 70)

    validate_crs(
        *datasets,
    )

    validate_bounds(
        *datasets,
    )

    validate_resolution(
        *datasets,
    )

    validate_nodata(
        *datasets,
    )

    logger.info(
        "Raster alignment validation completed."
    )


# BAGIAN 6
# =========================================================
# SECTION 6
# Pipeline
# =========================================================

def run() -> None:
    """
    Execute raster preprocessing pipeline.
    """

    logger.info("=" * 70)
    logger.info("START RASTER PREPROCESSING")
    logger.info("=" * 70)

    with (

        open_raster(SENTINEL_INPUT) as sentinel_ds,

        open_raster(VIIRS_INPUT) as viirs_ds,

        open_raster(BURNED_INPUT) as burned_ds,

    ):

        # -------------------------------------------------
        # Input Validation
        # -------------------------------------------------

        validate_input(

            sentinel_ds,

            expected_band=SENTINEL_BAND_COUNT,

            expected_descriptions=SENTINEL_BANDS,

        )

        validate_input(

            viirs_ds,

            expected_band=VIIRS_BAND_COUNT,

        )

        validate_input(

            burned_ds,

            expected_band=BURNED_BAND_COUNT,

        )

        # -------------------------------------------------
        # Raster Alignment
        # -------------------------------------------------

        alignment_summary(

            sentinel_ds,

            viirs_ds,

            burned_ds,

        )

       
    logger.info("=" * 70)

    logger.info(
        "PREPROCESSING FINISHED"
    )

    logger.info("=" * 70)


# =========================================================
# Main
# =========================================================

def main() -> None:
    """
    Program entry point.
    """

    try:

        run()

    except KeyboardInterrupt:

        logger.warning(
            "Process interrupted by user."
        )

    except Exception:

        logger.exception(
            "Unexpected error occurred."
        )

        raise


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":

    main()