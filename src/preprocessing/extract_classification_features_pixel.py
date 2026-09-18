# =============================================================================
# SECTION 1 — CONFIGURATION & INITIALIZATION
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path
import yaml

# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file not found:\n{CONFIG_PATH}"
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# =============================================================================
# INPUT FILES
# =============================================================================

INPUT_DIR = PROJECT_ROOT / config["paths"]["processed_data"]

SENTINEL_PATH = INPUT_DIR / "sentinel_processed.tif"

BURNED_PATH = INPUT_DIR / "burned_area_aligned.tif"

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = PROJECT_ROOT / config["paths"]["feature_data"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "classification_dataset.csv"

OUTPUT_PARQUET = OUTPUT_DIR / "classification_dataset.parquet"

OUTPUT_SUMMARY = OUTPUT_DIR / "classification_summary.json"

# =============================================================================
# REQUIRED SENTINEL BANDS
# =============================================================================

REQUIRED_BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
]

# =============================================================================
# LABEL DEFINITION
# =============================================================================

VALID_LABELS = (0, 1)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("Pixel-based Classification Feature Extraction")
logger.info("=" * 70)

logger.info(f"Project Root : {PROJECT_ROOT}")
logger.info(f"Sentinel     : {SENTINEL_PATH}")
logger.info(f"Burned Area  : {BURNED_PATH}")
logger.info(f"Output       : {OUTPUT_DIR}")

# =============================================================================
# SECTION 2A — ALIGN BURNED AREA TO SENTINEL GRID
# =============================================================================

from rasterio.warp import reproject, Resampling

ALIGNED_BURNED_PATH = OUTPUT_DIR.parent / "processed" / "burned_area_aligned.tif"


def align_burned_area() -> Path:
    """
    Reproject burned area raster so it matches
    Sentinel raster exactly.
    """

    logger.info("=" * 70)
    logger.info("SECTION 2A : ALIGN BURNED AREA")
    logger.info("=" * 70)

    if ALIGNED_BURNED_PATH.exists():

        logger.info("Aligned burned raster already exists.")

        return ALIGNED_BURNED_PATH

    with rasterio.open(SENTINEL_PATH) as sentinel:

        with rasterio.open(BURNED_PATH) as burned:

            profile = sentinel.profile.copy()

            profile.update(

                dtype=burned.dtypes[0],

                count=1,

                nodata=burned.nodata,

                compress="lzw",

            )

            with rasterio.open(
                ALIGNED_BURNED_PATH,
                "w",
                **profile,
            ) as dst:

                reproject(

                    source=rasterio.band(burned, 1),

                    destination=rasterio.band(dst, 1),

                    src_transform=burned.transform,

                    src_crs=burned.crs,

                    dst_transform=sentinel.transform,

                    dst_crs=sentinel.crs,

                    resampling=Resampling.nearest,

                )

    logger.info("✓ Burned raster aligned successfully")

    return ALIGNED_BURNED_PATH

# =============================================================================
# SECTION 2 — DATASET AUDIT
# =============================================================================

import rasterio
import numpy as np


def audit_input_datasets() -> None:
    """
    Validate input datasets before feature extraction.

    This function checks:
        - File existence
        - Raster readability
        - Sentinel band count
        - CRS consistency
        - Raster dimensions
        - Spatial resolution
        - Affine transform
        - Burned label values
    """

    logger.info("=" * 70)
    logger.info("SECTION 2 : DATASET AUDIT")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # File existence
    # -------------------------------------------------------------------------

    if not SENTINEL_PATH.exists():
        raise FileNotFoundError(f"Sentinel raster not found:\n{SENTINEL_PATH}")

    if not BURNED_PATH.exists():
        raise FileNotFoundError(f"Burned raster not found:\n{BURNED_PATH}")

    logger.info("✓ Input files found")

    # -------------------------------------------------------------------------
    # Open datasets
    # -------------------------------------------------------------------------

    with rasterio.open(SENTINEL_PATH) as sentinel, rasterio.open(BURNED_PATH) as burned:

        # ---------------------------------------------------------------------
        # Sentinel bands
        # ---------------------------------------------------------------------

        if sentinel.count != len(REQUIRED_BANDS):
            raise ValueError(
                f"Expected {len(REQUIRED_BANDS)} Sentinel bands "
                f"but found {sentinel.count}"
            )

        logger.info(f"✓ Sentinel bands : {sentinel.count}")

        # ---------------------------------------------------------------------
        # CRS
        # ---------------------------------------------------------------------

        if sentinel.crs != burned.crs:
            raise ValueError(
                f"CRS mismatch:\n"
                f"Sentinel : {sentinel.crs}\n"
                f"Burned   : {burned.crs}"
            )

        logger.info(f"✓ CRS : {sentinel.crs}")

        # ---------------------------------------------------------------------
        # Width & Height
        # ---------------------------------------------------------------------

        if sentinel.width != burned.width:
            raise ValueError("Raster width mismatch")

        if sentinel.height != burned.height:
            raise ValueError("Raster height mismatch")

        logger.info(
            f"✓ Raster size : {sentinel.width} × {sentinel.height}"
        )

        # ---------------------------------------------------------------------
        # Resolution
        # ---------------------------------------------------------------------

        if sentinel.res != burned.res:
            raise ValueError(
                f"Resolution mismatch:\n"
                f"{sentinel.res}\n"
                f"{burned.res}"
            )

        logger.info(f"✓ Resolution : {sentinel.res}")

        # ---------------------------------------------------------------------
        # Affine Transform
        # ---------------------------------------------------------------------

        if sentinel.transform != burned.transform:
            raise ValueError("Affine transform mismatch")

        logger.info("✓ Affine transform verified")

        # ---------------------------------------------------------------------
        # Burned Labels
        # ---------------------------------------------------------------------

        labels = np.unique(burned.read(1))

        labels = labels[~np.isnan(labels)]

        invalid = np.setdiff1d(labels, VALID_LABELS)

        if len(invalid) > 0:
            raise ValueError(
                f"Invalid burned labels detected: {invalid}"
            )

        logger.info(f"✓ Burn labels : {labels.tolist()}")

    logger.info("Dataset audit completed successfully.\n")



# =============================================================================
# SECTION 3 — OPEN RASTER & INITIALIZE RESOURCES
# =============================================================================

from dataclasses import dataclass
import rasterio


@dataclass
class RasterContext:
    """
    Container for raster datasets and metadata.
    """

    sentinel: rasterio.io.DatasetReader
    burned: rasterio.io.DatasetReader

    transform: rasterio.Affine
    crs: rasterio.crs.CRS

    width: int
    height: int

    resolution: tuple

    nodata_sentinel: float | int | None
    nodata_burned: float | int | None

    profile: dict


def initialize_rasters() -> RasterContext:
    """
    Open raster datasets and initialize shared metadata.

    Returns
    -------
    RasterContext
        Container shared across subsequent sections.
    """

    logger.info("=" * 70)
    logger.info("SECTION 3 : OPEN RASTER")
    logger.info("=" * 70)

    sentinel = rasterio.open(SENTINEL_PATH)

    burned = rasterio.open(BURNED_PATH)

    context = RasterContext(
        sentinel=sentinel,
        burned=burned,
        transform=sentinel.transform,
        crs=sentinel.crs,
        width=sentinel.width,
        height=sentinel.height,
        resolution=sentinel.res,
        nodata_sentinel=sentinel.nodata,
        nodata_burned=burned.nodata,
        profile=sentinel.profile,
    )

    logger.info("Sentinel raster opened successfully")
    logger.info("Burned raster opened successfully")

    logger.info(f"Raster Size : {context.width} x {context.height}")
    logger.info(f"Resolution  : {context.resolution}")
    logger.info(f"CRS         : {context.crs}")

    return context

# =============================================================================
# SECTION 3A — INITIALIZE STREAMING DATASET WRITER
# =============================================================================

from dataclasses import dataclass, field
from pathlib import Path

import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# DATASET STATISTICS
# =============================================================================

@dataclass
class DatasetStatistics:
    """
    Running statistics collected during pixel extraction.
    """

    total_pixels: int = 0
    valid_pixels: int = 0

    burned_pixels: int = 0
    non_burned_pixels: int = 0

    feature_stats: dict = field(default_factory=dict)

    def initialize(self, columns: list[str]) -> None:

        for col in columns:

            if col in ("latitude", "longitude", "label"):
                continue

            self.feature_stats[col] = {

                "count": 0,

                "sum": 0.0,

                "sum_square": 0.0,

                "min": np.inf,

                "max": -np.inf,

            }

    def update(self, dataframe: pd.DataFrame):

        if dataframe.empty:
            return

        self.valid_pixels += len(dataframe)

        self.burned_pixels += int((dataframe["label"] == 1).sum())

        self.non_burned_pixels += int((dataframe["label"] == 0).sum())

        for column, stat in self.feature_stats.items():

            values = dataframe[column].to_numpy(dtype=np.float64)

            stat["count"] += len(values)

            stat["sum"] += values.sum()

            stat["sum_square"] += np.square(values).sum()

            stat["min"] = min(stat["min"], values.min())

            stat["max"] = max(stat["max"], values.max())


# =============================================================================
# STREAMING DATASET WRITER
# =============================================================================

class StreamingDatasetWriter:
    """
    Stream pixel blocks directly into a Parquet dataset.
    """

    def __init__(
        self,
        output_path: Path,
        statistics: DatasetStatistics,
    ):

        self.output_path = output_path

        self.statistics = statistics

        self.writer = None

    def write(self, dataframe: pd.DataFrame):

        if dataframe.empty:
            return

        table = pa.Table.from_pandas(
            dataframe,
            preserve_index=False,
        )

        if self.writer is None:

            self.writer = pq.ParquetWriter(
                self.output_path,
                table.schema,
                compression="snappy",
            )

        self.writer.write_table(table)

        self.statistics.update(dataframe)

    def close(self):

        if self.writer is not None:

            self.writer.close()

# =============================================================================
# SECTION 4 — PIXEL EXTRACTION ENGINE
# =============================================================================

import numpy as np
from typing import Generator


def extract_pixel_blocks(
    context: RasterContext,
) -> Generator[dict, None, None]:
    """
    Extract valid Sentinel pixels and burned labels block-by-block.

    Yields
    ------
    dict
        Dictionary containing row/column indices,
        Sentinel bands, and burned labels.
    """

    logger.info("=" * 70)
    logger.info("SECTION 4 : PIXEL EXTRACTION")
    logger.info("=" * 70)

    sentinel = context.sentinel
    burned = context.burned

    total_blocks = 0
    total_pixels = 0

    for _, window in sentinel.block_windows(1):

        # ---------------------------------------------------------------------
        # Read Sentinel bands
        # Shape:
        # (6, rows, cols)
        # ---------------------------------------------------------------------

        bands = sentinel.read(window=window)

        # ---------------------------------------------------------------------
        # Read Burned Label
        # ---------------------------------------------------------------------

        label = burned.read(1, window=window)

        # ---------------------------------------------------------------------
        # Pixel Coordinates (row / col)
        # ---------------------------------------------------------------------

        rows, cols = np.indices(label.shape)

        rows += window.row_off
        cols += window.col_off

        # ---------------------------------------------------------------------
        # Build Valid Mask
        # ---------------------------------------------------------------------

        valid_mask = np.ones(label.shape, dtype=bool)

        if context.nodata_burned is not None:
            valid_mask &= label != context.nodata_burned

        if context.nodata_sentinel is not None:
            valid_mask &= np.all(
                bands != context.nodata_sentinel,
                axis=0,
            )

        # ---------------------------------------------------------------------
        # Skip Empty Block
        # ---------------------------------------------------------------------

        if valid_mask.sum() == 0:
            continue

        total_blocks += 1
        total_pixels += valid_mask.sum()

        yield {

            "rows": rows[valid_mask],

            "cols": cols[valid_mask],

            "B2": bands[0][valid_mask],

            "B3": bands[1][valid_mask],

            "B4": bands[2][valid_mask],

            "B8": bands[3][valid_mask],

            "B11": bands[4][valid_mask],

            "B12": bands[5][valid_mask],

            "label": label[valid_mask],

        }

    logger.info(f"Processed Blocks : {total_blocks:,}")
    logger.info(f"Valid Pixels     : {total_pixels:,}")

# =============================================================================
# SECTION 5 — PIXEL FEATURE ENGINEERING
# =============================================================================

import rasterio
import numpy as np


def engineer_pixel_features(
    pixel_block: dict,
    context: RasterContext,
) -> dict:
    """
    Generate pixel-level features from extracted Sentinel bands.

    Parameters
    ----------
    pixel_block : dict
        Output from extract_pixel_blocks().

    context : RasterContext
        Shared raster metadata.

    Returns
    -------
    dict
        Pixel features ready for dataset construction.
    """

    rows = pixel_block["rows"]
    cols = pixel_block["cols"]

    # -------------------------------------------------------------------------
    # Pixel Coordinates
    # -------------------------------------------------------------------------

    longitude, latitude = rasterio.transform.xy(
        context.transform,
        rows,
        cols,
        offset="center",
    )

    longitude = np.asarray(longitude, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Spectral Bands
    # -------------------------------------------------------------------------

    b2 = pixel_block["B2"].astype(np.float32)
    b3 = pixel_block["B3"].astype(np.float32)
    b4 = pixel_block["B4"].astype(np.float32)
    b8 = pixel_block["B8"].astype(np.float32)
    b11 = pixel_block["B11"].astype(np.float32)
    b12 = pixel_block["B12"].astype(np.float32)

    # -------------------------------------------------------------------------
    # NBR Calculation
    # -------------------------------------------------------------------------

    denominator = b8 + b12

    nbr = np.full_like(b8, np.nan, dtype=np.float32)

    valid = denominator != 0

    nbr[valid] = (
        (b8[valid] - b12[valid]) /
        denominator[valid]
    )

    nbr = np.nan_to_num(
        nbr,
        nan=np.nan,
        posinf=np.nan,
        neginf=np.nan,
    )

    # -------------------------------------------------------------------------
    # Return Features
    # -------------------------------------------------------------------------

    return {

        "latitude": latitude,

        "longitude": longitude,

        "B2": b2,

        "B3": b3,

        "B4": b4,

        "B8": b8,

        "B11": b11,

        "B12": b12,

        "NBR": nbr,

        "label": pixel_block["label"].astype(np.uint8),

    }

# =============================================================================
# SECTION 6 — PIXEL VALIDATION & CLEANING
# =============================================================================

import numpy as np


def validate_pixel_features(features: dict) -> dict:
    """
    Validate and clean engineered pixel features.

    Parameters
    ----------
    features : dict
        Output from engineer_pixel_features().

    Returns
    -------
    dict
        Clean pixel features.
    """

    logger.info("Cleaning pixel block...")

    # -------------------------------------------------------------------------
    # Initial Mask
    # -------------------------------------------------------------------------

    mask = np.ones(len(features["label"]), dtype=bool)

    # -------------------------------------------------------------------------
    # Label Validation
    # -------------------------------------------------------------------------

    mask &= np.isin(features["label"], VALID_LABELS)

    # -------------------------------------------------------------------------
    # Coordinate Validation
    # -------------------------------------------------------------------------

    mask &= (
        (features["latitude"] >= -90) &
        (features["latitude"] <= 90)
    )

    mask &= (
        (features["longitude"] >= -180) &
        (features["longitude"] <= 180)
    )

    # -------------------------------------------------------------------------
    # Spectral Validation
    # -------------------------------------------------------------------------

    spectral_features = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B11",
        "B12",
    ]

    for band in spectral_features:

        values = features[band]

        mask &= np.isfinite(values)

        mask &= (values >= 0)

        mask &= (values <= 10000)

    # -------------------------------------------------------------------------
    # NBR Validation
    # -------------------------------------------------------------------------

    mask &= np.isfinite(features["NBR"])

    # -------------------------------------------------------------------------
    # Apply Mask
    # -------------------------------------------------------------------------

    cleaned = {}

    for key, values in features.items():

        cleaned[key] = values[mask]

    logger.info(
        f"Valid Pixels : {len(cleaned['label']):,}"
    )

    return cleaned

# =============================================================================
# SECTION 7 — STREAMING DATASET BUILDER
# =============================================================================

from dataclasses import dataclass, field
import pandas as pd


class ClassificationDatasetBuilder:
    """
    Build classification dataset incrementally.
    """

    def __init__(self):

        self.blocks: list[pd.DataFrame] = []

        self.stats = DatasetStatistics()

    def append(self, features: dict):

        if len(features["label"]) == 0:
            return

        df = pd.DataFrame({

            "latitude": features["latitude"],

            "longitude": features["longitude"],

            "B2": features["B2"],

            "B3": features["B3"],

            "B4": features["B4"],

            "B8": features["B8"],

            "B11": features["B11"],

            "B12": features["B12"],

            "NBR": features["NBR"],

            "label": features["label"],

        })

        self.blocks.append(df)

        # ---------------------------------------------------------------------
        # Update Statistics
        # ---------------------------------------------------------------------

        self.stats.valid_pixels += len(df)

        self.stats.burned_pixels += (df["label"] == 1).sum()

        self.stats.non_burned_pixels += (df["label"] == 0).sum()

    def finalize(self) -> pd.DataFrame:

        logger.info("=" * 70)
        logger.info("Building Final Dataset")
        logger.info("=" * 70)

        dataset = pd.concat(
            self.blocks,
            ignore_index=True,
        )

        logger.info(f"Dataset Size : {len(dataset):,}")

        return dataset
    
# =============================================================================
# SECTION 8 — STREAMING DATASET STATISTICS
# =============================================================================

from dataclasses import dataclass, field
import json
import numpy as np


@dataclass
class FeatureStatistics:

    count: int = 0

    minimum: float = np.inf

    maximum: float = -np.inf

    total: float = 0.0

    total_square: float = 0.0

def update_statistics(stat: FeatureStatistics, values: np.ndarray):

    if len(values) == 0:
        return

    stat.count += len(values)

    stat.minimum = min(stat.minimum, values.min())

    stat.maximum = max(stat.maximum, values.max())

    stat.total += values.sum()

    stat.total_square += np.square(values).sum()

def summarize_statistics(stat: FeatureStatistics):

    mean = stat.total / stat.count

    variance = (
        stat.total_square / stat.count
    ) - (mean ** 2)

    variance = max(variance, 0)

    std = np.sqrt(variance)

    return {

        "count": stat.count,

        "min": float(stat.minimum),

        "max": float(stat.maximum),

        "mean": float(mean),

        "std": float(std),

    }

def export_summary(summary: dict):

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

# =============================================================================
# SECTION 9 — PIPELINE FINALIZATION
# =============================================================================

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassificationOutput:
    """
    Final output metadata produced by the extraction pipeline.
    """

    csv_path: Path
    parquet_path: Path
    summary_path: Path

    valid_pixels: int
    burned_pixels: int
    non_burned_pixels: int


def finalize_pipeline(
    writer,
    statistics,
) -> ClassificationOutput:
    """
    Finalize classification feature extraction pipeline.
    """

    logger.info("=" * 70)
    logger.info("SECTION 9 : PIPELINE FINALIZATION")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Close Writer
    # -------------------------------------------------------------------------

    writer.close()

    logger.info("✓ Dataset writer closed.")

    # -------------------------------------------------------------------------
    # Verify Outputs
    # -------------------------------------------------------------------------

    required_outputs = [
        OUTPUT_CSV,
        OUTPUT_PARQUET,
        OUTPUT_SUMMARY,
    ]

    for file in required_outputs:

        if not file.exists():

            raise FileNotFoundError(
                f"Expected output not found:\n{file}"
            )

        logger.info(f"✓ {file.name}")

    # -------------------------------------------------------------------------
    # File Sizes
    # -------------------------------------------------------------------------

    logger.info("-" * 70)

    for file in required_outputs:

        size_mb = file.stat().st_size / (1024 ** 2)

        logger.info(
            f"{file.name:<35} {size_mb:10.2f} MB"
        )

    logger.info("-" * 70)

    logger.info(
        f"Valid Pixels     : {statistics.valid_pixels:,}"
    )

    logger.info(
        f"Burned Pixels    : {statistics.burned_pixels:,}"
    )

    logger.info(
        f"Non Burned Pixel : {statistics.non_burned_pixels:,}"
    )

    logger.info("=" * 70)

    return ClassificationOutput(

        csv_path=OUTPUT_CSV,

        parquet_path=OUTPUT_PARQUET,

        summary_path=OUTPUT_SUMMARY,

        valid_pixels=statistics.valid_pixels,

        burned_pixels=statistics.burned_pixels,

        non_burned_pixels=statistics.non_burned_pixels,

    )

# =============================================================================
# SECTION 10 — RESOURCE CLEANUP & PIPELINE SHUTDOWN
# =============================================================================

import gc
import time


def cleanup_pipeline(
    context: RasterContext,
    writer: StreamingDatasetWriter,
    start_time: float,
) -> None:
    """
    Release all resources used during the classification
    feature extraction pipeline.
    """

    logger.info("=" * 70)
    logger.info("SECTION 10 : RESOURCE CLEANUP")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Close Raster Files
    # -------------------------------------------------------------------------

    try:

        context.sentinel.close()

        logger.info("✓ Sentinel raster closed.")

    except Exception as e:

        logger.warning(f"Failed closing Sentinel raster: {e}")

    try:

        context.burned.close()

        logger.info("✓ Burned raster closed.")

    except Exception as e:

        logger.warning(f"Failed closing Burned raster: {e}")

    # -------------------------------------------------------------------------
    # Close Dataset Writer
    # -------------------------------------------------------------------------

    try:

        writer.close()

        logger.info("✓ Dataset writer closed.")

    except Exception as e:

        logger.warning(f"Failed closing dataset writer: {e}")

    # -------------------------------------------------------------------------
    # Force Garbage Collection
    # -------------------------------------------------------------------------

    gc.collect()

    logger.info("✓ Memory released.")

    # -------------------------------------------------------------------------
    # Execution Time
    # -------------------------------------------------------------------------

    elapsed = time.time() - start_time

    hours = int(elapsed // 3600)

    minutes = int((elapsed % 3600) // 60)

    seconds = int(elapsed % 60)

    logger.info("-" * 70)

    logger.info(
        f"Execution Time : "
        f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    )

    logger.info("-" * 70)

    logger.info("Pipeline finished successfully.")

    logger.info("=" * 70)

# =============================================================================
# SECTION 11 — MAIN PIPELINE
# =============================================================================

import time
import pandas as pd


def process_pixel_block(
    pixel_block: dict,
    context: RasterContext,
    writer: StreamingDatasetWriter,
):
    """
    Process one extracted pixel block.
    """

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------

    features = engineer_pixel_features(
        pixel_block,
        context,
    )

    # -------------------------------------------------------------------------
    # Cleaning
    # -------------------------------------------------------------------------

    clean = validate_pixel_features(features)

    if len(clean["label"]) == 0:
        return

    # -------------------------------------------------------------------------
    # Convert to DataFrame
    # -------------------------------------------------------------------------

    dataframe = pd.DataFrame(clean)

    # -------------------------------------------------------------------------
    # Stream to Dataset
    # -------------------------------------------------------------------------

    writer.write(dataframe)


def main():
    """
    Pixel-based classification feature extraction pipeline.
    """

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("PIXEL-BASED CLASSIFICATION FEATURE EXTRACTION")
    logger.info("=" * 70)

    try:

        # ---------------------------------------------------------------------
        # Section 2
        # ---------------------------------------------------------------------

        audit_input_datasets()
        aligned_path = align_burned_area()

        # ---------------------------------------------------------------------
        # Section 3
        # ---------------------------------------------------------------------

        context = initialize_rasters()

        # ---------------------------------------------------------------------
        # Section 3A
        # ---------------------------------------------------------------------

        statistics = DatasetStatistics()

        statistics.initialize([
            "B2",
            "B3",
            "B4",
            "B8",
            "B11",
            "B12",
            "NBR",
        ])

        writer = StreamingDatasetWriter(
            OUTPUT_PARQUET,
            statistics,
        )

        # ---------------------------------------------------------------------
        # Section 4
        # ---------------------------------------------------------------------

        logger.info("Starting pixel extraction...")

        for pixel_block in extract_pixel_blocks(context):

            process_pixel_block(
                pixel_block,
                context,
                writer,
            )

        # ---------------------------------------------------------------------
        # Section 8
        # ---------------------------------------------------------------------

        export_summary(statistics)

        # ---------------------------------------------------------------------
        # Section 9
        # ---------------------------------------------------------------------

        finalize_pipeline(
            writer=writer,
            statistics=statistics,
        )

    finally:

        # ---------------------------------------------------------------------
        # Section 10
        # ---------------------------------------------------------------------

        cleanup_pipeline(
            context=context,
            writer=writer,
            start_time=start_time,
        )


if __name__ == "__main__":

    main()