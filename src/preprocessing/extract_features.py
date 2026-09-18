"""
extract_features.py

Feature extraction pipeline for wildfire susceptibility analysis.

Research
--------
Analisis Zona Rawan dan Area Terbakar Kebakaran Hutan dan Lahan
Menggunakan HDBSCAN dan XGBoost Berbasis Data Hotspot FIRMS dan
Citra Sentinel-2 di Kabupaten Ketapang Tahun 2025–2026.

Author : Zahra Aura Hisani
Python : >=3.11
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Polygon
from shapely.prepared import prep


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

# Raw raster exported from Google Earth Engine (GEE)
RASTER_DIR = DATA_DIR / "processed"

FEATURE_DIR = DATA_DIR / "features"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

METADATA_DIR = OUTPUT_DIR / "metadata"


# =============================================================================
# INPUT RASTERS
# =============================================================================

SENTINEL_PATH = RASTER_DIR / "sentinel_mosaic.tif"

VIIRS_PATH = RAW_DIR / "viirs_hotspot_stats.tif"

BURNED_AREA_PATH = RAW_DIR / "burned_area.tif"


# =============================================================================
# OUTPUT FILES
# =============================================================================

FEATURE_GRID_GPKG = FEATURE_DIR / "feature_grid.gpkg"

FEATURE_GRID_CSV = FEATURE_DIR / "feature_grid.csv"

REPORT_PATH = METADATA_DIR / "feature_extraction_report.txt"


# =============================================================================
# GRID CONFIGURATION
# =============================================================================

GRID_SIZE: float = 500.0          # meter

GRID_ID_PREFIX: str = "GRID"


# =============================================================================
# SENTINEL-2 BAND CONFIGURATION
# =============================================================================

SENTINEL_BANDS = {
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B8": 4,
    "B11": 5,
    "B12": 6,
}

SPECTRAL_FEATURES = [
    "Mean_B2",
    "Mean_B3",
    "Mean_B4",
    "Mean_B8",
    "Mean_B11",
    "Mean_B12",
]

NBR_FEATURES = [
    "Mean_NBR",
    "Std_NBR",
]


# =============================================================================
# VIIRS BAND CONFIGURATION
# =============================================================================

VIIRS_BANDS = {
    "Hotspot_Frequency": 1,
    "Confidence_Mean": 2,
    "FRP_Max": 3,
}


# =============================================================================
# BURNED AREA CONFIGURATION
# =============================================================================

BURN_RATIO_BAND = 1


# =============================================================================
# OUTPUT FEATURE ORDER
# =============================================================================

FEATURE_COLUMNS = [
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
    "Hotspot_Frequency",
    "Confidence_Mean",
    "FRP_Max",
    "Burn_Ratio",
]


# =============================================================================
# NUMERIC SETTINGS
# =============================================================================

NODATA_FILL = np.nan

FLOAT_PRECISION = 4


# =============================================================================
# LOGGER CONFIGURATION
# =============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ExtractionConfig:
    """
    Global configuration used throughout the feature extraction pipeline.
    """

    grid_size: float = GRID_SIZE

    sentinel_path: Path = SENTINEL_PATH
    viirs_path: Path = VIIRS_PATH
    burned_area_path: Path = BURNED_AREA_PATH

    gpkg_output: Path = FEATURE_GRID_GPKG
    csv_output: Path = FEATURE_GRID_CSV
    report_output: Path = REPORT_PATH

    nodata_fill: float = NODATA_FILL
    float_precision: int = FLOAT_PRECISION


CONFIG = ExtractionConfig()


# =============================================================================
# CREATE OUTPUT DIRECTORIES
# =============================================================================

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

METADATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# END OF SECTION 1
# =============================================================================

# =============================================================================
# SECTION 2A
# LOAD RASTER & BASIC VALIDATION
# =============================================================================

from dataclasses import dataclass

import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window


# =============================================================================
# RASTER DATASET CONTAINER
# =============================================================================

@dataclass(slots=True)
class RasterDatasets:
    """
    Container holding opened raster datasets.

    All datasets remain open during the feature extraction
    process and must be closed after processing finishes.
    """

    sentinel: DatasetReader
    viirs: DatasetReader
    burned_area: DatasetReader


# =============================================================================
# FILE VALIDATION
# =============================================================================

def validate_file(path: Path) -> None:
    """
    Validate that an input raster exists.

    Parameters
    ----------
    path : Path
        Raster file path.

    Raises
    ------
    FileNotFoundError
        If the raster file does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Input raster not found:\n{path}"
        )

    logger.info(
        "Raster found: %s",
        path.name,
    )


# =============================================================================
# OPEN RASTER
# =============================================================================

def open_raster(
    path: Path,
) -> DatasetReader:
    """
    Open raster safely.

    Parameters
    ----------
    path : Path
        Raster path.

    Returns
    -------
    DatasetReader
        Open raster dataset.
    """

    validate_file(path)

    try:
        dataset = rasterio.open(path)

    except Exception as exc:
        raise RuntimeError(
            f"Unable to open raster:\n{path}"
        ) from exc

    logger.info(
        "Opened raster: %s",
        path.name,
    )

    return dataset


# =============================================================================
# READABILITY VALIDATION
# =============================================================================

def validate_readability(
    dataset: DatasetReader,
    name: str,
) -> None:
    """
    Verify that the raster can actually be read.

    Only a single pixel is read, so this operation
    is extremely lightweight.

    Parameters
    ----------
    dataset : DatasetReader

    name : str
        Dataset name.
    """

    try:

        dataset.read(
            1,
            window=Window(
                col_off=0,
                row_off=0,
                width=1,
                height=1,
            ),
        )

    except Exception as exc:

        raise RuntimeError(
            f"{name} cannot be read."
        ) from exc

    logger.info(
        "%s readability validation passed.",
        name,
    )


# =============================================================================
# BAND VALIDATION
# =============================================================================

def validate_band_count(
    dataset: DatasetReader,
    expected: int,
    name: str,
) -> None:
    """
    Validate raster band count.

    Parameters
    ----------
    dataset : DatasetReader

    expected : int

    name : str

    Raises
    ------
    ValueError
        If the raster contains an unexpected
        number of bands.
    """

    if dataset.count != expected:

        raise ValueError(
            f"{name} should contain "
            f"{expected} band(s), "
            f"but found {dataset.count}."
        )

    logger.info(
        "%s band validation passed (%d band%s).",
        name,
        dataset.count,
        "" if dataset.count == 1 else "s",
    )


# =============================================================================
# CRS VALIDATION
# =============================================================================

def validate_projection(
    reference: DatasetReader,
    target: DatasetReader,
    reference_name: str,
    target_name: str,
) -> None:
    """
    Validate Coordinate Reference System (CRS).

    Parameters
    ----------
    reference : DatasetReader

    target : DatasetReader

    reference_name : str

    target_name : str
    """

    if reference.crs is None:

        raise ValueError(
            f"{reference_name} has no CRS."
        )

    if target.crs is None:

        raise ValueError(
            f"{target_name} has no CRS."
        )

    if reference.crs != target.crs:

        raise ValueError(
            "CRS mismatch detected.\n\n"
            f"{reference_name}: {reference.crs}\n"
            f"{target_name}: {target.crs}"
        )

    logger.info(
        "%s CRS matches %s.",
        target_name,
        reference_name,
    )

# =============================================================================
# DATASET INFORMATION
# =============================================================================

def log_dataset_information(
    name: str,
    dataset: DatasetReader,
) -> None:
    """
    Log raster metadata.

    Parameters
    ----------
    name : str
        Dataset name.

    dataset : DatasetReader
        Open raster dataset.
    """

    logger.info("-" * 70)
    logger.info("Dataset    : %s", name)
    logger.info("Bands      : %d", dataset.count)
    logger.info("CRS        : %s", dataset.crs)
    logger.info("Resolution : %.8f x %.8f",
                dataset.res[0],
                dataset.res[1])
    logger.info("Size       : %d x %d",
                dataset.width,
                dataset.height)
    logger.info("Bounds     : %s",
                dataset.bounds)
    logger.info("-" * 70)


# =============================================================================
# LOAD DATASETS
# =============================================================================

def load_datasets() -> RasterDatasets:
    """
    Open all raster datasets and perform
    metadata validation.

    Returns
    -------
    RasterDatasets
        Container containing all opened datasets.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("INPUT RASTER VALIDATION")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Open datasets
    # -------------------------------------------------------------------------

    sentinel = open_raster(
        CONFIG.sentinel_path
    )

    viirs = open_raster(
        CONFIG.viirs_path
    )

    burned = open_raster(
        CONFIG.burned_area_path
    )

    # -------------------------------------------------------------------------
    # Readability validation
    # -------------------------------------------------------------------------

    validate_readability(
        sentinel,
        "Sentinel",
    )

    validate_readability(
        viirs,
        "VIIRS",
    )

    validate_readability(
        burned,
        "Burned Area",
    )

    # -------------------------------------------------------------------------
    # Band validation
    # -------------------------------------------------------------------------

    validate_band_count(
        sentinel,
        expected=6,
        name="Sentinel",
    )

    validate_band_count(
        viirs,
        expected=3,
        name="VIIRS",
    )

    validate_band_count(
        burned,
        expected=1,
        name="Burned Area",
    )

    # -------------------------------------------------------------------------
    # CRS validation
    # -------------------------------------------------------------------------

    validate_projection(
        sentinel,
        viirs,
        "Sentinel",
        "VIIRS",
    )

    validate_projection(
        sentinel,
        burned,
        "Sentinel",
        "Burned Area",
    )

    # -------------------------------------------------------------------------
    # Dataset summary
    # -------------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 70)
    logger.info("INPUT RASTER SUMMARY")
    logger.info("=" * 70)

    log_dataset_information(
        "Sentinel",
        sentinel,
    )

    log_dataset_information(
        "VIIRS",
        viirs,
    )

    log_dataset_information(
        "Burned Area",
        burned,
    )

    logger.info("")
    logger.info("Raster validation completed successfully.")
    logger.info("=" * 70)

    return RasterDatasets(
        sentinel=sentinel,
        viirs=viirs,
        burned_area=burned,
    )


# =============================================================================
# CLOSE DATASETS
# =============================================================================

def close_datasets(
    datasets: RasterDatasets,
) -> None:
    """
    Close all opened raster datasets.

    Parameters
    ----------
    datasets : RasterDatasets
        Container of opened raster datasets.
    """

    logger.info("")
    logger.info("Closing raster datasets...")

    for dataset in (
        datasets.sentinel,
        datasets.viirs,
        datasets.burned_area,
    ):
        dataset.close()

    logger.info("All raster datasets closed.")


# =============================================================================
# END OF SECTION 2
# =============================================================================

# =============================================================================
# SECTION 3A
# COORDINATE REFERENCE SYSTEM & TRANSFORMATION
# =============================================================================

from dataclasses import dataclass

from pyproj import CRS
from pyproj import Transformer

from shapely.geometry import box
from shapely.geometry import Polygon
from shapely.ops import transform


# =============================================================================
# PROJECTED EXTENT CONTAINER
# =============================================================================

@dataclass(slots=True)
class ProjectedExtent:
    """
    Container holding projected AOI information.

    Attributes
    ----------
    bounds : tuple
        Bounding coordinates in projected CRS.

    polygon : Polygon
        AOI polygon in projected CRS.

    crs : CRS
        Projected Coordinate Reference System.

    forward : Transformer
        Geographic → projected transformer.

    inverse : Transformer
        Projected → geographic transformer.
    """

    bounds: tuple[float, float, float, float]
    polygon: Polygon
    crs: CRS
    forward: Transformer
    inverse: Transformer


# =============================================================================
# DETERMINE PROJECTED CRS
# =============================================================================

def get_utm_crs(
    dataset: DatasetReader,
) -> CRS:
    """
    Determine the appropriate UTM CRS from the
    centroid of the raster.

    Parameters
    ----------
    dataset : DatasetReader

    Returns
    -------
    CRS
        UTM coordinate reference system.
    """

    bounds = dataset.bounds

    longitude = (bounds.left + bounds.right) / 2
    latitude = (bounds.bottom + bounds.top) / 2

    zone = int((longitude + 180) // 6) + 1

    if latitude >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    crs = CRS.from_epsg(epsg)

    logger.info(
        "Projected CRS : EPSG:%d",
        epsg,
    )

    return crs


# =============================================================================
# CREATE TRANSFORMERS
# =============================================================================

def create_transformers(
    source_crs,
    target_crs,
) -> tuple[Transformer, Transformer]:
    """
    Create forward and inverse coordinate transformers.

    Parameters
    ----------
    source_crs
        Source CRS.

    target_crs
        Target CRS.

    Returns
    -------
    tuple[Transformer, Transformer]
        Forward and inverse transformers.
    """

    forward = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )

    inverse = Transformer.from_crs(
        target_crs,
        source_crs,
        always_xy=True,
    )

    return forward, inverse


# =============================================================================
# PROJECT AOI
# =============================================================================

def get_projected_extent(
    dataset: DatasetReader,
) -> ProjectedExtent:
    """
    Project raster extent into UTM coordinates.

    Parameters
    ----------
    dataset : DatasetReader

    Returns
    -------
    ProjectedExtent
        Projected AOI information.
    """

    utm_crs = get_utm_crs(
        dataset,
    )

    forward, inverse = create_transformers(
        dataset.crs,
        utm_crs,
    )

    geographic_extent = box(
        *dataset.bounds,
    )

    projected_polygon = transform(
        forward.transform,
        geographic_extent,
    )

    logger.info(
        "AOI successfully projected."
    )

    logger.info(
        "Projected Bounds : %s",
        projected_polygon.bounds,
    )

    return ProjectedExtent(
        bounds=projected_polygon.bounds,
        polygon=projected_polygon,
        crs=utm_crs,
        forward=forward,
        inverse=inverse,
    )


# =============================================================================
# END OF SECTION 3A
# =============================================================================
# =============================================================================
# GRID DIMENSION
# =============================================================================

def compute_grid_dimensions(
    projection: ProjectedExtent,
) -> tuple[int, int]:
    """
    Compute grid rows and columns.

    Parameters
    ----------
    projection : ProjectedExtent

    Returns
    -------
    tuple[int, int]
        Number of rows and columns.
    """

    xmin, ymin, xmax, ymax = projection.bounds

    width = xmax - xmin
    height = ymax - ymin

    cols = int(
        np.ceil(
            width / CONFIG.grid_size
        )
    )

    rows = int(
        np.ceil(
            height / CONFIG.grid_size
        )
    )

    logger.info("Grid Size      : %.0f meter",
                CONFIG.grid_size)

    logger.info("Grid Rows      : %d", rows)

    logger.info("Grid Columns   : %d", cols)

    logger.info("Estimated Grid : %d",
                rows * cols)

    return rows, cols


# =============================================================================
# GRID GENERATION
# =============================================================================

def generate_grid(
    datasets: RasterDatasets,
) -> list[Polygon]:
    """
    Generate grid covering the study area.

    Parameters
    ----------
    datasets : RasterDatasets

    Returns
    -------
    list[Polygon]
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("GENERATING GRID")
    logger.info("=" * 70)

    projection = get_projected_extent(
        datasets.sentinel
    )

    rows, cols = compute_grid_dimensions(
        projection
    )

    xmin, ymin, xmax, ymax = projection.bounds

    grid_size = CONFIG.grid_size

    prepared_extent = prep(
        projection.polygon
    )

    polygons: list[Polygon] = []

    generated = 0

    for row in range(rows):

        y_top = ymax - row * grid_size

        y_bottom = y_top - grid_size

        for col in range(cols):

            x_left = xmin + col * grid_size

            x_right = x_left + grid_size

            cell = box(
                x_left,
                y_bottom,
                x_right,
                y_top,
            )

            if not prepared_extent.intersects(cell):
                continue

            clipped = cell.intersection(
                projection.polygon
            )

            geographic = transform(
                projection.inverse.transform,
                clipped,
            )

            polygons.append(
                geographic
            )

            generated += 1

            if generated % 500 == 0:

                logger.info(
                    "Generated %d grids",
                    generated,
                )

    logger.info("")

    logger.info(
        "Grid generation completed."
    )

    logger.info(
        "Total Grid : %d",
        generated,
    )

    logger.info("=" * 70)

    return polygons


# =============================================================================
# END OF SECTION 3B
# =============================================================================

# =============================================================================
# BUILD GRID
# =============================================================================

def build_grid(
    datasets: RasterDatasets,
) -> gpd.GeoDataFrame:
    """
    Build grid GeoDataFrame.

    Parameters
    ----------
    datasets : RasterDatasets

    Returns
    -------
    GeoDataFrame
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("BUILDING GRID")
    logger.info("=" * 70)

    polygons = generate_grid(
        datasets
    )

    grid = gpd.GeoDataFrame(
        geometry=polygons,
        crs=datasets.sentinel.crs,
    )

    grid["grid_id"] = [
        f"{GRID_ID_PREFIX}_{i:05d}"
        for i in range(
            1,
            len(grid) + 1,
        )
    ]

    # ---------------------------------------------------------
    # Centroid
    # ---------------------------------------------------------

    projected = grid.to_crs(
        get_utm_crs(
            datasets.sentinel
        )
    )

    centroid = projected.centroid

    centroid = (
        gpd.GeoSeries(
            centroid,
            crs=projected.crs,
        )
        .to_crs(
            datasets.sentinel.crs
        )
    )

    grid["longitude"] = centroid.x

    grid["latitude"] = centroid.y

    grid = grid[
        [
            "grid_id",
            "latitude",
            "longitude",
            "geometry",
        ]
    ]

    logger.info(
        "Grid GeoDataFrame created."
    )

    return grid


# =============================================================================
# GRID VALIDATION
# =============================================================================

def validate_grid(
    grid: gpd.GeoDataFrame,
) -> None:
    """
    Validate generated grid.

    Parameters
    ----------
    grid : GeoDataFrame
    """

    if grid.empty:

        raise RuntimeError(
            "Generated grid is empty."
        )

    if not grid.geometry.is_valid.all():

        raise RuntimeError(
            "Invalid geometry detected."
        )

    if grid.geometry.is_empty.any():

        raise RuntimeError(
            "Empty geometry detected."
        )

    duplicated = grid.grid_id.duplicated().sum()

    if duplicated:

        raise RuntimeError(
            f"Duplicate grid id : {duplicated}"
        )

    logger.info(
        "Grid validation passed."
    )


# =============================================================================
# GRID SUMMARY
# =============================================================================

def log_grid_summary(
    grid: gpd.GeoDataFrame,
) -> None:
    """
    Log grid summary.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("GRID SUMMARY")
    logger.info("=" * 70)

    logger.info(
        "Total Grid : %d",
        len(grid),
    )

    logger.info(
        "Grid Size  : %.0f meter",
        CONFIG.grid_size,
    )

    logger.info(
        "CRS        : %s",
        grid.crs,
    )

    logger.info(
        "Extent     : %s",
        grid.total_bounds,
    )

    logger.info("=" * 70)


# =============================================================================
# MAIN GRID FUNCTION
# =============================================================================

def create_grid(
    datasets: RasterDatasets,
) -> gpd.GeoDataFrame:
    """
    Generate grid used for feature extraction.

    Parameters
    ----------
    datasets : RasterDatasets

    Returns
    -------
    GeoDataFrame
    """

    grid = build_grid(
        datasets
    )

    validate_grid(
        grid
    )

    log_grid_summary(
        grid
    )

    return grid


# =============================================================================
# END OF SECTION 3
# =============================================================================
# =============================================================================
# SECTION 4A
# RASTER WINDOW DEFINITION
# =============================================================================

from dataclasses import dataclass

from affine import Affine
from rasterio.features import geometry_window
from rasterio.io import DatasetReader
from rasterio.windows import Window


# =============================================================================
# RASTER WINDOW CONTAINER
# =============================================================================

@dataclass(slots=True)
class RasterWindow:
    """
    Container describing a valid raster window.

    Attributes
    ----------
    window : Window
        Pixel window used for raster reading.

    transform : Affine
        Affine transform associated with the window.
    """

    window: Window
    transform: Affine


# =============================================================================
# GEOMETRY TO WINDOW
# =============================================================================

def geometry_to_window(
    dataset: DatasetReader,
    geometry,
) -> RasterWindow:
    """
    Convert a polygon into a valid raster window.

    The resulting window is clipped to the raster extent to
    guarantee that every read operation remains within the
    dataset boundary.

    Parameters
    ----------
    dataset : DatasetReader
        Open raster dataset.

    geometry : shapely.geometry.base.BaseGeometry
        Grid geometry in the same CRS as the raster.

    Returns
    -------
    RasterWindow
        Valid raster window corresponding to the geometry.

    Raises
    ------
    ValueError
        If the geometry does not intersect the raster or the
        resulting window contains no pixels.
    """

    try:

        window = geometry_window(
            dataset,
            [geometry],
            pad_x=0,
            pad_y=0,
        )

    except Exception as exc:

        raise ValueError(
            "Geometry does not intersect the raster."
        ) from exc

    # -------------------------------------------------------------------------
    # Clip window to raster extent
    # -------------------------------------------------------------------------

    window = window.intersection(
        Window(
            col_off=0,
            row_off=0,
            width=dataset.width,
            height=dataset.height,
        )
    )

    # -------------------------------------------------------------------------
    # Validate window size
    # -------------------------------------------------------------------------

    if window.width <= 0 or window.height <= 0:

        raise ValueError(
            "Generated raster window is empty."
        )

    transform = dataset.window_transform(
        window
    )

    return RasterWindow(
        window=window,
        transform=transform,
    )


# =============================================================================
# END OF SECTION 4A
# =============================================================================
# =============================================================================
# SECTION 4B
# RASTER WINDOW READING
# =============================================================================

from dataclasses import dataclass

from affine import Affine


# =============================================================================
# RASTER BLOCK CONTAINER
# =============================================================================

@dataclass(slots=True)
class RasterBlock:
    """
    Container holding raster data extracted from a window.

    Attributes
    ----------
    data : np.ndarray
        Raster values with shape (band, row, column).

    transform : Affine
        Affine transform of the extracted block.

    nodata : float | None
        Original raster nodata value.
    """

    data: np.ndarray
    transform: Affine
    nodata: float | None


# =============================================================================
# REPLACE NODATA
# =============================================================================

def replace_nodata(
    array: np.ndarray,
    nodata: float | None,
) -> np.ndarray:
    """
    Replace raster nodata values with NaN.

    Parameters
    ----------
    array : np.ndarray
        Raster array.

    nodata : float |None
        Raster nodata value.

    Returns
    -------
    np.ndarray
        Float32 array with nodata replaced by NaN.
    """

    array = array.astype(
        np.float32,
        copy=False,
    )

    if nodata is not None:
        array[array == nodata] = np.nan

    return array


# =============================================================================
# READ RASTER WINDOW
# =============================================================================

def read_window(
    dataset: DatasetReader,
    raster_window: RasterWindow,
) -> RasterBlock:
    """
    Read every band contained in a raster window.

    Parameters
    ----------
    dataset : DatasetReader

    raster_window : RasterWindow

    Returns
    -------
    RasterBlock
        Raster data ready for feature extraction.
    """

    data = dataset.read(
        window=raster_window.window,
    )

    data = replace_nodata(
        data,
        dataset.nodata,
    )

    return RasterBlock(
        data=data,
        transform=raster_window.transform,
        nodata=dataset.nodata,
    )


# =============================================================================
# END OF SECTION 4B
# =============================================================================
# =============================================================================
# SECTION 4C
# SPECTRAL FEATURE PREPARATION
# =============================================================================

from dataclasses import dataclass


# =============================================================================
# SPECTRAL FEATURE CONTAINER
# =============================================================================

@dataclass(slots=True)
class SpectralFeatures:
    """
    Container holding Sentinel-2 spectral layers.

    Attributes
    ----------
    b2, b3, b4, b8, b11, b12 : np.ndarray
        Sentinel-2 spectral bands.

    nbr : np.ndarray
        Normalized Burn Ratio computed in memory.
    """

    b2: np.ndarray
    b3: np.ndarray
    b4: np.ndarray
    b8: np.ndarray
    b11: np.ndarray
    b12: np.ndarray
    nbr: np.ndarray


# =============================================================================
# PREPARE SPECTRAL FEATURES
# =============================================================================

def prepare_spectral_features(
    block: RasterBlock,
) -> SpectralFeatures:
    """
    Split Sentinel-2 raster cube into individual bands and
    compute the Normalized Burn Ratio (NBR).

    Parameters
    ----------
    block : RasterBlock
        Sentinel raster block with shape (6, rows, cols).

    Returns
    -------
    SpectralFeatures
        Spectral layers ready for statistical analysis.
    """

    if block.data.shape[0] != len(SENTINEL_BANDS):
        raise ValueError(
            "Invalid Sentinel raster block. "
            f"Expected {len(SENTINEL_BANDS)} bands, "
            f"but found {block.data.shape[0]}."
        )

    cube = block.data

    b2 = cube[SENTINEL_BANDS["B2"] - 1]
    b3 = cube[SENTINEL_BANDS["B3"] - 1]
    b4 = cube[SENTINEL_BANDS["B4"] - 1]
    b8 = cube[SENTINEL_BANDS["B8"] - 1]
    b11 = cube[SENTINEL_BANDS["B11"] - 1]
    b12 = cube[SENTINEL_BANDS["B12"] - 1]

    numerator = b8 - b12
    denominator = b8 + b12

    nbr = np.full_like(
        numerator,
        np.nan,
        dtype=np.float32,
    )

    np.divide(
        numerator,
        denominator,
        out=nbr,
        where=~np.isclose(denominator, 0.0),
    )

    return SpectralFeatures(
        b2=b2,
        b3=b3,
        b4=b4,
        b8=b8,
        b11=b11,
        b12=b12,
        nbr=nbr,
    )


# =============================================================================
# END OF SECTION 4C
# =============================================================================
# =============================================================================
# SECTION 4D
# SPECTRAL STATISTICS
# =============================================================================

from dataclasses import dataclass


# =============================================================================
# SPECTRAL STATISTICS CONTAINER
# =============================================================================

@dataclass(slots=True)
class SpectralStatistics:
    """
    Summary statistics computed from Sentinel-2 spectral features.

    Attributes
    ----------
    mean_b2, mean_b3, mean_b4, mean_b8,
    mean_b11, mean_b12 : float
        Mean reflectance of each Sentinel-2 band.

    mean_nbr : float
        Mean Normalized Burn Ratio.

    std_nbr : float
        Standard deviation of the Normalized Burn Ratio.
    """

    mean_b2: float
    mean_b3: float
    mean_b4: float
    mean_b8: float
    mean_b11: float
    mean_b12: float
    mean_nbr: float
    std_nbr: float


# =============================================================================
# CALCULATE SPECTRAL STATISTICS
# =============================================================================

def calculate_spectral_statistics(
    spectral: SpectralFeatures,
) -> SpectralStatistics:
    """
    Compute summary statistics from Sentinel-2 spectral features.

    Parameters
    ----------
    spectral : SpectralFeatures
        Spectral layers prepared in Section 4C.

    Returns
    -------
    SpectralStatistics
        Spectral statistics for one grid cell.
    """

    return SpectralStatistics(
        mean_b2=float(np.nanmean(spectral.b2)),
        mean_b3=float(np.nanmean(spectral.b3)),
        mean_b4=float(np.nanmean(spectral.b4)),
        mean_b8=float(np.nanmean(spectral.b8)),
        mean_b11=float(np.nanmean(spectral.b11)),
        mean_b12=float(np.nanmean(spectral.b12)),
        mean_nbr=float(np.nanmean(spectral.nbr)),
        std_nbr=float(np.nanstd(spectral.nbr)),
    )


# =============================================================================
# END OF SECTION 4D
# =============================================================================
# =============================================================================
# SECTION 5A
# FEATURE EXTRACTION PER GRID
# =============================================================================

from dataclasses import asdict
from dataclasses import dataclass


# =============================================================================
# FEATURE RECORD
# =============================================================================

@dataclass(slots=True)
class FeatureRecord:
    """
    Feature values extracted from a single grid.
    """

    grid_id: str

    latitude: float
    longitude: float

    Mean_B2: float
    Mean_B3: float
    Mean_B4: float
    Mean_B8: float
    Mean_B11: float
    Mean_B12: float

    Mean_NBR: float
    Std_NBR: float

    Hotspot_Frequency: float
    Confidence_Mean: float
    FRP_Max: float

    Burn_Ratio: float

    def to_dict(self) -> dict:
        """
        Convert record into dictionary.
        """
        return asdict(self)


# =============================================================================
# EXTRACT FEATURE FROM SINGLE GRID
# =============================================================================

def extract_grid_feature(
    grid_row,
    datasets: RasterDatasets,
) -> FeatureRecord:
    """
    Extract all numerical features from a single grid.

    Parameters
    ----------
    grid_row
        One row from the grid GeoDataFrame.

    datasets : RasterDatasets

    Returns
    -------
    FeatureRecord
    """

    # -------------------------------------------------------------------------
    # Sentinel
    # -------------------------------------------------------------------------

    sentinel_window = geometry_to_window(
        datasets.sentinel,
        grid_row.geometry,
    )

    sentinel_block = read_window(
        datasets.sentinel,
        sentinel_window,
    )

    spectral = prepare_spectral_features(
        sentinel_block,
    )

    spectral_stats = calculate_spectral_statistics(
        spectral,
    )

    # -------------------------------------------------------------------------
    # VIIRS
    # -------------------------------------------------------------------------

    viirs_window = geometry_to_window(
        datasets.viirs,
        grid_row.geometry,
    )

    viirs_block = read_window(
        datasets.viirs,
        viirs_window,
    )

    hotspot_frequency = float(
        np.nanmean(
            viirs_block.data[
                VIIRS_BANDS["Hotspot_Frequency"] - 1
            ]
        )
    )

    confidence_mean = float(
        np.nanmean(
            viirs_block.data[
                VIIRS_BANDS["Confidence_Mean"] - 1
            ]
        )
    )

    frp_max = float(
        np.nanmax(
            viirs_block.data[
                VIIRS_BANDS["FRP_Max"] - 1
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Burn Ratio
    # -------------------------------------------------------------------------

    burn_window = geometry_to_window(
        datasets.burned_area,
        grid_row.geometry,
    )

    burn_block = read_window(
        datasets.burned_area,
        burn_window,
    )

    burn_ratio = float(
        np.nanmean(
            burn_block.data[0]
        )
    )

    return FeatureRecord(
        grid_id=grid_row.grid_id,
        latitude=grid_row.latitude,
        longitude=grid_row.longitude,
        Mean_B2=spectral_stats.mean_b2,
        Mean_B3=spectral_stats.mean_b3,
        Mean_B4=spectral_stats.mean_b4,
        Mean_B8=spectral_stats.mean_b8,
        Mean_B11=spectral_stats.mean_b11,
        Mean_B12=spectral_stats.mean_b12,
        Mean_NBR=spectral_stats.mean_nbr,
        Std_NBR=spectral_stats.std_nbr,
        Hotspot_Frequency=hotspot_frequency,
        Confidence_Mean=confidence_mean,
        FRP_Max=frp_max,
        Burn_Ratio=burn_ratio,
    )


# =============================================================================
# END OF SECTION 5A
# =============================================================================
# =============================================================================
# SECTION 5B
# BUILD FEATURE TABLE
# =============================================================================

from dataclasses import asdict


# =============================================================================
# BUILD FEATURE TABLE
# =============================================================================

def build_feature_table(
    grid: gpd.GeoDataFrame,
    datasets: RasterDatasets,
) -> gpd.GeoDataFrame:
    """
    Extract features from every grid cell and build the
    final feature GeoDataFrame.

    Parameters
    ----------
    grid : GeoDataFrame

    datasets : RasterDatasets

    Returns
    -------
    GeoDataFrame
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("FEATURE EXTRACTION")
    logger.info("=" * 70)

    records: list[dict] = []

    total = len(grid)

    for index, row in enumerate(
        grid.itertuples(index=False),
        start=1,
    ):

        feature = extract_grid_feature(
            row,
            datasets,
        )

        records.append(
            asdict(feature)
        )

        if (
            index % 500 == 0
            or index == total
        ):

            logger.info(
                "Processed %d/%d grids (%.1f%%)",
                index,
                total,
                index / total * 100,
            )

    feature_table = gpd.GeoDataFrame(
        records,
        geometry=grid.geometry.values,
        crs=grid.crs,
    )

    feature_table = feature_table[
        FEATURE_COLUMNS + ["geometry"]
    ]

    logger.info("")

    logger.info(
        "Feature extraction completed."
    )

    logger.info(
        "Total records : %d",
        len(feature_table),
    )

    logger.info("=" * 70)

    return feature_table


# =============================================================================
# VALIDATE FEATURE TABLE
# =============================================================================

def validate_feature_table(
    feature_table: gpd.GeoDataFrame,
) -> None:
    """
    Validate extracted feature table.

    Parameters
    ----------
    feature_table : GeoDataFrame
    """

    if feature_table.empty:

        raise RuntimeError(
            "Feature table is empty."
        )

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in feature_table.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing feature columns:\n"
            + "\n".join(missing)
        )

    duplicated = (
        feature_table.grid_id
        .duplicated()
        .sum()
    )

    if duplicated:

        raise RuntimeError(
            f"Duplicate grid_id detected: {duplicated}"
        )

    logger.info(
        "Feature table validation passed."
    )


# =============================================================================
# END OF SECTION 5B
# =============================================================================
# =============================================================================
# SECTION 5C
# SAVE FEATURE OUTPUTS
# =============================================================================


# =============================================================================
# SAVE CSV
# =============================================================================

def save_csv(
    feature_table: gpd.GeoDataFrame,
) -> None:
    """
    Save feature table to CSV.

    Parameters
    ----------
    feature_table : GeoDataFrame
    """

    csv_table = feature_table.drop(
        columns="geometry"
    )

    csv_table = csv_table.round(
        CONFIG.float_precision
    )

    csv_table.to_csv(
        CONFIG.csv_output,
        index=False,
    )

    if not CONFIG.csv_output.exists():

        raise RuntimeError(
            "Failed to save CSV output."
        )

    logger.info(
        "CSV saved: %s",
        CONFIG.csv_output.name,
    )


# =============================================================================
# SAVE GEOPACKAGE
# =============================================================================

def save_geopackage(
    feature_table: gpd.GeoDataFrame,
) -> None:
    """
    Save feature table to GeoPackage.

    Parameters
    ----------
    feature_table : GeoDataFrame
    """

    gpkg_table = feature_table.copy()

    numeric_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in (
            "grid_id",
        )
    ]

    gpkg_table[numeric_columns] = (
        gpkg_table[numeric_columns]
        .round(CONFIG.float_precision)
    )

    gpkg_table.to_file(
        CONFIG.gpkg_output,
        driver="GPKG",
    )

    if not CONFIG.gpkg_output.exists():

        raise RuntimeError(
            "Failed to save GeoPackage."
        )

    logger.info(
        "GeoPackage saved: %s",
        CONFIG.gpkg_output.name,
    )


# =============================================================================
# SAVE ALL OUTPUTS
# =============================================================================

def save_feature_outputs(
    feature_table: gpd.GeoDataFrame,
) -> None:
    """
    Save every feature extraction output.

    Parameters
    ----------
    feature_table : GeoDataFrame
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 70)

    save_csv(
        feature_table,
    )

    save_geopackage(
        feature_table,
    )

    logger.info("")
    logger.info("Output files successfully created.")

    logger.info(
        "CSV        : %s",
        CONFIG.csv_output,
    )

    logger.info(
        "GeoPackage : %s",
        CONFIG.gpkg_output,
    )

    logger.info("=" * 70)


# =============================================================================
# END OF SECTION 5C
# =============================================================================
# =============================================================================
# SECTION 5D
# FEATURE EXTRACTION REPORT
# =============================================================================

from datetime import datetime


# =============================================================================
# BUILD REPORT
# =============================================================================

def build_report(
    feature_table: gpd.GeoDataFrame,
) -> str:
    """
    Build feature extraction report.

    Parameters
    ----------
    feature_table : GeoDataFrame

    Returns
    -------
    str
        Report text.
    """

    report = [
        "=" * 70,
        "FEATURE EXTRACTION REPORT",
        "=" * 70,
        "",
        f"Execution Time : {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "INPUT RASTERS",
        "-" * 70,
        f"Sentinel     : {CONFIG.sentinel_path.name}",
        f"VIIRS        : {CONFIG.viirs_path.name}",
        f"Burned Area  : {CONFIG.burned_area_path.name}",
        "",
        "GRID",
        "-" * 70,
        f"Grid Size    : {CONFIG.grid_size:.0f} meter",
        f"Total Grid   : {len(feature_table):,}",
        "",
        "FEATURES",
        "-" * 70,
        f"Total Feature: {len(FEATURE_COLUMNS)}",
        "",
    ]

    report.extend(
        f"  - {column}"
        for column in FEATURE_COLUMNS
    )

    report.extend([
        "",
        "OUTPUT FILES",
        "-" * 70,
        f"CSV         : {CONFIG.csv_output}",
        f"GeoPackage  : {CONFIG.gpkg_output}",
        "",
        "=" * 70,
    ])

    return "\n".join(report)


# =============================================================================
# SAVE REPORT
# =============================================================================

def save_report(
    feature_table: gpd.GeoDataFrame,
) -> None:
    """
    Generate and save feature extraction report.

    Parameters
    ----------
    feature_table : GeoDataFrame
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("GENERATING REPORT")
    logger.info("=" * 70)

    report = build_report(
        feature_table
    )

    CONFIG.report_output.write_text(
        report,
        encoding="utf-8",
    )

    if not CONFIG.report_output.exists():

        raise RuntimeError(
            "Failed to create report."
        )

    logger.info(
        "Report saved: %s",
        CONFIG.report_output.name,
    )

    logger.info("=" * 70)


# =============================================================================
# END OF SECTION 5D
# =============================================================================
# =============================================================================
# SECTION 5E
# MAIN PIPELINE
# =============================================================================

from time import perf_counter


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Run the complete feature extraction pipeline.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("FEATURE EXTRACTION PIPELINE")
    logger.info("=" * 70)

    start_time = perf_counter()

    datasets = load_datasets()

    try:

        # ---------------------------------------------------------------------
        # Generate Grid
        # ---------------------------------------------------------------------

        grid = create_grid(
            datasets
        )

        # ---------------------------------------------------------------------
        # Extract Features
        # ---------------------------------------------------------------------

        feature_table = build_feature_table(
            grid,
            datasets,
        )

        validate_feature_table(
            feature_table,
        )

        # ---------------------------------------------------------------------
        # Save Outputs
        # ---------------------------------------------------------------------

        save_feature_outputs(
            feature_table,
        )

        # ---------------------------------------------------------------------
        # Report
        # ---------------------------------------------------------------------

        save_report(
            feature_table,
        )

    finally:

        close_datasets(
            datasets,
        )

    elapsed = perf_counter() - start_time

    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 70)

    logger.info(
        "Execution Time : %.2f seconds",
        elapsed,
    )

    logger.info(
        "CSV Output     : %s",
        CONFIG.csv_output,
    )

    logger.info(
        "GeoPackage     : %s",
        CONFIG.gpkg_output,
    )

    logger.info(
        "Report         : %s",
        CONFIG.report_output,
    )

    logger.info("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()


# =============================================================================
# END OF SECTION 5
# =============================================================================