# BAGIAN 1
"""
=========================================================
Sentinel-2 Mosaic
=========================================================

Project :
Analisis Zona Rawan dan Area Terbakar Kebakaran Hutan
dan Lahan Menggunakan HDBSCAN dan XGBoost

Author  : Zahra Aura Hisani
Year    : 2026

Description
-----------
Menggabungkan seluruh tile Sentinel-2 hasil export
Google Earth Engine menjadi satu raster utuh.

Tahapan:
1. Mencari seluruh tile Sentinel
2. Validasi raster
3. Merge raster
4. Simpan GeoTIFF
5. Generate report

Output
------
data/processed/sentinel_mosaic.tif

Requirements
------------
rasterio
numpy
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import rasterio
from rasterio.coords import BoundingBox
from rasterio.io import DatasetReader
from rasterio.merge import merge

# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

REPORT_DIR = PROJECT_ROOT / "outputs" / "metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "sentinel_mosaic.tif"

REPORT_FILE = REPORT_DIR / "mosaic_report.txt"

SENTINEL_PATTERN = "sentinel_raw-*.tif"

# =========================================================
# LOGGER
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# =========================================================
# DATA CLASS
# =========================================================

@dataclass
class RasterInfo:
    """Metadata raster."""

    path: Path

    width: int

    height: int

    bands: int

    dtype: str

    crs: str

    resolution: tuple[float, float]

    bounds: BoundingBox

# =========================================================
# FIND SENTINEL TILES
# =========================================================

def find_tiles(raw_directory: Path) -> List[Path]:
    """
    Mencari seluruh tile Sentinel.

    Parameters
    ----------
    raw_directory : Path

    Returns
    -------
    List[Path]
    """

    logger.info("Searching Sentinel tiles...")

    files = sorted(raw_directory.glob(SENTINEL_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"Tidak ditemukan file '{SENTINEL_PATTERN}' "
            f"di folder:\n{raw_directory}"
        )

    logger.info("Found %d Sentinel tile(s).", len(files))

    for file in files:
        logger.info("   %s", file.name)

    return files

# =========================================================
# READ RASTER INFO
# =========================================================

def read_info(dataset: DatasetReader, path: Path) -> RasterInfo:
    """
    Membaca metadata raster.
    """

    return RasterInfo(
        path=path,
        width=dataset.width,
        height=dataset.height,
        bands=dataset.count,
        dtype=dataset.dtypes[0],
        crs=str(dataset.crs),
        resolution=dataset.res,
        bounds=dataset.bounds,
    )

# =========================================================
# VALIDATE
# =========================================================

def validate_tiles(tile_paths: List[Path]) -> List[RasterInfo]:
    """
    Memastikan seluruh tile memiliki:

    - CRS sama
    - Resolution sama
    - Band sama
    - Datatype sama

    Returns
    -------
    List[RasterInfo]
    """

    logger.info("Validating Sentinel tiles...")

    infos: List[RasterInfo] = []

    with ExitStack() as stack:

        datasets = [
            stack.enter_context(rasterio.open(path))
            for path in tile_paths
        ]

        for ds, path in zip(datasets, tile_paths):

            infos.append(read_info(ds, path))

    reference = infos[0]

    for info in infos[1:]:

        if info.crs != reference.crs:
            raise ValueError(
                f"CRS mismatch:\n"
                f"{reference.path.name}: {reference.crs}\n"
                f"{info.path.name}: {info.crs}"
            )

        if info.resolution != reference.resolution:
            raise ValueError(
                f"Resolution mismatch:\n"
                f"{reference.path.name}: {reference.resolution}\n"
                f"{info.path.name}: {info.resolution}"
            )

        if info.bands != reference.bands:
            raise ValueError(
                f"Band mismatch:\n"
                f"{reference.path.name}: {reference.bands}\n"
                f"{info.path.name}: {info.bands}"
            )

        if info.dtype != reference.dtype:
            raise ValueError(
                f"Datatype mismatch:\n"
                f"{reference.path.name}: {reference.dtype}\n"
                f"{info.path.name}: {info.dtype}"
            )

    logger.info("CRS          : %s", reference.crs)
    logger.info("Resolution   : %s", reference.resolution)
    logger.info("Band Count   : %d", reference.bands)
    logger.info("Datatype     : %s", reference.dtype)

    logger.info("Validation completed successfully.")

    return infos


# BAGIAN 2
# =========================================================
# MERGE TILES
# =========================================================

def merge_tiles(tile_paths: List[Path]):
    """
    Menggabungkan seluruh tile Sentinel menjadi satu raster.

    Parameters
    ----------
    tile_paths : List[Path]

    Returns
    -------
    tuple
        mosaic_array, transform, metadata
    """

    logger.info("=" * 60)
    logger.info("Merging Sentinel tiles...")
    logger.info("=" * 60)

    with ExitStack() as stack:

        datasets = [
            stack.enter_context(rasterio.open(path))
            for path in tile_paths
        ]

        mosaic_array, mosaic_transform = merge(datasets)

        metadata = datasets[0].meta.copy()

    logger.info("Merge completed successfully.")

    logger.info(
        "Output Size : %d x %d",
        mosaic_array.shape[2],
        mosaic_array.shape[1],
    )

    return mosaic_array, mosaic_transform, metadata

# =========================================================
# SAVE MOSAIC
# =========================================================

def save_mosaic(
    mosaic_array,
    transform,
    metadata,
    output_path: Path,
):
    """
    Menyimpan hasil mosaic ke GeoTIFF.

    Parameters
    ----------
    mosaic_array
        Hasil merge raster.

    transform
        Transform hasil merge.

    metadata
        Metadata raster.

    output_path : Path
    """

    logger.info("Saving mosaic...")

    metadata.update(
        driver="GTiff",
        height=mosaic_array.shape[1],
        width=mosaic_array.shape[2],
        transform=transform,
        compress="LZW",
        tiled=True,
        BIGTIFF="YES",
    )

    with rasterio.open(output_path, "w", **metadata) as dst:
        dst.write(mosaic_array)

    logger.info("Mosaic saved successfully.")
    logger.info("Output : %s", output_path)

# =========================================================
# WRITE REPORT
# =========================================================

def write_report(
    infos: List[RasterInfo],
    mosaic_array,
    output_path: Path,
):
    """
    Membuat laporan proses mosaic.
    """

    logger.info("Writing report...")

    with open(REPORT_FILE, "w", encoding="utf-8") as report:

        report.write("=" * 60 + "\n")
        report.write("SENTINEL MOSAIC REPORT\n")
        report.write("=" * 60 + "\n\n")

        report.write(
            f"Created : "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        report.write(f"Input Tiles : {len(infos)}\n\n")

        report.write("-" * 60 + "\n")
        report.write("INPUT FILES\n")
        report.write("-" * 60 + "\n")

        for idx, info in enumerate(infos, start=1):

            report.write(f"\n[{idx}] {info.path.name}\n")
            report.write(f"Width      : {info.width}\n")
            report.write(f"Height     : {info.height}\n")
            report.write(f"Bands      : {info.bands}\n")
            report.write(f"Datatype   : {info.dtype}\n")
            report.write(f"CRS        : {info.crs}\n")
            report.write(
                f"Resolution : {info.resolution}\n"
            )

        report.write("\n")
        report.write("=" * 60 + "\n")
        report.write("OUTPUT MOSAIC\n")
        report.write("=" * 60 + "\n")

        report.write(
            f"Filename : {output_path.name}\n"
        )

        report.write(
            f"Width    : {mosaic_array.shape[2]}\n"
        )

        report.write(
            f"Height   : {mosaic_array.shape[1]}\n"
        )

        report.write(
            f"Bands    : {mosaic_array.shape[0]}\n"
        )

        report.write("Compression : LZW\n")
        report.write("BIGTIFF     : YES\n")

        report.write("\n")

        report.write("=" * 60 + "\n")
        report.write("STATUS : SUCCESS\n")
        report.write("=" * 60 + "\n")

    logger.info("Report saved.")

# BAGIAN 3
# =========================================================
# PIPELINE
# =========================================================

def run() -> Path:
    """
    Menjalankan seluruh proses mosaic Sentinel.

    Returns
    -------
    Path
        Lokasi file mosaic yang dihasilkan.
    """

    logger.info("=" * 70)
    logger.info("START SENTINEL MOSAIC")
    logger.info("=" * 70)

    # Cari tile
    tile_paths = find_tiles(RAW_DIR)

    # Validasi
    infos = validate_tiles(tile_paths)

    # Merge
    mosaic_array, transform, metadata = merge_tiles(tile_paths)

    # Simpan
    save_mosaic(
        mosaic_array=mosaic_array,
        transform=transform,
        metadata=metadata,
        output_path=OUTPUT_FILE,
    )

    # Report
    write_report(
        infos=infos,
        mosaic_array=mosaic_array,
        output_path=OUTPUT_FILE,
    )

    logger.info("=" * 70)
    logger.info("SENTINEL MOSAIC FINISHED")
    logger.info("=" * 70)

    return OUTPUT_FILE

try:

    from src.visualization.visualize_raster import visualize_file

    logger.info("Generating visualization...")

    visualize_file(
        raster_path=OUTPUT_FILE,
        output_dir=PROJECT_ROOT
        / "outputs"
        / "visualization",
    )

    logger.info("Visualization completed.")

except ImportError:

    logger.warning(
        "visualize_raster.py tidak ditemukan."
    )

except Exception as e:

    logger.warning(
        "Visualization skipped: %s",
        e,
    )

# =========================================================
# MAIN
# =========================================================

def main():

    try:

        run()

    except KeyboardInterrupt:

        logger.warning(
            "Process cancelled by user."
        )

    except Exception as e:

        logger.exception(
            "Unexpected error occurred."
        )

        raise e

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()