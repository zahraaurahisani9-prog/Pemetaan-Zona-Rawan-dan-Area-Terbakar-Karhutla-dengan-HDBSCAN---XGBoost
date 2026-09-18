"""
=========================================================
Cloud Probability Mosaic
=========================================================

Project :
Analisis Zona Rawan dan Area Terbakar Kebakaran Hutan
dan Lahan Menggunakan HDBSCAN dan XGBoost

Author  : Zahra Aura Hisani
Year    : 2026

Description
-----------
Menggabungkan seluruh tile Cloud Probability hasil
export Google Earth Engine menjadi satu raster utuh.

Tahapan
--------
1. Mencari seluruh tile Cloud Probability
2. Validasi raster
3. Merge raster
4. Simpan GeoTIFF
5. Generate report

Output
------
data/processed/cloud_probability_mosaic.tif

Requirements
------------
rasterio
numpy
"""
# BAGIAN 1 & 2

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

# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REPORT_DIR = PROJECT_ROOT / "outputs" / "metadata"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =========================================================
# INPUT / OUTPUT
# =========================================================

CLOUD_PATTERN = "cloud_probability-*.tif"

OUTPUT_FILE = (
    PROCESSED_DIR
    / "cloud_probability_mosaic.tif"
)

REPORT_FILE = (
    REPORT_DIR
    / "cloud_mosaic_report.txt"
)

# =========================================================
# OUTPUT SETTINGS
# =========================================================

COMPRESS = "LZW"

BIGTIFF = "YES"

BAND_NAME = "probability"

# =========================================================
# LOGGER
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

logger = logging.getLogger(__name__)

# =========================================================
# SECTION 2
# DATA CLASS
# =========================================================

@dataclass(slots=True)
class CloudRasterInfo:
    """
    Metadata Cloud Probability raster.
    """

    path: Path
    width: int
    height: int
    bands: int
    dtype: str
    crs: str
    resolution: tuple[float, float]
    bounds: BoundingBox


# =========================================================
# FIND CLOUD TILES
# =========================================================

def find_tiles(
    raw_directory: Path,
) -> List[Path]:
    """
    Mencari seluruh tile Cloud Probability.

    Parameters
    ----------
    raw_directory : Path
        Direktori data mentah.

    Returns
    -------
    List[Path]
        Daftar file Cloud Probability.
    """

    logger.info(
        "Searching Cloud Probability tiles..."
    )

    files = sorted(
        raw_directory.glob(CLOUD_PATTERN)
    )

    if not files:

        raise FileNotFoundError(
            f"Tidak ditemukan file "
            f"'{CLOUD_PATTERN}'\n"
            f"pada folder:\n{raw_directory}"
        )

    logger.info(
        "Found %d tile(s).",
        len(files),
    )

    for file in files:

        logger.info(
            "   %s",
            file.name,
        )

    return files


# =========================================================
# READ RASTER INFO
# =========================================================

def read_info(
    dataset: DatasetReader,
    path: Path,
) -> CloudRasterInfo:
    """
    Membaca metadata raster tanpa
    membaca seluruh isi raster.

    Parameters
    ----------
    dataset : DatasetReader

    path : Path

    Returns
    -------
    CloudRasterInfo
    """

    return CloudRasterInfo(
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
# VALIDATE CLOUD TILES
# =========================================================

def validate_tiles(
    tile_paths: List[Path],
) -> List[CloudRasterInfo]:
    """
    Memastikan seluruh tile Cloud Probability
    memiliki metadata yang konsisten.

    Validasi:

    - CRS
    - Resolution
    - Band Count
    - Data Type

    Parameters
    ----------
    tile_paths : List[Path]

    Returns
    -------
    List[CloudRasterInfo]
    """

    logger.info(
        "Validating Cloud Probability tiles..."
    )

    infos: List[CloudRasterInfo] = []

    with ExitStack() as stack:

        datasets = [

            stack.enter_context(
                rasterio.open(path)
            )

            for path in tile_paths
        ]

        for dataset, path in zip(
            datasets,
            tile_paths,
        ):

            infos.append(
                read_info(
                    dataset,
                    path,
                )
            )

    reference = infos[0]

    for info in infos[1:]:

        if info.crs != reference.crs:

            raise ValueError(
                "CRS mismatch\n"
                f"{reference.path.name} : "
                f"{reference.crs}\n"
                f"{info.path.name} : "
                f"{info.crs}"
            )

        if (
            info.resolution
            != reference.resolution
        ):

            raise ValueError(
                "Resolution mismatch\n"
                f"{reference.path.name} : "
                f"{reference.resolution}\n"
                f"{info.path.name} : "
                f"{info.resolution}"
            )

        if info.bands != reference.bands:

            raise ValueError(
                "Band mismatch\n"
                f"{reference.path.name} : "
                f"{reference.bands}\n"
                f"{info.path.name} : "
                f"{info.bands}"
            )

        if info.dtype != reference.dtype:

            raise ValueError(
                "Datatype mismatch\n"
                f"{reference.path.name} : "
                f"{reference.dtype}\n"
                f"{info.path.name} : "
                f"{info.dtype}"
            )

    logger.info(
        "CRS          : %s",
        reference.crs,
    )

    logger.info(
        "Resolution   : %s",
        reference.resolution,
    )

    logger.info(
        "Band Count   : %d",
        reference.bands,
    )

    logger.info(
        "Datatype     : %s",
        reference.dtype,
    )

    logger.info(
        "Validation completed successfully."
    )

    return infos

# BAGIAN 3 & 4
# =========================================================
# SECTION 3
# MERGE CLOUD TILES
# =========================================================

from rasterio.merge import merge


def merge_tiles(
    tile_paths: List[Path],
):
    """
    Menggabungkan seluruh tile Cloud Probability.

    Parameters
    ----------
    tile_paths : List[Path]

    Returns
    -------
    tuple
        mosaic_array,
        mosaic_transform,
        metadata
    """

    logger.info("=" * 60)
    logger.info("Merging Cloud Probability tiles...")
    logger.info("=" * 60)

    with ExitStack() as stack:

        datasets = [
            stack.enter_context(
                rasterio.open(path)
            )
            for path in tile_paths
        ]

        mosaic_array, mosaic_transform = merge(
            datasets
        )

        metadata = datasets[0].meta.copy()

    logger.info("Merge completed successfully.")

    logger.info(
        "Output Size : %d x %d",
        mosaic_array.shape[2],
        mosaic_array.shape[1],
    )

    return (
        mosaic_array,
        mosaic_transform,
        metadata,
    )


# =========================================================
# SAVE CLOUD MOSAIC
# =========================================================

def save_cloud_mosaic(
    mosaic_array,
    transform,
    metadata,
    output_path: Path,
):
    """
    Menyimpan hasil mosaic Cloud Probability.

    Parameters
    ----------
    mosaic_array
        Hasil merge raster.

    transform

    metadata

    output_path : Path
    """

    logger.info("Saving cloud mosaic...")

    metadata.update(
        driver="GTiff",
        height=mosaic_array.shape[1],
        width=mosaic_array.shape[2],
        transform=transform,
        compress=COMPRESS,
        tiled=True,
        BIGTIFF=BIGTIFF,
    )

    with rasterio.open(
        output_path,
        "w",
        **metadata,
    ) as dst:

        dst.write(mosaic_array)

        dst.set_band_description(
            1,
            BAND_NAME,
        )

    logger.info("Cloud mosaic saved.")

    logger.info(
        "Output : %s",
        output_path,
    )


# =========================================================
# WRITE REPORT
# =========================================================

def write_report(
    infos: List[CloudRasterInfo],
    mosaic_array,
    output_path: Path,
):
    """
    Membuat laporan proses
    Cloud Probability Mosaic.
    """

    logger.info(
        "Writing report..."
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        report.write("=" * 60 + "\n")
        report.write(
            "CLOUD PROBABILITY MOSAIC REPORT\n"
        )
        report.write("=" * 60 + "\n\n")

        report.write(
            "Created : "
            f"{datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
        )

        report.write(
            f"Input Tiles : {len(infos)}\n\n"
        )

        report.write("-" * 60 + "\n")
        report.write("INPUT FILES\n")
        report.write("-" * 60 + "\n")

        for idx, info in enumerate(
            infos,
            start=1,
        ):

            report.write(
                f"\n[{idx}] {info.path.name}\n"
            )

            report.write(
                f"Width      : {info.width}\n"
            )

            report.write(
                f"Height     : {info.height}\n"
            )

            report.write(
                f"Bands      : {info.bands}\n"
            )

            report.write(
                f"Datatype   : {info.dtype}\n"
            )

            report.write(
                f"CRS        : {info.crs}\n"
            )

            report.write(
                f"Resolution : "
                f"{info.resolution}\n"
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

        report.write(
            f"Band Name : {BAND_NAME}\n"
        )

        report.write(
            f"Compression : {COMPRESS}\n"
        )

        report.write(
            f"BIGTIFF     : {BIGTIFF}\n"
        )

        report.write("\n")

        report.write("=" * 60 + "\n")
        report.write("STATUS : SUCCESS\n")
        report.write("=" * 60 + "\n")

    logger.info(
        "Report saved."
    )


# =========================================================
# SECTION 4
# PIPELINE
# =========================================================

def run() -> Path:
    """
    Menjalankan seluruh proses
    Cloud Probability Mosaic.

    Returns
    -------
    Path
        Lokasi file output.
    """

    logger.info("=" * 70)
    logger.info(
        "START CLOUD PROBABILITY MOSAIC"
    )
    logger.info("=" * 70)

    # Cari seluruh tile

    tile_paths = find_tiles(
        RAW_DIR,
    )

    # Validasi

    infos = validate_tiles(
        tile_paths,
    )

    # Merge

    (
        mosaic_array,
        transform,
        metadata,
    ) = merge_tiles(
        tile_paths,
    )

    # Simpan

    save_cloud_mosaic(
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
    logger.info(
        "CLOUD MOSAIC FINISHED"
    )
    logger.info("=" * 70)

    return OUTPUT_FILE


# =========================================================
# MAIN
# =========================================================

def main():
    """
    Entry point program.
    """

    try:

        run()

    except KeyboardInterrupt:

        logger.warning(
            "Process cancelled by user."
        )

    except Exception:

        logger.exception(
            "Unexpected error occurred."
        )

        raise


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()