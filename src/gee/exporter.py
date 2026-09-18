"""
Earth Engine Exporter
Export seluruh dataset mentah ke Google Drive
"""

import ee

from config.settings import (
    AOI,
    START_DATE,
    END_DATE,
)

from src.gee.loader import (
    load_sentinel,
    load_cloud,
    load_viirs,
    load_modis,
)

# ==========================
# Konfigurasi Google Drive
# ==========================

DRIVE_FOLDER = "Karhutla_Raw_Data"


# ==========================================================
# Sentinel-2
# ==========================================================

def export_sentinel():

    image = (
        load_sentinel()
        .median()
        .select(["B2", "B3", "B4", "B8", "B11", "B12"])
        .clip(AOI)
    )

    return ee.batch.Export.image.toDrive(
        image=image,
        description="Sentinel_Raw_2025_2026",
        folder=DRIVE_FOLDER,
        fileNamePrefix="sentinel_raw",
        region=AOI.geometry(),
        scale=10,
        maxPixels=1e13,
        fileFormat="GeoTIFF"
    )


# ==========================================================
# Cloud Probability
# ==========================================================

def export_cloud():

    image = (
        load_cloud()
        .mean()
        .select("probability")
        .clip(AOI)
    )

    return ee.batch.Export.image.toDrive(
        image=image,
        description="Cloud_Probability_2025_2026",
        folder=DRIVE_FOLDER,
        fileNamePrefix="cloud_probability",
        region=AOI.geometry(),
        scale=10,
        maxPixels=1e13,
        fileFormat="GeoTIFF"
    )


# ==========================================================
# VIIRS Hotspot
# ==========================================================

def export_viirs():

    collection = load_viirs()

    hotspot = (
        collection
        .select(["confidence", "frp"])
        .mean()
        .clip(AOI)
    )

    return ee.batch.Export.image.toDrive(
        image=hotspot,
        description="VIIRS_Hotspot_2025_2026",
        folder=DRIVE_FOLDER,
        fileNamePrefix="viirs_hotspot",
        region=AOI.geometry(),
        scale=375,
        maxPixels=1e13,
        fileFormat="GeoTIFF"
    )


# ==========================================================
# MODIS Burned Area
# ==========================================================

def export_modis():

    image = (
        load_modis()
        .select("BurnDate")
        .max()
        .clip(AOI)
    )

    return ee.batch.Export.image.toDrive(
        image=image,
        description="MODIS_BurnedArea_2025_2026",
        folder=DRIVE_FOLDER,
        fileNamePrefix="burned_area",
        region=AOI.geometry(),
        scale=500,
        maxPixels=1e13,
        fileFormat="GeoTIFF"
    )


# ==========================================================
# Membuat daftar task
# ==========================================================

def build_export_tasks():

    return [
        ("Sentinel-2", export_sentinel()),
        ("Cloud Probability", export_cloud()),
        ("VIIRS Hotspot", export_viirs()),
        ("MODIS Burned Area", export_modis()),
    ]