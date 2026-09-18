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
with rasterio.open(SENTINEL_PATH) as src:

    band = src.read(1)

    print(band.min())

    print(band.max())