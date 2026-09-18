# =============================================================================
# SECTION 1 — IMPORT LIBRARIES
# =============================================================================

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# SECTION 2 — CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# Project Directory
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/mahasiswa1/zahra_project")

# -----------------------------------------------------------------------------
# Input Files
# -----------------------------------------------------------------------------
CLASSIFICATION_CSV = (
    PROJECT_ROOT /
    "data" /
    "features" /
    "classification_dataset.csv"
)

GRID_FILE = (
    PROJECT_ROOT /
    "data" /
    "shapefile" /
    "grid_500m.gpkg"
)

KECAMATAN_FILE = (
    PROJECT_ROOT /
    "data" /
    "shapefile" /
    "ketapang_kecamatan.shp"
)

# -----------------------------------------------------------------------------
# Output Directory
# -----------------------------------------------------------------------------
OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "visualization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MAP = OUTPUT_DIR / "new_burned_area_map.png"

# -----------------------------------------------------------------------------
# Visualization Settings
# -----------------------------------------------------------------------------
MAP_TITLE = (
    "Burned and Non-Burned Area Distribution\n"
    "Ketapang Regency (2025–2026)"
)

COLOR_BURNED = "#d73027"
COLOR_NOT_BURNED = "#1a9850"
BOUNDARY_COLOR = "black"

FIGURE_SIZE = (12, 12)
OUTPUT_DPI = 300

# =============================================================================
# SECTION 3 — LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("BURNED AREA VISUALIZATION")
logger.info("=" * 80)

logger.info("Input Classification : %s", CLASSIFICATION_CSV)
logger.info("Input Grid           : %s", GRID_FILE)
logger.info("Input Boundary       : %s", KECAMATAN_FILE)
logger.info("Output Map           : %s", OUTPUT_MAP)

logger.info("=" * 80)

# =============================================================================
# SECTION 4 — LOAD & PREPARE DATA
# =============================================================================

logger.info("Loading classification dataset...")

classification_df = pd.read_csv(CLASSIFICATION_CSV)

logger.info("Classification records : %d", len(classification_df))

logger.info("Loading grid dataset...")

grid_gdf = gpd.read_file(GRID_FILE)

logger.info("Grid polygons : %d", len(grid_gdf))

logger.info("Loading district boundary...")

kecamatan_gdf = gpd.read_file(KECAMATAN_FILE)

logger.info("District polygons : %d", len(kecamatan_gdf))

# -----------------------------------------------------------------------------
# Validate Required Column
# -----------------------------------------------------------------------------
required_columns = ["grid_id", "Burn_Label"]

missing_columns = [
    column for column in required_columns
    if column not in classification_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required column(s): {missing_columns}"
    )

if "grid_id" not in grid_gdf.columns:
    raise ValueError(
        "Column 'grid_id' not found in grid dataset."
    )

# -----------------------------------------------------------------------------
# Merge Classification Result with Grid
# -----------------------------------------------------------------------------
logger.info("Joining classification data with grid...")

grid_gdf = grid_gdf.merge(
    classification_df[["grid_id", "Burn_Label"]],
    on="grid_id",
    how="left"
)

# -----------------------------------------------------------------------------
# Handle Missing Burn_Label
# -----------------------------------------------------------------------------
missing_count = grid_gdf["Burn_Label"].isna().sum()

logger.info("Grid without Burn_Label : %d", missing_count)

# Isi seluruh nilai NaN sebagai Non-Burned (0)
grid_gdf["Burn_Label"] = (
    grid_gdf["Burn_Label"]
    .fillna(0)
    .astype(int)
)

# -----------------------------------------------------------------------------
# Ensure Same Coordinate Reference System (CRS)
# -----------------------------------------------------------------------------
if kecamatan_gdf.crs != grid_gdf.crs:
    kecamatan_gdf = kecamatan_gdf.to_crs(grid_gdf.crs)

logger.info("Coordinate Reference System : %s", grid_gdf.crs)

logger.info("Prepared polygons : %d", len(grid_gdf))

logger.info("Data preparation completed successfully.")

# =============================================================================
# SECTION 5 — CREATE & SAVE BURNED AREA MAP
# =============================================================================

logger.info("Creating burned area map...")

fig, ax = plt.subplots(figsize=FIGURE_SIZE)

# -------------------------------------------------------------------------
# Plot Burned / Non-Burned Grid
# -------------------------------------------------------------------------
grid_gdf.plot(
    column="Burn_Label",
    categorical=True,
    cmap=None,
    color=grid_gdf["Burn_Label"].map({
        0: COLOR_NOT_BURNED,
        1: COLOR_BURNED
    }),
    linewidth=0,
    ax=ax
)

# -------------------------------------------------------------------------
# Plot District Boundary
# -------------------------------------------------------------------------
kecamatan_gdf.boundary.plot(
    ax=ax,
    color=BOUNDARY_COLOR,
    linewidth=0.8
)

# -------------------------------------------------------------------------
# Title
# -------------------------------------------------------------------------
ax.set_title(
    MAP_TITLE,
    fontsize=16,
    fontweight="bold",
    pad=15
)

# -------------------------------------------------------------------------
# Axis
# -------------------------------------------------------------------------
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)

# -------------------------------------------------------------------------
# Legend
# -------------------------------------------------------------------------
legend_elements = [

    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label="Non-Burned",
        markerfacecolor=COLOR_NOT_BURNED,
        markersize=12
    ),

    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        label="Burned",
        markerfacecolor=COLOR_BURNED,
        markersize=12
    )

]

ax.legend(
    handles=legend_elements,
    title="Burn Status",
    loc="lower right",
    frameon=True
)

# -------------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------------
ax.set_aspect("equal")

plt.tight_layout()

# -------------------------------------------------------------------------
# Save Figure
# -------------------------------------------------------------------------
plt.savefig(
    OUTPUT_MAP,
    dpi=OUTPUT_DPI,
    bbox_inches="tight"
)

plt.close(fig)

logger.info("Visualization saved successfully.")
logger.info("Output : %s", OUTPUT_MAP)

# =============================================================================
# SECTION 6 — FINAL SUMMARY
# =============================================================================

logger.info("=" * 80)
logger.info("BURNED AREA VISUALIZATION COMPLETED")
logger.info("=" * 80)

logger.info("Total Grid Visualized : %d", len(grid_gdf))
logger.info("Burned Grid           : %d", (grid_gdf["Burn_Label"] == 1).sum())
logger.info("Non-Burned Grid       : %d", (grid_gdf["Burn_Label"] == 0).sum())

logger.info("-" * 80)
logger.info("Output Map : %s", OUTPUT_MAP)
logger.info("=" * 80)

print("\n")
print("=" * 80)
print(" Burned Area Visualization Completed Successfully ")
print("=" * 80)
print(f"Output Map : {OUTPUT_MAP}")
print("=" * 80)