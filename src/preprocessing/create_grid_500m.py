# =============================================================================
# CREATE 500 METER GRID
# =============================================================================

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

# =============================================================================
# PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DIR = PROJECT_ROOT / "data" / "shapefile"

AOI_PATH = VECTOR_DIR / "ketapang_kecamatan.shp"

OUTPUT_GRID = VECTOR_DIR / "grid_500m.gpkg"

# =============================================================================
# GRID PARAMETER
# =============================================================================

GRID_SIZE = 500  # meter

TARGET_CRS = "EPSG:32749"     # UTM Zone 49S

OUTPUT_CRS = "EPSG:4326"

# =============================================================================
# LOAD AOI
# =============================================================================

print("=" * 70)
print("LOAD AOI")
print("=" * 70)

aoi = gpd.read_file(AOI_PATH)

print(f"AOI Feature : {len(aoi)}")

# =============================================================================
# PROJECT TO UTM
# =============================================================================

aoi = aoi.to_crs(TARGET_CRS)

# =============================================================================
# CREATE GRID
# =============================================================================

print("=" * 70)
print("CREATE GRID")
print("=" * 70)

xmin, ymin, xmax, ymax = aoi.total_bounds

cols = np.arange(xmin, xmax + GRID_SIZE, GRID_SIZE)

rows = np.arange(ymin, ymax + GRID_SIZE, GRID_SIZE)

polygons = []

for x in cols[:-1]:

    for y in rows[:-1]:

        polygons.append(

            box(

                x,

                y,

                x + GRID_SIZE,

                y + GRID_SIZE,

            )

        )

grid = gpd.GeoDataFrame(

    geometry=polygons,

    crs=TARGET_CRS,

)

print(f"Grid Before Clip : {len(grid):,}")

# =============================================================================
# CLIP GRID
# =============================================================================

print("=" * 70)
print("CLIP GRID")
print("=" * 70)

grid = gpd.overlay(

    grid,

    aoi,

    how="intersection",

)

print(f"Grid After Clip : {len(grid):,}")

# =============================================================================
# GRID ID
# =============================================================================

grid = grid.reset_index(drop=True)

grid["grid_id"] = np.arange(

    1,

    len(grid) + 1,

)

# =============================================================================
# CENTROID
# =============================================================================

centroid = grid.geometry.centroid

grid["longitude"] = centroid.x

grid["latitude"] = centroid.y

# =============================================================================
# PROJECT BACK TO WGS84
# =============================================================================

grid = grid.to_crs(OUTPUT_CRS)

centroid = grid.geometry.centroid

grid["longitude"] = centroid.x

grid["latitude"] = centroid.y

# =============================================================================
# SAVE
# =============================================================================

print("=" * 70)
print("SAVE GRID")
print("=" * 70)

grid.to_file(

    OUTPUT_GRID,

    driver="GPKG",

)

print(f"Output : {OUTPUT_GRID}")

print("=" * 70)
print("GRID CREATED SUCCESSFULLY")
print("=" * 70)