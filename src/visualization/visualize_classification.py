from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# ==========================================================
# CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path("/home/mahasiswa1/zahra_project")

INPUT_GPKG = PROJECT_ROOT / "data/features/classification_dataset.gpkg"

# jika ternyata gpkg belum memiliki Burn_Label,
# otomatis akan join dari csv
INPUT_CSV = PROJECT_ROOT / "data/features/classification_dataset.csv"

KECAMATAN = PROJECT_ROOT / "data/shapefile/ketapang_kecamatan.shp"

OUTPUT_DIR = PROJECT_ROOT / "data/results/classification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading classification dataset ...")

gdf = gpd.read_file(INPUT_GPKG)

# apabila Burn_Label belum ada
if "Burn_Label" not in gdf.columns:

    print("Joining Burn_Label from CSV ...")

    df = pd.read_csv(INPUT_CSV)

    gdf = gdf.merge(
        df[["grid_id", "Burn_Label"]],
        on="grid_id",
        how="left"
    )

print(gdf.head())

# ==========================================================
# LOAD KECAMATAN
# ==========================================================

kecamatan = gpd.read_file(KECAMATAN)

if kecamatan.crs != gdf.crs:
    kecamatan = kecamatan.to_crs(gdf.crs)

# ==========================================================
# STATISTICS
# ==========================================================

stats = (
    gdf["Burn_Label"]
    .value_counts()
    .rename_axis("Burn_Label")
    .reset_index(name="Jumlah_Grid")
)

stats["Persentase"] = (
    stats["Jumlah_Grid"] /
    stats["Jumlah_Grid"].sum()
) * 100

stats.to_csv(
    OUTPUT_DIR / "burn_label_statistics.csv",
    index=False
)

print(stats)

# ==========================================================
# COLOR
# ==========================================================

color_map = {
    0: "#4CAF50",   # hijau
    1: "#D32F2F"    # merah
}

gdf["color"] = gdf["Burn_Label"].map(color_map)

# ==========================================================
# FIGURE
# ==========================================================

plt.style.use("default")

fig, ax = plt.subplots(
    figsize=(12,12),
    dpi=300
)

# batas kecamatan
kecamatan.boundary.plot(
    ax=ax,
    linewidth=0.8,
    color="black"
)

# grid
gdf.plot(
    ax=ax,
    color=gdf["color"],
    linewidth=0.03,
    edgecolor="none"
)

# ==========================================================
# LEGEND
# ==========================================================

legend_elements = [

    Line2D(
        [0],
        [0],
        marker='s',
        color='w',
        markerfacecolor="#D32F2F",
        markersize=12,
        label="Area Terbakar"
    ),

    Line2D(
        [0],
        [0],
        marker='s',
        color='w',
        markerfacecolor="#4CAF50",
        markersize=12,
        label="Tidak Terbakar"
    )

]

ax.legend(
    handles=legend_elements,
    loc="lower left",
    fontsize=11,
    frameon=True
)

# ==========================================================
# TITLE
# ==========================================================

ax.set_title(
    "Distribusi Area Terbakar dan Tidak Terbakar\nKabupaten Ketapang",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_aspect("equal")

plt.tight_layout()

# ==========================================================
# SAVE
# ==========================================================

plt.savefig(
    OUTPUT_DIR / "burn_label_map.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "burn_label_map.pdf",
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "burn_label_map.svg",
    bbox_inches="tight"
)

plt.close()

print("="*60)
print("Map saved successfully.")
print("="*60)