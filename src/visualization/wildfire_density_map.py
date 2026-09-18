"""
===============================================================================
WEIGHTED WILDFIRE DENSITY MAP
===============================================================================

Project
-------
Analisis Zona Rawan dan Area Terbakar Kebakaran Hutan dan Lahan
Menggunakan HDBSCAN dan XGBoost Berbasis Data Hotspot VIIRS
dan Citra Sentinel-2 di Kabupaten Ketapang Tahun 2025–2026

Description
-----------
Weighted Kernel Density Estimation (KDE) visualization using
interpreted HDBSCAN wildfire risk levels.

Output
------
07_weighted_kde_surface.png

Author
------
Zahra Aura Hisani
===============================================================================
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

from pathlib import Path
import logging

import numpy as np
import pandas as pd

import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy.stats import gaussian_kde
from rasterio.features import geometry_mask
from affine import Affine
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar

# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = PROJECT_ROOT / "data" / "results" / "hdbscan"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MAP_OUTPUT_DIR = OUTPUT_DIR / "maps"

VIS_OUTPUT_DIR = OUTPUT_DIR / "visualization"

MAP_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VIS_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# =============================================================================
# INPUT FILES
# =============================================================================

CLUSTERED_DATASET_FILE = (
    RESULT_DIR /
    "clustered_dataset.csv"
)

CLUSTER_INTERPRETATION_FILE = (
    RESULT_DIR /
    "cluster_interpretation.csv"
)

SHAPEFILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "shapefile"
    / "ketapang_kecamatan.shp"
)

# =============================================================================
# OUTPUT FILE
# =============================================================================

OUTPUT_MAP = (
    MAP_OUTPUT_DIR /
    "07_weighted_kde_surface.png"
)

# =============================================================================
# KDE PARAMETERS
# =============================================================================

GRID_SIZE = 500

KDE_BANDWIDTH = "scott"

KDE_PADDING = 0.05

SURFACE_ALPHA = 0.75

CONTOUR_LEVELS = 8

HOTSPOT_SIZE = 10

HOTSPOT_ALPHA = 0.30

FIGURE_SIZE = (12, 12)

FIGURE_DPI = 300

# =============================================================================
# RISK WEIGHT
# =============================================================================

RISK_WEIGHT = {

    "Very High": 5,

    "High": 4,

    "Moderate": 3,

    "Low": 2,

    "Very Low": 1,

    "Noise": 0,

}

# =============================================================================
# RISK COLOR
# =============================================================================

RISK_COLORS = [

    "#2E8B57",   # Very Low

    "#FFFF66",   # Low

    "#FDB863",   # Moderate

    "#E34A33",   # High

    "#7F0000",   # Very High

]

RISK_CMAP = ListedColormap(
    RISK_COLORS
)

# =============================================================================
# BOUNDARY STYLE
# =============================================================================

BOUNDARY_FACE_COLOR = "#F7F7F7"

BOUNDARY_EDGE_COLOR = "black"

BOUNDARY_LINEWIDTH = 0.8

# =============================================================================
# HOTSPOT STYLE
# =============================================================================

HOTSPOT_COLOR = "black"

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("WEIGHTED KDE WILDFIRE DENSITY MAP")
logger.info("=" * 80)

# =============================================================================
# LOAD HDBSCAN DATA
# =============================================================================

def load_hdbscan_data():
    """
    Load HDBSCAN clustering outputs and prepare GeoDataFrame.

    Returns
    -------
    GeoDataFrame
        Spatial hotspot dataset with interpreted risk level.
    """

    logger.info("=" * 80)
    logger.info("LOADING HDBSCAN DATA")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Check Input Files
    # -------------------------------------------------------------------------

    input_files = [
        CLUSTERED_DATASET_FILE,
        CLUSTER_INTERPRETATION_FILE,
        SHAPEFILE_PATH,
    ]

    for file in input_files:

        if not file.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file}"
            )

        logger.info(
            "Found : %s",
            file.name,
        )

    # -------------------------------------------------------------------------
    # Read CSV
    # -------------------------------------------------------------------------

    clustered_dataset = pd.read_csv(
        CLUSTERED_DATASET_FILE
    )

    cluster_interpretation = pd.read_csv(
        CLUSTER_INTERPRETATION_FILE
    )

    logger.info(
        "Clustered Dataset : %d records",
        len(clustered_dataset),
    )

    logger.info(
        "Cluster Interpretation : %d clusters",
        len(cluster_interpretation),
    )

    # -------------------------------------------------------------------------
    # Required Columns
    # -------------------------------------------------------------------------

    required_cluster_columns = [

        "Cluster",

        "latitude",

        "longitude",

    ]

    required_interpretation_columns = [

        "Cluster",

        "Risk_Level",

        "Activity_Rank",

    ]

    missing_cluster = [

        col
        for col in required_cluster_columns
        if col not in clustered_dataset.columns

    ]

    if missing_cluster:

        raise ValueError(
            "Missing columns in clustered_dataset.csv : "
            f"{missing_cluster}"
        )

    missing_interpretation = [

        col
        for col in required_interpretation_columns
        if col not in cluster_interpretation.columns

    ]

    if missing_interpretation:

        raise ValueError(
            "Missing columns in cluster_interpretation.csv : "
            f"{missing_interpretation}"
        )

    # -------------------------------------------------------------------------
    # Merge Risk Interpretation
    # -------------------------------------------------------------------------

    logger.info(
        "Merging clustering result with risk interpretation..."
    )

    clustered_dataset = clustered_dataset.merge(

        cluster_interpretation[
            [
                "Cluster",
                "Risk_Level",
                "Activity_Rank",
            ]
        ],

        on="Cluster",

        how="left",

        validate="many_to_one",

    )

    missing = clustered_dataset[
        "Risk_Level"
    ].isna().sum()

    if missing > 0:

        raise ValueError(
            f"{missing} hotspot(s) have no Risk_Level."
        )

    logger.info(
        "Merge completed successfully."
    )

    # -------------------------------------------------------------------------
    # Create GeoDataFrame
    # -------------------------------------------------------------------------

    hotspot_gdf = gpd.GeoDataFrame(

        clustered_dataset,

        geometry=gpd.points_from_xy(

            clustered_dataset["longitude"],

            clustered_dataset["latitude"],

        ),

        crs="EPSG:4326",

    )

    logger.info(
        "GeoDataFrame successfully created."
    )

    # -------------------------------------------------------------------------
    # Dataset Summary
    # -------------------------------------------------------------------------

    logger.info("-" * 80)

    logger.info(
        "Total Hotspots : %d",
        len(hotspot_gdf),
    )

    logger.info(
        "Unique Clusters : %d",
        hotspot_gdf["Cluster"].nunique(),
    )

    logger.info(
        "Risk Levels : %s",
        ", ".join(
            sorted(
                hotspot_gdf[
                    "Risk_Level"
                ].unique()
            )
        ),
    )

    logger.info(
        "Coordinate Reference System : %s",
        hotspot_gdf.crs,
    )

    logger.info("-" * 80)

    return hotspot_gdf


# =============================================================================
# LOAD ADMINISTRATIVE BOUNDARY
# =============================================================================

def load_boundary():
    """
    Load Ketapang administrative boundary.

    Returns
    -------
    GeoDataFrame
        Administrative boundary.
    """

    logger.info("=" * 80)
    logger.info("LOADING ADMINISTRATIVE BOUNDARY")
    logger.info("=" * 80)

    boundary = gpd.read_file(
        SHAPEFILE_PATH
    )

    if boundary.crs != "EPSG:4326":

        logger.info(
            "Reprojecting boundary to EPSG:4326..."
        )

        boundary = boundary.to_crs(
            "EPSG:4326"
        )

    logger.info(
        "Boundary loaded successfully."
    )

    logger.info(
        "Total Polygons : %d",
        len(boundary),
    )

    logger.info(
        "Boundary CRS : %s",
        boundary.crs,
    )

    xmin, ymin, xmax, ymax = boundary.total_bounds

    logger.info(
        "Extent : %.6f %.6f %.6f %.6f",
        xmin,
        ymin,
        xmax,
        ymax,
    )

    return boundary

# =============================================================================
# PREPARE WEIGHTED KDE
# =============================================================================

def prepare_weighted_kde(
    hotspot_gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
):
    """
    Prepare weighted dataset and analysis grid for KDE.

    Parameters
    ----------
    hotspot_gdf : GeoDataFrame
        Hotspot point dataset.

    boundary : GeoDataFrame
        Administrative boundary.

    Returns
    -------
    tuple

        hotspot_weighted

        coordinates

        weights

        grid_x

        grid_y
    """

    logger.info("=" * 80)
    logger.info("PREPARING WEIGHTED KDE")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Copy Dataset
    # -------------------------------------------------------------------------

    hotspot_weighted = hotspot_gdf.copy()

    # -------------------------------------------------------------------------
    # Assign Numerical Weight
    # -------------------------------------------------------------------------

    hotspot_weighted["Weight"] = (

        hotspot_weighted["Risk_Level"]

        .map(RISK_WEIGHT)

        .astype(float)

    )

    if hotspot_weighted["Weight"].isna().any():

        raise ValueError(
            "Unknown Risk_Level detected."
        )

    logger.info(
        "Risk weight successfully assigned."
    )

    # -------------------------------------------------------------------------
    # Remove Noise
    # -------------------------------------------------------------------------

    before = len(hotspot_weighted)

    hotspot_weighted = hotspot_weighted[
        hotspot_weighted["Weight"] > 0
    ].copy()

    after = len(hotspot_weighted)

    logger.info(
        "Noise removed : %d hotspot(s)",
        before - after,
    )

    logger.info(
        "Remaining hotspots : %d",
        after,
    )

    # -------------------------------------------------------------------------
    # Extract Coordinates
    # -------------------------------------------------------------------------

    x = hotspot_weighted.geometry.x.to_numpy()

    y = hotspot_weighted.geometry.y.to_numpy()

    weights = hotspot_weighted[
        "Weight"
    ].to_numpy()

    coordinates = np.vstack([

        x,

        y,

    ])

    logger.info(
        "Coordinate matrix prepared."
    )

    logger.info(
        "Coordinate shape : %s",
        coordinates.shape,
    )

    # -------------------------------------------------------------------------
    # Boundary Extent
    # -------------------------------------------------------------------------

    xmin, ymin, xmax, ymax = boundary.total_bounds

    x_pad = (xmax - xmin) * KDE_PADDING

    y_pad = (ymax - ymin) * KDE_PADDING

    xmin -= x_pad
    xmax += x_pad

    ymin -= y_pad
    ymax += y_pad

    logger.info(
        "Analysis extent prepared."
    )

    # -------------------------------------------------------------------------
    # Create Analysis Grid
    # -------------------------------------------------------------------------

    grid_x, grid_y = np.meshgrid(

        np.linspace(

            xmin,

            xmax,

            GRID_SIZE,

        ),

        np.linspace(

            ymin,

            ymax,

            GRID_SIZE,

        ),

    )

    logger.info(
        "Grid successfully generated."
    )

    logger.info(
        "Grid Resolution : %d × %d",
        GRID_SIZE,
        GRID_SIZE,
    )

    logger.info(
        "Total Grid Cells : %s",
        f"{GRID_SIZE * GRID_SIZE:,}",
    )

    logger.info("-" * 80)

    return (

        hotspot_weighted,

        coordinates,

        weights,

        grid_x,

        grid_y,

    )

# =============================================================================
# GENERATE WEIGHTED KDE SURFACE
# =============================================================================

def generate_weighted_kde_surface(
    hotspot_weighted: gpd.GeoDataFrame,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
):
    """
    Generate weighted KDE surface from every wildfire risk level.

    Returns
    -------
    ndarray
        Weighted density surface (normalized)

    dict
        Individual KDE surface
    """

    logger.info("=" * 80)
    logger.info("GENERATING WEIGHTED KDE SURFACE")
    logger.info("=" * 80)

    grid_coordinate = np.vstack(
        (
            grid_x.ravel(),
            grid_y.ravel(),
        )
    )

    weighted_surface = np.zeros(
        grid_x.shape,
        dtype=float,
    )

    kde_surfaces = {}

    risk_order = [

        "Very Low",

        "Low",

        "Moderate",

        "High",

        "Very High",

    ]

    for risk in risk_order:

        subset = hotspot_weighted[
            hotspot_weighted["Risk_Level"] == risk
        ]

        logger.info(
            "%-10s : %5d hotspot(s)",
            risk,
            len(subset),
        )

        if len(subset) < 2:

            logger.warning(
                "%s skipped (<2 hotspot).",
                risk,
            )

            kde_surfaces[risk] = np.zeros(
                grid_x.shape,
                dtype=float,
            )

            continue

        xy = np.vstack(
            (
                subset.geometry.x.values,
                subset.geometry.y.values,
            )
        )

        kde = gaussian_kde(

            xy,

            bw_method=KDE_BANDWIDTH,

        )

        density = kde(
            grid_coordinate
        ).reshape(
            grid_x.shape
        )

        weight = RISK_WEIGHT[risk]

        weighted_density = density * weight

        kde_surfaces[risk] = weighted_density

        weighted_surface += weighted_density

        logger.info(
            "%-10s Weight : %d",
            risk,
            weight,
        )

    maximum = weighted_surface.max()

    if maximum > 0:

        weighted_surface /= maximum

    logger.info("-" * 80)

    logger.info(
        "Weighted KDE completed."
    )

    logger.info(
        "Surface Size : %d x %d",
        weighted_surface.shape[0],
        weighted_surface.shape[1],
    )

    logger.info(
        "Maximum Density : %.4f",
        weighted_surface.max(),
    )

    logger.info("-" * 80)

    return (

        weighted_surface,

        kde_surfaces,

    )

# =============================================================================
# CREATE WILDFIRE DENSITY MAP
# =============================================================================

def create_wildfire_density_map(
    weighted_surface: np.ndarray,
    hotspot_data: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
):
    """
    Create wildfire density visualization.

    Parameters
    ----------
    weighted_surface : ndarray
        Weighted KDE surface.

    hotspot_data : GeoDataFrame
        Hotspot point data.

    boundary : GeoDataFrame
        Ketapang administrative boundary.

    grid_x, grid_y : ndarray
        KDE grid coordinate.

    Returns
    -------
    fig : matplotlib.figure.Figure

    ax : matplotlib.axes.Axes
    """

    logger.info("=" * 80)
    logger.info("CREATING WILDFIRE DENSITY MAP")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Boundary Geometry
    # -------------------------------------------------------------------------

    boundary_geom = boundary.geometry.union_all()

    # -------------------------------------------------------------------------
    # Affine Transform
    # -------------------------------------------------------------------------

    pixel_width = (
        grid_x[0, 1] - grid_x[0, 0]
    )

    pixel_height = (
        grid_y[1, 0] - grid_y[0, 0]
    )

    transform = Affine(
        pixel_width,
        0,
        grid_x.min(),

        0,
        pixel_height,
        grid_y.min()
    )

    # -------------------------------------------------------------------------
    # Geometry Mask
    # -------------------------------------------------------------------------

    mask = geometry_mask(

        [boundary_geom],

        transform=transform,

        invert=True,

        out_shape=weighted_surface.shape,

        all_touched=False,

    )

    masked_surface = np.where(

        mask,

        weighted_surface,

        np.nan,

    )

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(

        figsize=FIGURE_SIZE,

        dpi=FIGURE_DPI,

    )

    extent = (

        grid_x.min(),
        grid_x.max(),
        grid_y.min(),
        grid_y.max(),

    )

    # -------------------------------------------------------------------------
    # Density Surface
    # -------------------------------------------------------------------------

    density_image = ax.imshow(

        masked_surface,

        extent=extent,

        origin="lower",

        cmap="RdYlGn_r",

        interpolation="bilinear",

        alpha=0.85,

        zorder=1,

    )

    # -------------------------------------------------------------------------
    # Density Contour
    # -------------------------------------------------------------------------

    contour_levels = np.linspace(

        np.nanmin(masked_surface),

        np.nanmax(masked_surface),

        12,

    )

    ax.contour(

        grid_x,

        grid_y,

        masked_surface,

        levels=contour_levels,

        colors="black",

        linewidths=0.35,

        alpha=0.35,

        zorder=2,

    )

    # -------------------------------------------------------------------------
    # Boundary Fill
    # -------------------------------------------------------------------------

    boundary.plot(

        ax=ax,

        facecolor="none",

        edgecolor="black",

        linewidth=1.0,

        zorder=3,

    )

    # -------------------------------------------------------------------------
    # Hotspot Overlay
    # -------------------------------------------------------------------------

    ax.scatter(

        hotspot_data.geometry.x,

        hotspot_data.geometry.y,

        s=5,

        c="black",

        alpha=0.35,

        linewidth=0,

        zorder=4,

    )

    # -------------------------------------------------------------------------
    # Axis
    # -------------------------------------------------------------------------

    ax.set_xlim(

        boundary.total_bounds[0],

        boundary.total_bounds[2],

    )

    ax.set_ylim(

        boundary.total_bounds[1],

        boundary.total_bounds[3],

    )

    ax.set_aspect("equal")

    ax.set_xlabel("Longitude")

    ax.set_ylabel("Latitude")

    ax.grid(

        linestyle="--",

        alpha=0.30,

    )

    logger.info("Wildfire density map created successfully.")

    return (

        fig,

        ax,

        density_image,

    )


# =============================================================================
# ADD MAP ELEMENTS
# =============================================================================

def add_map_elements(
    fig,
    ax,
    density_image,
    boundary: gpd.GeoDataFrame,
):
    """
    Add cartographic elements to wildfire density map.

    Parameters
    ----------
    fig : matplotlib.figure.Figure

    ax : matplotlib.axes.Axes

    density_image :
        Result of imshow()

    boundary : GeoDataFrame
        Administrative boundary.

    Returns
    -------
    fig
    ax
    """

    logger.info("=" * 80)
    logger.info("ADDING MAP ELEMENTS")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    ax.set_title(
        "Wildfire Risk Density Map\n"
        "Ketapang Regency (2025–2026)",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    # -------------------------------------------------------------------------
    # Colorbar
    # -------------------------------------------------------------------------

    colorbar = fig.colorbar(
        density_image,
        ax=ax,
        shrink=0.82,
        pad=0.02,
    )

    colorbar.set_label(
        "Normalized Wildfire Density",
        fontsize=11,
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    hotspot_legend = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Hotspot",
        markerfacecolor="black",
        markersize=6,
        alpha=0.5,
    )

    boundary_legend = Line2D(
        [0],
        [0],
        color="black",
        linewidth=1.2,
        label="Administrative Boundary",
    )

    ax.legend(
        handles=[
            hotspot_legend,
            boundary_legend,
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        fontsize=10,
    )

    # -------------------------------------------------------------------------
    # North Arrow
    # -------------------------------------------------------------------------

    ax.annotate(
        "N",
        xy=(0.965, 0.93),
        xytext=(0.965, 0.84),
        xycoords="axes fraction",
        ha="center",
        fontsize=15,
        fontweight="bold",
        arrowprops=dict(
            facecolor="black",
            width=4,
            headwidth=12,
        ),
    )

    # -------------------------------------------------------------------------
    # Scale Bar
    # -------------------------------------------------------------------------

    scalebar = ScaleBar(
        dx=111320,
        units="m",
        location="lower right",
        box_alpha=0.85,
        scale_loc="bottom",
    )

    ax.add_artist(scalebar)

    # -------------------------------------------------------------------------
    # CRS Information
    # -------------------------------------------------------------------------

    crs_name = (
        boundary.crs.to_string()
        if boundary.crs is not None
        else "Unknown CRS"
    )

    ax.text(
        0.995,
        0.01,
        f"CRS : {crs_name}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="dimgray",
    )

    # -------------------------------------------------------------------------
    # Tick Style
    # -------------------------------------------------------------------------

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    plt.tight_layout()

    logger.info("Map elements added successfully.")

    return fig, ax

# =============================================================================
# SAVE OUTPUT FIGURE
# =============================================================================

def save_output_figure(
    fig,
    output_dir: Path = OUTPUT_DIR,
    filename: str = "wildfire_density_map.png",
):
    """
    Save wildfire density map to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object.

    output_dir : Path
        Output directory.

    filename : str
        Output image filename.

    Returns
    -------
    Path
        Saved image path.
    """

    logger.info("=" * 80)
    logger.info("SAVING OUTPUT FIGURE")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Create Output Directory
    # -------------------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / filename

    # -------------------------------------------------------------------------
    # Save Figure
    # -------------------------------------------------------------------------

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )

    logger.info(
        "Output image saved successfully."
    )

    logger.info(
        "Location : %s",
        output_path,
    )

    # -------------------------------------------------------------------------
    # Close Figure
    # -------------------------------------------------------------------------

    plt.close(fig)

    logger.info(
        "Figure closed successfully."
    )

    logger.info("=" * 80)

    return output_path

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Main execution pipeline.
    """

    logger.info("=" * 80)
    logger.info("WILDFIRE DENSITY MAPPING")
    logger.info("=" * 80)

    try:

        # ---------------------------------------------------------------------
        # Load Data
        # ---------------------------------------------------------------------

        hotspot_data = load_hdbscan_data()

        boundary = load_boundary()

        # ---------------------------------------------------------------------
        # Prepare KDE
        # ---------------------------------------------------------------------

        (
            hotspot_weighted,
            coordinates,
            weights,
            grid_x,
            grid_y,
        ) = prepare_weighted_kde(
            hotspot_data,
            boundary,
        )

        # ---------------------------------------------------------------------
        # Generate Weighted KDE Surface
        # ---------------------------------------------------------------------

        (
            weighted_surface,
            kde_surfaces,
        ) = generate_weighted_kde_surface(
            hotspot_weighted,
            grid_x,
            grid_y,
        )

        # ---------------------------------------------------------------------
        # Create Density Map
        # ---------------------------------------------------------------------

        (
            fig,
            ax,
            density_image,
        ) = create_wildfire_density_map(
            weighted_surface,
            hotspot_weighted,
            boundary,
            grid_x,
            grid_y,
        )

        # ---------------------------------------------------------------------
        # Add Map Elements
        # ---------------------------------------------------------------------

        fig, ax = add_map_elements(
            fig,
            ax,
            density_image,
            boundary,
        )

        # ---------------------------------------------------------------------
        # Save Figure
        # ---------------------------------------------------------------------

        output_path = save_output_figure(
            fig,
        )

        logger.info("-" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("Output : %s", output_path)
        logger.info("-" * 80)

    except Exception as error:

        logger.exception(
            "Pipeline failed : %s",
            error,
        )

        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()