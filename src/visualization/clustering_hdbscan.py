# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_NAME: str = "VIIRS Hotspot HDBSCAN Visualization"

RANDOM_SEED: int = 42

DPI: int = 300

FIGURE_FORMAT: str = "png"

FIGURE_SIZE: tuple[int, int] = (12, 10)

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"

RESULT_DIR: Path = DATA_DIR / "results" / "hdbscan"

OUTPUT_DIR: Path = RESULT_DIR / "visualization"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# INPUT FILES
# =============================================================================

CLUSTERED_DATASET_CSV: Path = RESULT_DIR / "clustered_dataset.csv"

CLUSTER_INTERPRETATION_CSV: Path = RESULT_DIR / "cluster_interpretation.csv"

EVALUATION_SUMMARY_CSV: Path = RESULT_DIR / "evaluation_summary.csv"

KECAMATAN_SHP: Path = (
    DATA_DIR
    / "shapefile"
    / "ketapang_kecamatan.shp"
)

# =============================================================================
# OUTPUT FIGURES
# =============================================================================

ORIGINAL_HOTSPOT_MAP: Path = OUTPUT_DIR / "01_original_hotspot_map.png"

CLUSTER_MAP: Path = OUTPUT_DIR / "02_hdbscan_cluster_map.png"

RISK_LEVEL_MAP: Path = OUTPUT_DIR / "03_risk_level_map.png"

FEATURE_SPACE_SCATTER: Path = OUTPUT_DIR / "04_feature_space_scatter.png"

CLUSTER_SIZE_DISTRIBUTION: Path = (
    OUTPUT_DIR / "05_cluster_size_distribution.png"
)

ACTIVITY_PROFILE: Path = (
    OUTPUT_DIR / "06_activity_profile.png"
)

# =============================================================================
# RISK LEVEL CONFIGURATION
# =============================================================================

RISK_LEVELS: tuple[str, ...] = (
    "Very High",
    "High",
    "Medium",
    "Low",
    "Noise",
)

RISK_COLORS: dict[str, str] = {
    "Very High": "#d73027",
    "High": "#fc8d59",
    "Medium": "#fee08b",
    "Low": "#91cf60",
    "Noise": "#bdbdbd",
}

# =============================================================================
# MAP CONFIGURATION
# =============================================================================

POINT_SIZE: int = 12

POINT_ALPHA: float = 0.80

BOUNDARY_COLOR: str = "black"

BOUNDARY_WIDTH: float = 0.6

# =============================================================================
# SECTION 2 — LOAD RESULTS
# =============================================================================

import logging

import geopandas as gpd
import pandas as pd

# =============================================================================
# LOGGER CONFIGURATION
# =============================================================================

logger = logging.getLogger("HDBSCAN_VISUALIZATION")

if not logger.handlers:
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

logger.propagate = False


# =============================================================================
# LOAD VISUALIZATION RESULTS
# =============================================================================

def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    """
    Load all visualization inputs.

    Returns
    -------
    tuple
        clustered_dataset,
        cluster_interpretation,
        evaluation_summary,
        kecamatan_boundary
    """

    logger.info("=" * 80)
    logger.info("SECTION 2 — LOAD RESULTS")
    logger.info("=" * 80)

    input_files = {
        "Clustered Dataset": CLUSTERED_DATASET_CSV,
        "Cluster Interpretation": CLUSTER_INTERPRETATION_CSV,
        "Evaluation Summary": EVALUATION_SUMMARY_CSV,
        "Kecamatan Boundary": KECAMATAN_SHP,
    }

    logger.info("Checking required input files...")

    for name, file_path in input_files.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"\n{name} not found:\n{file_path}"
            )

        logger.info("✓ %s", name)

    logger.info("Reading clustered dataset...")

    clustered_df = pd.read_csv(CLUSTERED_DATASET_CSV)

    logger.info("Reading cluster interpretation...")

    interpretation_df = pd.read_csv(CLUSTER_INTERPRETATION_CSV)

    logger.info("Reading evaluation summary...")

    evaluation_df = pd.read_csv(EVALUATION_SUMMARY_CSV)

    logger.info("Reading kecamatan boundary...")

    kecamatan_gdf = gpd.read_file(KECAMATAN_SHP)

    logger.info("-" * 80)

    logger.info(
        f"Cluster Interpretation     : {len(interpretation_df):,} clusters"
    )
    logger.info(
        f"Clustered Dataset          : {len(clustered_df):,} rows × {clustered_df.shape[1]} columns"
    )
    logger.info(
        f"Evaluation Summary         : {len(evaluation_df):,} rows"
    )

    logger.info(
        f"Kecamatan Boundary         : {len(kecamatan_gdf):,} polygons"
    )

    logger.info("=" * 80)

    return (
        clustered_df,
        interpretation_df,
        evaluation_df,
        kecamatan_gdf,
    )

# =============================================================================
# SECTION 3 — PREPARE VISUALIZATION DATA
# =============================================================================

from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


# =============================================================================
# PREPARE VISUALIZATION DATA
# =============================================================================

def prepare_visualization_data(
    clustered_df: pd.DataFrame,
    interpretation_df: pd.DataFrame,
    kecamatan_gdf: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Prepare all datasets required for visualization.

    Parameters
    ----------
    clustered_df : pd.DataFrame
        Clustered hotspot dataset.

    interpretation_df : pd.DataFrame
        Cluster interpretation results.

    kecamatan_gdf : gpd.GeoDataFrame
        Kecamatan boundary.

    Returns
    -------
    GeoDataFrame
        Hotspot GeoDataFrame.

    GeoDataFrame
        Kecamatan boundary with matching CRS.
    """

    logger.info("=" * 80)
    logger.info("SECTION 3 — PREPARE VISUALIZATION DATA")
    logger.info("=" * 80)

    logger.info("Merging clustered dataset with cluster interpretation...")

    visualization_df = clustered_df.merge(
        interpretation_df[
        [
        "Cluster",
        "Activity_Rank",
        "Mean_Hotspot_Frequency",
        "Mean_FRP_Max",
        ]
        ],
        left_on="Cluster",
        right_on="Cluster",
        how="left",
    )

    logger.info("Creating risk level labels...")

    max_rank = visualization_df["Activity_Rank"].max()

    q10 = np.ceil(max_rank * 0.10)
    q30 = np.ceil(max_rank * 0.30)
    q60 = np.ceil(max_rank * 0.60)

    def assign_risk_level(row):

        if row["Cluster"] == -1:
            return "Noise"

        rank = row["Activity_Rank"]

        if rank <= q10:
            return "Very High"

        elif rank <= q30:
            return "High"

        elif rank <= q60:
            return "Medium"

        else:
            return "Low"

    visualization_df["Risk_Level"] = visualization_df.apply(
        assign_risk_level,
        axis=1,
    )

    logger.info("Converting hotspot dataset into GeoDataFrame...")

    hotspot_gdf = gpd.GeoDataFrame(
        visualization_df,
        geometry=gpd.points_from_xy(
            visualization_df["longitude"],
            visualization_df["latitude"],
        ),
        crs="EPSG:4326",
    )

    logger.info("Matching coordinate reference system...")

    hotspot_gdf = hotspot_gdf.to_crs(kecamatan_gdf.crs)

    logger.info("-" * 80)

    logger.info(
    f"Total Hotspots        : {len(hotspot_gdf):,}"
    )

    logger.info(
        "Risk Levels           : %s",
        hotspot_gdf["Risk_Level"].value_counts().to_dict(),
    )

    logger.info(
        "Coordinate System     : %s",
        hotspot_gdf.crs,
    )

    logger.info("=" * 80)

    return hotspot_gdf, kecamatan_gdf

# SECTION 4

# =============================================================================
# ORIGINAL HOTSPOT MAP
# =============================================================================

from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def create_original_hotspot_map(
    hotspot_gdf: gpd.GeoDataFrame,
    kecamatan_gdf: gpd.GeoDataFrame,
) -> None:
    """
    Generate the original VIIRS hotspot distribution map before HDBSCAN
    clustering.

    Parameters
    ----------
    hotspot_gdf : GeoDataFrame
        Original hotspot locations.

    kecamatan_gdf : GeoDataFrame
        Kecamatan administrative boundaries.
    """

    logger.info("-" * 80)
    logger.info("Creating Original Hotspot Map...")

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE,
        dpi=DPI,
        constrained_layout=True,
    )

    # -------------------------------------------------------------------------
    # Administrative Boundary
    # -------------------------------------------------------------------------

    kecamatan_gdf.boundary.plot(
        ax=ax,
        edgecolor=BOUNDARY_COLOR,
        linewidth=BOUNDARY_WIDTH,
        zorder=1,
    )

    # -------------------------------------------------------------------------
    # Original Hotspots
    # -------------------------------------------------------------------------

    hotspot_gdf.plot(
        ax=ax,
        color="#d73027",
        edgecolor="black",
        linewidth=0.20,
        markersize=POINT_SIZE,
        alpha=POINT_ALPHA,
        zorder=2,
    )

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    ax.set_title(
        (
            "Original VIIRS Hotspot Distribution\n"
            "Kabupaten Ketapang (2025–2026)"
        ),
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    # -------------------------------------------------------------------------
    # Axis Labels
    # -------------------------------------------------------------------------

    if hotspot_gdf.crs.is_geographic:

        ax.set_xlabel("Longitude (°E)", fontsize=12)
        ax.set_ylabel("Latitude (°S)", fontsize=12)

    else:

        ax.set_xlabel("Easting (m)", fontsize=12)
        ax.set_ylabel("Northing (m)", fontsize=12)

    # -------------------------------------------------------------------------
    # Grid
    # -------------------------------------------------------------------------

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.40,
    )

    # -------------------------------------------------------------------------
    # Equal Aspect
    # -------------------------------------------------------------------------

    ax.set_aspect("equal")

    # -------------------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------------------

    hotspot_legend = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="VIIRS Hotspot",
        markerfacecolor="#d73027",
        markeredgecolor="black",
        markersize=8,
    )

    boundary_legend = Line2D(
        [0],
        [0],
        color="black",
        linewidth=1.0,
        label="Kecamatan Boundary",
    )

    ax.legend(
        handles=[
            hotspot_legend,
            boundary_legend,
        ],
        loc="upper right",
        frameon=True,
    )

    # -------------------------------------------------------------------------
    # North Arrow
    # -------------------------------------------------------------------------

    ax.annotate(
        "N",
        xy=(0.95, 0.90),
        xytext=(0.95, 0.78),
        xycoords="axes fraction",
        ha="center",
        fontsize=14,
        fontweight="bold",
        arrowprops=dict(
            facecolor="black",
            width=3,
            headwidth=10,
        ),
    )

    # -------------------------------------------------------------------------
    # Scale Bar
    # -------------------------------------------------------------------------

    if not hotspot_gdf.crs.is_geographic:

        scalebar = AnchoredSizeBar(
            ax.transData,
            10000,
            "10 km",
            "lower left",
            pad=0.5,
            color="black",
            frameon=False,
            size_vertical=150,
            fontproperties=fm.FontProperties(size=10),
        )

        ax.add_artist(scalebar)

    # -------------------------------------------------------------------------
    # Total Hotspot Annotation
    # -------------------------------------------------------------------------

    hotspot_info = AnchoredText(
        f"Total Hotspots : {len(hotspot_gdf):,}",
        loc="upper left",
        frameon=True,
        prop=dict(size=10),
    )

    ax.add_artist(hotspot_info)

    # -------------------------------------------------------------------------
    # Remove Top/Right Border
    # -------------------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Save Figure
    # -------------------------------------------------------------------------

    plt.savefig(
        ORIGINAL_HOTSPOT_MAP,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        f"Original hotspot map saved to:\n{ORIGINAL_HOTSPOT_MAP}"
    )

from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib.colors import to_hex
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def create_cluster_map(
    hotspot_gdf: gpd.GeoDataFrame,
    kecamatan_gdf: gpd.GeoDataFrame,
) -> None:
    """
    Visualize HDBSCAN clustering result.

    Parameters
    ----------
    hotspot_gdf : GeoDataFrame
        Hotspot dataset with HDBSCAN cluster labels.

    kecamatan_gdf : GeoDataFrame
        Kecamatan administrative boundary.
    """

    logger.info("-" * 80)
    logger.info("Creating HDBSCAN Cluster Map...")

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE,
        dpi=DPI,
        constrained_layout=True,
    )

    # ------------------------------------------------------------------
    # Boundary
    # ------------------------------------------------------------------

    kecamatan_gdf.boundary.plot(
        ax=ax,
        edgecolor=BOUNDARY_COLOR,
        linewidth=BOUNDARY_WIDTH,
        zorder=1,
    )

    # ------------------------------------------------------------------
    # Cluster Colors
    # ------------------------------------------------------------------

    clusters = sorted(
        hotspot_gdf["Cluster"].unique()
    )

    valid_clusters = [
        c for c in clusters
        if c != -1
    ]

    cmap = mpl.colormaps["tab20"].resampled(
    max(len(valid_clusters), 1)
    )

    cluster_colors = {
        cluster: to_hex(cmap(i))
        for i, cluster in enumerate(valid_clusters)
    }

    cluster_colors[-1] = "#BDBDBD"

    # ------------------------------------------------------------------
    # Plot Noise
    # ------------------------------------------------------------------

    noise = hotspot_gdf[
        hotspot_gdf["Cluster"] == -1
    ]

    if not noise.empty:

        noise.plot(
            ax=ax,
            color=cluster_colors[-1],
            edgecolor="black",
            linewidth=0.15,
            markersize=POINT_SIZE,
            alpha=0.60,
            zorder=2,
        )

    # ------------------------------------------------------------------
    # Plot Every Cluster
    # ------------------------------------------------------------------

    for cluster in valid_clusters:

        subset = hotspot_gdf[
            hotspot_gdf["Cluster"] == cluster
        ]

        subset.plot(
            ax=ax,
            color=cluster_colors[cluster],
            edgecolor="black",
            linewidth=0.15,
            markersize=POINT_SIZE,
            alpha=POINT_ALPHA,
            zorder=3,
        )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    legend_handles = []

    for cluster in valid_clusters:

        legend_handles.append(

            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                label=f"Cluster {cluster}",
                markerfacecolor=cluster_colors[cluster],
                markeredgecolor="black",
                markersize=7,
            )

        )

    if not noise.empty:

        legend_handles.append(

            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                label="Noise",
                markerfacecolor=cluster_colors[-1],
                markeredgecolor="black",
                markersize=7,
            )

        )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        frameon=True,
        title="HDBSCAN Cluster",
    )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    ax.set_title(
        "HDBSCAN Cluster Distribution\n"
        "Kabupaten Ketapang (2025–2026)",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    # ------------------------------------------------------------------
    # Axis
    # ------------------------------------------------------------------

    if hotspot_gdf.crs.is_geographic:

        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°S)")

    else:

        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    ax.set_aspect("equal")

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
    )

    # ------------------------------------------------------------------
    # North Arrow
    # ------------------------------------------------------------------

    ax.annotate(
        "N",
        xy=(0.95, 0.90),
        xytext=(0.95, 0.78),
        xycoords="axes fraction",
        ha="center",
        fontsize=14,
        fontweight="bold",
        arrowprops=dict(
            facecolor="black",
            width=3,
            headwidth=10,
        ),
    )

    # ------------------------------------------------------------------
    # Scale Bar
    # ------------------------------------------------------------------

    if not hotspot_gdf.crs.is_geographic:

        scalebar = AnchoredSizeBar(
            ax.transData,
            10000,
            "10 km",
            "lower left",
            pad=0.5,
            color="black",
            frameon=False,
            size_vertical=150,
            fontproperties=fm.FontProperties(size=10),
        )

        ax.add_artist(scalebar)

    # ------------------------------------------------------------------
    # Information Box
    # ------------------------------------------------------------------

    info = AnchoredText(
        (
            f"Total Clusters : {len(valid_clusters)}\n"
            f"Noise Pixels : {len(noise):,}"
        ),
        loc="upper left",
        frameon=True,
        prop=dict(size=10),
    )

    ax.add_artist(info)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    plt.savefig(
        CLUSTER_MAP,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        f"HDBSCAN cluster map saved to:\n{CLUSTER_MAP}"
    )

from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def create_risk_level_map(
    hotspot_gdf: gpd.GeoDataFrame,
    kecamatan_gdf: gpd.GeoDataFrame,
) -> None:
    """
    Visualize wildfire risk levels derived from HDBSCAN cluster
    interpretation.

    Parameters
    ----------
    hotspot_gdf : GeoDataFrame
        Hotspot dataset with Risk_Level attribute.

    kecamatan_gdf : GeoDataFrame
        Kecamatan administrative boundaries.
    """

    logger.info("-" * 80)
    logger.info("Creating Wildfire Risk Level Map...")

    fig, ax = plt.subplots(
        figsize=FIGURE_SIZE,
        dpi=DPI,
        constrained_layout=True,
    )

def generate_maps(
    hotspot_gdf: gpd.GeoDataFrame,
    kecamatan_gdf: gpd.GeoDataFrame,
) -> None:

    logger.info("=" * 80)
    logger.info("SECTION 4 — GENERATE MAPS")
    logger.info("=" * 80)

    create_original_hotspot_map(
        hotspot_gdf,
        kecamatan_gdf,
    )

    create_cluster_map(
        hotspot_gdf,
        kecamatan_gdf,
    )

    create_risk_level_map(
        hotspot_gdf,
        kecamatan_gdf,
    )

    logger.info("Map generation completed successfully.")
    


from matplotlib.offsetbox import AnchoredText


def create_feature_space_scatter(
    hotspot_gdf: gpd.GeoDataFrame,
) -> None:
    """
    Visualize hotspot distribution in feature space using
    Hotspot Frequency and Maximum Fire Radiative Power (FRP),
    colored by wildfire risk level.
    """

    logger.info("-" * 80)
    logger.info("Creating Feature Space Scatter Plot...")

    fig, ax = plt.subplots(
        figsize=(10, 8),
        dpi=DPI,
        constrained_layout=True,
    )

    # ------------------------------------------------------------------
    # Plot Each Risk Level
    # ------------------------------------------------------------------

    for level in RISK_LEVELS:

        subset = hotspot_gdf[
            hotspot_gdf["Risk_Level"] == level
        ]

        if subset.empty:
            continue

        ax.scatter(
            subset["Hotspot_Frequency"],
            subset["FRP_Max"],
            s=18,
            c=RISK_COLORS[level],
            edgecolors="black",
            linewidths=0.20,
            alpha=0.70,
            label=level,
        )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    ax.set_title(
        (
            "Feature Space Distribution\n"
            "Hotspot Frequency vs Maximum FRP"
        ),
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    # ------------------------------------------------------------------
    # Axis Labels
    # ------------------------------------------------------------------

    ax.set_xlabel(
        "Hotspot Frequency",
        fontsize=12,
    )

    ax.set_ylabel(
        "Maximum Fire Radiative Power (FRP)",
        fontsize=12,
    )

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    ax.legend(
        title="Risk Level",
        frameon=True,
        fontsize=9,
    )

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    info = AnchoredText(
        f"Total Pixels : {len(hotspot_gdf):,}",
        loc="upper left",
        frameon=True,
        prop=dict(size=10),
    )

    ax.add_artist(info)

    # ------------------------------------------------------------------
    # Cosmetic
    # ------------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    plt.savefig(
        FEATURE_SPACE_SCATTER,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        f"Feature space scatter saved to:\n{FEATURE_SPACE_SCATTER}"
    )

def create_cluster_size_distribution(
    hotspot_gdf: gpd.GeoDataFrame,
) -> None:
    """
    Visualize the number of pixels contained in each HDBSCAN cluster.
    """

    logger.info("-" * 80)
    logger.info("Creating Cluster Size Distribution...")

    # ------------------------------------------------------------------
    # Calculate Cluster Size
    # ------------------------------------------------------------------

    cluster_size = (
        hotspot_gdf.groupby("Cluster")
        .size()
        .reset_index(name="Pixel_Count")
    )

    # ------------------------------------------------------------------
    # Create Labels
    # ------------------------------------------------------------------

    cluster_size["Cluster_Label"] = cluster_size["Cluster"].apply(
        lambda x: "Noise" if x == -1 else f"Cluster {x}"
    )

    # ------------------------------------------------------------------
    # Sort by Pixel Count
    # ------------------------------------------------------------------

    cluster_size = cluster_size.sort_values(
        by="Pixel_Count",
        ascending=False,
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Assign Colors
    # ------------------------------------------------------------------

    colors = [
        "#BDBDBD" if c == -1 else "#1F77B4"
        for c in cluster_size["Cluster"]
    ]

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7),
        dpi=DPI,
        constrained_layout=True,
    )

    bars = ax.bar(
        cluster_size["Cluster_Label"],
        cluster_size["Pixel_Count"],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )

    # ------------------------------------------------------------------
    # Labels on Bars
    # ------------------------------------------------------------------

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------

    ax.set_title(
        (
            "Cluster Size Distribution\n"
            "HDBSCAN Clustering Result"
        ),
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    # ------------------------------------------------------------------
    # Axis Labels
    # ------------------------------------------------------------------

    ax.set_xlabel(
        "Cluster",
        fontsize=12,
    )

    ax.set_ylabel(
        "Number of Pixels",
        fontsize=12,
    )

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
    )

    ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Rotate Labels
    # ------------------------------------------------------------------

    plt.xticks(
        rotation=45,
        ha="right",
    )

    # ------------------------------------------------------------------
    # Cosmetic
    # ------------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    plt.savefig(
        CLUSTER_SIZE_DISTRIBUTION,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        f"Cluster size distribution saved to:\n"
        f"{CLUSTER_SIZE_DISTRIBUTION}"
    )



# SECTION 5

def create_activity_profile(
    hotspot_gdf: gpd.GeoDataFrame,
) -> None:
    """
    Compare hotspot activity profile across wildfire risk levels.

    The comparison is based on the HDBSCAN cluster interpretation,
    not on recalculated clustering results.
    """

    logger.info("-" * 80)
    logger.info("Creating Activity Profile...")

    # ------------------------------------------------------------------
    # Prepare Summary
    # ------------------------------------------------------------------

    activity_profile = (
        hotspot_gdf.groupby("Risk_Level")[
            [
                "Mean_Hotspot_Frequency",
                "Mean_FRP_Max",
            ]
        ]
        .mean()
        .reindex(RISK_LEVELS)
        .fillna(0)
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 7),
        dpi=DPI,
        constrained_layout=True,
    )

    x = np.arange(len(activity_profile))

    width = 0.36

    bars1 = ax.bar(
        x - width / 2,
        activity_profile["Mean_Hotspot_Frequency"],
        width,
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.4,
        label="Mean Hotspot Frequency",
    )

    bars2 = ax.bar(
        x + width / 2,
        activity_profile["Mean_FRP_Max"],
        width,
        color="#DD8452",
        edgecolor="black",
        linewidth=0.4,
        label="Mean FRP Max",
    )

    # ------------------------------------------------------------------
    # Value Labels
    # ------------------------------------------------------------------

    for bars in (bars1, bars2):

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # ------------------------------------------------------------------
    # Axis
    # ------------------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        activity_profile.index,
        rotation=0,
    )

    ax.set_xlabel(
        "Risk Level",
        fontsize=12,
    )

    ax.set_ylabel(
        "Mean Value",
        fontsize=12,
    )

    ax.set_title(
        (
            "Wildfire Activity Profile\n"
            "Mean Hotspot Frequency and Maximum FRP\n"
            "Based on HDBSCAN Cluster Interpretation"
        ),
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    ax.legend(
        frameon=True,
        fontsize=9,
    )

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
    )

    ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Cosmetic
    # ------------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    plt.savefig(
        ACTIVITY_PROFILE,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        f"Activity profile saved to:\n{ACTIVITY_PROFILE}"
    )


def generate_statistical_visualizations(
    hotspot_gdf: gpd.GeoDataFrame,
) -> None:
    """
    Generate all statistical visualizations.
    """

    logger.info("=" * 80)
    logger.info("SECTION 5 — GENERATE STATISTICAL VISUALIZATIONS")
    logger.info("=" * 80)

    create_feature_space_scatter(
        hotspot_gdf,
    )

    create_cluster_size_distribution(
        hotspot_gdf,
    )

    create_activity_profile(
        hotspot_gdf,
    )

    logger.info(
        "Statistical visualizations generated successfully."
    )

# =============================================================================
# SECTION 6 — EXPORT RESULTS
# =============================================================================

def export_results() -> None:
    """
    Verify and summarize all exported visualization results.
    """

    logger.info("=" * 80)
    logger.info("SECTION 6 — EXPORT RESULTS")
    logger.info("=" * 80)

    output_files = {
        "Original Hotspot Map": ORIGINAL_HOTSPOT_MAP,
        "HDBSCAN Cluster Map": CLUSTER_MAP,
        "Risk Level Map": RISK_LEVEL_MAP,
        "Feature Space Scatter": FEATURE_SPACE_SCATTER,
        "Cluster Size Distribution": CLUSTER_SIZE_DISTRIBUTION,
        "Activity Profile": ACTIVITY_PROFILE,
    }

    logger.info("Checking exported visualization files...")

    exported = 0

    for name, path in output_files.items():

        if path.exists():

            logger.info(
                "✓ %-30s : %s",
                name,
                path.name,
            )

            exported += 1

        else:

            logger.warning(
                "✗ %-30s : Not Found",
                name,
            )

    logger.info("-" * 80)

    logger.info(
        "Successfully exported : %d/%d figures",
        exported,
        len(output_files),
    )

    logger.info(
        "Output Directory : %s",
        OUTPUT_DIR,
    )

    logger.info("=" * 80)

    logger.info("Visualization pipeline completed successfully.")

# =============================================================================
# MAIN
# =============================================================================

def main():

    (
        clustered_df,
        interpretation_df,
        evaluation_df,
        kecamatan_gdf,
    ) = load_results()

    hotspot_gdf, kecamatan_gdf = prepare_visualization_data(
        clustered_df,
        interpretation_df,
        kecamatan_gdf,
    )

    generate_maps(
        hotspot_gdf,
        kecamatan_gdf,
    )

    generate_statistical_visualizations(
        hotspot_gdf,
    )

    export_results()


if __name__ == "__main__":
    main()