# BAGIAN 1
"""
============================================================
PROJECT

Analisis Zona Rawan dan Area Terbakar Karhutla

FILE
visualize_raster.py

DESKRIPSI
Visualisasi otomatis seluruh raster hasil Google Earth Engine.

Dataset yang didukung:

1. Sentinel-2 (6 Band)
2. VIIRS Hotspot Statistics (3 Band)
3. MODIS Burned Area (1 Band)

Output:

outputs/
└── visualization/
      *.png

Author :
Zahra Aura Hisani

============================================================
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.enums import Resampling

# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualization"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# PARAMETER
# ============================================================

PREVIEW_SIZE = 1500

HIST_SAMPLE_SIZE = 100000

DPI = 300

PERCENTILE_MIN = 2

PERCENTILE_MAX = 98

# ============================================================
# NAMA BAND
# ============================================================

SENTINEL_BANDS = [
    "B2 (Blue)",
    "B3 (Green)",
    "B4 (Red)",
    "B8 (NIR)",
    "B11 (SWIR-1)",
    "B12 (SWIR-2)"
]

VIIRS_BANDS = [
    "Hotspot Frequency",
    "Confidence Mean",
    "Maximum FRP"
]

# ============================================================
# READ PREVIEW
# ============================================================

def read_preview(src):

    """
    Membaca raster dalam ukuran kecil agar
    proses visualisasi jauh lebih cepat.
    """

    return src.read(

        out_shape=(
            src.count,
            PREVIEW_SIZE,
            PREVIEW_SIZE
        ),

        resampling=Resampling.bilinear

    )

# ============================================================
# STATISTIK
# ============================================================

def get_statistics(img):

    """
    Mengembalikan statistik raster.
    """

    valid = img[np.isfinite(img)]

    if len(valid) == 0:

        return {

            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan

        }

    return {

        "min": float(valid.min()),

        "max": float(valid.max()),

        "mean": float(valid.mean()),

        "std": float(valid.std()),

        "median": float(np.median(valid))

    }

# ============================================================
# HISTOGRAM SAMPLE
# ============================================================

def sample_pixels(img):

    """
    Mengambil sebagian pixel untuk histogram.

    Histogram tidak perlu memakai seluruh
    pixel karena jumlahnya sangat besar.
    """

    valid = img[np.isfinite(img)]

    if len(valid) <= HIST_SAMPLE_SIZE:

        return valid

    idx = np.random.choice(

        len(valid),

        HIST_SAMPLE_SIZE,

        replace=False

    )

    return valid[idx]

# ============================================================
# STRETCH SINGLE BAND
# ============================================================

def stretch_single(img):

    """
    Contrast Stretch
    Untuk visualisasi grayscale.
    """

    img = img.astype(np.float32)

    mask = np.isfinite(img)

    if mask.sum() == 0:

        return np.zeros_like(img)

    valid = img[mask]

    p2 = np.percentile(
        valid,
        PERCENTILE_MIN
    )

    p98 = np.percentile(
        valid,
        PERCENTILE_MAX
    )

    if p98 <= p2:

        return np.zeros_like(img)

    img = (img - p2) / (p98 - p2)

    img = np.clip(
        img,
        0,
        1
    )

    img[~mask] = 0

    return img

# ============================================================
# STRETCH RGB
# ============================================================

def stretch_rgb(red, green, blue):

    stack = np.stack([
        red,
        green,
        blue
    ]).astype(np.float32)

    mask = np.isfinite(stack)

    valid = stack[mask]

    if len(valid) == 0:
        return np.zeros(
            (stack.shape[1], stack.shape[2], 3),
            dtype=np.float32
        )

    p2 = np.percentile(
        valid,
        PERCENTILE_MIN
    )

    p98 = np.percentile(
        valid,
        PERCENTILE_MAX
    )

    if p98 <= p2:

        p98 = p2 + 1

    stack = (stack - p2) / (p98 - p2)

    stack = np.clip(
        stack,
        0,
        1
    )

    stack = np.moveaxis(
        stack,
        0,
        -1
    )

    stack[np.isnan(stack)] = 0

    return stack

# ============================================================
# METADATA TEXT
# ============================================================

def create_metadata(src, filename):

    """
    Membuat text metadata
    untuk ditampilkan pada gambar.
    """

    text = ""

    text += f"File : {filename}\n"

    text += f"Bands : {src.count}\n"

    text += f"Size : {src.width} x {src.height}\n"

    text += f"CRS : {src.crs}\n"

    text += f"Resolution : {src.res[0]:.8f}\n"

    text += f"Data Type : {src.dtypes[0]}"

    return text

# BAGIAN 2
# ============================================================
# VISUALIZE SENTINEL
# ============================================================

def visualize_sentinel(src, preview, outfile):

    """
    Visualisasi Sentinel-2
    (6 band)
    """

    fig = plt.figure(figsize=(20, 16))

    gs = fig.add_gridspec(
        5,
        2,
        height_ratios=[1.2, 1, 1, 1, 1]
    )

    # ========================================================
    # TRUE COLOR
    # ========================================================

    true_rgb = stretch_rgb(
        preview[2],      # B4
        preview[1],      # B3
        preview[0]       # B2
    )

    ax = fig.add_subplot(gs[0, 0])

    ax.imshow(true_rgb)

    ax.set_title(
        "True Color (B4 - B3 - B2)",
        fontsize=12,
        fontweight="bold"
    )

    ax.axis("off")

    # ========================================================
    # FALSE COLOR
    # ========================================================

    false_rgb = stretch_rgb(
        preview[3],      # B8
        preview[2],      # B4
        preview[1]       # B3
    )

    ax = fig.add_subplot(gs[0, 1])

    ax.imshow(false_rgb)

    ax.set_title(
        "False Color (B8 - B4 - B3)",
        fontsize=12,
        fontweight="bold"
    )

    ax.axis("off")

    # ========================================================
    # BAND PREVIEW
    # ========================================================

    band_positions = [

        (1,0),
        (1,1),

        (2,0),
        (2,1),

        (3,0),
        (3,1)

    ]

    for band_idx, (r, c) in enumerate(band_positions):

        ax = fig.add_subplot(gs[r, c])

        img = stretch_single(preview[band_idx])

        ax.imshow(
            img,
            cmap="gray"
        )

        ax.set_title(
            SENTINEL_BANDS[band_idx],
            fontsize=11
        )

        ax.axis("off")

    # ========================================================
    # HISTOGRAM
    # ========================================================

    ax_hist = fig.add_subplot(gs[4,0])

    colors = [
        "blue",
        "green",
        "red",
        "purple",
        "orange",
        "brown"
    ]

    for i in range(6):

        sample = sample_pixels(preview[i])

        ax_hist.hist(
            sample,
            bins=120,
            alpha=0.35,
            color=colors[i],
            label=SENTINEL_BANDS[i]
        )

    ax_hist.legend(
        fontsize=8,
        loc="upper right"
    )

    ax_hist.set_title(
        "Histogram",
        fontsize=11
    )

    ax_hist.grid(True)

    # ========================================================
    # METADATA
    # ========================================================

    ax_info = fig.add_subplot(gs[4,1])

    ax_info.axis("off")

    info = create_metadata(
        src,
        outfile.stem
    )

    info += "\n\n"

    info += "STATISTICS\n"
    info += "-" * 35 + "\n"

    for i in range(6):

        stat = get_statistics(
            preview[i]
        )

        info += (
            f"{SENTINEL_BANDS[i]}\n"
            f"Min    : {stat['min']:.2f}\n"
            f"Max    : {stat['max']:.2f}\n"
            f"Mean   : {stat['mean']:.2f}\n"
            f"Std    : {stat['std']:.2f}\n\n"
        )

    ax_info.text(

        0,

        1,

        info,

        va="top",

        fontsize=9,

        family="monospace"

    )

    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout()

    plt.savefig(

        outfile,

        dpi=DPI,

        bbox_inches="tight"

    )

    plt.close()

# BAGIAN 3
# ============================================================
# VISUALIZE VIIRS
# ============================================================

def visualize_viirs(src, preview, outfile):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18,10)
    )

    cmaps = [
        "hot",
        "viridis",
        "plasma"
    ]

    for i in range(3):

        img = stretch_single(preview[i])

        axes[0,i].imshow(
            img,
            cmap=cmaps[i]
        )

        axes[0,i].set_title(
            VIIRS_BANDS[i],
            fontsize=11,
            fontweight="bold"
        )

        axes[0,i].axis("off")

        sample = sample_pixels(preview[i])

        axes[1,i].hist(
            sample,
            bins=120,
            color="darkred",
            alpha=0.7
        )

        axes[1,i].grid(True)

        stat = get_statistics(preview[i])

        axes[1,i].set_title(
            f"""
Mean : {stat['mean']:.2f}

Std  : {stat['std']:.2f}
"""
        )

    plt.suptitle(
        outfile.stem,
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# VISUALIZE BURNED AREA
# ============================================================

def visualize_burned(src, preview, outfile):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14,6)
    )

    axes[0].imshow(

        stretch_single(preview[0]),

        cmap="YlOrRd"

    )

    axes[0].set_title(
        "MODIS Burned Area",
        fontsize=12,
        fontweight="bold"
    )

    axes[0].axis("off")

    sample = sample_pixels(preview[0])

    axes[1].hist(

        sample,

        bins=120,

        color="darkorange",

        alpha=0.8

    )

    stat = get_statistics(preview[0])

    axes[1].set_title(
        f"""
Histogram

Min  : {stat['min']:.2f}

Max  : {stat['max']:.2f}

Mean : {stat['mean']:.2f}

Std  : {stat['std']:.2f}
"""
    )

    axes[1].grid(True)

    plt.suptitle(
        outfile.stem,
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

# MAIN
# ============================================================
# MAIN
# ============================================================

def main():

    tif_files = sorted(
        RAW_DIR.glob("*.tif")
    )

    print("="*70)
    print("VISUALIZE RASTER")
    print("="*70)
    print()

    print(f"Jumlah Raster : {len(tif_files)}")
    print()

    for i, tif in enumerate(tif_files, start=1):

        print(
            f"[{i}/{len(tif_files)}] {tif.name}"
        )

        outfile = OUTPUT_DIR / f"{tif.stem}.png"

        try:

            with rasterio.open(tif) as src:

                preview = read_preview(src)

                if src.count == 6:

                    visualize_sentinel(
                        src,
                        preview,
                        outfile
                    )

                elif src.count == 3:

                    visualize_viirs(
                        src,
                        preview,
                        outfile
                    )

                elif src.count == 1:

                    visualize_burned(
                        src,
                        preview,
                        outfile
                    )

                else:

                    print(
                        f"Band {src.count} belum didukung."
                    )

        except Exception as e:

            print(
                f"Gagal memproses {tif.name}"
            )

            print(e)

    print()

    print("="*70)

    print("SELESAI")

    print(f"Hasil berada di : {OUTPUT_DIR}")

    print("="*70)


# ============================================================

if __name__ == "__main__":

    main()