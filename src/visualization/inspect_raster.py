from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

# ======================================================
# Folder
# ======================================================

RAW_DIR = Path("data/raw")
OUT_DIR = Path("outputs/metadata")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# Menyimpan hasil
# ======================================================

metadata_rows = []
statistics_rows = []

report = []

report.append("=" * 80)
report.append("HASIL INSPEKSI DATA RASTER")
report.append("=" * 80)
report.append("")

# ======================================================
# Loop semua raster
# ======================================================

for tif in sorted(RAW_DIR.glob("*.tif")):

    print(f"\nMembaca : {tif.name}")

    with rasterio.open(tif) as src:

        # ---------------------------------------------
        # Metadata
        # ---------------------------------------------

        metadata_rows.append({

            "filename": tif.name,

            "bands": src.count,

            "width": src.width,

            "height": src.height,

            "crs": str(src.crs),

            "resolution_x": src.res[0],

            "resolution_y": src.res[1],

            "dtype": ",".join(src.dtypes),

            "left": src.bounds.left,

            "bottom": src.bounds.bottom,

            "right": src.bounds.right,

            "top": src.bounds.top

        })

        report.append("=" * 80)
        report.append(f"FILE : {tif.name}")
        report.append("=" * 80)

        report.append(f"Jumlah Band : {src.count}")
        report.append(f"Ukuran      : {src.width} x {src.height}")
        report.append(f"CRS         : {src.crs}")
        report.append(f"Resolusi    : {src.res}")
        report.append("")

        # ---------------------------------------------
        # Statistik tiap band
        # ---------------------------------------------

        for band in range(1, src.count + 1):

            img = src.read(band)

            valid = img[np.isfinite(img)]

            minimum = float(valid.min())
            maximum = float(valid.max())
            mean = float(valid.mean())
            median = float(np.median(valid))
            std = float(valid.std())

            p2 = float(np.percentile(valid,2))
            p98 = float(np.percentile(valid,98))

            nodata = int((~np.isfinite(img)).sum())

            statistics_rows.append({

                "filename": tif.name,

                "band": band,

                "minimum": minimum,

                "maximum": maximum,

                "mean": mean,

                "median": median,

                "std": std,

                "percentile_2": p2,

                "percentile_98": p98,

                "nodata": nodata

            })

            report.append(f"Band {band}")
            report.append(f"  Minimum        : {minimum}")
            report.append(f"  Maximum        : {maximum}")
            report.append(f"  Mean           : {mean}")
            report.append(f"  Median         : {median}")
            report.append(f"  Std            : {std}")
            report.append(f"  Percentile 2   : {p2}")
            report.append(f"  Percentile 98  : {p98}")
            report.append(f"  NoData         : {nodata}")
            report.append("")

# ======================================================
# Simpan CSV
# ======================================================

metadata_df = pd.DataFrame(metadata_rows)

statistics_df = pd.DataFrame(statistics_rows)

metadata_df.to_csv(
    OUT_DIR / "raster_metadata.csv",
    index=False
)

statistics_df.to_csv(
    OUT_DIR / "raster_statistics.csv",
    index=False
)

# ======================================================
# Simpan TXT
# ======================================================

with open(
    OUT_DIR / "inspect_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(report))

print("\n")
print("=" * 80)
print("INSPEKSI SELESAI")
print("=" * 80)

print("Metadata : outputs/metadata/raster_metadata.csv")

print("Statistik: outputs/metadata/raster_statistics.csv")

print("Report   : outputs/metadata/inspect_report.txt")