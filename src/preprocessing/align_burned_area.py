"""
Align burned_area.tif to Sentinel grid.

Output:
    data/processed/burned_area_aligned.tif
"""

from pathlib import Path

import rasterio
from rasterio.warp import reproject, Resampling

# ==========================================================
# PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "processed"

SENTINEL_PATH = INPUT_DIR / "sentinel_processed.tif"

BURNED_PATH = INPUT_DIR / "burned_area.tif"

OUTPUT_PATH = INPUT_DIR / "burned_area_aligned.tif"

# ==========================================================
# MAIN
# ==========================================================


def main():

    print("=" * 70)
    print("ALIGN BURNED AREA TO SENTINEL GRID")
    print("=" * 70)

    with rasterio.open(SENTINEL_PATH) as sentinel:

        with rasterio.open(BURNED_PATH) as burned:

            profile = sentinel.profile.copy()

            profile.update(

                driver="GTiff",

                count=1,

                dtype=burned.dtypes[0],

                nodata=burned.nodata,

                compress="lzw"

            )

            with rasterio.open(
                OUTPUT_PATH,
                "w",
                **profile
            ) as dst:

                reproject(

                    source=rasterio.band(burned, 1),

                    destination=rasterio.band(dst, 1),

                    src_transform=burned.transform,

                    src_crs=burned.crs,

                    dst_transform=sentinel.transform,

                    dst_crs=sentinel.crs,

                    resampling=Resampling.nearest,

                )

    print(f"\n✓ Output saved : {OUTPUT_PATH}")

    with rasterio.open(OUTPUT_PATH) as out:

        print(f"Size       : {out.width} x {out.height}")

        print(f"Resolution : {out.res}")

        print(f"CRS        : {out.crs}")

        print(f"Bands      : {out.count}")


if __name__ == "__main__":

    main()