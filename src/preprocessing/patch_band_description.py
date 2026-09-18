from pathlib import Path

import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RASTER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentinel_mosaic.tif"
)

BAND_NAMES = (
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
)


def main():

    with rasterio.open(RASTER, "r+") as ds:

        print("Before :")
        print(ds.descriptions)

        for index, name in enumerate(BAND_NAMES, start=1):
            ds.set_band_description(index, name)

        print()

        print("After :")
        print(ds.descriptions)


if __name__ == "__main__":
    main()