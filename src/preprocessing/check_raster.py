from pathlib import Path
import rasterio

DATA_DIR = Path("data/raw")

for tif in sorted(DATA_DIR.glob("*.tif")):

    print("=" * 60)
    print(f"Nama File : {tif.name}")

    with rasterio.open(tif) as src:

        print(f"Ukuran     : {src.width} x {src.height}")
        print(f"Band       : {src.count}")
        print(f"CRS        : {src.crs}")
        print(f"Resolution : {src.res}")
        print(f"Datatype   : {src.dtypes}")
        print(f"Bounds     : {src.bounds}")