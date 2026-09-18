from pathlib import Path

import rasterio

import matplotlib.pyplot as plt

RAW_DIR = Path("data/raw")

OUT_DIR = Path("outputs/bands/cek")

OUT_DIR.mkdir(parents=True, exist_ok=True)

for tif in sorted(RAW_DIR.glob("*.tif")):

    with rasterio.open(tif) as src:

        fig, axes = plt.subplots(
            1,
            src.count,
            figsize=(5 * src.count, 5)
        )

        if src.count == 1:
            axes = [axes]

        for i in range(src.count):

            band = src.read(i + 1)

            axes[i].imshow(band, cmap="gray")

            axes[i].set_title(f"Band {i+1}")

            axes[i].axis("off")

        plt.tight_layout()

        outfile = OUT_DIR / f"{tif.stem}.png"

        plt.savefig(outfile, dpi=300)

        plt.close()

        print(outfile)