from pathlib import Path

import pandas as pd

# =============================================================================
# PATH
# =============================================================================

INPUT_CSV = Path("data/features/feature_grid.csv")
OUTPUT_CSV = Path("data/features/final_dataset.csv")

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 60)
print("DATASET AWAL")
print("=" * 60)
print(f"Total Grid : {len(df):,}")

# =============================================================================
# HAPUS GRID TANPA FITUR SENTINEL
# =============================================================================

sentinel_features = [
    "Mean_B2",
    "Mean_B3",
    "Mean_B4",
    "Mean_B8",
    "Mean_B11",
    "Mean_B12",
    "Mean_NBR",
]

df = df.dropna(
    subset=sentinel_features
).reset_index(drop=True)

print(f"Grid Valid : {len(df):,}")

# =============================================================================
# SIMPAN
# =============================================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print("=" * 60)
print("SELESAI")
print("=" * 60)
print(f"Output : {OUTPUT_CSV}")