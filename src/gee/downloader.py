from pathlib import Path
import gdown
from zahra_project.src.gee import downloader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw"

FOLDER_ID = "1rS0HNl0Mk10rZO__jjDq0gRyJGxLau3P"


def download_all():

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

    print("=" * 60)
    print("Mengunduh seluruh isi folder Google Drive...")
    print("=" * 60)

    gdown.download_folder(
        url=url,
        output=str(DOWNLOAD_DIR),
        quiet=False,
        use_cookies=False
    )

    print("\nSemua file berhasil diunduh.")

