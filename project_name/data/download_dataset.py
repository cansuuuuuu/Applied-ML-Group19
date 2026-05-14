import kagglehub
import shutil
from pathlib import Path

# Download dataset
path = kagglehub.dataset_download(
    "nakendraprasathk/cloud-image-classification-dataset"
)

destination = Path("project_name/data/raw_dataset")

shutil.copytree(path, destination, dirs_exist_ok=True)

print("Dataset copied to:", destination)
