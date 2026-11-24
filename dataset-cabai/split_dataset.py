
import random
import shutil
from pathlib import Path

# Folder sumber: seluruh gambar asli Mendeley Data
SOURCE_DIR = Path("dataset_cabai/train")

# Folder hasil pembagian baru
OUTPUT_DIR = Path("dataset_cabai_baru")

# Rasio pembagian dataset
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Supaya pembagian bisa diulang dengan hasil yang sama
random.seed(42)

# Format gambar yang didukung
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Folder sumber tidak ditemukan: {SOURCE_DIR}"
    )

class_folders = [
    folder for folder in SOURCE_DIR.iterdir()
    if folder.is_dir()
]

if not class_folders:
    raise ValueError("Folder sumber tidak memiliki folder kelas.")

# Periksa total rasio
if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 0.0001:
    raise ValueError("Total rasio harus sama dengan 1.")

# Buat folder hasil
for split in ["train", "val", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

total_images = 0

for class_folder in sorted(class_folders):
    class_name = class_folder.name

    images = [
        file for file in class_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)

    if total < 3:
        raise ValueError(
            f"Kelas {class_name} hanya memiliki {total} gambar."
        )

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    print(f"\nKelas: {class_name}")

    for split_name, split_images in split_data.items():
        destination = OUTPUT_DIR / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(image, destination / image.name)

        print(f"{split_name}: {len(split_images)} gambar")

    total_images += total

print("\n================================")
print("PEMBAGIAN DATASET SELESAI")
print("================================")
print(f"Total gambar sumber: {total_images}")
print(f"Folder hasil: {OUTPUT_DIR.resolve()}")
