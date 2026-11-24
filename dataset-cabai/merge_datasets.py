import os
import glob
import random
import shutil
from pathlib import Path

random.seed(42)

old_base = r"C:\machine-learning\dataset-cabai\dataset_cabai\train"
kaggle_base = r"C:\cabai_web\chilli-plant-diseases\Chilli Plant Diseases Dataset(Augmented)\Chilli Plant Diseases Dataset"
output_dir = Path(r"C:\machine-learning\dataset-cabai\dataset_cabai_unified")

mapping = {
    "Antraknosa Patek": [
        os.path.join(kaggle_base, "*", "Chilli__Anthracnos", "*")
    ],
    "Bercak Daun Cercospora": [
        os.path.join(old_base, "Bercak Daun Cercospora", "*"),
        os.path.join(kaggle_base, "*", "Chilli__Leaf_Spot", "*")
    ],
    "Bercak Putih": [
        os.path.join(old_base, "Bercak Putih", "*")
    ],
    "Bintik Daun Bakteri": [
        os.path.join(old_base, "Bintik Daun Bakteri", "*")
    ],
    "Virus Keriting Daun": [
        os.path.join(old_base, "Daun Keriting", "*"),
        os.path.join(kaggle_base, "*", "Chilli__Leaf_Curl_Virus", "*")
    ],
    "Virus Kuning Gemini": [
        os.path.join(kaggle_base, "*", "Chilli __Yellowish", "*")
    ],
    "Hama Kutu Kebul": [
        os.path.join(kaggle_base, "*", "Chilli __Whitefly", "*")
    ],
    "Rebah Semai Damping Off": [
        os.path.join(kaggle_base, "*", "Chilli__Damping_Off", "*")
    ],
    "Defisiensi Nutrisi": [
        os.path.join(old_base, "Defisiensi Nutrisi", "*"),
        os.path.join(kaggle_base, "*", "Chilli__Veinal_Mottle_Virus", "*")
    ],
    "Tanaman dan Daun Sehat": [
        os.path.join(old_base, "Daun Sehat", "*"),
        os.path.join(kaggle_base, "*", "Chilli___healthy", "*")
    ]
}

CAP_PER_CLASS = {
    "Defisiensi Nutrisi": 600,
    "Tanaman dan Daun Sehat": 600,
    "Virus Keriting Daun": 650,
    "Bercak Daun Cercospora": 650
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

for split in ["train", "val", "test"]:
    (output_dir / split).mkdir(parents=True, exist_ok=True)

total_merged = 0

print("Memulai proses penggabungan dataset...")
for cls_name, patterns in mapping.items():
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat))
    
    # Filter hanya file gambar valid
    valid_files = [
        f for f in all_files 
        if os.path.isfile(f) and Path(f).suffix.lower() in VALID_EXTENSIONS
    ]
    # Hapus duplikat path
    valid_files = list(set(valid_files))
    random.shuffle(valid_files)

    # Batasi jika kelas terlalu besar untuk menjaga keseimbangan
    cap = CAP_PER_CLASS.get(cls_name, len(valid_files))
    chosen_files = valid_files[:cap]
    
    n_total = len(chosen_files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val
    
    splits = {
        "train": chosen_files[:n_train],
        "val": chosen_files[n_train:n_train + n_val],
        "test": chosen_files[n_train + n_val:]
    }
    
    print(f"\nKelas: {cls_name} (Total: {n_total})")
    for s_name, s_files in splits.items():
        dest_dir = output_dir / s_name / cls_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(s_files):
            ext = Path(src).suffix.lower()
            dst = dest_dir / f"{cls_name[:8].strip()}_{i:04d}{ext}"
            shutil.copy2(src, dst)
        print(f" - {s_name}: {len(s_files)} gambar")
        
    total_merged += n_total

print("\n" + "="*40)
print(f"PENGGABUNGAN SELESAI! Total gambar gabungan: {total_merged}")
print(f"Lokasi dataset baru: {output_dir}")
print("="*40)
