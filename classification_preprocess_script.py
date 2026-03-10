import random
import shutil
from pathlib import Path

# =========================
# Settings
# =========================
root = Path("data")
data_dir = root / "DATA"
test_dir = root / "TEST"
csv_file = root / "ds1_labels.csv"  
out_root = Path("classify_dataset") 
train_ratio = 0.8 # percentage of data used to train
val_ratio = 0.2 # percentage of data used for validation
random.seed(100)
exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

assert data_dir.exists(), f"Missing folder: {data_dir}"
assert test_dir.exists(), f"Missing folder: {test_dir}"
assert abs(train_ratio + val_ratio - 1.0) < 1e-9, "train_ratio + val_ratio must equal 1"

# =========================
# Helpers
# =========================
def get_class_dirs(parent: Path):
    """Return numeric class subfolders sorted by class id."""
    return sorted(
        [p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name)
    )

def get_images(folder: Path):
    """Return image files in a folder."""
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

def copy_files(files, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst_dir / f.name)

# =========================
# Create output structure
# =========================
for split in ["train", "val", "test"]:
    (out_root / split).mkdir(parents=True, exist_ok=True)

# ==============================
# Build train/val from data/DATA
# ===========================
for class_dir in get_class_dirs(data_dir):
    class_name = class_dir.name
    images = get_images(class_dir)

    if not images:
        print(f"Warning: no images found in {class_dir}")
        continue

    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)

    # make sure val is not empty when possible
    if n >= 2:
        n_train = max(1, min(n_train, n - 1))

    train_files = images[:n_train]
    val_files = images[n_train:]

    copy_files(train_files, out_root / "train" / class_name)
    copy_files(val_files, out_root / "val" / class_name)

    print(f"DATA class {class_name}: total={n}, train={len(train_files)}, val={len(val_files)}")

# =========================
# Build test from data/TEST
# =========================
for class_dir in get_class_dirs(test_dir):
    class_name = class_dir.name
    images = get_images(class_dir)

    if not images:
        print(f"Warning: no images found in {class_dir}")
        continue

    copy_files(images, out_root / "test" / class_name)
    print(f"TEST class {class_name}: total={len(images)}")

# =========================
# Copy CSV if exists
# =========================
if csv_file.exists():
    shutil.copy2(csv_file, out_root / csv_file.name)
    print(f"Copied CSV: {csv_file.name}")
else:
    print(f"Warning: CSV file not found: {csv_file}")

print(f"\nNew dataset created at: {out_root.resolve()}")