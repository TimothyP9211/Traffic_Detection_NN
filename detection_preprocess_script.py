import random
import shutil
from pathlib import Path

# =========================
# Settings
# =========================
root = Path("data") # may likely change this later to seperate dataset with zoomed out images
data_dir = root / "DATA"
test_dir = root / "TEST"
out_root = Path("detect_dataset")
train_ratio = 0.8
val_ratio = 0.2
random.seed(100)
exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

assert data_dir.exists(), f"Missing folder: {data_dir}"
assert test_dir.exists(), f"Missing folder: {test_dir}"
assert abs(train_ratio + val_ratio - 1.0) < 1e-9, "train_ratio + val_ratio must equal 1"

# =========================
# Helpers
# =========================
def get_class_dirs(parent: Path):
    return sorted(
        [p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name)
    )

def get_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

def prepare_split_dirs(base: Path, split: str):
    (base / "images" / split).mkdir(parents=True, exist_ok=True)
    (base / "labels" / split).mkdir(parents=True, exist_ok=True)

def safe_unique_name(class_name: str, img_path: Path) -> str:
    return f"{class_name}_{img_path.stem}{img_path.suffix.lower()}"

def copy_image_and_label(src_img: Path, dst_img: Path, dst_label: Path):
    shutil.copy2(src_img, dst_img)

    # single-class full-image box:
    # class x_center y_center width height
    with open(dst_label, "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 1.0 1.0\n")

# =========================
# Create output structure
# =========================
for split in ["train", "val", "test"]:
    prepare_split_dirs(out_root, split)

# =========================
# Build train/val from DATA
# =========================
for class_dir in get_class_dirs(data_dir):
    images = get_images(class_dir)

    if not images:
        print(f"Warning: no images in {class_dir}")
        continue

    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)

    if n >= 2:
        n_train = max(1, min(n_train, n - 1))

    train_files = images[:n_train]
    val_files = images[n_train:]

    for split, files in [("train", train_files), ("val", val_files)]:
        for img_path in files:
            out_name = safe_unique_name(class_dir.name, img_path)
            dst_img = out_root / "images" / split / out_name
            dst_label = out_root / "labels" / split / f"{Path(out_name).stem}.txt"
            copy_image_and_label(img_path, dst_img, dst_label)

    print(f"DATA class {class_dir.name}: total={n}, train={len(train_files)}, val={len(val_files)}")

# =========================
# Build test from TEST
# =========================
for class_dir in get_class_dirs(test_dir):
    images = get_images(class_dir)

    if not images:
        print(f"Warning: no images in {class_dir}")
        continue

    for img_path in images:
        out_name = safe_unique_name(class_dir.name, img_path)
        dst_img = out_root / "images" / "test" / out_name
        dst_label = out_root / "labels" / "test" / f"{Path(out_name).stem}.txt"
        copy_image_and_label(img_path, dst_img, dst_label)

    print(f"TEST class {class_dir.name}: total={len(images)}")

# =========================
# Write detect_data.yaml
# =========================
yaml_text = f"""path: {out_root.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: sign
"""

with open(out_root / "detect_data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_text)

print(f"\\nDone. Single-class detection dataset written to: {out_root.resolve()}")
print(f"YAML file: {out_root / 'detect_data.yaml'}")