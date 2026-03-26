"""
Rename all frames in frames directory to img_{frame#} for easier processing
"""

import os

files = sorted(os.listdir("classify_dataset/val/5"))

for i, f in enumerate(files):
    if f.endswith(".jpg") or f.endswith(".png"):
        new_name = f"val_stop_sign_{i:04d}.jpg"
        os.rename(f"classify_dataset/val/5/{f}", f"classify_dataset/val/5/{new_name}")