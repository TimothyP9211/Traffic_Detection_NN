import csv
import cv2
import os
import time
import numpy as np
from os import listdir
from ultralytics import YOLO

CLASS_NUM = 6

# ================ #
# TESTING DETECTOR 
# ================ #

# test the detection model on a sample image (change train to the appropriate run)
final_detector = YOLO("runs/detect/train12/weights/best.pt")

dataset_path = "./detect_dataset/images/test"
detect_dataset = []
for fname in listdir(dataset_path):
    path = os.path.join(dataset_path, fname)
    if os.path.isfile(path):
        detect_dataset.append(path)

# plot results for detect dataset test images
for i, path in enumerate(detect_dataset):
    results = final_detector(path, conf=0.25)
    r = results[0]
    annotated = r.plot()

    # save output images with bounding boxes drawn
    out_path = os.path.join("output_detect", f"{i}.png")
    cv2.imwrite(out_path, annotated)
    print("image saved to:", out_path)