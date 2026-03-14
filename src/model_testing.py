import csv
import cv2
import os
import time
import numpy as np
from os import listdir
from ultralytics import YOLO

CLASS_NUM = 51

# ================== #
# TESTING CLASSIFIER
# ================== #

# load sign class names from CSV file
id_to_name = {}
with open("classify_dataset/ds1_labels.csv", newline='', encoding='utf8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        id_to_name[str(row["ClassId"])] = row["Name"]

# test the classification model on a sample image
final_classifier = YOLO("runs/classify/train/weights/best.pt")

total_predictions = 0
class_data_set = []
for sign_class in range(CLASS_NUM):
    test_signs = listdir("./data/TEST/{}".format(sign_class))
    for sign_file in test_signs:
        path = "./data/TEST/{}/{}".format(sign_class, sign_file)
        class_data_set.append((path, sign_class))
        total_predictions += 1

# Per-class tracking
per_class = {}
for c in range(CLASS_NUM):
    per_class[c] = {'tp': 0, 'fp': 0, 'fn': 0}

correct_predictions = 0
incorrect_predictions = 0
inference_times = []

# Confusion matrix: rows = true, cols = predicted
confusion = np.zeros((CLASS_NUM, CLASS_NUM), dtype=int)

for curr_path, true_class in class_data_set:
    t0 = time.time()
    results = final_classifier(curr_path)
    t1 = time.time()
    inference_times.append(t1 - t0)

    result = results[0]
    pred_idx = result.probs.top1
    pred_label = result.names[pred_idx]
    pred_class = int(pred_label)

    confusion[true_class][pred_class] += 1

    if pred_label == str(true_class):
        correct_predictions += 1
        per_class[true_class]['tp'] += 1
    else:
        print("Incorrect Prediction:", id_to_name[pred_label])
        print("Confidence:", float(result.probs.top1conf))
        print("True class:", id_to_name[str(true_class)])
        incorrect_predictions += 1
        per_class[true_class]['fn'] += 1
        per_class[pred_class]['fp'] += 1

print("\n" + "=" * 60)
print("CLASSIFICATION RESULTS")
print("=" * 60)
print("Total predictions:", total_predictions)
print("Correct predictions:", correct_predictions)
print("Incorrect predictions:", incorrect_predictions)
print("Accuracy:", correct_predictions / total_predictions if total_predictions > 0 else 0)

# Per-class metrics
print(f"\n{'Class':>6} {'Name':<30} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-" * 60)
for c in range(CLASS_NUM):
    tp = per_class[c]['tp']
    fp = per_class[c]['fp']
    fn = per_class[c]['fn']
    if tp + fp + fn == 0:
        continue
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    name = id_to_name.get(str(c), f'class_{c}')
    print(f'{c:>6} {name:<30} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}')

# Timing
if inference_times:
    avg_ms = np.mean(inference_times) * 1000
    p95_ms = np.percentile(inference_times, 95) * 1000
    print(f"\nAvg inference time: {avg_ms:.1f} ms")
    print(f"P95 inference time: {p95_ms:.1f} ms")

# Save confusion matrix
np.savetxt("confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
print("\nConfusion matrix saved to confusion_matrix.csv")

# ================ #
# TESTING DETECTOR 
# ================ #

# test the detection model on a sample image (change train to the appropriate run)
final_detector = YOLO("runs/detect/train11/weights/best.pt")

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