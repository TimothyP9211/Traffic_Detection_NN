import csv
import os
import cv2
from os import listdir
from ultralytics import YOLO

CLASS_NUM = 51
DETECT_NUM = 1

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
data_set = []
for sign_class in range(CLASS_NUM):
    test_signs = listdir("./data/TEST/{}".format(sign_class))
    for sign_file in test_signs:
        path = "./data/TEST/{}/{}".format(sign_class, sign_file)
        data_set.append((path, sign_class))
        total_predictions += 1

correct_predictions = 0
incorrect_predictions = 0 
for curr_path, true_class in data_set:
    results = final_classifier(curr_path)
    result = results[0]
    pred_idx = result.probs.top1
    pred_label = result.names[pred_idx]
    if pred_label == str(true_class):
        print("Correct prediction")
        correct_predictions += 1
    else: 
        print("Incorrect Prediction:", id_to_name[pred_label])
        print("Confidence:", float(result.probs.top1conf))
        print("True class:", id_to_name[str(true_class)])
        incorrect_predictions += 1

print("=====================================")
print("Total predictions:", total_predictions)
print("Correct predictions:", correct_predictions)
print("Incorrect predictions:", incorrect_predictions)
print("Accuracy:", correct_predictions / total_predictions if total_predictions > 0 else 0)
print("=====================================")

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

# run detector and save images with drawn bounding boxes
for i, path in enumerate(detect_dataset):
    results = final_detector(path, conf=0.25)
    r = results[0]
    annotated = r.plot()

    # save output image with boxes drawn
    out_path = os.path.join("output_detect", f"{i}.png")
    cv2.imwrite(out_path, annotated)
    print("image saved to:", out_path)