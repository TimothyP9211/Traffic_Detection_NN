import csv
from ultralytics import YOLO

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
results = final_classifier("gostraight_21_test_sign.png")
result = results[0]

pred_idx = result.probs.top1
pred_label = result.names[pred_idx]
print("Predicted:", id_to_name[pred_label])
print("Confidence:", float(result.probs.top1conf))

# ================ #
# TESTING DETECTOR 
# ================ #

# test the detection model on a sample image
final_detector = YOLO("runs/detect/train4/weights/best.pt")
results = final_detector("gostraight_21_test_sign.png", conf=0.25)
for i, r in enumerate(results):
    print("Boxes:", r.boxes.xyxy)
    print("Confidences:", r.boxes.conf)
    print("Classes:", r.boxes.cls)

    # draw bounding boxes on the image and save it
    if i == 0:
        file = r.save(filename=str("output_detect/detector_output.png"))
        print("image saved to:", file)