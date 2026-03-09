import csv
from ultralytics import YOLO

# classifier and detector training parameters
CLASS_EPOCHS = 30
CLASS_IMGSZ = 256
DETECT_EPOCHS = 40
DETECT_IMGSZ = 640

# load the pre-trained YOLOv8n classification model
classifier = YOLO("yolov8n-cls.pt")  
detector = YOLO("yolov8n.pt")

# train the model using the specified dataset and parameters
classifier.train(data="classify_dataset", epochs=CLASS_EPOCHS, imgsz=CLASS_IMGSZ)  
detector.train(data="traffic_sign_single_detect/data.yaml", epochs=DETECT_EPOCHS, imgsz=DETECT_IMGSZ)

# evaluate the model on the test set
metrics = classifier.val(data="classify_dataset", split="test")
print(metrics)

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
final_detector = YOLO("runs/detect/train/weights/best.pt")
results = final_detector("gostraight_21_test_sign.png", conf=0.25)
for r in results:
    print("Boxes:", r.boxes.xyxy)
    print("Confidences:", r.boxes.conf)
    print("Classes:", r.boxes.cls)