import csv
from ultralytics import YOLO

# load the pre-trained YOLOv8n classification model
model = YOLO("yolov8n-cls.pt")  

# train the model using the specified dataset and parameters
model.train(data="classify_dataset", epochs=30, imgsz=256)  

# evaluate the model on the test set
metrics = model.val(data="classify_dataset", split="test")
print(metrics)

# ============== #
# TESTING #
# ============== #

# load sign class names from CSV file
id_to_name = {}
with open("classify_dataset/ds1_labels.csv", newline='', encoding='utf8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        id_to_name[str(row["ClassId"])] = row["Name"]

# test the model on a sample image
final_model = YOLO("runs/classify/train/weights/best.pt")
results = final_model("gostraight_21_test_sign.png")
result = results[0]

pred_idx = result.probs.top1
pred_label = result.names[pred_idx]
print("Predicted:", id_to_name[pred_label])
print("Confidence:", float(result.probs.top1conf))