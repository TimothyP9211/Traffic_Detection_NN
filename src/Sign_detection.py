from ultralytics import YOLO

# Load the pre-trained YOLOv8n classification model
model = YOLO("yolov8n-cls.pt")  

# Train the model using the specified dataset and parameters
model.train(data="dataset_ready", epochs=100, imgsz=224, batch=16, name="yolo_sign_detection", device=0)  
