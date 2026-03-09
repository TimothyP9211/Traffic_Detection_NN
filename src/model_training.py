from ultralytics import YOLO

# classifier and detector training parameters
CLASS_EPOCHS = 30
CLASS_IMGSZ = 256
DETECT_EPOCHS = 20
DETECT_IMGSZ = 640

# load the pre-trained YOLOv8n classification model
classifier = YOLO("yolov8n-cls.pt")  
detector = YOLO("yolov8n.pt")

# train the model using the specified dataset and parameters
# classifier.train(data="classify_dataset", epochs=CLASS_EPOCHS, imgsz=CLASS_IMGSZ)  
detector.train(data="detect_dataset/detect_data.yaml", epochs=DETECT_EPOCHS, imgsz=DETECT_IMGSZ)

# evaluate the model on the test set
classifier_metrics = classifier.val(data="classify_dataset", split="test")
detector_metrics = detector.val(data="detect_dataset/detect_data.yaml", split="test")
print(classifier_metrics)
print(detector_metrics)