from ultralytics import YOLO

# classifier and detector training parameters
CLASS_EPOCHS = 30
CLASS_IMGSZ = 256
DETECT_EPOCHS = 50
DETECT_IMGSZ = 640

# load the pre-trained YOLOv8n classification model
classifier = YOLO("yolov8n-cls.pt")
detector = YOLO("yolov8n.pt")

# # train the classifier with built-in augmentation
# classifier.train(
#     data="Data/classify_dataset",
#     epochs=CLASS_EPOCHS,
#     imgsz=CLASS_IMGSZ,
#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,
#     translate=0.1,
#     scale=0.5,
#     erasing=0.3,
# )

# train the detector with augmentation
detector.train(
    data="Data/detect_dataset/detect_data.yaml",
    epochs=DETECT_EPOCHS,
    imgsz=DETECT_IMGSZ,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    mosaic=1.0,
    flipud=0.5,
    fliplr=0.5,
)


# evaluate the model on the test set
classifier_metrics = classifier.val(data="classify_dataset", split="test")
print(classifier_metrics)

detector_metrics = detector.val(data="detect_dataset/detect_data.yaml", split="test")
print(detector_metrics)