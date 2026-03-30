import os
import cv2
import numpy as np

from pathlib import Path
from ultralytics import YOLO
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Bag recording and video output 
BAG_PATH = "/home/tpeng02/sim_ws/src/scripts/demo"
IMAGE_TOPIC = "/camera/color/image_raw"
OUTPUT_MP4 = "YOLO_demo_vid.mp4"

# Detector and Classifier
DETECTOR_WEIGHTS = "/home/tpeng02/sim_ws/src/scripts/models/detect.pt"
CLASSIFIER_WEIGHTS = "/home/tpeng02/sim_ws/src/scripts/models/classify.pt"

# Vid configs
OUTPUT_FPS = 20.0
DETECT_CONF = 0.65 # draw boxes only when confidence is above this
FONT_SCALE = 0.6
BOX_THICKNESS = 2
DRAW_CONFIDENCE = True

# Detector classes
DETECTOR_NAMES = {
    0: "White speed sign",
    1: "Stop sign",
    2: "Pedestrain crossing",
    3: "Parking sign",
    4: "Red speed sign",
}

# Classifier classes
CLASSIFIER_NAMES = {
    0: "Red max 10",
    1: "Red max 60",
    2: "White max 30",
    3: "White max 50",
    4: "Pedestrian Croswalk",
    5: "Stop Sign",
}

# Speed sign classes sent to classifier
SPEED_CLASS_GROUPS = {
    0: {2, 3},  # White speed sign -> White max 30 or White max 50
    4: {0, 1},  # Red speed sign   -> Red max 10 or Red max 60
}

def image_msg_to_bgr(msg) -> np.ndarray:
    """
    Convert sensor message data to OpenCV BGR.
    """
    h = msg.height
    w = msg.width
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "bgr8":
        return data.reshape(h, w, 3).copy()

    if enc == "rgb8":
        img = data.reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if enc == "bgra8":
        img = data.reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if enc == "rgba8":
        img = data.reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")

def clip_box(x1, y1, x2, y2, w, h):
    """
    Clip the demensions of the rectangular box.
    """
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2

def choose_speed_label(det_cls: int, crop_bgr: np.ndarray, classifier: YOLO):
    """
    Run classifier only on speed-sign subset (red/white).
    """
    cls_results = classifier(crop_bgr, verbose=False)
    r = cls_results[0]

    probs = r.probs.data.cpu().numpy()
    allowed = SPEED_CLASS_GROUPS[det_cls]

    best_idx = None
    best_score = -1.0
    for idx in allowed:
        score = float(probs[idx])
        if score > best_score:
            best_score = score
            best_idx = idx

    return CLASSIFIER_NAMES[best_idx], best_score


def draw_box_with_label(img, x1, y1, x2, y2, label, score=None):
    CLASS_COLORS = {
        "Red max 10": (0, 0, 255),              # red
        "Red max 60": (0, 0, 180),              # darker red
        "White max 30": (255, 255, 255),        # white
        "White max 50": (200, 200, 200),        # light gray
        "Pedestrain crossing": (0, 255, 255),   # yellow
        "Pedestrian Croswalk": (0, 255, 255),   # same, for classifier label
        "Stop sign": (255, 0, 0),               # blue
        "Stop Sign": (255, 0, 0),               # same
        "Parking sign": (255, 255, 0),          # cyan
        "White speed sign": (255, 255, 255),    # white
        "Red speed sign": (0, 0, 255),          # red
    }

    color = CLASS_COLORS.get(label, (0, 255, 0))  # default green

    cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    text = label
    if score is not None and DRAW_CONFIDENCE:
        text = f"{label} {score:.2f}"

    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2
    )

    y_top = max(0, y1 - th - baseline - 6)
    y_bot = y_top + th + baseline + 6
    x_right = min(img.shape[1] - 1, x1 + tw + 8)

    cv2.rectangle(img, (x1, y_top), (x_right, y_bot), color, -1)

    # Use black text on bright boxes, white text on dark boxes
    brightness = sum(color)
    text_color = (0, 0, 0) if brightness > 500 else (255, 255, 255)

    cv2.putText(
        img,
        text,
        (x1 + 4, y_bot - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        text_color,
        2,
        cv2.LINE_AA,
    )

def main():
    bag_path = Path(BAG_PATH)
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path not found: {BAG_PATH}")
    if not os.path.exists(DETECTOR_WEIGHTS):
        raise FileNotFoundError(f"Detector weights not found: {DETECTOR_WEIGHTS}")
    if not os.path.exists(CLASSIFIER_WEIGHTS):
        raise FileNotFoundError(f"Classifier weights not found: {CLASSIFIER_WEIGHTS}")

    detector = YOLO(DETECTOR_WEIGHTS)
    classifier = YOLO(CLASSIFIER_WEIGHTS)

    typestore = get_typestore(Stores.ROS2_FOXY)
    writer = None
    frame_count = 0

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == IMAGE_TOPIC]
        if not connections:
            topics = sorted({c.topic for c in reader.connections})
            raise RuntimeError(
                f"Topic not found: {IMAGE_TOPIC}\nAvailable topics:\n" + "\n".join(topics)
            )

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Decode ROS image message
            if connection.msgtype == "sensor_msgs/msg/Image":
                frame = image_msg_to_bgr(msg)
            else:
                raise ValueError(f"Unsupported message type on topic: {connection.msgtype}")

            # print("frame shape:", frame.shape)

            h, w = frame.shape[:2]

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(OUTPUT_MP4, fourcc, OUTPUT_FPS, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video: {OUTPUT_MP4}")

            det_results = detector(frame, conf=DETECT_CONF, verbose=False)
            r = det_results[0]
            print("num boxes:", 0 if r.boxes is None else len(r.boxes))
            annotated = frame.copy()

            # Keep track of the number and positions of drawn boxes
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()

                for box, det_cls, det_conf in zip(boxes, classes, confs):
                    print("raw detection:", det_cls, det_conf, box)
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, w, h)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    final_label = DETECTOR_NAMES.get(det_cls, f"class_{det_cls}")
                    final_score = float(det_conf)

                    if det_cls in SPEED_CLASS_GROUPS:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size != 0:
                            final_label, final_score = choose_speed_label(det_cls, crop, classifier)

                    draw_box_with_label(annotated, x1, y1, x2, y2, final_label, final_score)
            else:
                print("No detections on this frame")

            writer.write(annotated)
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames...")

    if writer is not None:
        writer.release()

    print(f"Done. Saved video to: {OUTPUT_MP4}")
    print(f"Frames processed: {frame_count}")

if __name__ == "__main__":
    main()