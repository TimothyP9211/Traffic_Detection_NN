import os
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom

# directory paths
images_dir = Path("images")
labels_dir = Path("labels")
output_dir = Path("xml_labels")

# class ID to name mapping
class_names = {
    0: "Red max 10",
    1: "Red max 60",
    2: "White max 30",
    3: "White max 50",
    4: "Pedestrian Crosswalk",
    5: "Stop Sign",
}

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x_center = xc * img_w
    y_center = yc * img_h
    box_w = w * img_w
    box_h = h * img_h

    xmin = int(round(x_center - box_w / 2))
    ymin = int(round(y_center - box_h / 2))
    xmax = int(round(x_center + box_w / 2))
    ymax = int(round(y_center + box_h / 2))

    xmin = max(0, min(xmin, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    xmax = max(1, min(xmax, img_w))
    ymax = max(1, min(ymax, img_h))

    return xmin, ymin, xmax, ymax


def prettify_xml(elem):
    rough_string = ET.tostring(elem, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")


def create_voc_xml(image_path, img_w, img_h, img_d, objects):
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = image_path.parent.name

    filename = ET.SubElement(annotation, "filename")
    filename.text = image_path.name

    path_elem = ET.SubElement(annotation, "path")
    path_elem.text = str(image_path.resolve())

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_w)
    height = ET.SubElement(size, "height")
    height.text = str(img_h)
    depth = ET.SubElement(size, "depth")
    depth.text = str(img_d)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")

        name = ET.SubElement(obj_elem, "name")
        name.text = obj["name"]

        pose = ET.SubElement(obj_elem, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj_elem, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj_elem, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj_elem, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(obj["xmin"])
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(obj["ymin"])
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(obj["xmax"])
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(obj["ymax"])

    return annotation

output_dir.mkdir(parents=True, exist_ok=True)

image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts]

converted = 0
skipped = 0

for img_path in image_paths:
    label_path = labels_dir / f"{img_path.stem}.txt"

    if not label_path.exists():
        print(f"Skipping {img_path.name}: no matching label file")
        skipped += 1
        continue

    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size
            img_d = len(img.getbands())
    except Exception as e:
        print(f"Skipping {img_path.name}: cannot open image ({e})")
        skipped += 1
        continue

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Skipping {img_path.name}: cannot read label ({e})")
        skipped += 1
        continue

    objects = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            print(f"Bad label format in {label_path.name}: {line}")
            continue

        try:
            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            print(f"Bad numeric values in {label_path.name}: {line}")
            continue

        xmin, ymin, xmax, ymax = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)

        if xmax <= xmin or ymax <= ymin:
            print(f"Invalid box in {label_path.name}: {line}")
            continue

        class_name = class_names.get(class_id, str(class_id))

        objects.append({
            "name": class_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    annotation = create_voc_xml(img_path, img_w, img_h, img_d, objects)
    xml_str = prettify_xml(annotation)
    out_path = output_dir / f"{img_path.stem}.xml"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    converted += 1
    print(f"Saved: {out_path}")

print(f"\nDone.")
print(f"Converted: {converted}")
print(f"Skipped: {skipped}")