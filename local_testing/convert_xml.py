import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split

# ================= PATH =================
IMG_DIR = "images"
XML_DIR = "annotations"

OUT_IMG_TRAIN = "dataset/images/train"
OUT_IMG_VAL   = "dataset/images/val"
OUT_LBL_TRAIN = "dataset/labels/train"
OUT_LBL_VAL   = "dataset/labels/val"

for p in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LBL_TRAIN, OUT_LBL_VAL]:
    os.makedirs(p, exist_ok=True)

# ================= CLASS =================
CLASS_NAME = "With Helmet"
CLASS_ID = 0

# ================= READ XML =================
xml_files = [f for f in os.listdir(XML_DIR) if f.endswith(".xml")]
train_xmls, val_xmls = train_test_split(xml_files, test_size=0.2, random_state=42)

def clamp(val, minv, maxv):
    return max(minv, min(val, maxv))

def convert(xml_file, img_out, lbl_out):
    xml_path = os.path.join(XML_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    img_path = os.path.join(IMG_DIR, filename)

    if not os.path.exists(img_path):
        return  

    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name != CLASS_NAME:
            continue

        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        # ===== CLAMP BBOX =====
        xmin = clamp(xmin, 0, w - 1)
        xmax = clamp(xmax, 0, w - 1)
        ymin = clamp(ymin, 0, h - 1)
        ymax = clamp(ymax, 0, h - 1)

        if xmax <= xmin or ymax <= ymin:
            continue  

        # ===== YOLO FORMAT =====
        x_c = ((xmin + xmax) / 2) / w
        y_c = ((ymin + ymax) / 2) / h
        bw  = (xmax - xmin) / w
        bh  = (ymax - ymin) / h

        # ===== CHECK NORMALIZED =====
        if not (0 < x_c < 1 and 0 < y_c < 1 and 0 < bw <= 1 and 0 < bh <= 1):
            continue

        yolo_lines.append(
            f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"
        )

    if len(yolo_lines) == 0:
        return  

    shutil.copy(img_path, img_out)

    txt_name = os.path.splitext(filename)[0] + ".txt"
    with open(os.path.join(lbl_out, txt_name), "w") as f:
        f.write("\n".join(yolo_lines))

# ================= RUN =================
for xml in train_xmls:
    convert(xml, OUT_IMG_TRAIN, OUT_LBL_TRAIN)

for xml in val_xmls:
    convert(xml, OUT_IMG_VAL, OUT_LBL_VAL)

print("Convert XML Done")
