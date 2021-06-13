import numpy as np
import cv2, pydicom
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut
import csv
import json

if __name__ == '__main__':
    # dcm = pydicom.read_file("./test3.dcm")
    # img = dcm.pixel_array
    # img = img / 4095
    # img = apply_voi_lut(img, dcm)

    with open("D:/Dataset/image/covid19/train_image_level.csv", "r") as reader:
        lines = csv.reader(reader)
        head = next(lines)
        print(head)
        for line in lines:
            id, boxes, label, study_instance_uid = line
            boxes = boxes.replace("'", "\"")
            boxes = json.loads(boxes)
            label = [l.split(" ")[0] for l in label.split("opacity ") if l != ""]
            print({
                "name": id,
                "uid": study_instance_uid,
                "bbox": [{
                    "x1": float(box["x"]),
                    "y1": float(box["y"]),
                    "x2": float(box["x"]) + float(box["width"]),
                    "y2": float(box["y"]) + float(box["height"]),
                    "width": float(box["width"]),
                    "height": float(box["height"]),
                    "opacity": int(label[i])
                } for i, box in enumerate(boxes)]
            })
            break
