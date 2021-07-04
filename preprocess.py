import csv
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from shutil import copyfile

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm

from commons import root_path, study_classes, image_classes

study_path_map, image_path_map, dst_path_map = {}, {}, {}


def save_classification_data():
    with open(root_path + "/train_study_level.csv", "r") as study_reader:
        lines = csv.reader(study_reader)
        next(lines)
        lines = [line for line in lines]

        def load(line):
            key = line[0].replace("_study", "")
            class_index = np.asarray(line[1:]).astype("float32").argmax()
            src_path = study_path_map[key]
            dst_path = f"{root_path}/classification/{study_classes[class_index]}/{src_path.split('/')[-1][:-4]}.jpg"
            dst_path_map[key] = dst_path

            dcm = pydicom.read_file(src_path)
            img = dcm.pixel_array / np.max(np.asarray(dcm.pixel_array))
            img = np.asarray(apply_voi_lut(img, dcm) * 255.)
            cv2.imwrite(dst_path, img)

        futures = []
        thread_pool_executor = ThreadPoolExecutor(max_workers=16)

        for line in lines:
            futures.append(thread_pool_executor.submit(load, line))
        for future in tqdm(futures):
            future.result()


def save_detection_data():
    futures = []
    thread_pool_executor = ThreadPoolExecutor(max_workers=16)

    classification_data_path = glob(root_path + "/classification/**/*.jpg")

    def file_load(path):
        path = path.replace("\\", "/")
        if not os.path.exists(f"{root_path}/detection/{path.split('/')[-1]}"):
            copyfile(path, f"{root_path}/detection/{path.split('/')[-1]}")

    for path in classification_data_path:
        futures.append(thread_pool_executor.submit(file_load, path))
    for future in tqdm(futures):
        future.result()
    futures = []

    with open(root_path + "/train_image_level.csv", "r") as image_reader:
        lines = csv.reader(image_reader)
        next(lines)
        lines = [line for line in lines]

        def load(line):
            filename = line[0].replace("_image", "")

            dcm = pydicom.read_file(image_path_map[filename])
            img = dcm.pixel_array / np.max(np.asarray(dcm.pixel_array))
            img = np.asarray(apply_voi_lut(img, dcm) * 255.)
            if filename not in dst_path_map or not os.path.exists(dst_path_map[filename]):
                cv2.imwrite(f"{root_path}/detection/{filename}.jpg", img)

            label = line[2].split(" ")
            detection_label_lines = []
            for i in range(0, len(label), 6):
                yolo_class_label, _, x1, y1, x2, y2 = label[i:i + 6]
                class_index = image_classes.index(yolo_class_label)
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                detection_label_lines.append(f"{class_index} {x1 / img.shape[1]} {y1 / img.shape[0]} {x2 / img.shape[1]} {y2 / img.shape[0]}\n")

            with open(f"{root_path}/detection/{filename}.txt", "w") as detection_label_writer:
                detection_label_writer.writelines(detection_label_lines)

        for line in lines:
            futures.append(thread_pool_executor.submit(load, line))
        for future in tqdm(futures):
            future.result()


def preprocess():
    if not os.path.exists(root_path + "/classification"):
        os.makedirs(root_path + "/classification")
    if not os.path.exists(root_path + "/detection"):
        os.makedirs(root_path + "/detection")

    for category in study_classes:
        if not os.path.exists(root_path + "/classification/" + category):
            os.makedirs(root_path + "/classification/" + category)

    paths = glob(f"{root_path}/train/*/*/*.dcm")
    paths = [path.replace("\\", "/") for path in paths]
    for path in paths:
        study_path_map[path.split("/")[-3]] = path
        image_path_map[path.split("/")[-1][:-4]] = path

    """
    Save to directory for classification
    """
    save_classification_data()

    """
    Save to directory for object detection
    """
    save_detection_data()


if __name__ == '__main__':
    preprocess()
