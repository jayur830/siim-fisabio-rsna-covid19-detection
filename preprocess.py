import os
import csv
import json
import cv2
import pydicom
import numpy as np

from glob import glob
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut

from concurrent.futures import ThreadPoolExecutor

if __name__ == '__main__':
    root_path = "D:/Dataset/image/covid19"

    if not os.path.exists(root_path + "/imgs/train"):
        os.makedirs(root_path + "/imgs/train")
    if not os.path.exists(root_path + "/imgs/test"):
        os.makedirs(root_path + "/imgs/test")
    categories = ["negative", "typical", "indeterminate", "atypical"]
    for category in categories:
        if not os.path.exists(root_path + "/imgs/train/" + category):
            os.makedirs(root_path + "/imgs/train/" + category)
        if not os.path.exists(root_path + "/imgs/test/" + category):
            os.makedirs(root_path + "/imgs/test/" + category)

    # dcm_list = glob(root_path + "/train/**/*.dcm", recursive=True)
    #
    # executor = ThreadPoolExecutor(max_workers=16)
    # futures = []
    #
    # def load(_filepath):
    #     _filepath = _filepath.replace("\\", "/")
    #
    #     dcm = pydicom.read_file(_filepath)
    #     img = dcm.pixel_array / 65535
    #     img = apply_voi_lut(img, dcm)
    #     cv2.imwrite(_filepath[_filepath.rindex("/"):], img)
    #     cv2.imshow("train", cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA))
    #     cv2.waitKey()
    #
    # for filepath in tqdm(dcm_list):
    #     futures.append(executor.submit(load, filepath))
    # for future in tqdm(futures):
    #     future.result()

    # with open(root_path + "/train_image_level.csv", "r") as image_reader:
    #     image_lines = csv.reader(image_reader)
    #     image_header = next(image_lines)
    #     print(image_header)
    #     for line in image_lines:
    #         filename = line[0][:line[0].index("_image")] + ".dcm"
    #         boxes = None if line[1] == "" else json.loads(line[1].replace("'", "\""))
    #         label = line[2]
    #         instance_id = line[3]
    #         file_list = glob(root_path + "/train/**/" + filename, recursive=True)
    #         if len(file_list) == 0:
    #             continue
    #         for filepath in file_list:
    #             filepath = filepath.replace("\\", "/")
    #             print(filepath)
    #             with open(root_path + "/train_study_level.csv", "r") as study_reader:
    #                 study_lines = csv.reader(study_reader)
    #                 study_header = next(study_lines)
    #                 print(study_header)
    #
    #             # dcm = pydicom.read_file(filepath)
    #             # img = dcm.pixel_array / 65535
    #             # img = apply_voi_lut(img, dcm)
    #             # cv2.imwrite(root_path + "/imgs/train" + filepath[filepath.rindex("/"):], img)
    #         input()

    with open(root_path + "/train_study_level.csv", "r") as study_reader:
        study_lines = csv.reader(study_reader)
        study_header = next(study_lines)
        print(study_header)
        for line in study_lines:
            filepath = line[0][:line[0].index("_study")]
            category = categories[np.asarray(line[1:]).astype("float32").argmax()]
            print(filepath, category)
            print(glob(root_path + "/train/" + filepath + "/**", recursive=True))
            input()
