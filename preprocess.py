import os
import cv2
import pydicom
import numpy as np

from glob import glob
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut

from concurrent.futures import ThreadPoolExecutor

if __name__ == '__main__':
    root_path = "D:/Dataset/image/covid19"

    # categories = ["negative", "typical", "indeterminate", "atypical"]
    # for category in categories:
    #     if os.path.exists(root_path + "/imgs/train/" + category):
    #         os.makedirs(root_path + "/imgs/train/" + category)
    #     if os.path.exists(root_path + "/imgs/test/" + category):
    #         os.makedirs(root_path + "/imgs/test/" + category)

    dcm_list = glob(root_path + "/train/**/*.dcm", recursive=True)

    executor = ThreadPoolExecutor(max_workers=16)
    futures = []

    def load(_filepath):
        dcm = pydicom.read_file(_filepath.replace("\\", "/"))
        img = dcm.pixel_array / 65535
        img = apply_voi_lut(img, dcm)
        cv2.imshow("train", cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA))
        cv2.waitKey()

    for filepath in tqdm(dcm_list):
        futures.append(executor.submit(load, filepath))
    for future in tqdm(futures):
        future.result()
