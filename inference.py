import tensorflow as tf
import numpy as np
import cv2
import pydicom

from glob import glob
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut
from concurrent.futures import ThreadPoolExecutor

from commons import root_path, study_classes, image_classes, anchors, target_size, grid_ratio
from utils import convert_to_real, high_confidence_vector

if __name__ == '__main__':
    paths = glob(f"{root_path}/test/*/*/*.dcm")
    paths = [path.replace("\\", "/") for path in paths]

    classifier = tf.keras.models.load_model("./classifier.h5", compile=False)
    detector = tf.keras.models.load_model("./detector.h5", compile=False)

    inference_results, futures = [], []
    thread_pool_executor = ThreadPoolExecutor(max_workers=16)

    def inference(_filepath):
        dcm = pydicom.read_file(_filepath)
        img = dcm.pixel_array / np.max(np.asarray(dcm.pixel_array))
        img = np.asarray(apply_voi_lut(img, dcm) * 255.)
        img = cv2.resize(src=img, dsize=(256, 256), interpolation=cv2.INTER_AREA)

        inference_results.append(f"{_filepath.split('/')[-3]}_study,{study_classes[np.asarray(classifier(img)).argmax()]} 1 0 0 1 1")

        for vector in high_confidence_vector(np.asarray(detector(img))[0], anchors):
            c_x, c_y, t_x, t_y, t_w, t_h, anchor_width, anchor_height, class_index = vector
            x1, y1, x2, y2 = convert_to_real(
                target_width=target_size[0],
                target_height=target_size[1],
                grid_width=grid_ratio[0],
                grid_height=grid_ratio[1],
                anchor_width=anchor_width,
                anchor_height=anchor_height,
                c_x=c_x,
                c_y=c_y,
                t_x=t_x,
                t_y=t_y,
                t_w=t_w,
                t_h=t_h)
            inference_results.append(f"{image_classes[class_index]} {x1} {y1} {x2} {y2}")

    for filepath in paths:
        futures.append(thread_pool_executor.submit(inference, filepath))
    for future in tqdm(futures):
        future.result()

    sorted(inference_results, key=lambda result: result[0][result[0].find("_study"):] if str(result[0]).find("_study") != -1 else result[0][result[0].find("_image"):])

    with open("submission.csv", "w") as submission_writer:
        submission_writer.writelines(inference_results)
