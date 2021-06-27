import tensorflow as tf
import numpy as np
import cv2
import random
import math

from glob import glob
from copy import copy
from concurrent.futures import ThreadPoolExecutor

from utils import convert_to_yolo


class YOLODataGenerator(tf.keras.utils.Sequence):
    """
    :param x_paths
    :param y_paths
    :param target_size: (width, height)
    :param grid_ratio: (grid_width_ratio, grid_height_ratio)
    :param batch_size
    :param shuffle
    """
    def __init__(self,
                 x_paths: str,
                 y_paths: str,
                 num_classes: int,
                 target_size: (int, int) = (224, 224),
                 grid_ratio: (int, int) = (28, 28),
                 batch_size: int = 32,
                 shuffle: bool = True,
                 color: str = "rgb"):
        self.__x_path_ext, self.__y_path_ext = x_paths[x_paths.rindex(".") + 1:], y_paths[y_paths.rindex(".") + 1:]
        self.__x_path_list, self.__y_path_list = glob(x_paths), glob(y_paths)
        self.__x_paths, self.__y_paths = None, None
        self.__num_classes = num_classes
        self.__target_width, self.__target_height = target_size
        self.__grid_ratio = np.asarray(grid_ratio)
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__color = color
        self.on_epoch_end()

    def __len__(self):
        return math.floor(len(self.__x_path_list) / self.__batch_size)

    def __getitem__(self, index):
        batch_x, batch_y = [], []
        executor = ThreadPoolExecutor(max_workers=16)

        futures = []
        for x_path in self.__x_path_list[index * self.__batch_size:(index + 1) * self.__batch_size]:
            futures.append(executor.submit(self.__load, x_path, batch_x, batch_y))
        for future in futures:
            future.result()

        return np.asarray(batch_x), np.asarray(batch_y)

    def on_epoch_end(self):
        if self.__shuffle:
            random.shuffle(self.__x_path_list)
            random.shuffle(self.__y_path_list)
        self.__x_paths, self.__y_paths = copy(self.__x_path_list), copy(self.__y_path_list)

    def __load(self, x_path, batch_x, batch_y):
        # Load image
        img_type = cv2.IMREAD_ANYCOLOR
        if self.__color == "rgb":
            img_type = cv2.IMREAD_COLOR
        elif self.__color == "grayscale":
            img_type = cv2.IMREAD_GRAYSCALE
        img = cv2.imread(x_path, img_type)
        img = cv2.resize(
            src=img,
            dsize=(self.__target_width, self.__target_height),
            interpolation=cv2.INTER_AREA) / 255.
        batch_x.append(img)

        # Create bounding box labels for YOLO
        path = x_path[:x_path.index(self.__x_path_ext) - 1]
        with open(path + "." + self.__y_path_ext, "r") as label_reader:
            bboxes = [label_filename[:-1].split(" ") for label_filename in label_reader.readlines()]
            label_tensor = np.zeros(shape=(self.__grid_ratio[1], self.__grid_ratio[0], 5 + self.__num_classes))

            for bbox in bboxes:
                class_num, x, y, w, h = bbox
                class_num, x, y, w, h = int(class_num), float(x), float(y), float(w), float(h)
                grid_x, grid_y, x, y, w, h = convert_to_yolo(self.__grid_ratio[0], self.__grid_ratio[1], x, y, w, h)
                label_tensor[grid_y, grid_x, 0] = x
                label_tensor[grid_y, grid_x, 1] = y
                label_tensor[grid_y, grid_x, 2] = w
                label_tensor[grid_y, grid_x, 3] = h
                label_tensor[grid_y, grid_x, 4] = 1.
                label_tensor[grid_y, grid_x, 5 + class_num] = 1.

            batch_y.append(label_tensor)


if __name__ == '__main__':
    with open("E:/Dataset/image/coco/classes.txt", "r") as reader:
        classes = [line[:-1] for line in reader.readlines()]
    yolo_data_generator = YOLODataGenerator(
        x_paths="E:/Dataset/image/coco/train2017/*.jpg",
        y_paths="E:/Dataset/image/coco/train2017/*.txt",
        num_classes=len(classes),
        target_size=(416, 416),
        grid_ratio=(52, 52),
        batch_size=256)

    x, y = yolo_data_generator[0]
    print(x.shape, y.shape)