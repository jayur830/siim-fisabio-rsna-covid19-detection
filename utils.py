import numpy as np


def convert_to_yolo(grid_width: int, grid_height: int, x: float, y: float, w: float, h: float):
    grid_x, grid_y = int(x * grid_width), int(y * grid_height)
    x, y = x * grid_width - grid_x, y * grid_height - grid_y
    return grid_x, grid_y, x, y, w, h


def convert_to_real(
        target_width: int,
        target_height: int,
        grid_width: int,
        grid_height: int,
        anchor_width: float,
        anchor_height: float,
        c_x: int,
        c_y: int,
        t_x: float,
        t_y: float,
        t_w: float,
        t_h: float):
    b_x = (t_x + c_x) * target_width / grid_width
    b_y = (t_y + c_y) * target_height / grid_height
    b_w = t_w * target_width * anchor_width / grid_width
    b_h = t_h * target_height * anchor_height / grid_height
    return int(b_x - b_w * .5), int(b_y - b_h * .5), int(b_x + b_w * .5), int(b_y + b_h * .5)


def high_confidence_vector(yolo_tensor: np.ndarray, anchors: [[float]], threshold: float = .5):
    if len(yolo_tensor.shape) != 3:
        return []
    vectors = []
    for h in range(yolo_tensor.shape[0]):
        for w in range(yolo_tensor.shape[1]):
            for n in range(len(anchors)):
                if yolo_tensor[h, w, n * 5 + 4] >= threshold:
                    vector = [w, h] + yolo_tensor[h, w, n * 5:n * 5 + 4].tolist() + [anchors[n][0], anchors[n][1]]
                    if yolo_tensor.shape[-1] > 5 * len(anchors):
                        vector += [yolo_tensor[h, w, 5 * len(anchors):].argmax()]
                    vectors.append(vector)
    return vectors
