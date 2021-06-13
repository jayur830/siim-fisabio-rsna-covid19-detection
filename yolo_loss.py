import tensorflow as tf

from tensorflow.python.framework.ops import convert_to_tensor_v2


def sum_squared_error(y_true, y_pred, axis=None):
    return tf.reduce_sum(tf.square(y_true - y_pred), axis=axis)


def yolo_loss_iou(y_true, y_pred):
    anchor_width, anchor_height = 5, 4

    confidence_channel = y_true[:, :, :, 4]

    y_true_shape = tf.cast(tf.shape(y_true), dtype="float32")
    num_samples, grid_width, grid_height = y_true_shape[0], y_true_shape[2], y_true_shape[1]
    anchor_width, anchor_height = anchor_width / grid_width, anchor_height / grid_height

    x_range, y_range = \
        tf.range(grid_width, dtype="float32"), \
        tf.transpose(tf.reshape(tf.range(grid_height, dtype="float32"), shape=(1, grid_height, 1)))

    c_x = confidence_channel * (x_range + tf.zeros(shape=(num_samples, grid_height, grid_width)))
    c_y = confidence_channel * (y_range + tf.zeros(shape=(num_samples, grid_height, grid_width)))

    y_true_x = y_true[:, :, :, 0] + c_x
    y_true_y = y_true[:, :, :, 1] + c_y
    y_true_w = y_true[:, :, :, 2] * grid_width
    y_true_h = y_true[:, :, :, 3] * grid_height

    y_true_x1 = y_true_x - y_true_w * .5
    y_true_y1 = y_true_y - y_true_h * .5
    y_true_x2 = y_true_x + y_true_w * .5
    y_true_y2 = y_true_y + y_true_h * .5

    y_pred_x = y_pred[:, :, :, 0] + c_x
    y_pred_y = y_pred[:, :, :, 1] + c_y
    y_pred_w = anchor_width * y_pred[:, :, :, 2] * grid_width
    y_pred_h = anchor_height * y_pred[:, :, :, 3] * grid_height

    y_pred_x1 = y_pred_x - y_pred_w * .5
    y_pred_y1 = y_pred_y - y_pred_h * .5
    y_pred_x2 = y_pred_x + y_pred_w * .5
    y_pred_y2 = y_pred_y + y_pred_h * .5

    min_x2 = tf.minimum(y_true_x2, y_pred_x2)
    max_x1 = tf.maximum(y_true_x1, y_pred_x1)
    min_y2 = tf.minimum(y_true_y2, y_pred_y2)
    max_y1 = tf.maximum(y_true_y1, y_pred_y1)

    intersection_width = tf.maximum(min_x2 - max_x1, 0)
    intersection_height = tf.maximum(min_y2 - max_y1, 0)
    intersection = intersection_width * intersection_height

    y_true_area = y_true_w * y_true_h
    y_pred_area = y_pred_w * y_pred_h

    return intersection / (y_true_area + y_pred_area - intersection)


class YOLOLoss(tf.keras.losses.Loss):
    def __init__(self,
                 anchors: [[float]],
                 lambda_coord: float = 5.,
                 lambda_noobj: float = .5):
        super(YOLOLoss, self).__init__()
        self.__anchors = anchors
        self.__lambda_coord = lambda_coord
        self.__lambda_noobj = lambda_noobj

    def call(self, y_true, y_pred):
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        return self.__localization_loss(y_true, y_pred) + \
               self.__confidence_loss(y_true, y_pred) + \
               self.__classification_loss(y_true, y_pred)

    def __localization_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        xywh_loss = 0

        for n in range(len(self.__anchors)):
            confidence_channel = y_true[:, :, :, n * 5 + 4]

            xy_loss = self.__lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, n * 5:n * 5 + 2], y_pred[:, :, :, n * 5:n * 5 + 2], axis=-1) * confidence_channel)
            wh_loss = self.__lambda_coord * tf.reduce_sum(sum_squared_error(y_true[:, :, :, n * 5 + 2:n * 5 + 4] ** .5, y_pred[:, :, :, n * 5 + 2:n * 5 + 4] ** .5, axis=-1) * confidence_channel)

            xywh_loss += xy_loss + wh_loss

        return xywh_loss
        # return 0

    def __confidence_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        confidence_loss = 0

        for n in range(len(self.__anchors)):
            confidence_channel = y_true[:, :, :, n * 5 + 4]

            confidence_loss += tf.reduce_sum(
                tf.square(y_true[:, :, :, n * 5 + 4] - y_pred[:, :, :, n * 5 + 4]) * tf.where(
                    tf.cast(confidence_channel, dtype=tf.bool),
                    tf.ones(shape=tf.shape(input=confidence_channel)),
                    tf.ones(shape=tf.shape(input=confidence_channel)) * self.__lambda_noobj))

        return confidence_loss
        # return 0

    def __classification_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        confidence_channel = y_true[:, :, :, 4]

        class_loss = tf.reduce_sum(sum_squared_error(y_true[:, :, :, 5 * len(self.__anchors):], y_pred[:, :, :, 5 * len(self.__anchors):], axis=-1) * confidence_channel)

        return class_loss
        # return 0