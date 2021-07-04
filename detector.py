import tensorflow as tf

from yolo_loss import YOLOLoss
from yolo_output_layer import YOLOOutput


def model(anchors, num_classes):
    # (416, 416, 1)
    input_layer = tf.keras.layers.Input(shape=(416, 416, 1))
    # (416, 416, 1) -> (208, 208, 8)
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (208, 208, 8) -> (104, 104, 16)
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (104, 104, 16) -> (52, 52, 32)
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (52, 52, 32) -> (26, 26, 64)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (26, 26, 64) -> (13, 13, 128)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (13, 13, 128) -> (13, 13, 5 * anchors + num_classes)
    x = tf.keras.layers.Conv2D(
        filters=5 * len(anchors) + num_classes,
        kernel_size=1,
        kernel_initializer="he_normal")(x)
    x = YOLOOutput(num_anchors=len(anchors))(x)

    m = tf.keras.models.Model(input_layer, x)
    m.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-2),
        loss=YOLOLoss(anchors))
    m.summary()

    return m
