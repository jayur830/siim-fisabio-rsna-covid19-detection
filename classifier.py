import tensorflow as tf


def model():
    # (256, 256, 1)
    input_layer = tf.keras.layers.Input(shape=(256, 256, 1))
    # (256, 256, 1) -> (128, 128, 8)
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (128, 128, 8) -> (64, 64, 16)
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (64, 64, 16) -> (32, 32, 32)
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (32, 32, 32) -> (16, 16, 64)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # (16, 16, 64) -> (8, 8, 128)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        strides=2,
        kernel_initializer="he_normal",
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=1e-2)(x)
    # Global Average Pooling
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    # (128,) -> (4,)
    x = tf.keras.layers.Dense(
        units=4,
        kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Softmax()(x)

    m = tf.keras.models.Model(input_layer, x)
    m.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-2),
        loss=tf.losses.categorical_crossentropy,
        metrics=["acc"])
    m.summary()

    return m
