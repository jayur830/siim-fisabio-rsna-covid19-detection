import tensorflow as tf


class YOLOOutput(tf.keras.layers.Layer):
    def __init__(self, num_anchors, name="YOLOOutput", **kwargs):
        super(YOLOOutput, self).__init__(name=name)
        self.__num_anchors = num_anchors
        super(YOLOOutput, self).__init__(**kwargs)

    def get_config(self):
        config = super(YOLOOutput, self).get_config()
        config.update({
            "name": "YOLOOutput",
            "trainable": True,
            "dynamic": False,
            "num_anchors": self.__num_anchors
        })
        return config

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, **kwargs):
        concat_x = []
        for n in range(self.__num_anchors):
            concat_x.append(tf.sigmoid(inputs[:, :, :, n * 5:n * 5 + 2]))
            concat_x.append(tf.exp(inputs[:, :, :, n * 5 + 2:n * 5 + 4]))
            concat_x.append(tf.sigmoid(inputs[:, :, :, n * 5 + 4:n * 5 + 5]))
        concat_x.append(tf.sigmoid(inputs[:, :, :, 5 * self.__num_anchors:]))
        return tf.concat(concat_x, axis=-1)
