import tensorflow as tf

from classifier import model
from commons import root_path

if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    batch_size, epochs = 256, 1

    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255., validation_split=.2)
    train_gen = generator.flow_from_directory(
        directory=root_path + "/classification",
        color_mode="grayscale",
        batch_size=batch_size)

    classifier = model()
    classifier.fit(
        x=train_gen,
        epochs=epochs)

    classifier.save(filepath="./classifier.h5", include_optimizer=False)
