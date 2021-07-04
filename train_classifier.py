from tensorflow.keras.preprocessing.image import ImageDataGenerator

from classifier import model
from commons import root_path

if __name__ == '__main__':
    batch_size, epochs = 256, 1

    generator = ImageDataGenerator(rescale=1. / 255., validation_split=.2)
    train_gen = generator.flow_from_directory(
        directory=root_path + "/classification",
        color_mode="grayscale",
        batch_size=batch_size)

    classifier = model()
    classifier.fit(
        x=train_gen,
        epochs=epochs)

    classifier.save(filepath="./classifier.h5", include_optimizer=False)
