from tensorflow.keras.preprocessing.image import ImageDataGenerator

from classifier import model

if __name__ == '__main__':
    batch_size, epochs = 256, 100

    generator = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=.2,
        height_shift_range=.2,
        brightness_range=.2,
        shear_range=.2,
        zoom_range=.2,
        rescale=1. / 255.)
    train_gen = generator.flow_from_directory(
        directory="",
        color_mode="grayscale",
        batch_size=batch_size)

    classifier = model()
    classifier.fit(

        batch_size=batch_size,
        epochs=epochs,
        validation_split=.2)
