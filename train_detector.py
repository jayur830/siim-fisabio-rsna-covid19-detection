import tensorflow as tf

from commons import root_path, image_classes, anchors, target_size, grid_ratio
from data_generator import YOLODataGenerator
from detector import model
from utils import gpu_init

if __name__ == '__main__':
    gpu_init()

    batch_size, epochs = 2, 100

    train_gen = YOLODataGenerator(
        x_paths=f"{root_path}/detection/*.jpg",
        y_paths=f"{root_path}/detection/*.txt",
        anchors=anchors,
        num_classes=len(image_classes),
        target_size=target_size,
        grid_ratio=grid_ratio,
        batch_size=batch_size,
        color="grayscale")

    detector = model(anchors, len(image_classes))
    detector.fit(
        x=train_gen,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath="./detection_checkpoint/detector-{epoch:02d}-{val_loss:.5f}.h5",
                monitor="val_loss",
                save_best_only=True,
                mode="min")
        ])

    detector.save(filepath="./detector.h5", include_optimizer=False)
