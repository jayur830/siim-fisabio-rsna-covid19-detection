from commons import root_path, anchors, image_classes, target_size, grid_ratio
from data_generator import YOLODataGenerator

if __name__ == '__main__':
    train_gen = YOLODataGenerator(
        x_paths=f"{root_path}/detection/*.jpg",
        y_paths=f"{root_path}/detection/*.txt",
        anchors=anchors,
        num_classes=len(image_classes),
        target_size=target_size,
        grid_ratio=grid_ratio,
        batch_size=8,
        color="grayscale")

    for batch, label in train_gen:
        print(batch.shape, label.shape)
        input()
