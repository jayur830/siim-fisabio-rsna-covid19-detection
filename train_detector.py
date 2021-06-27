from data_generator import YOLODataGenerator
from detector import model

if __name__ == '__main__':
    root_path = "D:/Dataset/image/covid19"
    batch_size, epochs = 256, 100

    train_gen = YOLODataGenerator(
        x_paths="E:/Dataset/image/coco/train2017/*.jpg",
        y_paths="E:/Dataset/image/coco/train2017/*.txt",
        num_classes=len(["negative", "typical", "indeterminate", "atypical"]),
        target_size=(416, 416),
        grid_ratio=(52, 52),
        batch_size=batch_size)

    detector = model([[3., 3.], [4., 5.], [5., 4.]])
    detector.fit(
        x=train_gen,
        epochs=epochs,
        validation_split=.2)
