import torch
import ultralytics
from ultralytics import YOLO

ultralytics.checks()
print(f"Cuda: {torch.cuda.is_available()}")



if __name__ == "__main__":
    #dataset_file = "Data/TinyYoloMulticlass/dataset.yaml"
    dataset_file = "Data/YoloBBox640/data.yaml"

    # pretrained model
    pretrained_model = "yolo11n.pt"
    device = "cuda" # or "cpu"
    img_size = 640
    epochs = 20
    batch_size = 8

    model = YOLO(pretrained_model, task="detect")   # 

    model.train(
        data=dataset_file,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        device=device,
    )