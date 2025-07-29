from ultralytics import YOLO
import torch
import os

# Ensure custom module path is added
import sys

sys.path.append('/FIOD/models')


def main():
    # Load custom YOLOv8 model with fog-pass filters
    model = YOLO('/FIOD/models/yolov8_fog.yaml')

    # Print model architecture to verify
    print(model.model)

    # Train model (basic training, will extend with custom losses later)
    results = model.train(
        data='/content/drive/MyDrive/FIOD_dataset/data/data.yaml',
        epochs=10,
        imgsz=[1024, 2048],
        batch=16,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='/content/drive/MyDrive/FIOD_dataset/runs',
        name='exp',
        exist_ok=True
    )
    print("Training completed!")


if __name__ == '__main__':
    main()
