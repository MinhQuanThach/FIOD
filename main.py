
from ultralytics import YOLO
import torch
import os

def main():
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use yolov8n for simplicity

    # Train model
    results = model.train(
        data='/content/drive/MyDrive/FIOD_dataset/data/data.yaml',
        epochs=1,  # Small number for testing
        imgsz=640,
        batch=16,  # Adjust based on Colab GPU
        device=0 if torch.cuda.is_available() else 'cpu',
        project='/content/drive/MyDrive/FIOD_dataset/runs',
        name='exp',
        exist_ok=True
    )
    print("Training completed!")

if __name__ == '__main__':
    main()
