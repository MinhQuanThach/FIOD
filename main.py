import torch
import sys

sys.path.append('/content/FIOD/models')
from models.custom_yolo import FogYOLO


def main():
    # Load custom YOLOv8 model with fog-pass filters
    model = FogYOLO('/content/FIOD/models/yolov8_fog.yaml', task='detect')

    # Print model architecture to verify
    print("Model architecture:")
    print(model.model)

    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 1024, 2048).to(model.device)
    with torch.no_grad():
        outputs, fog_factors = model.model(dummy_input)
        print("Detection outputs shape:", [o.shape for o in outputs])
        print("Fog factors shapes:", [f.shape for f in fog_factors])

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