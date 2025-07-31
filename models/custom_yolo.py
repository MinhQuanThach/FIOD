import torch
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.tasks import DetectionModel
from models.fog_pass_filter import FogPassFilter


class FogYOLO(YOLO):
    def __init__(self, model='yolov8_fog.yaml', task='detect'):
        super().__init__(model, task)
        self.fog_factors = []  # Store fog factors during forward pass

    def _initialize_model(self, cfg):
        # Override to use custom DetectionModel with fog-pass filters
        self.model = FogDetectionModel(cfg, nc=self.nc)
        return self.model


class FogDetectionModel(DetectionModel):
    def __init__(self, cfg='yolov8_fog.yaml', nc=None, verbose=True):
        super().__init__(cfg, nc, verbose)

    def forward(self, x, *args, **kwargs):
        """
        Forward pass with fog-pass filter outputs collected as side branches.

        Args:
            x (torch.Tensor): Input tensor (batch_size, channels, height, width).

        Returns:
            tuple: (detection_outputs, fog_factors)
        """
        self.fog_factors = []  # Reset fog factors
        outputs = []
        for i, module in enumerate(self.model):
            if isinstance(module, FogPassFilter):
                # Compute fog factors as side branch
                fog_factor = module(outputs[i - 1])
                self.fog_factors.append(fog_factor)
                outputs.append(outputs[i - 1])  # Pass through previous layer's output
            else:
                x = module(x if i == 0 else outputs[-1])
                outputs.append(x)

        # Return detection outputs and fog factors
        detection_outputs = outputs[-1]
        return detection_outputs, self.fog_factors