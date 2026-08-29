import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath("."))

from models.csrnet.model import CSRNet


class DensityEstimator:

    def __init__(self, model_path, device="cpu"):

        self.device = torch.device(device)

        self.model = CSRNet(load_weights=True)

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print("CSRNet model loaded successfully!")

    def create_heatmap(self, density_map):
        """
        Convert CSRNet density map into a cleaner colored heatmap.
        """

        # Normalize to 0-255
        normalized = cv2.normalize(
            density_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        normalized = normalized.astype(np.uint8)

        # Remove weak density values (noise)
        _, normalized = cv2.threshold(
            normalized,
            40,              # Try values between 30-60 if needed
            255,
            cv2.THRESH_TOZERO
        )

        # Smooth the heatmap slightly
        normalized = cv2.GaussianBlur(
            normalized,
            (7, 7),
            0
        )

        # Apply color map
        heatmap = cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET
        )

        return heatmap

    def predict(self, image_path):
        """
        Predict crowd density and count.
        """

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image = Image.open(image_path).convert("RGB")

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            density_map = self.model(image_tensor)

        crowd_count = density_map.sum().item()

        return density_map.squeeze().cpu().numpy(), crowd_count