import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from visualization import Visualizer

image = cv2.imread("data/frames/frame_0000.jpg")

visualizer = Visualizer()

result = visualizer.draw_information(
    image,
    crowd_count=1432.05,
    motion=0.2234,
    risk_score=0.5347,
    risk_level="MEDIUM",
)

cv2.imwrite(
    "outputs/frame_with_information.jpg",
    result,
)

print("Visualization saved successfully!")
