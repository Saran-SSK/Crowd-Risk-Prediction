import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from optical_flow import OpticalFlowEstimator

estimator = OpticalFlowEstimator()

magnitude, avg_motion = estimator.compute_flow(
    "data/frames/frame_0000.jpg",
    "data/frames/frame_0001.jpg",
)

print(f"Average Motion: {avg_motion:.4f}")

os.makedirs("outputs/flow_maps", exist_ok=True)

plt.figure(figsize=(8, 6))
plt.imshow(magnitude, cmap="inferno")
plt.colorbar(label="Motion Magnitude")
plt.title(f"Average Motion: {avg_motion:.4f}")

output_path = "outputs/flow_maps/frame_0000_flow.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Flow map saved to: {output_path}")
