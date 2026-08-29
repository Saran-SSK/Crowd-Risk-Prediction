import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from density_estimator import DensityEstimator

estimator = DensityEstimator(
    model_path="models/csrnet/weights.pth"
)

density_map, crowd_count = estimator.predict(
    "data/frames/frame_0000.jpg"
)

print(f"Estimated Crowd Count: {crowd_count:.2f}")
print(f"Density Map Shape: {density_map.shape}")

# Create output directory
os.makedirs("outputs/density_maps", exist_ok=True)

# Save density map
plt.figure(figsize=(8, 6))
plt.imshow(density_map, cmap="jet")
plt.colorbar(label="Density")
plt.title(f"Estimated Count: {crowd_count:.2f}")

output_path = "outputs/density_maps/frame_0000_density.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Density map saved to: {output_path}")
