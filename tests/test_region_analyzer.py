import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from density_estimator import DensityEstimator
from region_analyzer import RegionAnalyzer

# Load CSRNet model
print("Loading CSRNet model...")
density = DensityEstimator(
    model_path="models/csrnet/weights.pth"
)

# Test on frame_0000.jpg
frame_path = "data/frames/frame_0000.jpg"
print(f"Processing {frame_path}...\n")

# Get density map and crowd count from CSRNet
density_map, crowd_count = density.predict(frame_path)

print(f"Original CSRNet crowd_count: {crowd_count:.4f}\n")

# Initialize RegionAnalyzer with 3x3 grid
region_analyzer = RegionAnalyzer(rows=3, cols=3)

# Analyze density map by regions
regions = region_analyzer.analyze_density(density_map)

# Print all regions
print("Region-wise analysis (3x3 grid):")
print("=" * 70)
print(f"{'Region ID':<12} {'Row':<5} {'Col':<5} {'Density Count':<15} {'Bbox (r_start, r_end, c_start, c_end)'}")
print("=" * 70)

total_regional_density = 0.0

for region in regions:
    print(f"{region['region_id']:<12} {region['row']:<5} {region['column']:<5} {region['density_count']:<15.4f} {region['bbox']}")
    total_regional_density += region['density_count']

print("=" * 70)
print(f"Sum of all regional density counts: {total_regional_density:.4f}")
print(f"Original CSRNet crowd_count:        {crowd_count:.4f}")
print(f"Difference:                         {abs(total_regional_density - crowd_count):.10f}")
print("=" * 70)

# Verify the sum matches
if abs(total_regional_density - crowd_count) < 0.001:
    print("\n✓ SUCCESS: Regional sum matches CSRNet crowd_count (within tolerance)")
else:
    print("\n✗ WARNING: Regional sum differs from CSRNet crowd_count")
