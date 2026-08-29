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

# Test on frame_0000.jpg and frame_0001.jpg
prev_frame_path = "data/frames/frame_0000.jpg"
curr_frame_path = "data/frames/frame_0001.jpg"
print(f"Processing {prev_frame_path} and {curr_frame_path}...\n")

# Get density maps and crowd counts from CSRNet
prev_density_map, prev_crowd_count = density.predict(prev_frame_path)
curr_density_map, curr_crowd_count = density.predict(curr_frame_path)

print(f"Previous frame CSRNet crowd_count: {prev_crowd_count:.4f}")
print(f"Current frame CSRNet crowd_count:  {curr_crowd_count:.4f}\n")

# Initialize RegionAnalyzer with 3x3 grid
region_analyzer = RegionAnalyzer(rows=3, cols=3)

# Analyze density by regions for both frames
prev_regions = region_analyzer.analyze_density(prev_density_map)
curr_regions = region_analyzer.analyze_density(curr_density_map)

# Calculate density changes
density_changes = region_analyzer.analyze_density_change(prev_regions, curr_regions)

# Verify sums
sum_prev_regions = sum(r['density_count'] for r in prev_regions)
sum_curr_regions = sum(r['density_count'] for r in curr_regions)

print("Verification:")
print(f"Sum of previous regional densities: {sum_prev_regions:.4f}")
print(f"Previous CSRNet crowd_count:        {prev_crowd_count:.4f}")
print(f"Difference:                         {abs(sum_prev_regions - prev_crowd_count):.10f}")
print()
print(f"Sum of current regional densities: {sum_curr_regions:.4f}")
print(f"Current CSRNet crowd_count:        {curr_crowd_count:.4f}")
print(f"Difference:                         {abs(sum_curr_regions - curr_crowd_count):.10f}")
print()

# Print all density changes
print("Region-wise density change analysis (3x3 grid):")
print("=" * 100)
print(f"{'Region ID':<12} {'Prev':<10} {'Curr':<10} {'Change':<10} {'Ratio':<12} {'Trend':<10} {'Bbox'}")
print("=" * 100)

for change in density_changes:
    trend = "→" if change['density_change'] == 0 else ("↑" if change['density_change'] > 0 else "↓")
    ratio_str = f"{change['density_change_ratio']:.4f}" if change['density_change_ratio'] != float('inf') else "inf"
    print(f"{change['region_id']:<12} {change['previous_density']:<10.2f} {change['current_density']:<10.2f} {change['density_change']:<10.2f} {ratio_str:<12} {trend:<10} {change['bbox']}")

print("=" * 100)

# Verify the sums match
if abs(sum_prev_regions - prev_crowd_count) < 0.001 and abs(sum_curr_regions - curr_crowd_count) < 0.001:
    print("\n✓ SUCCESS: Regional sums match CSRNet crowd_counts (within tolerance)")
else:
    print("\n✗ WARNING: Regional sums differ from CSRNet crowd_counts")
