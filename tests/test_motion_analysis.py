import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from optical_flow import OpticalFlowEstimator
from region_analyzer import RegionAnalyzer

# Initialize OpticalFlowEstimator
print("Initializing OpticalFlowEstimator...")
flow = OpticalFlowEstimator()

# Test on frame_0000.jpg and frame_0001.jpg
frame1_path = "data/frames/frame_0000.jpg"
frame2_path = "data/frames/frame_0001.jpg"
print(f"Processing {frame1_path} and {frame2_path}...\n")

# Compute optical flow
flow_magnitude, avg_motion = flow.compute_flow(frame1_path, frame2_path)

print(f"Global average motion: {avg_motion:.4f}\n")

# Initialize RegionAnalyzer with 3x3 grid
region_analyzer = RegionAnalyzer(rows=3, cols=3)

# Analyze motion by regions
motion_regions = region_analyzer.analyze_motion(flow_magnitude)

# Print all regions
print("Region-wise motion analysis (3x3 grid):")
print("=" * 70)
print(f"{'Region ID':<12} {'Row':<5} {'Col':<5} {'Average Motion':<15} {'Bbox (r_start, r_end, c_start, c_end)'}")
print("=" * 70)

total_regional_motion = 0.0

for region in motion_regions:
    print(f"{region['region_id']:<12} {region['row']:<5} {region['column']:<5} {region['average_motion']:<15.4f} {region['bbox']}")
    total_regional_motion += region['average_motion']

print("=" * 70)
print(f"Average of all regional motions: {total_regional_motion / len(motion_regions):.4f}")
print(f"Global average motion:            {avg_motion:.4f}")
print(f"Difference:                        {abs(total_regional_motion / len(motion_regions) - avg_motion):.10f}")
print("=" * 70)

# Verify the average matches
if abs(total_regional_motion / len(motion_regions) - avg_motion) < 0.01:
    print("\n✓ SUCCESS: Regional average matches global average motion (within tolerance)")
else:
    print("\n✗ WARNING: Regional average differs from global average motion")
