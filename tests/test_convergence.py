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

# Compute optical flow vectors
flow_x, flow_y, magnitude, avg_motion = flow.compute_flow_vectors(frame1_path, frame2_path)

print(f"Global average motion: {avg_motion:.4f}\n")

# Initialize RegionAnalyzer with 3x3 grid
region_analyzer = RegionAnalyzer(rows=3, cols=3)

# Analyze convergence by regions
convergence_regions = region_analyzer.analyze_convergence(flow_x, flow_y)

# Find region with highest convergence score
max_convergence_region = max(convergence_regions, key=lambda r: r['convergence_score'])

# Print all regions
print("Region-wise convergence analysis (3x3 grid):")
print("=" * 100)
print(f"{'Region ID':<12} {'Avg Flow X':<12} {'Avg Flow Y':<12} {'Convergence':<12} {'Bbox'}")
print("=" * 100)

for region in convergence_regions:
    print(f"{region['region_id']:<12} {region['average_flow_x']:<12.4f} {region['average_flow_y']:<12.4f} {region['convergence_score']:<12.4f} {region['bbox']}")

print("=" * 100)
print(f"\nRegion with highest convergence: {max_convergence_region['region_id']}")
print(f"Convergence score: {max_convergence_region['convergence_score']:.4f}")
print(f"Average flow: ({max_convergence_region['average_flow_x']:.4f}, {max_convergence_region['average_flow_y']:.4f})")
print()

print("Convergence score calculation explanation:")
print("- For each pixel in a region, calculate the inward direction vector (pixel → region center)")
print("- Normalize both the optical flow vector and the inward direction vector")
print("- Calculate cosine similarity between them: dot product of normalized vectors")
print("- Positive similarity = inward motion, Negative similarity = outward motion")
print("- Convergence score = average of all positive cosine similarities in the region")
print("- Score range: 0.0 (no inward convergence) to 1.0 (strong inward convergence)")
