import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from density_estimator import DensityEstimator
from optical_flow import OpticalFlowEstimator
from region_analyzer import RegionAnalyzer
from risk_assessment import RiskAssessment

# Initialize components
print("Loading CSRNet model...")
density = DensityEstimator(model_path="models/csrnet/weights.pth")

print("Initializing OpticalFlowEstimator...")
flow = OpticalFlowEstimator()

print("Initializing RegionAnalyzer...")
region_analyzer = RegionAnalyzer(rows=3, cols=3)

print("Initializing RiskAssessment...")
risk = RiskAssessment(max_density=2000, max_motion=2.0)

# Test on frame_0000.jpg and frame_0001.jpg
prev_frame_path = "data/frames/frame_0000.jpg"
curr_frame_path = "data/frames/frame_0001.jpg"
print(f"\nProcessing {prev_frame_path} and {curr_frame_path}...\n")

# Get density maps from CSRNet
prev_density_map, prev_crowd_count = density.predict(prev_frame_path)
curr_density_map, curr_crowd_count = density.predict(curr_frame_path)

print(f"Previous frame CSRNet crowd_count: {prev_crowd_count:.4f}")
print(f"Current frame CSRNet crowd_count:  {curr_crowd_count:.4f}\n")

# Get optical flow vectors
flow_x, flow_y, magnitude, avg_motion = flow.compute_flow_vectors(prev_frame_path, curr_frame_path)
print(f"Global average motion: {avg_motion:.4f}\n")

# Analyze regions for both frames
prev_regions = region_analyzer.analyze_density(prev_density_map)
curr_regions = region_analyzer.analyze_density(curr_density_map)
motion_regions = region_analyzer.analyze_motion(magnitude)
convergence_regions = region_analyzer.analyze_convergence(flow_x, flow_y)

# Calculate density changes
density_changes = region_analyzer.analyze_density_change(prev_regions, curr_regions)

# Fuse signals and calculate region-wise risk
region_risks = []

for i in range(len(prev_regions)):
    # Extract data for this region
    region_id = prev_regions[i]['region_id']
    density_count = curr_regions[i]['density_count']
    average_motion = motion_regions[i]['average_motion']
    density_change_ratio = density_changes[i]['density_change_ratio']
    convergence_score = convergence_regions[i]['convergence_score']
    
    # Calculate region risk
    risk_result = risk.calculate_region_risk(
        density_count=density_count,
        average_motion=average_motion,
        density_change_ratio=density_change_ratio,
        convergence_score=convergence_score,
        region_id=region_id
    )
    
    region_risks.append(risk_result)

# Print region-wise risk table
print("Region-wise Risk Fusion (3x3 grid):")
print("=" * 110)
print(f"{'Region':<10} {'Density':<10} {'Motion':<10} {'Dens Change %':<13} {'Convergence':<12} {'Risk Score':<11} {'Risk Level':<10}")
print("=" * 110)

for risk_result in region_risks:
    density_change_pct = risk_result['density_change_ratio'] * 100
    print(f"{risk_result['region_id']:<10} {risk_result['density_count']:<10.2f} {risk_result['average_motion']:<10.4f} {density_change_pct:<13.2f} {risk_result['convergence_score']:<12.4f} {risk_result['risk_score']:<11.4f} {risk_result['risk_level']:<10}")

print("=" * 110)

# Find regions with highest values
max_risk_region = max(region_risks, key=lambda r: r['risk_score'])
max_density_region = max(region_risks, key=lambda r: r['density_count'])
max_motion_region = max(region_risks, key=lambda r: r['average_motion'])
max_convergence_region = max(region_risks, key=lambda r: r['convergence_score'])

print(f"\nRegion with highest risk score:      {max_risk_region['region_id']} (Score: {max_risk_region['risk_score']:.4f}, Level: {max_risk_region['risk_level']})")
print(f"Region with highest density:         {max_density_region['region_id']} (Density: {max_density_region['density_count']:.2f})")
print(f"Region with highest motion:          {max_motion_region['region_id']} (Motion: {max_motion_region['average_motion']:.4f})")
print(f"Region with highest convergence:     {max_convergence_region['region_id']} (Convergence: {max_convergence_region['convergence_score']:.4f})")

# Verify all risk scores are in [0, 1] range
all_scores_valid = all(0 <= r['risk_score'] <= 1 for r in region_risks)
print(f"\nAll risk scores in [0, 1] range: {all_scores_valid}")

# Print fusion weights explanation
print("\nRisk fusion weights:")
print("  Density (40%):       Regional crowd count normalized by max_density")
print("  Motion (15%):        Average optical flow magnitude normalized by max_motion")
print("  Density Trend (20%): Density change ratio (only increases, clipped to [0,1])")
print("  Convergence (25%):   Inward motion strength (0-1 from cosine similarity)")
print("\nRisk level thresholds:")
print("  Score < 0.30:  LOW")
print("  Score < 0.60:  MEDIUM")
print("  Score >= 0.60: HIGH")
print("\nNote: Risk level is a model-generated crowd-risk indicator, not a stampede prediction.")
