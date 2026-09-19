import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# ==============================
# Paths
# ==============================

CSV_PATH = "outputs/results.csv"
OUTPUT_FOLDER = "outputs/graphs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==============================
# Load results
# ==============================

df = pd.read_csv(CSV_PATH)

# One row per frame for global statistics.
# The CSV contains 9 regional rows per frame,
# so we keep only the first row for global values.
global_df = df.drop_duplicates(subset=["frame"]).copy()

global_df = global_df.reset_index(drop=True)

global_df["frame_number"] = range(1, len(global_df) + 1)


# ==============================
# Graph 1:
# Global Risk Score
# ==============================

plt.figure(figsize=(10, 5))

plt.plot(
    global_df["frame_number"],
    global_df["global_risk_score"],
    linewidth=2,
    label="Global Risk Score",
)

# Risk thresholds
plt.axhline(y=0.30, linestyle="--", label="LOW / MEDIUM threshold")

plt.axhline(y=0.60, linestyle="--", label="MEDIUM / HIGH threshold")

plt.xlabel("Frame Pair")
plt.ylabel("Risk Score")

plt.title("Global Crowd Risk Score Across Frame Pairs")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


# ==============================
# Save graph
# ==============================

output_path = os.path.join(OUTPUT_FOLDER, "global_risk_score.png")

plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print()
print("Graph generated successfully!")
print(f"Saved to: {output_path}")


# ==============================
# Graph 2:
# Average Regional Risk Score
# ==============================

region_order = [
    "R0_C0",
    "R0_C1",
    "R0_C2",
    "R1_C0",
    "R1_C1",
    "R1_C2",
    "R2_C0",
    "R2_C1",
    "R2_C2",
]

regional_risk = (
    df.groupby("region_id")["risk_score"]
    .mean()
    .reindex(region_order)
)

plt.figure(figsize=(10, 5))

plt.bar(
    regional_risk.index,
    regional_risk.values,
)

plt.axhline(
    y=0.30,
    linestyle="--",
    label="LOW / MEDIUM threshold",
)

plt.axhline(
    y=0.60,
    linestyle="--",
    label="MEDIUM / HIGH threshold",
)

plt.xlabel("Region ID")
plt.ylabel("Average Risk Score")

plt.title("Average Regional Risk Score")

plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

average_regional_risk_path = os.path.join(
    OUTPUT_FOLDER,
    "average_regional_risk.png",
)

plt.savefig(
    average_regional_risk_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print("Graph 2 generated successfully!")
print(f"Saved to: {average_regional_risk_path}")


# ==============================
# Graph 3:
# Regional Risk Component Comparison
# ==============================

component_columns = [
    "density_score",
    "motion_score",
    "density_trend_score",
    "convergence_score",
]

component_labels = [
    "Density Score",
    "Motion Score",
    "Density Trend Score",
    "Convergence Score",
]

regional_components = (
    df.groupby("region_id")[component_columns]
    .mean()
    .reindex(region_order)
)

x_positions = range(len(region_order))
bar_width = 0.2

plt.figure(figsize=(12, 6))

for index, column in enumerate(component_columns):
    offsets = [
        position + (index - 1.5) * bar_width
        for position in x_positions
    ]

    plt.bar(
        offsets,
        regional_components[column],
        width=bar_width,
        label=component_labels[index],
    )

plt.xlabel("Region ID")
plt.ylabel("Component Score")

plt.title("Regional Risk Component Comparison")

plt.xticks(list(x_positions), region_order)
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

regional_components_path = os.path.join(
    OUTPUT_FOLDER,
    "regional_risk_components.png",
)

plt.savefig(
    regional_components_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print("Graph 3 generated successfully!")
print(f"Saved to: {regional_components_path}")


# ==============================
# Graph 4:
# Average Crowd Density by Region
# ==============================

regional_density = (
    df.groupby("region_id")["density_count"]
    .mean()
    .reindex(region_order)
)

highest_density_region = regional_density.idxmax()

bar_colors = [
    "tab:red" if region_id == highest_density_region else "tab:blue"
    for region_id in regional_density.index
]

plt.figure(figsize=(10, 5))

bars = plt.bar(
    regional_density.index,
    regional_density.values,
    color=bar_colors,
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.xlabel("Region ID")
plt.ylabel("Average Crowd Count / Density")

plt.title("Average Crowd Density by Region")

plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

average_regional_density_path = os.path.join(
    OUTPUT_FOLDER,
    "average_regional_density.png",
)

plt.savefig(
    average_regional_density_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print("Graph 4 generated successfully!")
print(f"Saved to: {average_regional_density_path}")


# ==============================
# Graph 5:
# Average Crowd Motion by Region
# ==============================

regional_motion = (
    df.groupby("region_id")["average_motion"]
    .mean()
    .reindex(region_order)
)

highest_motion_region = regional_motion.idxmax()

bar_colors = [
    "tab:red" if region_id == highest_motion_region else "tab:blue"
    for region_id in regional_motion.index
]

plt.figure(figsize=(10, 5))

bars = plt.bar(
    regional_motion.index,
    regional_motion.values,
    color=bar_colors,
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.xlabel("Region ID")
plt.ylabel("Average Motion")

plt.title("Average Crowd Motion by Region")

plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

average_regional_motion_path = os.path.join(
    OUTPUT_FOLDER,
    "average_regional_motion.png",
)

plt.savefig(
    average_regional_motion_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print("Graph 5 generated successfully!")
print(f"Saved to: {average_regional_motion_path}")


# ==============================
# Graph 6:
# Average Crowd Convergence by Region
# ==============================

regional_convergence = (
    df.groupby("region_id")["convergence_score"]
    .mean()
    .reindex(region_order)
)

highest_convergence_region = regional_convergence.idxmax()

bar_colors = [
    "tab:red" if region_id == highest_convergence_region else "tab:blue"
    for region_id in regional_convergence.index
]

plt.figure(figsize=(10, 5))

bars = plt.bar(
    regional_convergence.index,
    regional_convergence.values,
    color=bar_colors,
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.xlabel("Region ID")
plt.ylabel("Average Convergence Score")

plt.title("Average Crowd Convergence by Region")

plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

average_regional_convergence_path = os.path.join(
    OUTPUT_FOLDER,
    "average_regional_convergence.png",
)

plt.savefig(
    average_regional_convergence_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print("Graph 6 generated successfully!")
print(f"Saved to: {average_regional_convergence_path}")


# ==============================
# Average Density Change by Region
# ==============================

density_trend_output_folder = "outputs/visualizations"

os.makedirs(density_trend_output_folder, exist_ok=True)

regional_density_change = (
    df.groupby("region_id")["density_change_ratio"]
    .mean()
    .reindex(region_order)
)

largest_positive_region = regional_density_change.idxmax()
largest_negative_region = regional_density_change.idxmin()

density_change_colors = []
for region_id in regional_density_change.index:
    if region_id == largest_positive_region:
        density_change_colors.append("tab:green")
    elif region_id == largest_negative_region:
        density_change_colors.append("tab:red")
    else:
        density_change_colors.append("tab:blue")

plt.figure(figsize=(10, 5))

bars = plt.bar(
    regional_density_change.index,
    regional_density_change.values,
    color=density_change_colors,
)

plt.axhline(
    y=0,
    linestyle="--",
    color="black",
    linewidth=1,
)

for bar in bars:
    height = bar.get_height()
    text_offset = 3 if height >= 0 else -12

    plt.annotate(
        f"{height:.4f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, text_offset),
        textcoords="offset points",
        ha="center",
        va="bottom" if height >= 0 else "top",
        fontsize=9,
    )

plt.xlabel("Region ID")
plt.ylabel("Average Density Change Ratio")

plt.title("Average Density Change by Region")

plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

regional_density_trend_path = os.path.join(
    density_trend_output_folder,
    "regional_density_trend.png",
)

plt.savefig(
    regional_density_trend_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print("Average density change visualization generated successfully!")
print(f"Saved to: {regional_density_trend_path}")


# ==============================
# NEW VISUALIZATION 1: Risk Distribution
# ==============================

plt.figure(figsize=(10, 6))

# Get all risk scores from the dataframe
all_risk_scores = df["risk_score"].values

# Create histogram
plt.hist(all_risk_scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')

# Add threshold lines (from risk_assessment.py: LOW < 0.30, MEDIUM < 0.60, HIGH >= 0.60)
plt.axvline(x=0.30, color='orange', linestyle='--', linewidth=2, label='LOW/MEDIUM threshold (0.30)')
plt.axvline(x=0.60, color='red', linestyle='--', linewidth=2, label='MEDIUM/HIGH threshold (0.60)')

# Add count annotations
low_count = (all_risk_scores < 0.30).sum()
medium_count = ((all_risk_scores >= 0.30) & (all_risk_scores < 0.60)).sum()
high_count = (all_risk_scores >= 0.60).sum()

plt.text(0.02, 0.95, f'LOW: {low_count}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
plt.text(0.35, 0.95, f'MEDIUM: {medium_count}', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
plt.text(0.70, 0.95, f'HIGH: {high_count}', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.xlabel('Risk Score')
plt.ylabel('Frequency')
plt.title('Distribution of Risk Scores Across All Regions and Frames')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

risk_distribution_path = os.path.join(OUTPUT_FOLDER, "risk_distribution.png")
plt.savefig(risk_distribution_path, dpi=300, bbox_inches="tight")
plt.show()

print()
print("Risk Distribution visualization generated successfully!")
print(f"Saved to: {risk_distribution_path}")


# ==============================
# NEW VISUALIZATION 2: 3×3 Regional Visualization
# ==============================

# Get the most recent frame's regional data for the 3x3 visualization
most_recent_frame = df["frame"].iloc[-1]
recent_frame_data = df[df["frame"] == most_recent_frame].copy()

# Create a 3x3 grid visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Create a 3x3 array of risk scores for heatmap
risk_grid = np.zeros((3, 3))
risk_level_grid = np.empty((3, 3), dtype=object)

for region_data in recent_frame_data.itertuples():
    region_id = region_data.region_id
    row = int(region_id.split("_")[0][1])
    col = int(region_id.split("_")[1][1])
    risk_grid[row, col] = region_data.risk_score
    risk_level_grid[row, col] = region_data.risk_level

# Create colormap based on risk levels
risk_colors = {"LOW": "#2ecc71", "MEDIUM": "#f1c40f", "HIGH": "#e74c3c"}
color_grid = np.array([[risk_colors[risk_level_grid[row, col]] for col in range(3)] for row in range(3)])

# Display the grid
table = ax.table(cellText=[[f"{risk_level_grid[row, col]}\n({risk_grid[row, col]:.3f})" 
                            for col in range(3)] for row in range(3)],
                 rowLabels=["Row 0", "Row 1", "Row 2"],
                 colLabels=["Col 0", "Col 1", "Col 2"],
                 cellLoc='center',
                 loc='center',
                 cellColours=color_grid)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 2)

# Style the table
for i in range(3):
    for j in range(3):
        cell = table[(i+1, j)]
        cell.set_facecolor(color_grid[i, j])
        if risk_level_grid[i, j] == "HIGH":
            cell.set_text_props(weight='bold', color='white')
        elif risk_level_grid[i, j] == "MEDIUM":
            cell.set_text_props(weight='bold', color='black')
        else:
            cell.set_text_props(weight='bold', color='black')

ax.axis('off')
plt.title(f'3×3 Regional Risk Visualization\nFrame: {most_recent_frame}', fontsize=14, fontweight='bold')
plt.tight_layout()

regional_3x3_path = os.path.join(OUTPUT_FOLDER, "regional_3x3_visualization.png")
plt.savefig(regional_3x3_path, dpi=300, bbox_inches="tight")
plt.show()

print()
print("3×3 Regional Visualization generated successfully!")
print(f"Saved to: {regional_3x3_path}")


# ==============================
# NEW VISUALIZATION 3: CSRNet Density-Map Output
# ==============================

# Load the actual density map from the existing output
density_map_path = "outputs/density_maps/frame_0000_density.png"

if os.path.exists(density_map_path):
    density_img = cv2.imread(density_map_path)
    density_img_rgb = cv2.cvtColor(density_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(density_img_rgb)
    plt.title('CSRNet Density Map Output (Frame 0000)', fontsize=14, fontweight='bold')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.colorbar(label='Density Magnitude')
    plt.tight_layout()
    
    csrnet_density_path = os.path.join(OUTPUT_FOLDER, "csrnet_density_map.png")
    plt.savefig(csrnet_density_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    print()
    print("CSRNet Density Map visualization generated successfully!")
    print(f"Saved to: {csrnet_density_path}")
else:
    print()
    print(f"Warning: Density map not found at {density_map_path}")
    print("CSRNet Density Map visualization skipped.")


# ==============================
# NEW VISUALIZATION 4: Farneback Optical-Flow Output
# ==============================

# Load the actual optical flow map from the existing output
flow_map_path = "outputs/flow_maps/frame_0000_flow.png"

if os.path.exists(flow_map_path):
    flow_img = cv2.imread(flow_map_path)
    flow_img_rgb = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(flow_img_rgb)
    plt.title('Farneback Optical Flow Output (Frame 0000)', fontsize=14, fontweight='bold')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.colorbar(label='Flow Magnitude')
    plt.tight_layout()
    
    farneback_flow_path = os.path.join(OUTPUT_FOLDER, "farneback_optical_flow.png")
    plt.savefig(farneback_flow_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    print()
    print("Farneback Optical Flow visualization generated successfully!")
    print(f"Saved to: {farneback_flow_path}")
else:
    print()
    print(f"Warning: Flow map not found at {flow_map_path}")
    print("Farneback Optical Flow visualization skipped.")


# ==============================
# NEW VISUALIZATION 5: Annotated Final Output Frame
# ==============================

# Load an actual final annotated frame from the existing output
final_frame_path = "outputs/final_frames/frame_0000.jpg"

if os.path.exists(final_frame_path):
    final_frame = cv2.imread(final_frame_path)
    final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(14, 10))
    plt.imshow(final_frame_rgb)
    plt.title('Annotated Final Output Frame (Frame 0000)', fontsize=14, fontweight='bold')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.tight_layout()
    
    annotated_frame_path = os.path.join(OUTPUT_FOLDER, "annotated_final_output.png")
    plt.savefig(annotated_frame_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    print()
    print("Annotated Final Output Frame visualization generated successfully!")
    print(f"Saved to: {annotated_frame_path}")
else:
    print()
    print(f"Warning: Final frame not found at {final_frame_path}")
    print("Annotated Final Output Frame visualization skipped.")
