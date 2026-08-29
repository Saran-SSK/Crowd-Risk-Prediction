import os
import pandas as pd
import matplotlib.pyplot as plt

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
