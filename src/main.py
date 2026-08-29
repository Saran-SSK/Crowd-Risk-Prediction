import csv
import os
from collections import Counter

import cv2
import numpy as np

from density_estimator import DensityEstimator
from optical_flow import OpticalFlowEstimator
from region_analyzer import RegionAnalyzer
from risk_assessment import RiskAssessment
from visualization import Visualizer


FRAME_FOLDER = "data/frames"
OUTPUT_FOLDER = "outputs/final_frames"
RESULTS_CSV = "outputs/results.csv"
SUMMARY_TXT = "outputs/summary.txt"

REGION_IDS = [f"R{row}_C{col}" for row in range(3) for col in range(3)]
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]


def mean_value(values):
    return float(np.mean(values)) if values else 0.0


def min_value(values):
    return float(np.min(values)) if values else 0.0


def max_value(values):
    return float(np.max(values)) if values else 0.0


def format_float(value):
    return f"{float(value):.4f}"


def write_results_csv(results):
    fieldnames = [
        "frame",
        "region_id",
        "density_count",
        "average_motion",
        "density_change_ratio",
        "convergence_score",
        "density_score",
        "motion_score",
        "density_trend_score",
        "risk_score",
        "risk_level",
        "global_crowd_count",
        "global_motion",
        "global_risk_score",
        "global_risk_level",
    ]

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def build_summary(
    total_frames,
    frame_pairs_processed,
    global_records,
    regional_history,
    highest_regional_record,
    highest_risk_region_counts,
):
    crowd_counts = [record["crowd_count"] for record in global_records]
    global_motions = [record["average_motion"] for record in global_records]
    global_risks = [record["risk_score"] for record in global_records]
    global_risk_distribution = Counter(
        record["risk_level"] for record in global_records
    )

    region_averages = {}
    for region_id in REGION_IDS:
        records = regional_history[region_id]
        region_averages[region_id] = {
            "density": mean_value([r["density_count"] for r in records]),
            "motion": mean_value([r["average_motion"] for r in records]),
            "density_change": mean_value(
                [r["density_change_ratio"] for r in records]
            ),
            "convergence": mean_value([r["convergence_score"] for r in records]),
            "risk": mean_value([r["risk_score"] for r in records]),
        }

    highest_density_region = max(
        REGION_IDS, key=lambda region_id: region_averages[region_id]["density"]
    )
    highest_motion_region = max(
        REGION_IDS, key=lambda region_id: region_averages[region_id]["motion"]
    )
    highest_convergence_region = max(
        REGION_IDS, key=lambda region_id: region_averages[region_id]["convergence"]
    )
    most_frequent_highest_risk_region = max(
        REGION_IDS, key=lambda region_id: highest_risk_region_counts[region_id]
    )

    lines = [
        "========================================",
        "FULL VIDEO CROWD RISK SUMMARY",
        "========================================",
        "",
        f"Total frames: {total_frames}",
        f"Total frame pairs processed: {frame_pairs_processed}",
        "",
        "GLOBAL STATISTICS",
        "",
        f"Average crowd count: {format_float(mean_value(crowd_counts))}",
        f"Minimum crowd count: {format_float(min_value(crowd_counts))}",
        f"Maximum crowd count: {format_float(max_value(crowd_counts))}",
        "",
        f"Average motion: {format_float(mean_value(global_motions))}",
        f"Minimum motion: {format_float(min_value(global_motions))}",
        f"Maximum motion: {format_float(max_value(global_motions))}",
        "",
        f"Average global risk: {format_float(mean_value(global_risks))}",
        f"Minimum global risk: {format_float(min_value(global_risks))}",
        f"Maximum global risk: {format_float(max_value(global_risks))}",
        "",
        "Global risk distribution:",
        f"LOW: {global_risk_distribution['LOW']}",
        f"MEDIUM: {global_risk_distribution['MEDIUM']}",
        f"HIGH: {global_risk_distribution['HIGH']}",
        "",
        "REGIONAL STATISTICS",
        "",
    ]

    for region_id in REGION_IDS:
        records = regional_history[region_id]
        risk_distribution = Counter(record["risk_level"] for record in records)

        lines.extend(
            [
                region_id,
                "",
                f"Average density: {format_float(region_averages[region_id]['density'])}",
                f"Average motion: {format_float(region_averages[region_id]['motion'])}",
                f"Average density change: {format_float(region_averages[region_id]['density_change'])}",
                f"Average convergence: {format_float(region_averages[region_id]['convergence'])}",
                f"Average risk score: {format_float(region_averages[region_id]['risk'])}",
                "",
                f"LOW count: {risk_distribution['LOW']}",
                f"MEDIUM count: {risk_distribution['MEDIUM']}",
                f"HIGH count: {risk_distribution['HIGH']}",
                "",
            ]
        )

    lines.extend(
        [
            f"Highest-risk region overall: {highest_regional_record['region_id']}",
            f"Frame: {highest_regional_record['frame']}",
            f"Risk score: {format_float(highest_regional_record['risk_score'])}",
            f"Risk level: {highest_regional_record['risk_level']}",
            "",
            f"Highest-density region: {highest_density_region}",
            f"Average density: {format_float(region_averages[highest_density_region]['density'])}",
            "",
            f"Highest-motion region: {highest_motion_region}",
            f"Average motion: {format_float(region_averages[highest_motion_region]['motion'])}",
            "",
            f"Highest-convergence region: {highest_convergence_region}",
            f"Average convergence: {format_float(region_averages[highest_convergence_region]['convergence'])}",
            "",
            f"Most frequently highest-risk region: {most_frequent_highest_risk_region}",
        ]
    )

    return "\n".join(lines), region_averages, global_risk_distribution


def write_summary(summary_text):
    os.makedirs(os.path.dirname(SUMMARY_TXT), exist_ok=True)
    with open(SUMMARY_TXT, "w") as summary_file:
        summary_file.write(summary_text)
        summary_file.write("\n")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    density = DensityEstimator(model_path="models/csrnet/weights.pth")
    flow = OpticalFlowEstimator()
    region_analyzer = RegionAnalyzer(rows=3, cols=3)
    risk = RiskAssessment()
    visualizer = Visualizer()

    frame_files = sorted(
        [
            filename
            for filename in os.listdir(FRAME_FOLDER)
            if filename.lower().endswith(".jpg")
        ]
    )

    if len(frame_files) < 2:
        raise ValueError("At least two frames are required for pair processing.")

    total_frames = len(frame_files)
    total_pairs = min(100, len(frame_files) - 1)

    print(f"Total frames: {total_frames}")
    print(f"Processing {total_pairs} consecutive frame pairs...")
    print()

    all_results = []
    global_records = []
    regional_history = {region_id: [] for region_id in REGION_IDS}
    highest_risk_region_counts = {region_id: 0 for region_id in REGION_IDS}
    highest_regional_record = None

    for i in range(total_pairs):
        frame1_name = frame_files[i]
        frame2_name = frame_files[i + 1]
        frame1_path = os.path.join(FRAME_FOLDER, frame1_name)
        frame2_path = os.path.join(FRAME_FOLDER, frame2_name)

        try:
            if i % 25 == 0 or i == total_pairs - 1:
                print(f"Processing {i + 1}/{total_pairs}")

            image = cv2.imread(frame1_path)
            if image is None:
                raise ValueError(f"Could not read frame: {frame1_path}")

            density_map, crowd_count = density.predict(frame1_path)
            density_map_next, _ = density.predict(frame2_path)

            heatmap = density.create_heatmap(density_map)
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            blended = cv2.addWeighted(image, 0.65, heatmap, 0.35, 0)

            flow_x, flow_y, magnitude, avg_motion = flow.compute_flow_vectors(
                frame1_path,
                frame2_path,
            )

            curr_regions = region_analyzer.analyze_density(density_map)
            next_regions = region_analyzer.analyze_density(density_map_next)
            motion_regions = region_analyzer.analyze_motion(magnitude)
            convergence_regions = region_analyzer.analyze_convergence(flow_x, flow_y)
            density_changes = region_analyzer.analyze_density_change(
                curr_regions,
                next_regions,
            )

            region_risks = []
            for j, curr_region in enumerate(curr_regions):
                risk_result = risk.calculate_region_risk(
                    density_count=curr_region["density_count"],
                    average_motion=motion_regions[j]["average_motion"],
                    density_change_ratio=density_changes[j]["density_change_ratio"],
                    convergence_score=convergence_regions[j]["convergence_score"],
                    region_id=curr_region["region_id"],
                )
                region_risks.append(risk_result)

            max_risk_region = max(region_risks, key=lambda r: r["risk_score"])
            max_risk_region_id = max_risk_region["region_id"]
            highest_risk_region_counts[max_risk_region_id] += 1

            global_risk = risk.calculate_risk(crowd_count, avg_motion)
            global_records.append(
                {
                    "frame": frame1_name,
                    "crowd_count": crowd_count,
                    "average_motion": avg_motion,
                    "risk_score": global_risk["risk_score"],
                    "risk_level": global_risk["risk_level"],
                }
            )

            output = visualizer.draw_information(
                blended,
                crowd_count,
                avg_motion,
                global_risk["risk_score"],
                global_risk["risk_level"],
            )
            output = visualizer.draw_regions(
                output,
                region_risks,
                density_map.shape,
                max_risk_region_id,
            )

            output_name = os.path.join(OUTPUT_FOLDER, frame1_name)
            if not cv2.imwrite(output_name, output):
                raise ValueError(f"Could not write output frame: {output_name}")

            for region_risk in region_risks:
                row = {
                    "frame": frame1_name,
                    "region_id": region_risk["region_id"],
                    "density_count": region_risk["density_count"],
                    "average_motion": region_risk["average_motion"],
                    "density_change_ratio": region_risk["density_change_ratio"],
                    "convergence_score": region_risk["convergence_score"],
                    "density_score": region_risk["density_score"],
                    "motion_score": region_risk["motion_score"],
                    "density_trend_score": region_risk["density_trend_score"],
                    "risk_score": region_risk["risk_score"],
                    "risk_level": region_risk["risk_level"],
                    "global_crowd_count": crowd_count,
                    "global_motion": avg_motion,
                    "global_risk_score": global_risk["risk_score"],
                    "global_risk_level": global_risk["risk_level"],
                }
                all_results.append(row)
                regional_history[region_risk["region_id"]].append(row)

                if (
                    highest_regional_record is None
                    or region_risk["risk_score"] > highest_regional_record["risk_score"]
                ):
                    highest_regional_record = row

            print(
                f"{frame1_name} -> Global: {global_risk['risk_level']} "
                f"({global_risk['risk_score']:.3f}) | Highest Region: "
                f"{max_risk_region_id} ({max_risk_region['risk_score']:.3f})"
            )

        except Exception as exc:
            print("ERROR during frame processing")
            print(f"Frame number: {i}")
            print(f"Filename: {frame1_name} -> {frame2_name}")
            print(f"Exception: {exc}")
            raise

    write_results_csv(all_results)
    summary_text, region_averages, global_risk_distribution = build_summary(
        total_frames,
        total_pairs,
        global_records,
        regional_history,
        highest_regional_record,
        highest_risk_region_counts,
    )
    write_summary(summary_text)

    crowd_counts = [record["crowd_count"] for record in global_records]
    global_motions = [record["average_motion"] for record in global_records]
    global_risks = [record["risk_score"] for record in global_records]

    highest_density_region = max(
        REGION_IDS, key=lambda region_id: region_averages[region_id]["density"]
    )
    highest_motion_region = max(
        REGION_IDS, key=lambda region_id: region_averages[region_id]["motion"]
    )
    highest_convergence_region = max(
        REGION_IDS, key=lambda region_id: region_averages[region_id]["convergence"]
    )
    most_frequent_highest_risk_region = max(
        REGION_IDS, key=lambda region_id: highest_risk_region_counts[region_id]
    )

    print()
    print("========================================")
    print("FULL PROCESSING COMPLETE")
    print("========================================")
    print()
    print(f"Frame pairs processed: {total_pairs}")
    print()
    print(f"Results CSV:\n{RESULTS_CSV}")
    print()
    print(f"Summary:\n{SUMMARY_TXT}")
    print()
    print(f"Output frames:\n{OUTPUT_FOLDER}/")
    print()
    print("Final global statistics:")
    print(f"Average crowd count: {format_float(mean_value(crowd_counts))}")
    print(f"Minimum crowd count: {format_float(min_value(crowd_counts))}")
    print(f"Maximum crowd count: {format_float(max_value(crowd_counts))}")
    print(f"Average motion: {format_float(mean_value(global_motions))}")
    print(f"Minimum motion: {format_float(min_value(global_motions))}")
    print(f"Maximum motion: {format_float(max_value(global_motions))}")
    print(f"Average global risk: {format_float(mean_value(global_risks))}")
    print(f"Minimum global risk: {format_float(min_value(global_risks))}")
    print(f"Maximum global risk: {format_float(max_value(global_risks))}")
    print()
    print("Global risk distribution:")
    print(f"LOW: {global_risk_distribution['LOW']}")
    print(f"MEDIUM: {global_risk_distribution['MEDIUM']}")
    print(f"HIGH: {global_risk_distribution['HIGH']}")
    print()
    print("Final regional statistics:")
    for region_id in REGION_IDS:
        risk_distribution = Counter(
            record["risk_level"] for record in regional_history[region_id]
        )
        print(
            f"{region_id}: "
            f"avg density={region_averages[region_id]['density']:.4f}, "
            f"avg motion={region_averages[region_id]['motion']:.4f}, "
            f"avg density change={region_averages[region_id]['density_change']:.4f}, "
            f"avg convergence={region_averages[region_id]['convergence']:.4f}, "
            f"avg risk={region_averages[region_id]['risk']:.4f}, "
            f"LOW={risk_distribution['LOW']}, "
            f"MEDIUM={risk_distribution['MEDIUM']}, "
            f"HIGH={risk_distribution['HIGH']}"
        )
    print()
    print("Highest-risk frame/region:")
    print(f"Frame: {highest_regional_record['frame']}")
    print(f"Region: {highest_regional_record['region_id']}")
    print(f"Risk score: {format_float(highest_regional_record['risk_score'])}")
    print(f"Risk level: {highest_regional_record['risk_level']}")
    print()
    print(f"Highest-density region: {highest_density_region}")
    print(
        f"Average density: {format_float(region_averages[highest_density_region]['density'])}"
    )
    print(f"Highest-motion region: {highest_motion_region}")
    print(
        f"Average motion: {format_float(region_averages[highest_motion_region]['motion'])}"
    )
    print(f"Highest-convergence region: {highest_convergence_region}")
    print(
        "Average convergence: "
        f"{format_float(region_averages[highest_convergence_region]['convergence'])}"
    )
    print(f"Most frequently highest-risk region: {most_frequent_highest_risk_region}")


if __name__ == "__main__":
    main()
