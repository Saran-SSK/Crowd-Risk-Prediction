import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from video_generator import VideoGenerator

generator = VideoGenerator()

generator.create_video(
    frames_folder="outputs/final_frames",
    output_video="outputs/crowd_risk_output.mp4",
    fps=25
)
