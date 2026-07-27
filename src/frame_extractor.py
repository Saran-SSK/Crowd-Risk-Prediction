import cv2
import os


class FrameExtractor:
    def __init__(self, video_path, output_folder):
        self.video_path = video_path
        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("Error: Unable to open video.")
            return

        frame_count = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_name = os.path.join(
                self.output_folder,
                f"frame_{frame_count:04d}.jpg"
            )

            cv2.imwrite(frame_name, frame)
            frame_count += 1

        cap.release()

        print(f"Extraction completed.")
        print(f"Total Frames Extracted: {frame_count}")


if __name__ == "__main__":

    video_path = "data/videos/crowd.mp4"
    output_folder = "data/frames"

    extractor = FrameExtractor(video_path, output_folder)
    extractor.extract_frames()