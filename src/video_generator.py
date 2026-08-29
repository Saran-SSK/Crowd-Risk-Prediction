import cv2
import os


class VideoGenerator:

    def create_video(
        self,
        frames_folder,
        output_video,
        fps=25
    ):

        frame_files = sorted([
            f for f in os.listdir(frames_folder)
            if f.endswith(".jpg")
        ])

        if not frame_files:
            print("No frames found!")
            return

        first_frame = cv2.imread(
            os.path.join(frames_folder, frame_files[0])
        )

        height, width, _ = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_video,
            fourcc,
            fps,
            (width, height)
        )

        for frame_name in frame_files:

            frame = cv2.imread(
                os.path.join(frames_folder, frame_name)
            )

            writer.write(frame)

        writer.release()

        print(f"\nVideo saved to: {output_video}")