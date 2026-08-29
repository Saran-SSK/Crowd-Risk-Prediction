import cv2
import numpy as np


class OpticalFlowEstimator:
    def __init__(self):
        pass

    def compute_flow(self, frame1_path, frame2_path):
        # Read images
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)

        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute Farneback Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )

        average_motion = np.mean(magnitude)

        return magnitude, average_motion

    def compute_flow_vectors(self, frame1_path, frame2_path):
        """
        Compute optical flow and return the flow vectors (x and y components).
        
        This method returns the full optical flow vectors, which are needed for
        convergence analysis to determine the direction of motion.
        
        Args:
            frame1_path (str): Path to the first frame
            frame2_path (str): Path to the second frame
        
        Returns:
            tuple: (flow_x, flow_y, magnitude, average_motion)
                   - flow_x (np.ndarray): Horizontal flow component (H, W)
                   - flow_y (np.ndarray): Vertical flow component (H, W)
                   - magnitude (np.ndarray): Flow magnitude (H, W)
                   - average_motion (float): Average flow magnitude across the frame
        """
        # Read images
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)

        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute Farneback Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Extract flow components
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        magnitude, angle = cv2.cartToPolar(
            flow_x,
            flow_y
        )

        average_motion = np.mean(magnitude)

        return flow_x, flow_y, magnitude, average_motion