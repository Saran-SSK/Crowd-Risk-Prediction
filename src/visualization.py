import cv2


class Visualizer:

    def draw_information(
        self,
        image,
        crowd_count,
        motion,
        risk_score,
        risk_level
    ):

        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (20, 20),
            (380, 180),
            (40, 40, 40),
            -1
        )

        alpha = 0.6

        image = cv2.addWeighted(
            overlay,
            alpha,
            image,
            1 - alpha,
            0
        )

        color = (0, 255, 0)

        if risk_level == "MEDIUM":
            color = (0, 255, 255)

        elif risk_level == "HIGH":
            color = (0, 0, 255)

        cv2.putText(
            image,
            "Crowd Analytics Dashboard",
            (35, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Crowd Count : {crowd_count:.0f}",
            (35, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.putText(
            image,
            f"Motion : {motion:.3f}",
            (35,115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.putText(
            image,
            f"Risk Score : {risk_score:.3f}",
            (35,145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.putText(
            image,
            f"Status : {risk_level}",
            (35,175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        return image

    def draw_regions(
        self,
        image,
        region_risks,
        density_map_shape,
        max_risk_region_id
    ):
        """
        Draw regional analysis visualization on the image.
        
        This method draws region boundaries, IDs, risk levels, and scores for each region.
        The highest-risk region is highlighted with a thicker boundary.
        
        Args:
            image (np.ndarray): Input image to draw on
            region_risks (list): List of region risk dictionaries from RiskAssessment.calculate_region_risk()
            density_map_shape (tuple): Shape of the density map (height, width) for coordinate scaling
            max_risk_region_id (str): ID of the region with highest risk score
        
        Returns:
            np.ndarray: Image with regional visualization drawn
        """
        density_h, density_w = density_map_shape
        frame_h, frame_w = image.shape[:2]
        scale_x = frame_w / density_w
        scale_y = frame_h / density_h

        # Define colors for risk levels
        risk_colors = {
            "LOW": (0, 255, 0),      # Green
            "MEDIUM": (0, 255, 255), # Yellow/Cyan
            "HIGH": (0, 0, 255)      # Red
        }

        for risk_data in region_risks:
            region_id = risk_data["region_id"]
            risk_level = risk_data["risk_level"]
            risk_score = risk_data["risk_score"]
            
            # Get region bbox (this would need to be passed or calculated)
            # For now, we'll use the density map shape to infer grid positions
            # The region_id format is "R{row}_C{col}"
            row = int(region_id.split("_")[0][1])
            col = int(region_id.split("_")[1][1])
            
            # Calculate region boundaries in density map coordinates
            region_height = density_h // 3
            region_width = density_w // 3
            
            row_start = row * region_height
            row_end = (row + 1) * region_height if row < 2 else density_h
            col_start = col * region_width
            col_end = (col + 1) * region_width if col < 2 else density_w
            
            # Scale to original frame coordinates
            x1 = int(col_start * scale_x)
            y1 = int(row_start * scale_y)
            x2 = int(col_end * scale_x)
            y2 = int(row_end * scale_y)
            
            # Determine boundary thickness
            thickness = 3 if region_id == max_risk_region_id else 1
            
            # Draw region boundary with risk color
            boundary_color = risk_colors.get(risk_level, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), boundary_color, thickness)
            
            # Draw region ID
            text_x = x1 + 10
            text_y = y1 + 25
            cv2.putText(
                image,
                region_id,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw risk level and score
            risk_text = f"{risk_level} ({risk_score:.2f})"
            cv2.putText(
                image,
                risk_text,
                (text_x, text_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                boundary_color,
                2
            )

        # Add regional summary panel
        panel_x = frame_w - 280
        panel_y = 20
        panel_w = 260
        panel_h = 80
        
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (40, 40, 40),
            -1
        )
        
        alpha = 0.6
        image = cv2.addWeighted(
            overlay,
            alpha,
            image,
            1 - alpha,
            0
        )
        
        # Draw regional summary text
        cv2.putText(
            image,
            "Regional Analysis",
            (panel_x + 15, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            image,
            f"Max Risk: {max_risk_region_id}",
            (panel_x + 15, panel_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
        # Find max risk score
        max_score = max(r["risk_score"] for r in region_risks)
        cv2.putText(
            image,
            f"Max Score: {max_score:.3f}",
            (panel_x + 15, panel_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
        return image