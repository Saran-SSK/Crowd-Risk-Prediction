class RiskAssessment:

    def __init__(
        self,
        max_density=2000,
        max_motion=2.0
    ):

        self.max_density = max_density
        self.max_motion = max_motion

    def calculate_risk(self, crowd_count, average_motion):

        # Normalize values
        density_score = min(crowd_count / self.max_density, 1.0)
        motion_score = min(average_motion / self.max_motion, 1.0)

        # Dynamic weighted score
        risk_score = (
            0.75 * density_score +
            0.25 * motion_score
        )

        # Rule-based classification
        if density_score >= 0.80 and motion_score >= 0.20:
            level = "HIGH"

        elif density_score >= 0.65:
            level = "MEDIUM"

        elif motion_score >= 0.35:
            level = "MEDIUM"

        else:
            level = "LOW"

        return {
            "crowd_count": crowd_count,
            "average_motion": average_motion,
            "density_score": density_score,
            "motion_score": motion_score,
            "risk_score": risk_score,
            "risk_level": level
        }

    def calculate_region_risk(
        self,
        density_count,
        average_motion,
        density_change_ratio,
        convergence_score,
        region_id=None
    ):
        """
        Calculate region-wise risk by fusing density, motion, density trend, and convergence signals.
        
        This method combines multiple crowd analysis signals into a single risk score for a specific
        region. The fusion uses interpretable weights to balance the contribution of each signal.
        
        Args:
            density_count (float): Regional crowd count from CSRNet density map
            average_motion (float): Average optical flow magnitude in the region
            density_change_ratio (float): Ratio of density change (current - previous) / previous
            convergence_score (float): Convergence score (0-1) indicating inward motion strength
            region_id (str, optional): Region identifier for output
        
        Returns:
            dict: Dictionary containing risk assessment components:
                  - region_id (str): Region identifier
                  - density_count (float): Input crowd count
                  - average_motion (float): Input motion magnitude
                  - density_change_ratio (float): Input density change ratio
                  - convergence_score (float): Input convergence score
                  - density_score (float): Normalized density (0-1)
                  - motion_score (float): Normalized motion (0-1)
                  - density_trend_score (float): Normalized density trend (0-1, only for increases)
                  - risk_score (float): Weighted fusion score (0-1)
                  - risk_level (str): Classification (LOW/MEDIUM/HIGH)
        """
        # Normalize density safely
        if self.max_density > 0:
            density_score = min(max(density_count / self.max_density, 0.0), 1.0)
        else:
            density_score = 0.0
        
        # Normalize motion safely
        if self.max_motion > 0:
            motion_score = min(max(average_motion / self.max_motion, 0.0), 1.0)
        else:
            motion_score = 0.0
        
        # Normalize density trend - only consider increases (positive change)
        # Negative changes (density decreasing) do not increase risk
        density_trend_score = max(density_change_ratio, 0.0)
        # Clip to [0, 1] to handle extreme ratios
        density_trend_score = min(density_trend_score, 1.0)
        
        # Normalize convergence score (already in [0, 1] range from RegionAnalyzer)
        convergence_score_normalized = max(min(convergence_score, 1.0), 0.0)
        
        # Fusion weights (must sum to 1.0)
        weight_density = 0.40
        weight_motion = 0.15
        weight_density_trend = 0.20
        weight_convergence = 0.25
        
        # Calculate weighted risk score
        risk_score = (
            weight_density * density_score +
            weight_motion * motion_score +
            weight_density_trend * density_trend_score +
            weight_convergence * convergence_score_normalized
        )
        
        # Clip risk score to [0, 1] range
        risk_score = max(min(risk_score, 1.0), 0.0)
        
        # Rule-based classification for region-wise risk
        if risk_score < 0.30:
            level = "LOW"
        elif risk_score < 0.60:
            level = "MEDIUM"
        else:
            level = "HIGH"
        
        return {
            "region_id": region_id,
            "density_count": density_count,
            "average_motion": average_motion,
            "density_change_ratio": density_change_ratio,
            "convergence_score": convergence_score,
            "density_score": density_score,
            "motion_score": motion_score,
            "density_trend_score": density_trend_score,
            "risk_score": risk_score,
            "risk_level": level
        }