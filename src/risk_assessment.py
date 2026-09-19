class RiskAssessment:
    """
    Crowd Risk Assessment Module.
    
    Provides both global (frame-level) and regional (grid-level) risk assessment
    by fusing crowd density, motion magnitude, temporal density trend, and flow convergence.
    
    DOCUMENTATION OF EQUATIONS AND DESIGN CHOICES:
    
    1. GLOBAL RISK EQUATION:
       D_global = clip(crowd_count / C_global_max, 0, 1)
       M_global = clip(average_motion / M_global_max, 0, 1)
       R_global = 0.65 * D_global + 0.35 * M_global
       
       Global Classification:
         LOW    : R_global < 0.30
         MEDIUM : 0.30 <= R_global < 0.60
         HIGH   : R_global >= 0.60
       
       ENGINEERING DESIGN PARAMETERS:
       - C_global_max = 2000.0: Engineering reference capacity for whole-frame crowd count.
       - M_global_max = 0.80: Engineering reference optical-flow speed (pixels/frame).
         This is NOT a scientifically validated universal panic threshold; it reflects
         an upper reference speed for walking crowds observed in Farneback optical flow.
       - Weights (0.65 density, 0.35 motion): Initial heuristic engineering weights selected
         to prioritize structural crowd accumulation while remaining responsive to kinetics.
         They are NOT claimed to be empirically or universally validated constants.
    
    2. REGIONAL RISK EQUATION:
       D_region = clip(density_count / C_region_max, 0, 1)
       M_region = clip(average_motion / M_region_max, 0, 1)
       T_region = clip(max(density_change_ratio, 0) / trend_reference, 0, 1)
       C_region = clip(convergence_score, 0, 1)
       
       R_region = 0.40 * D_region + 0.20 * M_region + 0.15 * T_region + 0.25 * C_region
       
       Regional Classification:
         LOW    : R_region < 0.30
         MEDIUM : 0.30 <= R_region < 0.60
         HIGH   : R_region >= 0.60
       
       ENGINEERING DESIGN PARAMETERS:
       - C_region_max = 800.0: Engineering reference value used to normalize regional
         crowd density to [0, 1]. This value can be calibrated using annotated crowd-density
         data in future work.
       - M_region_max = 0.80: Regional optical-flow reference speed (pixels/frame).
       - trend_reference = 0.05: Engineering reference for expected significant frame-to-frame
         density increase (5% increase per frame interval at ~30 fps). Decreasing density
         does not contribute to hazard.
       - Weights (0.40 D, 0.20 M, 0.15 T, 0.25 C): Heuristic engineering fusion weights.
    """

    def __init__(
        self,
        max_density=2000.0,
        max_motion=0.80,
        c_region_max=800.0,
        m_region_max=0.80,
        trend_reference=0.05,
    ):
        # Global normalization references (Engineering Reference Values)
        self.max_density = float(max_density)      # C_global_max
        self.max_motion = float(max_motion)        # M_global_max
        
        # Regional normalization references (Engineering Reference Values)
        self.c_region_max = float(c_region_max)    # C_region_max
        self.m_region_max = float(m_region_max)    # M_region_max
        self.trend_reference = float(trend_reference)

    def calculate_risk(self, crowd_count, average_motion):
        """
        Calculate global crowd risk score and classification level.
        
        Formula:
          D = clip(crowd_count / max_density, 0, 1)
          M = clip(average_motion / max_motion, 0, 1)
          R_global = 0.65 * D + 0.35 * M
        """
        # 1. Normalize global density safely
        if self.max_density > 0:
            density_score = min(max(float(crowd_count) / self.max_density, 0.0), 1.0)
        else:
            density_score = 0.0

        # 2. Normalize global motion safely
        if self.max_motion > 0:
            motion_score = min(max(float(average_motion) / self.max_motion, 0.0), 1.0)
        else:
            motion_score = 0.0

        # 3. Weighted global fusion (heuristic engineering weights)
        weight_density = 0.65
        weight_motion = 0.35
        risk_score = weight_density * density_score + weight_motion * motion_score
        risk_score = max(min(risk_score, 1.0), 0.0)

        # 4. Direct classification derived from global risk_score
        # Consistent with visualization thresholds: LOW < 0.30, MEDIUM [0.30, 0.60), HIGH >= 0.60
        if risk_score < 0.30:
            level = "LOW"
        elif risk_score < 0.60:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return {
            "crowd_count": float(crowd_count),
            "average_motion": float(average_motion),
            "density_score": float(density_score),
            "motion_score": float(motion_score),
            "risk_score": float(risk_score),
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
        Calculate region-wise crowd risk by fusing density, motion, density trend, and convergence.
        
        Formula:
          D = clip(density_count / c_region_max, 0, 1)
          M = clip(average_motion / m_region_max, 0, 1)
          T = clip(max(density_change_ratio, 0) / trend_reference, 0, 1)
          C = clip(convergence_score, 0, 1)
          R_region = 0.40 * D + 0.20 * M + 0.15 * T + 0.25 * C
        """
        # 1. Normalize regional density using dedicated regional capacity reference
        if self.c_region_max > 0:
            density_score = min(max(float(density_count) / self.c_region_max, 0.0), 1.0)
        else:
            density_score = 0.0

        # 2. Normalize regional motion using regional motion reference
        if self.m_region_max > 0:
            motion_score = min(max(float(average_motion) / self.m_region_max, 0.0), 1.0)
        else:
            motion_score = 0.0

        # 3. Normalize density trend (only positive increases contribute to hazard)
        if self.trend_reference > 0:
            raw_increase = max(float(density_change_ratio), 0.0)
            density_trend_score = min(max(raw_increase / self.trend_reference, 0.0), 1.0)
        else:
            density_trend_score = 0.0

        # 4. Normalize convergence score (already [0, 1] from RegionAnalyzer)
        convergence_score_normalized = max(min(float(convergence_score), 1.0), 0.0)

        # 5. Fusion weights (sum to 1.0, heuristic engineering weights)
        weight_density = 0.40
        weight_motion = 0.20
        weight_density_trend = 0.15
        weight_convergence = 0.25

        risk_score = (
            weight_density * density_score
            + weight_motion * motion_score
            + weight_density_trend * density_trend_score
            + weight_convergence * convergence_score_normalized
        )
        risk_score = max(min(risk_score, 1.0), 0.0)

        # 6. Direct classification derived from regional risk_score
        if risk_score < 0.30:
            level = "LOW"
        elif risk_score < 0.60:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return {
            "region_id": region_id,
            "density_count": float(density_count),
            "average_motion": float(average_motion),
            "density_change_ratio": float(density_change_ratio),
            "convergence_score": float(convergence_score),
            "density_score": float(density_score),
            "motion_score": float(motion_score),
            "density_trend_score": float(density_trend_score),
            "risk_score": float(risk_score),
            "risk_level": level
        }