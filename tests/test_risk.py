import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from risk_assessment import RiskAssessment

risk_engine = RiskAssessment()

result = risk_engine.calculate_risk(
    crowd_count=1432.05,
    average_motion=0.2234
)

print("\n----- Risk Assessment -----")

for key, value in result.items():
    print(f"{key}: {value}")
