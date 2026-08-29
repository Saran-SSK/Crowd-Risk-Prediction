import sys
import os

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models.csrnet.model import CSRNet

model = CSRNet(load_weights=False)

print(model)
print("\nCSRNet Loaded Successfully!")
