import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load metrics
with open('models/deepfake_efficientnet_b0_metrics.json', 'r') as f:
    metrics = json.load(f)

# Convert confusion matrix to numpy array
cm = np.array(metrics['confusion_matrix'])

# Plot and save confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Deepfake'])
disp.plot(cmap='Blues', values_format='d')
plt.title('EfficientNet-B0 Confusion Matrix (Latest)')
plt.savefig('static/plots/deepfake_efficientnet_b0_confusion_matrix.png')
plt.show()
