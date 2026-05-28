import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Load your CSV
input_csv = "datasets/faceforensics/features_video_classical_cnn.csv"
df = pd.read_csv(input_csv)

# Separate features and label
X = df.drop(["label"], axis=1)
y = df["label"]

# Apply undersampling to balance classes
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# Combine back to DataFrame
balanced_df = X_res.copy()
balanced_df["label"] = y_res

# Save to new CSV
output_csv = "datasets/faceforensics/features_video_classical_cnn_balanced.csv"
balanced_df.to_csv(output_csv, index=False)
print(f"Balanced CSV saved as {output_csv}")
