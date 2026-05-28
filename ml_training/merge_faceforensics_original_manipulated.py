import pandas as pd

# Paths to the CSVs
original_csv = "datasets/faceforensics/features_video_original_only.csv"
manipulated_csv = "datasets/faceforensics/features_video_classical_cnn.csv"  # or your manipulated-only CSV
output_csv = "datasets/faceforensics/features_video_merged.csv"

# Load CSVs
df_original = pd.read_csv(original_csv)
df_manipulated = pd.read_csv(manipulated_csv)

# Concatenate
df_merged = pd.concat([df_original, df_manipulated], ignore_index=True)

# Save merged CSV
df_merged.to_csv(output_csv, index=False)
print(f"Merged CSV saved to {output_csv}")
