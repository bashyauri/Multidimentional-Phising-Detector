import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Voice model
voice_csv = 'datasets/voice/DATASET-balanced.csv'
df_voice = pd.read_csv(voice_csv)
X_voice = df_voice.drop('LABEL', axis=1)
y_voice = df_voice['LABEL'].map({'FAKE': 1, 'REAL': 0})
voice_model = RandomForestClassifier(n_estimators=100, random_state=42)
voice_model.fit(X_voice, y_voice)
joblib.dump(voice_model, 'models/voice_model_balanced.pkl')
print('Saved models/voice_model_balanced.pkl')

# Video model
video_csv = 'datasets/faceforensics/features_video_classical_cnn.csv'
df_video = pd.read_csv(video_csv)
X_video = df_video.drop(['path', 'label'], axis=1)
y_video = df_video['label'].astype(int)
video_model = RandomForestClassifier(n_estimators=100, random_state=42)
video_model.fit(X_video, y_video)
joblib.dump(video_model, 'models/deepfake_model_legacy_random_forest.pkl')
print('Saved models/deepfake_model_legacy_random_forest.pkl')
