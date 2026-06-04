import numpy as np
import librosa
import io

VOICE_FEATURE_COLUMNS = [
    "chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth",
    "rolloff", "zero_crossing_rate",
    *[f"mfcc{i}" for i in range(1, 21)],
]


def extract_voice_features_from_bytes(file_bytes, filename=None):
    """
    Extract features matching datasets/voice/DATASET-balanced.csv.
    """
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)

    target_len = 3 * sr
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = [
        float(np.mean(chroma)),
        float(np.mean(rms)),
        float(np.mean(spectral_centroid)),
        float(np.mean(spectral_bandwidth)),
        float(np.mean(rolloff)),
        float(np.mean(zero_crossing_rate)),
        *[float(np.mean(row)) for row in mfcc],
    ]
    return np.array(features, dtype=np.float32)
