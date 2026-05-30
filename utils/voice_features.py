import numpy as np
import librosa
import io

def extract_voice_features_from_bytes(file_bytes, filename=None):
    """
    Extracts features from raw audio bytes for voice deepfake detection.
    Returns a 1D numpy array of features suitable for classical ML models.
    """
    # Load audio from bytes
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)
    # Pad/trim to 3 seconds
    target_len = 3 * sr
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    # Feature extraction (MFCCs, Chroma, Spectral Contrast, Tonnetz)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    # Aggregate (mean, std)
    features = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        chroma.mean(axis=1), chroma.std(axis=1),
        contrast.mean(axis=1), contrast.std(axis=1),
        tonnetz.mean(axis=1), tonnetz.std(axis=1)
    ])
    return features
