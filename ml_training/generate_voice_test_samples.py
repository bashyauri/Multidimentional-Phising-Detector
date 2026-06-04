import math
import random
import wave
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "datasets" / "voice" / "test_samples"
SAMPLE_RATE = 16000
DURATION_SECONDS = 3.0


def _write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(samples, -0.98, 0.98)
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def _smooth_envelope(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    syllables = rng.uniform(0.15, 0.9, size=10)
    envelope = np.interp(
        np.linspace(0, len(syllables) - 1, length),
        np.arange(len(syllables)),
        syllables,
    )
    attack = np.linspace(0.0, 1.0, min(1200, length))
    release = np.linspace(1.0, 0.0, min(1600, length))
    envelope[: len(attack)] *= attack
    envelope[-len(release) :] *= release
    return envelope


def _voice_like_signal(seed: int, fake: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SAMPLE_RATE * DURATION_SECONDS)
    t = np.arange(n) / SAMPLE_RATE

    base_freq = rng.uniform(105, 190)
    vibrato = 1.0 + 0.018 * np.sin(2 * math.pi * rng.uniform(4.5, 6.5) * t)
    phase = 2 * math.pi * np.cumsum(base_freq * vibrato) / SAMPLE_RATE

    signal = np.zeros_like(t)
    for harmonic in range(1, 7):
        signal += (1.0 / harmonic) * np.sin(harmonic * phase + rng.uniform(0, math.pi))

    formants = [
        (rng.uniform(500, 800), rng.uniform(0.40, 0.75)),
        (rng.uniform(1100, 1700), rng.uniform(0.20, 0.45)),
        (rng.uniform(2200, 3100), rng.uniform(0.08, 0.25)),
    ]
    for freq, weight in formants:
        signal += weight * np.sin(2 * math.pi * freq * t + rng.uniform(0, math.pi))

    signal *= _smooth_envelope(n, seed + 100)
    signal += rng.normal(0, 0.012, size=n)

    if fake:
        # Add common synthetic artifacts: quantization, tremolo, jitter, and repeats.
        tremolo = 0.65 + 0.35 * np.sin(2 * math.pi * rng.uniform(18, 32) * t)
        signal *= tremolo
        signal += 0.10 * np.sin(2 * math.pi * rng.uniform(2600, 3800) * t)
        signal = np.round(signal * 18) / 18
        for start in range(SAMPLE_RATE // 2, n - 900, SAMPLE_RATE // 3):
            signal[start : start + 450] = signal[start - 450 : start]

    signal /= max(np.max(np.abs(signal)), 1e-6)
    return 0.75 * signal


def main() -> None:
    random.seed(42)
    for idx in range(1, 11):
        clean = _voice_like_signal(seed=idx, fake=False)
        fake = _voice_like_signal(seed=idx + 1000, fake=True)
        _write_wav(OUT_DIR / "legitimate" / f"legitimate_voice_{idx:02d}.wav", clean)
        _write_wav(OUT_DIR / "fake" / f"fake_voice_{idx:02d}.wav", fake)

    print(f"Generated 20 WAV test samples in {OUT_DIR}")
    print("Use these only for upload/endpoint testing, not as research evaluation data.")


if __name__ == "__main__":
    main()
