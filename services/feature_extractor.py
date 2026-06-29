from pathlib import Path
import os
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "music_transformer_chords_numba"))
import librosa
import numpy as np


class FeatureExtractor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def load_audio(self, path: str | Path) -> tuple[np.ndarray, int]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        y, sr = librosa.load(path, sr=self.sample_rate)
        return y, sr

    def extract_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        return chroma.T  # (time, 12)
