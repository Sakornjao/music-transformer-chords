import librosa

class FeatureExtractor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def load_audio(self, path):
        y, sr = librosa.load(path, sr=self.sample_rate)
        return y, sr

    def extract_chroma(self, y, sr):
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        return chroma.T  # (time, 12)