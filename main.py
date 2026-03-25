from services.feature_extractor import FeatureExtractor
from services.chord_predictor import ChordPredictor

class MusicChordPipeline:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.feature_extractor = FeatureExtractor()
        self.chord_predictor = ChordPredictor()

    def run(self):
        print("Loading audio...")
        y, sr = self.feature_extractor.load_audio(self.audio_path)

        print("Extracting chroma...")
        chroma = self.feature_extractor.extract_chroma(y, sr)

        print("Predicting chords...")
        chords = self.chord_predictor.predict(chroma)

        return chords


if __name__ == "__main__":
    AUDIO_PATH = "input/accompaniment.wav"

    pipeline = MusicChordPipeline(AUDIO_PATH)
    chords = pipeline.run()

    for i, chord in enumerate(chords[:20]):
        print(f"{i}: {chord}")