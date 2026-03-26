import librosa
from services.feature_extractor import FeatureExtractor
from services.chord_predictor import ChordPredictor
from services.rhythm_detector import RhythmDetector


class MusicAnalysisPipeline:
    def __init__(self, audio_path):
        self.audio_path = audio_path

        self.feature_extractor = FeatureExtractor()
        self.chord_predictor = ChordPredictor()
        self.rhythm_detector = RhythmDetector()

    def run(self):
        print("Loading audio...")
        y, sr = self.feature_extractor.load_audio(self.audio_path)

        print("Extracting chroma...")
        chroma = self.feature_extractor.extract_chroma(y, sr)

        print("Predicting chords (indices)...")
        chord_indices = self.chord_predictor.predict_with_indices(chroma)

        print("Detecting rhythm...")
        rhythm = self.rhythm_detector.detect(self.audio_path)

        # Convert beats → time (seconds)
        beat_times = librosa.frames_to_time(rhythm["beats"], sr=sr)

        # Align chords with beats + time signature
        aligned_chords = self._align_chords_with_beats(
            chord_indices,
            beat_times,
            sr,
            len(y),
            rhythm["time_signature"]
        )

        return {
            "tempo": rhythm["tempo"],
            "time_signature": rhythm["time_signature"],
            "aligned_chords": aligned_chords
        }

    def _align_chords_with_beats(self, chord_indices, beat_times, sr, total_samples, time_signature):
        labels = self.chord_predictor.chord_labels

        # Parse time signature
        try:
            beats_per_bar = int(time_signature.split("/")[0])
        except:
            beats_per_bar = 4

        aligned = []
        bar_number = 1
        hop_length = 512

        for i in range(len(beat_times) - 1):
            start = beat_times[i]
            end = beat_times[i + 1]

            # Map time → frame index
            frame_idx = int((start * sr) / hop_length)
            frame_idx = min(frame_idx, len(chord_indices) - 1)

            chord = labels[chord_indices[frame_idx]]

            aligned.append({
                "bar": bar_number,
                "chord": chord,
                "start": round(start, 2),
                "end": round(end, 2)
            })

            # Move to next bar
            if (i + 1) % beats_per_bar == 0:
                bar_number += 1

        return aligned


if __name__ == "__main__":
    AUDIO_PATH = "input/accompaniment.wav"

    pipeline = MusicAnalysisPipeline(AUDIO_PATH)
    result = pipeline.run()

    print("\n=== RESULT ===")
    print(f"Tempo: {result['tempo']} BPM")
    print(f"Time Signature: {result['time_signature']}")

    print("\nChord Timeline:")
    for item in result["aligned_chords"][:16]:
        print(f"[Bar {item['bar']}] {item['chord']} ({item['start']}s → {item['end']}s)")