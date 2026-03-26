import librosa
import numpy as np


class RhythmDetector:
    def __init__(self):
        pass

    def detect(self, audio_path):
        y, sr = librosa.load(audio_path)

        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

        # Safe tempo conversion
        tempo = float(tempo.item() if hasattr(tempo, "item") else tempo)

        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Estimate time signature
        time_signature = self._estimate_time_signature(onset_env, beats)

        return {
            "tempo": tempo,
            "beats": beats.tolist(),
            "time_signature": time_signature
        }

    def _estimate_time_signature(self, onset_env, beats):
        if len(beats) < 8:
            return "Unknown"

        beats = beats[beats < len(onset_env)]
        beat_strengths = onset_env[beats]

        max_val = np.max(beat_strengths)
        if max_val > 0:
            beat_strengths = beat_strengths / max_val

        scores = {
            "3/4": self._pattern_score(beat_strengths, 3),
            "4/4": self._pattern_score(beat_strengths, 4),
            "6/8": self._pattern_score(beat_strengths, 6),
            "12/8": self._pattern_score(beat_strengths, 12),
        }

        return max(scores, key=scores.get)

    def _pattern_score(self, strengths, group_size):
        score = 0
        count = 0

        for i in range(0, len(strengths) - group_size, group_size):
            group = strengths[i:i+group_size]

            strong = group[0]
            others = np.mean(group[1:])

            score += (strong * 1.5) - others
            count += 1

        return score / (count + 1e-6)