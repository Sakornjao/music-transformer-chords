import librosa
import numpy as np

TIME_SIGNATURE_BEATS_PER_BAR = {
    "2/4": 2,
    "3/4": 3,
    "4/4": 4,
    "6/8": 2,
    "12/8": 4,
}


class RhythmDetector:
    def detect(self, y: np.ndarray, sr: int) -> dict:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo.item() if hasattr(tempo, "item") else tempo)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        time_signature = self._estimate_time_signature(onset_env, beats)

        return {
            "tempo": tempo,
            "beats": beats.tolist(),
            "time_signature": time_signature
        }

    def _estimate_time_signature(self, onset_env: np.ndarray, beats: np.ndarray) -> str:
        if len(beats) < 8:
            return "Unknown"

        beats = beats[beats < len(onset_env)]
        if len(beats) == 0:
            return "Unknown"

        beat_strengths = onset_env[beats]

        max_val = np.max(beat_strengths)
        if max_val > 0:
            beat_strengths = beat_strengths / max_val

        scores = {
            "3/4": self._pattern_score(beat_strengths, TIME_SIGNATURE_BEATS_PER_BAR["3/4"]),
            "4/4": self._pattern_score(beat_strengths, TIME_SIGNATURE_BEATS_PER_BAR["4/4"]),
        }

        best_meter = max(scores, key=scores.get)
        if best_meter == "4/4" and self._has_compound_subdivision(onset_env, beats):
            return "12/8"

        return best_meter

    def _pattern_score(self, strengths: np.ndarray, group_size: int) -> float:
        score = 0
        count = 0

        for i in range(0, len(strengths) - group_size, group_size):
            group = strengths[i:i+group_size]

            strong = group[0]
            others = np.mean(group[1:])

            score += (strong * 1.5) - others
            count += 1

        return score / (count + 1e-6)

    def _has_compound_subdivision(self, onset_env: np.ndarray, beats: np.ndarray) -> bool:
        third_subdivisions = []
        half_subdivisions = []
        beat_strengths = []

        for left, right in zip(beats[:-1], beats[1:]):
            span = right - left
            if span < 3:
                continue

            first_third = min(len(onset_env) - 1, int(round(left + span / 3)))
            second_third = min(len(onset_env) - 1, int(round(left + 2 * span / 3)))
            half = min(len(onset_env) - 1, int(round(left + span / 2)))

            beat_strengths.append(onset_env[left])
            third_subdivisions.append((onset_env[first_third] + onset_env[second_third]) / 2)
            half_subdivisions.append(onset_env[half])

        if len(third_subdivisions) < 8:
            return False

        beat_level = float(np.median(beat_strengths))
        third_level = float(np.median(third_subdivisions))
        half_level = float(np.median(half_subdivisions))
        if beat_level <= 0:
            return False

        return third_level > half_level * 1.35 and third_level > beat_level * 0.12
