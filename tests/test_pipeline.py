import unittest

import numpy as np

from main import MusicAnalysisPipeline, format_bar
from services.chord_predictor import ChordPredictor
from services.report_renderer import render_report
from services.rhythm_detector import RhythmDetector


class _FakeChordPredictor:
    chord_labels = ["C", "D", "E"]

    def predict_segment_index(self, chroma_segment):
        if len(chroma_segment) == 0:
            return 0
        return int(np.argmax(np.mean(chroma_segment, axis=0)))


class MusicAnalysisPipelineTest(unittest.TestCase):
    def test_aligns_chords_to_bars(self):
        pipeline = MusicAnalysisPipeline.__new__(MusicAnalysisPipeline)
        pipeline.chord_predictor = _FakeChordPredictor()

        aligned = pipeline._align_chords_with_beats(
            chroma=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ]),
            beat_times=np.array([0, 1, 2, 3, 4]),
            sr=512,
            time_signature="2/4",
        )

        self.assertEqual(
            aligned,
            [
                {"bar": 1, "chord": "C", "start": 0, "end": 1},
                {"bar": 1, "chord": "D", "start": 1, "end": 2},
                {"bar": 2, "chord": "E", "start": 2, "end": 3},
                {"bar": 2, "chord": "D", "start": 3, "end": 4},
            ],
        )

    def test_invalid_time_signature_falls_back_to_four_four(self):
        pipeline = MusicAnalysisPipeline.__new__(MusicAnalysisPipeline)
        pipeline.chord_predictor = _FakeChordPredictor()

        aligned = pipeline._align_chords_with_beats(
            chroma=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ]),
            beat_times=np.array([0, 1, 2, 3, 4, 5]),
            sr=512,
            time_signature="Unknown",
        )

        self.assertEqual(aligned[0]["bar"], 1)
        self.assertEqual(aligned[3]["bar"], 1)
        self.assertEqual(aligned[4]["bar"], 2)

    def test_empty_chord_indices_returns_empty_alignment(self):
        pipeline = MusicAnalysisPipeline.__new__(MusicAnalysisPipeline)
        pipeline.chord_predictor = _FakeChordPredictor()

        aligned = pipeline._align_chords_with_beats(
            chroma=np.array([]),
            beat_times=np.array([0, 1, 2]),
            sr=512,
            time_signature="4/4",
        )

        self.assertEqual(aligned, [])

    def test_formats_whole_bar(self):
        bar = [
            {"bar": 1, "chord": "C", "start": 0, "end": 1},
            {"bar": 1, "chord": "D", "start": 1, "end": 2},
        ]

        self.assertEqual(format_bar(bar), "[Bar 1] 0s -> 2s: C 0s-1s | D 1s-2s")


class RhythmDetectorTest(unittest.TestCase):
    def test_short_beat_sequence_returns_unknown_time_signature(self):
        detector = RhythmDetector()
        onset_env = np.ones(16)
        beats = np.array([1, 2, 3])

        self.assertEqual(detector._estimate_time_signature(onset_env, beats), "Unknown")


class ChordPredictorTest(unittest.TestCase):
    def _chroma(self, notes):
        chroma = np.zeros((1, 12))
        chroma[0, notes] = 1
        return chroma

    def test_template_fallback_predicts_c_major(self):
        predictor = ChordPredictor()

        self.assertEqual(predictor.predict(self._chroma([0, 4, 7])), ["C"])

    def test_template_fallback_predicts_a_minor(self):
        predictor = ChordPredictor()

        self.assertEqual(predictor.predict(self._chroma([9, 0, 4])), ["Am"])

    def test_template_fallback_predicts_dominant_seventh(self):
        predictor = ChordPredictor()

        self.assertEqual(predictor.predict(self._chroma([4, 8, 11, 2])), ["E7"])

    def test_template_fallback_predicts_major_seventh(self):
        predictor = ChordPredictor()

        self.assertEqual(predictor.predict(self._chroma([0, 4, 7, 11])), ["Cmaj7"])

    def test_template_fallback_predicts_half_diminished(self):
        predictor = ChordPredictor()

        self.assertIn(predictor.predict(self._chroma([6, 9, 0, 4]))[0], ["F#m7b5", "Am6"])

    def test_predict_segment_uses_average_chroma(self):
        predictor = ChordPredictor()
        c_major = self._chroma([0, 4, 7])[0]
        g_major = self._chroma([7, 11, 2])[0]
        segment = np.array([c_major, c_major, g_major])

        chord_index = predictor.predict_segment_index(segment)

        self.assertEqual(predictor.chord_labels[chord_index], "C")

    def test_template_fallback_exposes_advanced_labels(self):
        predictor = ChordPredictor()

        self.assertIn("Cmaj7", predictor.chord_labels)
        self.assertIn("F#m7b5", predictor.chord_labels)


class ReportRendererTest(unittest.TestCase):
    def test_renders_html_report(self):
        result = {
            "tempo": 120.0,
            "time_signature": "4/4",
            "aligned_chords": [
                {"bar": 1, "chord": "C", "start": 0, "end": 1},
                {"bar": 1, "chord": "G", "start": 1, "end": 2},
            ],
        }

        html = render_report(result, "input/song.wav")

        self.assertIn("Chord Detection Report", html)
        self.assertIn("song.wav", html)
        self.assertIn("<strong>120.00</strong>", html)
        self.assertIn("<strong>4/4</strong>", html)
        self.assertIn("<strong>C</strong>", html)
        self.assertIn("<strong>G</strong>", html)
        self.assertIn("<audio id=\"audio\"", html)
        self.assertIn("class=\"chord-block\"", html)
        self.assertIn("data-start=\"0\"", html)
        self.assertIn("id=\"syncOffset\"", html)
        self.assertIn('"bar": 1', html)
        self.assertNotIn("&quot;bar&quot;", html)


if __name__ == "__main__":
    unittest.main()
