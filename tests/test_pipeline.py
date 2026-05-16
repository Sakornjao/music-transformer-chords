import unittest

import numpy as np

from main import MusicAnalysisPipeline, format_bar
from services.chord_predictor import ChordPredictor
from services.report_renderer import render_report
from services.rhythm_detector import RhythmDetector


class _FakeChordPredictor:
    chord_labels = ["C", "D", "E"]


class MusicAnalysisPipelineTest(unittest.TestCase):
    def test_aligns_chords_to_bars(self):
        pipeline = MusicAnalysisPipeline.__new__(MusicAnalysisPipeline)
        pipeline.chord_predictor = _FakeChordPredictor()

        aligned = pipeline._align_chords_with_beats(
            chord_indices=np.array([0, 1, 2, 1, 0]),
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
            chord_indices=np.array([0, 1, 2, 1, 0]),
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
            chord_indices=np.array([]),
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
    def test_template_fallback_predicts_c_major(self):
        predictor = ChordPredictor()
        chroma = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]])

        self.assertEqual(predictor.predict(chroma), ["C"])

    def test_template_fallback_predicts_a_minor(self):
        predictor = ChordPredictor()
        chroma = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]])

        self.assertEqual(predictor.predict(chroma), ["Am"])


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


if __name__ == "__main__":
    unittest.main()
