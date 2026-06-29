import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import numpy as np
import soundfile as sf

from main import MusicAnalysisPipeline, format_bar, keep_first_chord_only
from services.chart_exporter import (
    _frequency_to_note_label,
    write_chord_chart,
    write_frequency_domain_chart,
    write_magnitude_spectrum_chart,
)
from services.chord_predictor import ChordPredictor
from services.report_renderer import render_report
from services.rhythm_detector import RhythmDetector
from services.sheet_exporter import write_chord_pdf, write_musicxml


class _FakeChordPredictor:
    chord_labels = ["C", "D", "E"]

    def predict_segment_index(self, chroma_segment):
        if len(chroma_segment) == 0:
            return 0
        return int(np.argmax(np.mean(chroma_segment, axis=0)))


class _FakeDemucsService:
    def separate_piano(self, audio_path):
        return Path("separated/htdemucs_6s/song/piano.wav")

    def separate_harmony(self, audio_path):
        return Path("separated/htdemucs_6s/song/harmony.wav")


class MusicAnalysisPipelineTest(unittest.TestCase):
    def test_piano_source_uses_separated_piano_audio(self):
        pipeline = MusicAnalysisPipeline.__new__(MusicAnalysisPipeline)
        pipeline.audio_path = Path("input/song.wav")
        pipeline.source = "piano_only"
        pipeline.demucs_service = _FakeDemucsService()

        self.assertEqual(
            pipeline._resolve_analysis_audio_path(),
            Path("separated/htdemucs_6s/song/piano.wav"),
        )

    def test_harmony_source_uses_combined_harmony_audio(self):
        pipeline = MusicAnalysisPipeline.__new__(MusicAnalysisPipeline)
        pipeline.audio_path = Path("input/song.wav")
        pipeline.source = "harmony"
        pipeline.demucs_service = _FakeDemucsService()

        self.assertEqual(
            pipeline._resolve_analysis_audio_path(),
            Path("separated/htdemucs_6s/song/harmony.wav"),
        )

    def test_normalizes_piano_only_source_aliases(self):
        pipeline = MusicAnalysisPipeline.__new__(MusicAnalysisPipeline)

        self.assertEqual(pipeline._normalize_source("piano"), "piano_only")
        self.assertEqual(pipeline._normalize_source("piano-only"), "piano_only")
        self.assertEqual(pipeline._normalize_source("harmonic"), "harmony")
        self.assertEqual(pipeline._normalize_source("instruments"), "harmony")
        self.assertEqual(pipeline._normalize_source("mix"), "mix")

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

    def test_keeps_first_chord_only(self):
        result = {
            "tempo": 120.0,
            "aligned_chords": [
                {"bar": 1, "chord": "C", "start": 0, "end": 1},
                {"bar": 1, "chord": "G", "start": 1, "end": 2},
            ],
        }

        filtered = keep_first_chord_only(result)

        self.assertEqual(filtered["aligned_chords"], [{"bar": 1, "chord": "C", "start": 0, "end": 1}])
        self.assertEqual(result["aligned_chords"][1]["chord"], "G")


class RhythmDetectorTest(unittest.TestCase):
    def test_short_beat_sequence_returns_unknown_time_signature(self):
        detector = RhythmDetector()
        onset_env = np.ones(16)
        beats = np.array([1, 2, 3])

        self.assertEqual(detector._estimate_time_signature(onset_env, beats), "Unknown")

    def test_estimates_four_four_from_four_pulse_accents(self):
        detector = RhythmDetector()
        onset_env = np.zeros(80)
        beats = np.arange(0, 64, 4)
        onset_env[beats] = np.tile([1.0, 0.2, 0.35, 0.2], 4)
        onset_env[beats[:-1] + 2] = 0.28

        self.assertEqual(detector._estimate_time_signature(onset_env, beats), "4/4")

    def test_estimates_twelve_eight_from_compound_four_pulse_accents(self):
        detector = RhythmDetector()
        onset_env = np.zeros(80)
        beats = np.arange(0, 64, 4)
        onset_env[beats] = np.tile([1.0, 0.2, 0.35, 0.2], 4)
        onset_env[beats[:-1] + 1] = 0.3
        onset_env[beats[:-1] + 3] = 0.3

        self.assertEqual(detector._estimate_time_signature(onset_env, beats), "12/8")

    def test_estimates_three_four_from_three_pulse_accents(self):
        detector = RhythmDetector()
        onset_env = np.zeros(32)
        beats = np.arange(18)
        onset_env[beats] = np.tile([1.0, 0.2, 0.25], 6)

        self.assertEqual(detector._estimate_time_signature(onset_env, beats), "3/4")


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
            "source": "harmony",
            "analysis_audio_path": "separated/htdemucs_6s/song/harmony.wav",
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
        self.assertIn("<strong>Harmony</strong>", html)
        self.assertIn("<strong>C</strong>", html)
        self.assertIn("<strong>G</strong>", html)
        self.assertIn("<audio id=\"audio\"", html)
        self.assertIn("id=\"audio-sources\"", html)
        self.assertIn("data-audio-source=\"mix\"", html)
        self.assertIn("data-audio-source=\"harmony\"", html)
        self.assertIn("class=\"chord-block\"", html)
        self.assertIn("data-start=\"0\"", html)
        self.assertIn("id=\"syncOffset\"", html)
        self.assertIn("id=\"pianoMode\"", html)
        self.assertIn("id=\"playPianoTimeline\"", html)
        self.assertIn("id=\"stopPianoTimeline\"", html)
        self.assertIn('"bar": 1', html)
        self.assertNotIn("&quot;bar&quot;", html)


class SheetExporterTest(unittest.TestCase):
    def test_writes_musicxml_with_chord_symbols_and_piano_notes(self):
        result = {
            "tempo": 120.0,
            "time_signature": "12/8",
            "aligned_chords": [
                {"bar": 1, "chord": "Cmaj7", "start": 0, "end": 1},
                {"bar": 1, "chord": "G7", "start": 1, "end": 2},
            ],
        }

        with TemporaryDirectory() as tmpdir:
            output_path = write_musicxml(result, "input/song.wav", Path(tmpdir) / "chords.musicxml")

            root = ET.parse(output_path).getroot()

        self.assertEqual(root.tag, "score-partwise")
        self.assertEqual(root.findtext(".//time/beats"), "12")
        self.assertEqual(root.findtext(".//time/beat-type"), "8")
        harmony_texts = [kind.attrib.get("text") for kind in root.findall(".//harmony/kind")]
        self.assertEqual(harmony_texts, ["Cmaj7", "G7"])
        self.assertGreaterEqual(len(root.findall(".//note/pitch")), 7)
        self.assertIsNotNone(root.find(".//note/dot"))

    def test_writes_pdf_chord_chart(self):
        result = {
            "tempo": 120.0,
            "time_signature": "12/8",
            "source": "harmony",
            "aligned_chords": [
                {"bar": 1, "chord": "Cmaj7", "start": 0, "end": 1},
                {"bar": 1, "chord": "G7", "start": 1, "end": 2},
            ],
        }

        with TemporaryDirectory() as tmpdir:
            output_path = write_chord_pdf(result, "input/song.wav", Path(tmpdir) / "chords.pdf")

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes()[:4], b"%PDF")


class ChartExporterTest(unittest.TestCase):
    def test_writes_matplotlib_chord_chart(self):
        result = {
            "tempo": 120.0,
            "time_signature": "4/4",
            "source": "harmony",
            "aligned_chords": [
                {"bar": 1, "chord": "C", "start": 0, "end": 1},
                {"bar": 1, "chord": "G7", "start": 1, "end": 2},
                {"bar": 2, "chord": "Am", "start": 2, "end": 3},
            ],
        }

        with TemporaryDirectory() as tmpdir:
            output_path = write_chord_chart(result, "input/song.wav", Path(tmpdir) / "chords.png")

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_writes_frequency_domain_chart(self):
        sample_rate = 22050
        time = np.linspace(0, 1, sample_rate, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * time)

        with TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sine.wav"
            output_path = Path(tmpdir) / "frequency.png"
            sf.write(audio_path, audio, sample_rate)

            written_path = write_frequency_domain_chart(audio_path, output_path)

            self.assertTrue(written_path.exists())
            self.assertEqual(written_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_writes_magnitude_spectrum_chart(self):
        sample_rate = 22050
        time = np.linspace(0, 1, sample_rate, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * time)

        with TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sine.wav"
            output_path = Path(tmpdir) / "magnitude.png"
            sf.write(audio_path, audio, sample_rate)

            written_path = write_magnitude_spectrum_chart(audio_path, output_path)

            self.assertTrue(written_path.exists())
            self.assertEqual(written_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_writes_sliced_magnitude_spectrum_chart(self):
        sample_rate = 22050
        time = np.linspace(0, 2, sample_rate * 2, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * time)

        with TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "sine.wav"
            output_path = Path(tmpdir) / "magnitude_slice.png"
            sf.write(audio_path, audio, sample_rate)

            written_path = write_magnitude_spectrum_chart(audio_path, output_path, start_time=0.5, end_time=1.0)

            self.assertTrue(written_path.exists())
            self.assertEqual(written_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_frequency_to_note_label(self):
        self.assertEqual(_frequency_to_note_label(440), "A4")
        self.assertEqual(_frequency_to_note_label(261.63), "C4")


if __name__ == "__main__":
    unittest.main()
