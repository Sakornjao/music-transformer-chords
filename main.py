import argparse
import logging
from pathlib import Path
from itertools import groupby

import librosa
import numpy as np

from services.feature_extractor import FeatureExtractor
from services.chord_predictor import ChordPredictor
from services.report_renderer import write_report
from services.rhythm_detector import RhythmDetector

logger = logging.getLogger(__name__)


class MusicAnalysisPipeline:
    def __init__(self, audio_path: str | Path, checkpoint_path: str | Path | None = None):
        self.audio_path = Path(audio_path)

        self.feature_extractor = FeatureExtractor()
        self.chord_predictor = ChordPredictor(checkpoint_path)
        self.rhythm_detector = RhythmDetector()

    def run(self) -> dict:
        logger.info("Loading audio from %s", self.audio_path)
        y, sr = self.feature_extractor.load_audio(self.audio_path)

        logger.info("Extracting chroma")
        chroma = self.feature_extractor.extract_chroma(y, sr)

        logger.info("Detecting rhythm")
        rhythm = self.rhythm_detector.detect(y, sr)

        beat_times = librosa.frames_to_time(rhythm["beats"], sr=sr)
        aligned_chords = self._align_chords_with_beats(
            chroma,
            beat_times,
            sr,
            rhythm["time_signature"]
        )

        return {
            "tempo": rhythm["tempo"],
            "time_signature": rhythm["time_signature"],
            "aligned_chords": aligned_chords
        }

    def _align_chords_with_beats(
        self,
        chroma: np.ndarray,
        beat_times: np.ndarray,
        sr: int,
        time_signature: str,
    ) -> list[dict]:
        labels = self.chord_predictor.chord_labels

        try:
            beats_per_bar = int(time_signature.split("/")[0])
        except (AttributeError, IndexError, ValueError):
            beats_per_bar = 4

        aligned = []
        bar_number = 1
        hop_length = 512

        if len(chroma) == 0:
            return aligned

        for i in range(len(beat_times) - 1):
            start = beat_times[i]
            end = beat_times[i + 1]

            start_frame = max(0, int((start * sr) / hop_length))
            end_frame = max(start_frame + 1, int((end * sr) / hop_length))
            end_frame = min(end_frame, len(chroma))
            chroma_segment = chroma[start_frame:end_frame]
            chord_index = self.chord_predictor.predict_segment_index(chroma_segment)
            chord = labels[chord_index]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze tempo, time signature, and chord timeline for an audio file.")
    parser.add_argument("audio_path", nargs="?", default="input/accompaniment.wav", help="Path to the audio file.")
    parser.add_argument("--checkpoint", help="Optional trained chord model checkpoint path.")
    parser.add_argument("--limit", type=int, default=16, help="Number of bars or chord events to print.")
    parser.add_argument("--report", help="Optional HTML report output path, for example reports/chords.html.")
    parser.add_argument(
        "--view",
        choices=["bars", "events"],
        default="bars",
        help="Print grouped bars or individual chord events.",
    )
    return parser.parse_args()


def format_bar(bar_items: list[dict]) -> str:
    bar_number = bar_items[0]["bar"]
    start = bar_items[0]["start"]
    end = bar_items[-1]["end"]
    chords = " | ".join(f"{item['chord']} {item['start']}s-{item['end']}s" for item in bar_items)
    return f"[Bar {bar_number}] {start}s -> {end}s: {chords}"


def print_chord_timeline(aligned_chords: list[dict], view: str, limit: int) -> None:
    print("\nChord Timeline:")
    if view == "events":
        for item in aligned_chords[:limit]:
            print(f"[Bar {item['bar']}] {item['chord']} ({item['start']}s -> {item['end']}s)")
        return

    bars = groupby(aligned_chords, key=lambda item: item["bar"])
    for index, (_, bar_items) in enumerate(bars):
        if index >= limit:
            break
        print(format_bar(list(bar_items)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    pipeline = MusicAnalysisPipeline(args.audio_path, args.checkpoint)
    result = pipeline.run()

    print("\n=== RESULT ===")
    print(f"Tempo: {result['tempo']} BPM")
    print(f"Time Signature: {result['time_signature']}")

    print_chord_timeline(result["aligned_chords"], args.view, args.limit)

    if args.report:
        report_path = write_report(result, args.audio_path, args.report)
        print(f"\nGUI report written to: {report_path}")


if __name__ == "__main__":
    main()
