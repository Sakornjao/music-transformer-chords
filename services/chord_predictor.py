from pathlib import Path
import logging

import numpy as np
import torch

from models.transformer import ChordTransformer

logger = logging.getLogger(__name__)

PITCH_CLASS_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODEL_CHORD_LABELS = PITCH_CLASS_LABELS + [f"{pitch}m" for pitch in PITCH_CLASS_LABELS]
CHORD_QUALITY_INTERVALS = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "mMaj7": [0, 3, 7, 11],
    "6": [0, 4, 7, 9],
    "m6": [0, 3, 7, 9],
    "add9": [0, 2, 4, 7],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "dim": [0, 3, 6],
    "dim7": [0, 3, 6, 9],
    "m7b5": [0, 3, 6, 10],
    "aug": [0, 4, 8],
}
CHORD_QUALITY_COMPLEXITY_PENALTIES = {
    "": 0.0,
    "m": 0.0,
    "7": 0.015,
    "maj7": 0.015,
    "m7": 0.015,
    "6": 0.02,
    "m6": 0.02,
    "add9": 0.025,
    "sus2": 0.025,
    "sus4": 0.025,
    "dim": 0.035,
    "aug": 0.035,
    "dim7": 0.05,
    "m7b5": 0.05,
    "mMaj7": 0.12,
}


class ChordPredictor:
    def __init__(self, checkpoint_path: str | Path | None = None):
        self.model = self._load_model(checkpoint_path) if checkpoint_path else None
        self.chord_labels = MODEL_CHORD_LABELS if self.model else self._build_advanced_chord_labels()
        self.templates = self._build_chord_templates() if self.model is None else None
        self.template_penalties = self._build_template_penalties() if self.model is None else None
        self._logged_template_mode = False

    def _load_model(self, checkpoint_path: str | Path | None) -> ChordTransformer:
        model = ChordTransformer()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)

        model.eval()
        return model

    def _build_chord_templates(self) -> np.ndarray:
        templates = []

        for root_pitch in range(12):
            for chord_intervals in CHORD_QUALITY_INTERVALS.values():
                template = np.zeros(12)
                for interval in chord_intervals:
                    template[(root_pitch + interval) % 12] = 1.0
                templates.append(template)

        templates = np.array(templates)
        return templates / np.linalg.norm(templates, axis=1, keepdims=True)

    def _build_advanced_chord_labels(self) -> list[str]:
        labels = []
        for pitch_label in PITCH_CLASS_LABELS:
            for quality_suffix in CHORD_QUALITY_INTERVALS:
                labels.append(f"{pitch_label}{quality_suffix}")
        return labels

    def _build_template_penalties(self) -> np.ndarray:
        penalties = []
        for _ in PITCH_CLASS_LABELS:
            for quality_suffix in CHORD_QUALITY_INTERVALS:
                penalties.append(CHORD_QUALITY_COMPLEXITY_PENALTIES[quality_suffix])
        return np.array(penalties)

    def predict(self, chroma: np.ndarray) -> list[str]:
        chord_indices = self.predict_with_indices(chroma)
        return [self.chord_labels[i] for i in chord_indices]

    def predict_segment_index(self, chroma_segment: np.ndarray) -> int:
        chroma_segment = np.asarray(chroma_segment, dtype=float)
        if chroma_segment.ndim == 1:
            averaged_chroma = chroma_segment.reshape(1, -1)
        elif len(chroma_segment) == 0:
            averaged_chroma = np.zeros((1, 12))
        else:
            averaged_chroma = np.mean(chroma_segment, axis=0, keepdims=True)

        return int(self.predict_with_indices(averaged_chroma)[0])

    def predict_with_indices(self, chroma: np.ndarray) -> np.ndarray:
        if self.model is None:
            if not self._logged_template_mode:
                logger.info("No checkpoint supplied; using deterministic advanced chroma-template chord estimation.")
                self._logged_template_mode = True
            return self._predict_with_templates(chroma)

        x = torch.as_tensor(chroma).unsqueeze(0).float()

        with torch.no_grad():
            output = self.model(x)

        return torch.argmax(output, dim=-1)[0].numpy()

    def _predict_with_templates(self, chroma: np.ndarray) -> np.ndarray:
        chroma = np.asarray(chroma, dtype=float)
        chroma_norms = np.linalg.norm(chroma, axis=1, keepdims=True)
        normalized_chroma = np.divide(chroma, chroma_norms, out=np.zeros_like(chroma), where=chroma_norms > 0)
        template_scores = (normalized_chroma @ self.templates.T) - self.template_penalties
        return np.argmax(template_scores, axis=1)
