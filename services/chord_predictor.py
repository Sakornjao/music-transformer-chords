from pathlib import Path
import logging

import numpy as np
import torch

from models.transformer import ChordTransformer

logger = logging.getLogger(__name__)


class ChordPredictor:
    def __init__(self, checkpoint_path: str | Path | None = None):
        self.chord_labels = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
            "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm"
        ]
        self.model = self._load_model(checkpoint_path) if checkpoint_path else None
        self.templates = self._build_chord_templates()

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
        major_template = np.array([1.0, 0.0, 0.0, 0.35, 0.8, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        minor_template = np.array([1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
        templates = []

        for root in range(12):
            templates.append(np.roll(major_template, root))
        for root in range(12):
            templates.append(np.roll(minor_template, root))

        templates = np.array(templates)
        return templates / np.linalg.norm(templates, axis=1, keepdims=True)

    def predict(self, chroma: np.ndarray) -> list[str]:
        preds = self.predict_with_indices(chroma)
        return [self.chord_labels[i] for i in preds]

    def predict_with_indices(self, chroma: np.ndarray) -> np.ndarray:
        if self.model is None:
            logger.info("No checkpoint supplied; using deterministic chroma-template chord estimation.")
            return self._predict_with_templates(chroma)

        x = torch.as_tensor(chroma).unsqueeze(0).float()

        with torch.no_grad():
            output = self.model(x)

        return torch.argmax(output, dim=-1)[0].numpy()

    def _predict_with_templates(self, chroma: np.ndarray) -> np.ndarray:
        chroma = np.asarray(chroma, dtype=float)
        norms = np.linalg.norm(chroma, axis=1, keepdims=True)
        normalized = np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > 0)
        scores = normalized @ self.templates.T
        return np.argmax(scores, axis=1)
