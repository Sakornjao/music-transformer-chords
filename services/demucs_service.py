import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
import torch

logger = logging.getLogger(__name__)

HARMONY_STEMS = ("piano", "guitar", "bass", "other")


class DemucsService:
    def __init__(self, model_name: str = "htdemucs_6s", output_dir: str | Path = "separated"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)

    def separate_stem(self, audio_path: str | Path, stem: str = "piano") -> Path:
        audio_path = Path(audio_path)
        expected_stem = self._expected_stem_path(audio_path, stem)
        if expected_stem.exists():
            logger.info("Using cached %s stem from %s", stem, expected_stem)
            return expected_stem

        logger.info("Separating %s stem with Demucs model %s", stem, self.model_name)
        try:
            self._run_demucs(audio_path)
        except Exception as exc:
            raise RuntimeError(f"Demucs failed while separating {audio_path}.") from exc

        if expected_stem.exists():
            return expected_stem

        fallback_stem = self._find_stem_path(audio_path, stem)
        if fallback_stem:
            return fallback_stem

        raise FileNotFoundError(
            f"Demucs completed, but could not find {stem}.wav under {self.output_dir}."
        )

    def separate_piano(self, audio_path: str | Path) -> Path:
        return self.separate_stem(audio_path, "piano")

    def separate_enhanced_piano(self, audio_path: str | Path) -> Path:
        audio_path = Path(audio_path)
        enhanced_path = self._expected_stem_path(audio_path, "piano_enhanced")
        if enhanced_path.exists():
            logger.info("Using cached enhanced piano stem from %s", enhanced_path)
            return enhanced_path

        piano_path = self.separate_piano(audio_path)
        audio, sample_rate = sf.read(piano_path, always_2d=True, dtype="float32")
        enhanced_audio = self.enhance_piano_audio(audio, sample_rate)

        enhanced_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(enhanced_path, enhanced_audio, sample_rate)
        return enhanced_path

    def separate_harmony(self, audio_path: str | Path) -> Path:
        audio_path = Path(audio_path)
        harmony_path = self._expected_stem_path(audio_path, "harmony")
        if harmony_path.exists():
            logger.info("Using cached harmony stem from %s", harmony_path)
            return harmony_path

        stem_paths = [self.separate_stem(audio_path, stem) for stem in HARMONY_STEMS]
        audio_parts = []
        sample_rate = None
        min_length = None

        for stem_path in stem_paths:
            audio, stem_sample_rate = sf.read(stem_path, always_2d=True, dtype="float32")
            if sample_rate is None:
                sample_rate = stem_sample_rate
            elif sample_rate != stem_sample_rate:
                raise ValueError(f"Stem sample rate mismatch in {stem_path}: {stem_sample_rate} != {sample_rate}")

            min_length = len(audio) if min_length is None else min(min_length, len(audio))
            audio_parts.append(audio)

        if not audio_parts or sample_rate is None or min_length is None:
            raise FileNotFoundError(f"No harmony stems found for {audio_path}.")

        mix = np.sum([audio[:min_length] for audio in audio_parts], axis=0)
        peak = float(np.max(np.abs(mix)))
        if peak > 0.98:
            mix = mix * (0.98 / peak)

        harmony_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(harmony_path, mix, sample_rate)
        return harmony_path

    def enhance_piano_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        enhanced = np.asarray(audio, dtype=np.float32)
        if enhanced.ndim == 1:
            enhanced = enhanced.reshape(-1, 1)

        enhanced = self._high_pass_filter(enhanced, sample_rate, cutoff_hz=70.0)
        enhanced = self._soft_noise_gate(enhanced, threshold_ratio=0.035)
        enhanced = self._normalize_peak(enhanced, target_peak=0.92)
        return np.asarray(enhanced, dtype=np.float32)

    def _expected_stem_path(self, audio_path: Path, stem: str) -> Path:
        return self.output_dir / self.model_name / audio_path.stem / f"{stem}.wav"

    def _find_stem_path(self, audio_path: Path, stem: str) -> Path | None:
        matches = sorted(self.output_dir.glob(f"*/{audio_path.stem}/{stem}.wav"))
        return matches[0] if matches else None

    def _high_pass_filter(self, audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
        nyquist = sample_rate / 2
        if cutoff_hz <= 0 or cutoff_hz >= nyquist:
            return audio

        sos = signal.butter(2, cutoff_hz / nyquist, btype="highpass", output="sos")
        return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)

    def _soft_noise_gate(self, audio: np.ndarray, threshold_ratio: float) -> np.ndarray:
        envelope = np.max(np.abs(audio), axis=1, keepdims=True)
        peak = float(np.max(envelope))
        if peak <= 0:
            return audio

        threshold = peak * threshold_ratio
        gate = np.clip((envelope - threshold) / max(threshold, 1e-8), 0.0, 1.0)
        gate = gate * gate * (3 - 2 * gate)
        return audio * gate

    def _normalize_peak(self, audio: np.ndarray, target_peak: float) -> np.ndarray:
        peak = float(np.max(np.abs(audio)))
        if peak <= 0:
            return audio
        return audio * (target_peak / peak)

    def _run_demucs(self, audio_path: Path) -> None:
        try:
            import certifi

            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        except ImportError:
            pass

        try:
            import demucs.separate
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("Demucs and Torchaudio must be installed before separating stems.") from exc

        original_argv = sys.argv
        original_load = torchaudio.load
        original_save = torchaudio.save
        torchaudio.load = self._load_audio_with_soundfile
        torchaudio.save = self._save_audio_with_soundfile
        sys.argv = [
            "demucs",
            "-n",
            self.model_name,
            "-o",
            str(self.output_dir),
            str(audio_path),
        ]
        try:
            demucs.separate.main()
        finally:
            sys.argv = original_argv
            torchaudio.load = original_load
            torchaudio.save = original_save

    @staticmethod
    def _load_audio_with_soundfile(path: str | Path, *args, **kwargs) -> tuple[torch.Tensor, int]:
        audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        return torch.from_numpy(audio.T.copy()), sample_rate

    @staticmethod
    def _save_audio_with_soundfile(
        path: str | Path,
        tensor: torch.Tensor,
        sample_rate: int,
        *args,
        **kwargs,
    ) -> None:
        audio = tensor.detach().cpu().numpy()
        if audio.ndim == 2:
            audio = audio.T
        audio = np.asarray(audio, dtype=np.float32)
        sf.write(path, audio, sample_rate)
