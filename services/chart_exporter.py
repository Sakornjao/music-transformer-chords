from pathlib import Path
import os
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

from services.chord_predictor import PITCH_CLASS_LABELS


CHORD_COLORS = {
    "C": "#2f80ed",
    "C#": "#5e72e4",
    "D": "#00a676",
    "D#": "#36a2eb",
    "E": "#f2c94c",
    "F": "#f2994a",
    "F#": "#eb5757",
    "G": "#9b51e0",
    "G#": "#bb6bd9",
    "A": "#27ae60",
    "A#": "#2d9cdb",
    "B": "#f26b8a",
}


def write_chord_chart(result: dict, audio_path: str | Path, output_path: str | Path) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "music_transformer_chords_matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chord_events = result.get("aligned_chords", [])
    title = f"{Path(audio_path).stem} chord analysis"
    subtitle = (
        f"Tempo {float(result.get('tempo', 0)):.2f} BPM | "
        f"Meter {result.get('time_signature', 'Unknown')} | "
        f"Source {str(result.get('source', 'mix')).replace('_', ' ').title()}"
    )

    bar_count = max((int(event.get("bar", 1)) for event in chord_events), default=1)
    figure_height = min(max(3.5, bar_count * 0.45 + 1.5), 14)
    fig, ax = plt.subplots(figsize=(14, figure_height))

    for chord_event in chord_events:
        start = float(chord_event.get("start", 0))
        end = float(chord_event.get("end", start))
        duration = max(end - start, 0.01)
        bar = int(chord_event.get("bar", 1))
        chord = str(chord_event.get("chord", ""))

        ax.barh(
            y=bar,
            width=duration,
            left=start,
            height=0.68,
            color=_root_color(chord),
            edgecolor="#1f252d",
            linewidth=0.6,
        )
        ax.text(
            start + duration / 2,
            bar,
            chord,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    ax.set_title(f"{title}\n{subtitle}", loc="left", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Bar")
    ax.set_yticks(range(1, bar_count + 1))
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d9d2c7", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    if not chord_events:
        ax.text(0.5, 0.5, "No chord events found", transform=ax.transAxes, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_frequency_domain_chart(audio_path: str | Path, output_path: str | Path) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "music_transformer_chords_matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mono_audio, sample_rate = _load_mono_audio(audio_path)
    frequencies, magnitude_db = _fft_magnitude_db(mono_audio, sample_rate)

    spectrogram_frequencies, spectrogram_times, spectrogram_power = signal.spectrogram(
        mono_audio,
        fs=sample_rate,
        window="hann",
        nperseg=2048,
        noverlap=1536,
        scaling="spectrum",
        mode="magnitude",
    )
    spectrogram_db = 20 * np.log10(np.maximum(spectrogram_power, 1e-10))

    max_frequency = min(8000, sample_rate / 2)
    spectrum_mask = frequencies <= max_frequency
    spectrogram_mask = spectrogram_frequencies <= max_frequency

    fig, (spectrum_ax, spectrogram_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [1, 1.4]},
    )
    fig.suptitle(f"{audio_path.stem} frequency-domain analysis", fontsize=15, fontweight="bold", x=0.01, ha="left")

    spectrum_ax.plot(frequencies[spectrum_mask], magnitude_db[spectrum_mask], color="#245c73", linewidth=0.9)
    spectrum_ax.set_title("FFT magnitude spectrum", loc="left", fontsize=11, fontweight="bold")
    spectrum_ax.set_xlabel("Frequency (Hz)")
    spectrum_ax.set_ylabel("Magnitude (dB)")
    spectrum_ax.grid(color="#d9d2c7", linewidth=0.8, alpha=0.8)

    image = spectrogram_ax.pcolormesh(
        spectrogram_times,
        spectrogram_frequencies[spectrogram_mask],
        spectrogram_db[spectrogram_mask],
        shading="auto",
        cmap="magma",
    )
    spectrogram_ax.set_title("STFT spectrogram", loc="left", fontsize=11, fontweight="bold")
    spectrogram_ax.set_xlabel("Time (seconds)")
    spectrogram_ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=spectrogram_ax, label="Magnitude (dB)")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_magnitude_spectrum_chart(
    audio_path: str | Path,
    output_path: str | Path,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "music_transformer_chords_matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mono_audio, sample_rate = _load_mono_audio(audio_path)
    mono_audio = _slice_audio(mono_audio, sample_rate, start_time, end_time)
    frequencies, magnitude_db = _fft_magnitude_db(mono_audio, sample_rate)

    max_frequency = min(8000, sample_rate / 2)
    spectrum_mask = frequencies <= max_frequency

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frequencies[spectrum_mask], magnitude_db[spectrum_mask], color="#245c73", linewidth=0.9)
    _label_magnitude_peaks(ax, frequencies[spectrum_mask], magnitude_db[spectrum_mask])
    title = f"{audio_path.stem} magnitude spectrum"
    if start_time is not None and end_time is not None:
        title = f"{title} ({start_time:.2f}s-{end_time:.2f}s)"
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(color="#d9d2c7", linewidth=0.8, alpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _load_mono_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
    mono_audio = np.mean(audio, axis=1)
    mono_audio = mono_audio - np.mean(mono_audio)
    return mono_audio, sample_rate


def _slice_audio(
    mono_audio: np.ndarray,
    sample_rate: int,
    start_time: float | None,
    end_time: float | None,
) -> np.ndarray:
    if start_time is None or end_time is None:
        return mono_audio

    start_sample = max(0, int(start_time * sample_rate))
    end_sample = min(len(mono_audio), int(end_time * sample_rate))
    if end_sample <= start_sample:
        return mono_audio
    return mono_audio[start_sample:end_sample]


def _fft_magnitude_db(mono_audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(mono_audio)
    frequencies = np.fft.rfftfreq(len(mono_audio), d=1 / sample_rate)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-10))
    return frequencies, magnitude_db


def _label_magnitude_peaks(ax, frequencies: np.ndarray, magnitude_db: np.ndarray, max_labels: int = 8) -> None:
    if len(frequencies) < 3:
        return

    valid_frequency_mask = frequencies >= 20
    valid_indices = np.flatnonzero(valid_frequency_mask)
    if len(valid_indices) < 3:
        return

    valid_magnitudes = magnitude_db[valid_frequency_mask]
    peak_indices, _ = signal.find_peaks(valid_magnitudes, distance=max(1, len(valid_magnitudes) // 160))
    if len(peak_indices) == 0:
        return

    strongest_peak_indices = peak_indices[np.argsort(valid_magnitudes[peak_indices])[-max_labels:]]
    strongest_peak_indices = strongest_peak_indices[np.argsort(valid_magnitudes[strongest_peak_indices])]
    y_span = float(np.max(magnitude_db) - np.min(magnitude_db))
    label_offset = max(y_span * 0.04, 3)

    for offset_index, local_peak_index in enumerate(strongest_peak_indices):
        spectrum_index = valid_indices[local_peak_index]
        frequency = float(frequencies[spectrum_index])
        magnitude = float(magnitude_db[spectrum_index])
        label = f"{_frequency_to_note_label(frequency)}\n{frequency:.0f} Hz"
        ax.scatter([frequency], [magnitude], color="#eb5757", s=22, zorder=3)
        ax.annotate(
            label,
            xy=(frequency, magnitude),
            xytext=(0, label_offset + (offset_index % 3) * 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f252d",
            arrowprops={"arrowstyle": "-", "color": "#eb5757", "linewidth": 0.7},
        )


def _frequency_to_note_label(frequency: float) -> str:
    if frequency <= 0:
        return ""

    midi_note = int(round(69 + 12 * np.log2(frequency / 440.0)))
    note_name = PITCH_CLASS_LABELS[midi_note % 12]
    octave = (midi_note // 12) - 1
    return f"{note_name}{octave}"


def _root_color(chord_name: str) -> str:
    root_label = next(
        (pitch for pitch in sorted(PITCH_CLASS_LABELS, key=len, reverse=True) if chord_name.startswith(pitch)),
        "",
    )
    return CHORD_COLORS.get(root_label, "#6b7280")
