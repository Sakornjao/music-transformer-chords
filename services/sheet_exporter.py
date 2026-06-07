from collections import defaultdict
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

from services.chord_predictor import CHORD_QUALITY_INTERVALS, PITCH_CLASS_LABELS
from services.rhythm_detector import TIME_SIGNATURE_BEATS_PER_BAR

PITCH_CLASS_OFFSETS = {pitch: index for index, pitch in enumerate(PITCH_CLASS_LABELS)}
QUALITY_TO_MUSICXML_KIND = {
    "": "major",
    "m": "minor",
    "7": "dominant",
    "maj7": "major-seventh",
    "m7": "minor-seventh",
    "mMaj7": "minor-major-seventh",
    "6": "major-sixth",
    "m6": "minor-sixth",
    "add9": "major",
    "sus2": "suspended-second",
    "sus4": "suspended-fourth",
    "dim": "diminished",
    "dim7": "diminished-seventh",
    "m7b5": "half-diminished",
    "aug": "augmented",
}


def write_musicxml(result: dict, audio_path: str | Path, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("score-partwise", version="3.1")
    _add_work(root, Path(audio_path).stem)
    _add_part_list(root)

    part = ET.SubElement(root, "part", id="P1")
    grouped_events = _group_events_by_bar(result.get("aligned_chords", []))
    time_signature = str(result.get("time_signature", "4/4"))
    divisions = 8

    for index, bar_events in enumerate(grouped_events, start=1):
        measure = ET.SubElement(part, "measure", number=str(index))
        if index == 1:
            _add_attributes(measure, time_signature, divisions)
            _add_tempo_direction(measure, result.get("tempo"))

        for chord_event in bar_events:
            chord_name = str(chord_event.get("chord", ""))
            duration = _event_duration(time_signature, divisions)
            _add_harmony(measure, chord_name)
            _add_piano_chord(measure, chord_name, duration, time_signature)

    xml_bytes = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_bytes).toprettyxml(indent="  ")
    output_path.write_text(pretty_xml, encoding="utf-8")
    return output_path


def write_chord_pdf(result: dict, audio_path: str | Path, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grouped_events = _group_events_by_bar(result.get("aligned_chords", []))
    title = f"{Path(audio_path).stem} chord chart"
    subtitle = (
        f"Tempo {float(result.get('tempo', 0)):.2f} BPM  |  "
        f"Meter {result.get('time_signature', 'Unknown')}  |  "
        f"Source {str(result.get('source', 'mix')).replace('_', ' ').title()}"
    )

    bars_per_page = 24
    page_streams = []
    for page_start in range(0, max(len(grouped_events), 1), bars_per_page):
        page_bars = grouped_events[page_start:page_start + bars_per_page]
        page_streams.append(_build_pdf_page_stream(title, subtitle, page_bars))

    output_path.write_bytes(_build_pdf_document(page_streams))
    return output_path


def _add_work(root: ET.Element, title: str) -> None:
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = f"{title} chord chart"


def _add_part_list(root: ET.Element) -> None:
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Piano Chords"


def _build_pdf_page_stream(title: str, subtitle: str, page_bars: list[list[dict]]) -> str:
    commands = [
        "BT /F1 18 Tf 40 560 Td " + _pdf_text(title) + " Tj ET",
        "BT /F1 10 Tf 40 536 Td " + _pdf_text(subtitle) + " Tj ET",
        "0.08 w",
    ]
    bars_per_row = 3
    bar_width = 240
    bar_height = 44
    left = 40
    top = 500
    x_gap = 20
    y_gap = 28

    for index, bar_events in enumerate(page_bars):
        row = index // bars_per_row
        col = index % bars_per_row
        x = left + col * (bar_width + x_gap)
        y = top - row * (bar_height + y_gap)
        commands.extend(_build_pdf_bar_commands(bar_events, x, y, bar_width, bar_height))

    return "\n".join(commands)


def _build_pdf_bar_commands(bar_events: list[dict], x: int, y: int, width: int, height: int) -> list[str]:
    if not bar_events:
        return []

    bar_number = int(bar_events[0].get("bar", 1))
    commands = [
        f"BT /F1 8 Tf {x} {y + 8} Td " + _pdf_text(f"Bar {bar_number}") + " Tj ET",
        f"{x} {y - height} {width} {height} re S",
    ]

    event_width = width / max(len(bar_events), 1)
    for index, chord_event in enumerate(bar_events):
        event_x = x + index * event_width
        if index:
            commands.append(f"{event_x:.2f} {y - height} m {event_x:.2f} {y} l S")

        chord = str(chord_event.get("chord", ""))
        text_x = event_x + (event_width / 2) - min(len(chord) * 3.1, event_width / 2 - 4)
        text_y = y - 26
        font_size = 11 if len(chord) <= 5 else 9
        commands.append(f"BT /F1 {font_size} Tf {text_x:.2f} {text_y} Td {_pdf_text(chord)} Tj ET")

    return commands


def _build_pdf_document(page_streams: list[str]) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids "
        + _pdf_array([3 + (index * 2) for index in range(len(page_streams))])
        + f" /Count {len(page_streams)} >>".encode("ascii"),
    ]

    for index, stream in enumerate(page_streams):
        page_object_id = 3 + index * 2
        stream_object_id = page_object_id + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 792 612] "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
            f"/Contents {stream_object_id} 0 R >>".encode("ascii")
        )
        stream_bytes = stream.encode("utf-8")
        objects.append(
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        )

    return _serialize_pdf_objects(objects)


def _serialize_pdf_objects(objects: list[bytes]) -> bytes:
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, content in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(content)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def _pdf_array(values: list[int]) -> bytes:
    return ("[" + " ".join(f"{value} 0 R" for value in values) + "]").encode("ascii")


def _pdf_text(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"({escaped})"


def _add_attributes(measure: ET.Element, time_signature: str, divisions: int) -> None:
    beats, beat_type = _parse_time_signature(time_signature)
    attributes = ET.SubElement(measure, "attributes")
    ET.SubElement(attributes, "divisions").text = str(divisions)

    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = "0"

    time = ET.SubElement(attributes, "time")
    ET.SubElement(time, "beats").text = str(beats)
    ET.SubElement(time, "beat-type").text = str(beat_type)

    clef = ET.SubElement(attributes, "clef")
    ET.SubElement(clef, "sign").text = "G"
    ET.SubElement(clef, "line").text = "2"


def _add_tempo_direction(measure: ET.Element, tempo: float | int | None) -> None:
    if tempo is None:
        return

    direction = ET.SubElement(measure, "direction", placement="above")
    direction_type = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(direction_type, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = str(round(float(tempo), 2))
    sound = ET.SubElement(direction, "sound")
    sound.set("tempo", str(round(float(tempo), 2)))


def _add_harmony(measure: ET.Element, chord_name: str) -> None:
    parsed = _parse_chord(chord_name)
    if parsed is None:
        return

    root_label, quality = parsed
    step, alter = _pitch_step_and_alter(root_label)
    harmony = ET.SubElement(measure, "harmony")
    root = ET.SubElement(harmony, "root")
    ET.SubElement(root, "root-step").text = step
    if alter:
        ET.SubElement(root, "root-alter").text = str(alter)

    kind = ET.SubElement(harmony, "kind")
    kind.text = QUALITY_TO_MUSICXML_KIND.get(quality, "major")
    kind.set("text", chord_name)

    if quality == "add9":
        degree = ET.SubElement(harmony, "degree")
        ET.SubElement(degree, "degree-value").text = "9"
        ET.SubElement(degree, "degree-alter").text = "0"
        ET.SubElement(degree, "degree-type").text = "add"


def _add_piano_chord(measure: ET.Element, chord_name: str, duration: int, time_signature: str) -> None:
    midi_notes = _chord_midi_notes(chord_name)
    if not midi_notes:
        note = ET.SubElement(measure, "note")
        ET.SubElement(note, "rest")
        ET.SubElement(note, "duration").text = str(duration)
        _add_note_type(note, time_signature)
        return

    for index, midi_note in enumerate(midi_notes):
        note = ET.SubElement(measure, "note")
        if index > 0:
            ET.SubElement(note, "chord")

        pitch = ET.SubElement(note, "pitch")
        step, alter, octave = _midi_to_pitch(midi_note)
        ET.SubElement(pitch, "step").text = step
        if alter:
            ET.SubElement(pitch, "alter").text = str(alter)
        ET.SubElement(pitch, "octave").text = str(octave)
        ET.SubElement(note, "duration").text = str(duration)
        _add_note_type(note, time_signature)


def _add_note_type(note: ET.Element, time_signature: str) -> None:
    if time_signature in {"6/8", "12/8"}:
        ET.SubElement(note, "type").text = "quarter"
        ET.SubElement(note, "dot")
        return

    ET.SubElement(note, "type").text = "quarter"


def _group_events_by_bar(chord_events: list[dict]) -> list[list[dict]]:
    grouped = defaultdict(list)
    for chord_event in chord_events:
        grouped[int(chord_event.get("bar", 1))].append(chord_event)
    return [grouped[bar_number] for bar_number in sorted(grouped)]


def _event_duration(time_signature: str, divisions: int) -> int:
    if time_signature in {"6/8", "12/8"}:
        return int(divisions * 1.5)
    return divisions


def _parse_time_signature(time_signature: str) -> tuple[int, int]:
    try:
        beats, beat_type = time_signature.split("/")
        return int(beats), int(beat_type)
    except (AttributeError, ValueError):
        return 4, 4


def _parse_chord(chord_name: str) -> tuple[str, str] | None:
    root_label = next(
        (pitch for pitch in sorted(PITCH_CLASS_LABELS, key=len, reverse=True) if chord_name.startswith(pitch)),
        None,
    )
    if root_label is None:
        return None

    quality = chord_name[len(root_label):]
    if quality not in CHORD_QUALITY_INTERVALS:
        return None
    return root_label, quality


def _chord_midi_notes(chord_name: str) -> list[int]:
    parsed = _parse_chord(chord_name)
    if parsed is None:
        return []

    root_label, quality = parsed
    root_midi = 48 + PITCH_CLASS_OFFSETS[root_label]
    midi_notes = []
    for interval in CHORD_QUALITY_INTERVALS[quality]:
        midi_note = root_midi + interval
        if quality == "add9" and interval == 2:
            midi_note += 12
        midi_notes.append(midi_note)
    return midi_notes


def _pitch_step_and_alter(pitch_label: str) -> tuple[str, int]:
    if pitch_label.endswith("#"):
        return pitch_label[0], 1
    return pitch_label, 0


def _midi_to_pitch(midi_note: int) -> tuple[str, int, int]:
    pitch_label = PITCH_CLASS_LABELS[midi_note % 12]
    step, alter = _pitch_step_and_alter(pitch_label)
    octave = (midi_note // 12) - 1
    return step, alter, octave
