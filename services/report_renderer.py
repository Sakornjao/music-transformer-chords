import json
import os
from html import escape
from itertools import groupby
from pathlib import Path


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

SOURCE_LABELS = {
    "mix": "Full mix",
    "harmony": "Harmony",
    "piano_only": "Piano only",
}


def _root_color(chord: str) -> str:
    root_label = next((candidate for candidate in sorted(CHORD_COLORS, key=len, reverse=True) if chord.startswith(candidate)), "")
    return CHORD_COLORS.get(root_label, "#6b7280")


def _group_by_bar(aligned_chords: list[dict]) -> list[list[dict]]:
    return [list(items) for _, items in groupby(aligned_chords, key=lambda item: item["bar"])]


def _render_bar(bar_items: list[dict]) -> str:
    bar_number = bar_items[0]["bar"]
    start = bar_items[0]["start"]
    end = bar_items[-1]["end"]

    chord_blocks = []
    for chord_event in bar_items:
        chord = escape(str(chord_event["chord"]))
        start_time = escape(str(chord_event["start"]))
        end_time = escape(str(chord_event["end"]))
        color = _root_color(str(chord_event["chord"]))
        chord_blocks.append(
            f"""
            <button class="chord-block" style="--chord-color: {color}" data-start="{start_time}" data-end="{end_time}">
                <strong>{chord}</strong>
                <span>{start_time}s - {end_time}s</span>
            </button>
            """
        )

    return f"""
    <section class="bar-card">
        <div class="bar-header">
            <h2>Bar {bar_number}</h2>
            <span>{start}s - {end}s</span>
        </div>
        <div class="chord-grid">
            {''.join(chord_blocks)}
        </div>
    </section>
    """


def render_html(
    result: dict,
    audio_path: str | Path,
    audio_src: str,
    css_src: str,
    js_src: str,
    original_audio_src: str | None = None,
) -> str:
    audio_name = escape(Path(audio_path).name)
    tempo = escape(f"{result['tempo']:.2f}")
    time_signature = escape(str(result["time_signature"]))
    source = str(result.get("source", "mix"))
    source_label = SOURCE_LABELS.get(source, source.replace("_", " ").title())
    bars = _group_by_bar(result["aligned_chords"])
    bar_markup = "\n".join(_render_bar(bar) for bar in bars)
    chord_events_json = json.dumps(result["aligned_chords"]).replace("</", "<\\/")
    audio_sources = _build_audio_sources(result, audio_src, original_audio_src)
    audio_sources_json = json.dumps(audio_sources).replace("</", "<\\/")
    audio_source_buttons = "\n".join(
        f'<button type="button" data-audio-source="{escape(source_name)}">{escape(SOURCE_LABELS[source_name])}</button>'
        for source_name, source_path in audio_sources.items()
        if source_path
    )

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Chord Detection Report</title>
    <link rel="stylesheet" href="{escape(css_src)}">
</head>
<body>
    <main>
        <header class="topbar">
            <div>
                <h1>Chord Detection Report</h1>
                <p class="audio-name">{audio_name}</p>
            </div>
            <div class="metrics">
                <div class="metric"><span>Tempo</span><strong>{tempo}</strong></div>
                <div class="metric"><span>Meter</span><strong>{time_signature}</strong></div>
                <div class="metric"><span>Bars</span><strong>{len(bars)}</strong></div>
                <div class="metric"><span>Source</span><strong>{source_label}</strong></div>
            </div>
        </header>
        <section class="player-panel">
            <audio id="audio" controls preload="metadata" src="{escape(audio_src)}"></audio>
            <div class="audio-source-toggle" data-active-source="{escape(source)}">
                {audio_source_buttons}
            </div>
            <div class="now-playing">
                <span>Now</span>
                <strong id="currentChord">Ready</strong>
                <small id="currentRange">Click a chord to jump</small>
            </div>
            <label class="sync-control">
                <span>Sync offset <strong id="syncOffsetValue">0.00s</strong></span>
                <input id="syncOffset" type="range" min="-2" max="2" step="0.05" value="0">
            </label>
            <label class="piano-mode">
                <input id="pianoMode" type="checkbox">
                <span>Piano check</span>
            </label>
            <div class="piano-actions">
                <button id="playPianoTimeline" type="button">Play piano timeline</button>
                <button id="stopPianoTimeline" type="button">Stop piano</button>
            </div>
        </section>
        <div class="timeline">
            {bar_markup}
        </div>
    </main>
    <script id="chord-events" type="application/json">{chord_events_json}</script>
    <script id="audio-sources" type="application/json">{audio_sources_json}</script>
    <script src="{escape(js_src)}"></script>
</body>
</html>
"""


def render_css() -> str:
    return """:root {
    color-scheme: light;
    --bg: #f6f4ef;
    --panel: #ffffff;
    --ink: #1f252d;
    --muted: #667085;
    --line: #d9d2c7;
    --accent: #245c73;
    --accent-ink: #ffffff;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

main {
    width: min(1120px, calc(100% - 32px));
    margin: 0 auto;
    padding: 32px 0 56px;
}

.topbar {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    align-items: flex-end;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--line);
}

h1 {
    margin: 0 0 8px;
    font-size: 32px;
    line-height: 1.1;
    font-weight: 760;
}

.audio-name {
    margin: 0;
    color: var(--muted);
    font-size: 15px;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(112px, 1fr));
    gap: 10px;
    min-width: 500px;
}

.metric {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 12px;
}

.metric span {
    display: block;
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0;
}

.metric strong {
    display: block;
    margin-top: 4px;
    font-size: 20px;
}

.player-panel {
    position: sticky;
    top: 0;
    z-index: 5;
    display: grid;
    grid-template-columns: minmax(240px, 1fr) minmax(180px, 240px) minmax(180px, 280px) minmax(180px, 240px);
    gap: 16px;
    align-items: center;
    margin-top: 18px;
    padding: 14px;
    background: color-mix(in srgb, var(--panel) 94%, var(--accent));
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: 0 10px 24px rgb(31 37 45 / 8%);
}

audio {
    width: 100%;
}

.audio-source-toggle {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 6px;
}

.audio-source-toggle button {
    appearance: none;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel);
    color: var(--muted);
    cursor: pointer;
    font: inherit;
    font-size: 13px;
    padding: 9px 10px;
}

.audio-source-toggle button.is-active {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--accent-ink);
}

.audio-source-toggle button:disabled {
    cursor: not-allowed;
    opacity: 0.45;
}

.now-playing {
    display: grid;
    gap: 3px;
    justify-items: end;
    text-align: right;
}

.now-playing span {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0;
}

.now-playing strong {
    font-size: 24px;
    line-height: 1.1;
}

.sync-control {
    display: grid;
    gap: 8px;
}

.sync-control span {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0;
}

.sync-control strong {
    color: var(--ink);
    font-size: 12px;
}

.sync-control input {
    width: 100%;
}

.piano-mode {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--muted);
    font-size: 13px;
    white-space: nowrap;
}

.piano-mode input {
    width: 16px;
    height: 16px;
}

.piano-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    grid-column: 1 / -1;
}

.piano-actions button {
    appearance: none;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--panel);
    color: var(--ink);
    cursor: pointer;
    font: inherit;
    font-size: 13px;
    padding: 8px 11px;
}

.piano-actions button:hover {
    border-color: var(--accent);
}

.timeline {
    display: grid;
    gap: 14px;
    margin-top: 24px;
}

.bar-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 16px;
}

.bar-header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: baseline;
    margin-bottom: 12px;
}

.bar-header h2 {
    margin: 0;
    font-size: 18px;
}

.bar-header span {
    color: var(--muted);
    font-size: 13px;
    white-space: nowrap;
}

.chord-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
    gap: 8px;
}

.chord-block {
    appearance: none;
    border: 0;
    text-align: left;
    color: var(--ink);
    min-height: 68px;
    border-left: 5px solid var(--chord-color);
    border-radius: 8px;
    background: color-mix(in srgb, var(--chord-color) 12%, white);
    padding: 10px;
    cursor: pointer;
    transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
}

.chord-block:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgb(31 37 45 / 10%);
}

.chord-block.is-active {
    background: var(--chord-color);
    color: var(--accent-ink);
    box-shadow: 0 10px 24px rgb(31 37 45 / 16%);
}

.chord-block strong {
    display: block;
    font-size: 21px;
    line-height: 1.1;
}

.chord-block span {
    display: block;
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
}

.chord-block.is-active span {
    color: rgb(255 255 255 / 82%);
}

@media (max-width: 760px) {
    main {
        width: min(100% - 20px, 1120px);
        padding-top: 20px;
    }

    .topbar {
        align-items: stretch;
        flex-direction: column;
    }

            .player-panel {
                position: static;
                grid-template-columns: 1fr;
    }

    .now-playing {
        justify-items: start;
        text-align: left;
    }

    .metrics {
        min-width: 0;
        grid-template-columns: repeat(2, 1fr);
    }

    h1 {
        font-size: 26px;
    }
}
"""


def render_js() -> str:
    return """const chordEvents = JSON.parse(document.getElementById("chord-events").textContent);
const audioSources = JSON.parse(document.getElementById("audio-sources").textContent);
const audioPlayer = document.getElementById("audio");
const currentChord = document.getElementById("currentChord");
const currentRange = document.getElementById("currentRange");
const chordButtons = Array.from(document.querySelectorAll(".chord-block"));
const audioSourceToggle = document.querySelector(".audio-source-toggle");
const audioSourceButtons = Array.from(document.querySelectorAll("[data-audio-source]"));
const syncOffsetInput = document.getElementById("syncOffset");
const syncOffsetValue = document.getElementById("syncOffsetValue");
const pianoModeInput = document.getElementById("pianoMode");
const playPianoTimelineButton = document.getElementById("playPianoTimeline");
const stopPianoTimelineButton = document.getElementById("stopPianoTimeline");
let audioContext;
let scheduledPianoNodes = [];
let pianoTimelineTimer;

function setActiveAudioSource(sourceName) {
    const nextSource = audioSources[sourceName];
    if (!nextSource) {
        return;
    }

    const currentTime = audioPlayer.currentTime;
    const wasPlaying = !audioPlayer.paused;
    const currentSource = audioPlayer.getAttribute("src");
    if (currentSource !== nextSource) {
        audioPlayer.src = nextSource;
        audioPlayer.addEventListener("loadedmetadata", () => {
            audioPlayer.currentTime = Math.min(currentTime, audioPlayer.duration || currentTime);
            if (wasPlaying) {
                audioPlayer.play();
            }
        }, { once: true });
    }
    audioSourceToggle.dataset.activeSource = sourceName;
    audioSourceButtons.forEach((button) => {
        const isActive = button.dataset.audioSource === sourceName;
        button.classList.toggle("is-active", isActive);
        button.disabled = !audioSources[button.dataset.audioSource];
    });
    if (wasPlaying && currentSource === nextSource) {
        audioPlayer.play();
    }
}

const pitchClassOffsets = {
    C: 0,
    "C#": 1,
    D: 2,
    "D#": 3,
    E: 4,
    F: 5,
    "F#": 6,
    G: 7,
    "G#": 8,
    A: 9,
    "A#": 10,
    B: 11,
};

const chordQualityIntervals = {
    "": [0, 4, 7],
    m: [0, 3, 7],
    "7": [0, 4, 7, 10],
    maj7: [0, 4, 7, 11],
    m7: [0, 3, 7, 10],
    mMaj7: [0, 3, 7, 11],
    "6": [0, 4, 7, 9],
    m6: [0, 3, 7, 9],
    add9: [0, 2, 4, 7],
    sus2: [0, 2, 7],
    sus4: [0, 5, 7],
    dim: [0, 3, 6],
    dim7: [0, 3, 6, 9],
    m7b5: [0, 3, 6, 10],
    aug: [0, 4, 8],
};

const chordQualitySuffixes = Object.keys(chordQualityIntervals).sort((left, right) => right.length - left.length);

function getSyncOffset() {
    return Number(syncOffsetInput.value);
}

function findCurrentEvent(time) {
    const correctedTime = time - getSyncOffset();
    return chordEvents.find((chordEvent) => correctedTime >= chordEvent.start && correctedTime < chordEvent.end);
}

function findButtonEvent(button) {
    return chordEvents.find((chordEvent) =>
        Number(button.dataset.start) === chordEvent.start && Number(button.dataset.end) === chordEvent.end
    );
}

function parseChord(chordName) {
    const root = Object.keys(pitchClassOffsets)
        .sort((left, right) => right.length - left.length)
        .find((pitchClass) => chordName.startsWith(pitchClass));

    if (!root) {
        return null;
    }

    const quality = chordName.slice(root.length);
    const matchedQuality = chordQualitySuffixes.find((suffix) => suffix === quality);
    if (matchedQuality === undefined) {
        return null;
    }

    return {
        root,
        intervals: chordQualityIntervals[matchedQuality],
    };
}

function midiToFrequency(midiNote) {
    return 440 * (2 ** ((midiNote - 69) / 12));
}

function getAudioContext() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioContext = audioContext || new AudioContextClass();
    return audioContext;
}

function stopScheduledPiano() {
    scheduledPianoNodes.forEach((node) => {
        try {
            node.stop();
        } catch {
            // The node may have already finished.
        }
    });
    scheduledPianoNodes = [];
    window.clearTimeout(pianoTimelineTimer);
}

function playPianoChord(chordName, duration = 1.25, startDelay = 0) {
    const parsedChord = parseChord(chordName);
    if (!parsedChord) {
        return;
    }

    const context = getAudioContext();
    const now = context.currentTime + startDelay;
    const rootMidi = 48 + pitchClassOffsets[parsedChord.root];
    const output = context.createGain();
    output.gain.setValueAtTime(0.18, now);
    output.connect(context.destination);

    parsedChord.intervals.forEach((interval, index) => {
        const midiNote = rootMidi + interval + (interval < 3 ? 12 : 0);
        const oscillator = context.createOscillator();
        const noteGain = context.createGain();
        const startTime = now + index * 0.015;
        const endTime = now + duration;

        oscillator.type = "triangle";
        oscillator.frequency.setValueAtTime(midiToFrequency(midiNote), startTime);
        noteGain.gain.setValueAtTime(0, startTime);
        noteGain.gain.linearRampToValueAtTime(0.9, startTime + 0.025);
        noteGain.gain.exponentialRampToValueAtTime(0.18, startTime + 0.35);
        noteGain.gain.exponentialRampToValueAtTime(0.001, endTime);

        oscillator.connect(noteGain);
        noteGain.connect(output);
        oscillator.start(startTime);
        oscillator.stop(endTime + 0.05);
        scheduledPianoNodes.push(oscillator);
    });
}

function findEventAtSongTime(songTime) {
    return chordEvents.find((chordEvent) => songTime >= chordEvent.start && songTime < chordEvent.end);
}

function playPianoTimeline() {
    stopScheduledPiano();
    audioPlayer.pause();

    const songStartTime = Math.max(0, audioPlayer.currentTime - getSyncOffset());
    const firstEvent = findEventAtSongTime(songStartTime) || chordEvents.find((chordEvent) => chordEvent.end >= songStartTime);
    if (!firstEvent) {
        return;
    }

    chordEvents
        .filter((chordEvent) => chordEvent.end >= songStartTime)
        .forEach((chordEvent) => {
            const startDelay = Math.max(0, chordEvent.start - songStartTime);
            const duration = Math.max(0.2, chordEvent.end - Math.max(chordEvent.start, songStartTime));
            playPianoChord(chordEvent.chord, duration, startDelay);
        });

    setActiveChord(firstEvent);
    const timelineStart = performance.now();
    const updateActiveChord = () => {
        const elapsedSeconds = (performance.now() - timelineStart) / 1000;
        const activeEvent = findEventAtSongTime(songStartTime + elapsedSeconds);
        setActiveChord(activeEvent);
        if (activeEvent) {
            pianoTimelineTimer = window.setTimeout(updateActiveChord, 80);
        }
    };
    updateActiveChord();
}

function setActiveChord(chordEvent) {
    chordButtons.forEach((button) => {
        const isActive = chordEvent
            && Number(button.dataset.start) === chordEvent.start
            && Number(button.dataset.end) === chordEvent.end;
        button.classList.toggle("is-active", isActive);
    });

    if (!chordEvent) {
        currentChord.textContent = "Ready";
        currentRange.textContent = "Click a chord to jump";
        return;
    }

    currentChord.textContent = `Bar ${chordEvent.bar} - ${chordEvent.chord}`;
    currentRange.textContent = `${chordEvent.start}s - ${chordEvent.end}s`;
}

chordButtons.forEach((button) => {
    button.addEventListener("click", () => {
        const chordName = button.querySelector("strong").textContent;
        if (pianoModeInput.checked) {
            audioPlayer.pause();
            stopScheduledPiano();
            const chordEvent = findButtonEvent(button);
            const chordDuration = chordEvent ? Math.max(0.2, chordEvent.end - chordEvent.start) : 1.25;
            playPianoChord(chordName, chordDuration);
            setActiveChord(chordEvent);
            return;
        }

        audioPlayer.currentTime = Math.max(0, Number(button.dataset.start) + getSyncOffset());
        audioPlayer.play();
    });
});

audioPlayer.addEventListener("timeupdate", () => {
    setActiveChord(findCurrentEvent(audioPlayer.currentTime));
});

syncOffsetInput.addEventListener("input", () => {
    syncOffsetValue.textContent = `${getSyncOffset().toFixed(2)}s`;
    setActiveChord(findCurrentEvent(audioPlayer.currentTime));
});

playPianoTimelineButton.addEventListener("click", playPianoTimeline);

stopPianoTimelineButton.addEventListener("click", () => {
    stopScheduledPiano();
    setActiveChord(findCurrentEvent(audioPlayer.currentTime));
});

audioSourceButtons.forEach((button) => {
    button.disabled = !audioSources[button.dataset.audioSource];
    button.addEventListener("click", () => {
        stopScheduledPiano();
        setActiveAudioSource(button.dataset.audioSource);
        setActiveChord(findCurrentEvent(audioPlayer.currentTime));
    });
});

setActiveAudioSource(audioSourceToggle.dataset.activeSource || "mix");
"""


def render_report(result: dict, audio_path: str | Path, audio_src: str | None = None) -> str:
    return render_html(result, audio_path, audio_src or str(audio_path), "chord_report.css", "chord_report.js")


def _build_audio_sources(result: dict, audio_src: str, original_audio_src: str | None) -> dict[str, str]:
    source_paths = dict(result.get("audio_sources", {}))
    sources = {
        "mix": original_audio_src or audio_src,
        "harmony": source_paths.get("harmony"),
        "piano_only": source_paths.get("piano_only"),
    }

    active_source = str(result.get("source", "mix"))
    if active_source in sources:
        sources[active_source] = audio_src

    return {source_name: source_path for source_name, source_path in sources.items() if source_path}


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start)


def write_report(result: dict, audio_path: str | Path, output_path: str | Path) -> Path:
    html_path = Path(output_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    css_path = html_path.with_suffix(".css")
    js_path = html_path.with_suffix(".js")
    base_dir = html_path.parent or Path(".")
    analysis_audio_path = Path(result.get("analysis_audio_path", audio_path))
    audio_src = _relative_path(analysis_audio_path, base_dir)
    original_audio_src = _relative_path(Path(audio_path), base_dir)
    result = dict(result)
    result["audio_sources"] = _relative_audio_sources(result, base_dir)
    css_src = _relative_path(css_path, base_dir)
    js_src = _relative_path(js_path, base_dir)

    html_path.write_text(
        render_html(result, audio_path, audio_src, css_src, js_src, original_audio_src),
        encoding="utf-8",
    )
    css_path.write_text(render_css(), encoding="utf-8")
    js_path.write_text(render_js(), encoding="utf-8")
    return html_path


def _relative_audio_sources(result: dict, base_dir: Path) -> dict[str, str]:
    audio_sources = {}
    analysis_path = Path(result.get("analysis_audio_path", ""))
    if result.get("source") == "harmony" and analysis_path:
        audio_sources["harmony"] = _relative_path(analysis_path, base_dir)
        piano_path = analysis_path.with_name("piano.wav")
        if piano_path.exists():
            audio_sources["piano_only"] = _relative_path(piano_path, base_dir)
    elif result.get("source") == "piano_only" and analysis_path:
        audio_sources["piano_only"] = _relative_path(analysis_path, base_dir)
        harmony_path = analysis_path.with_name("harmony.wav")
        if harmony_path.exists():
            audio_sources["harmony"] = _relative_path(harmony_path, base_dir)

    return audio_sources
