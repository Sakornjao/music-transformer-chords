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


def _root_color(chord: str) -> str:
    root = chord.replace("m", "")
    return CHORD_COLORS.get(root, "#6b7280")


def _group_by_bar(aligned_chords: list[dict]) -> list[list[dict]]:
    return [list(items) for _, items in groupby(aligned_chords, key=lambda item: item["bar"])]


def _render_bar(bar_items: list[dict]) -> str:
    bar_number = bar_items[0]["bar"]
    start = bar_items[0]["start"]
    end = bar_items[-1]["end"]

    chord_blocks = []
    for item in bar_items:
        chord = escape(str(item["chord"]))
        start_time = escape(str(item["start"]))
        end_time = escape(str(item["end"]))
        color = _root_color(str(item["chord"]))
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
) -> str:
    audio_name = escape(Path(audio_path).name)
    tempo = escape(f"{result['tempo']:.2f}")
    time_signature = escape(str(result["time_signature"]))
    bars = _group_by_bar(result["aligned_chords"])
    bar_markup = "\n".join(_render_bar(bar) for bar in bars)

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
            </div>
        </header>
        <section class="player-panel">
            <audio id="audio" controls preload="metadata" src="{escape(audio_src)}"></audio>
            <div class="now-playing">
                <span>Now</span>
                <strong id="currentChord">Ready</strong>
                <small id="currentRange">Click a chord to jump</small>
            </div>
        </section>
        <div class="timeline">
            {bar_markup}
        </div>
    </main>
    <script id="chord-events" type="application/json">{escape(json.dumps(result["aligned_chords"]))}</script>
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
    grid-template-columns: repeat(3, minmax(112px, 1fr));
    gap: 10px;
    min-width: 380px;
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
    grid-template-columns: 1fr minmax(220px, 320px);
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
        grid-template-columns: repeat(3, 1fr);
    }

    h1 {
        font-size: 26px;
    }
}
"""


def render_js() -> str:
    return """const events = JSON.parse(document.getElementById("chord-events").textContent);
const audio = document.getElementById("audio");
const currentChord = document.getElementById("currentChord");
const currentRange = document.getElementById("currentRange");
const blocks = Array.from(document.querySelectorAll(".chord-block"));

function findCurrentEvent(time) {
    return events.find((event) => time >= event.start && time < event.end);
}

function setActive(event) {
    blocks.forEach((block) => {
        const isActive = event
            && Number(block.dataset.start) === event.start
            && Number(block.dataset.end) === event.end;
        block.classList.toggle("is-active", isActive);
    });

    if (!event) {
        currentChord.textContent = "Ready";
        currentRange.textContent = "Click a chord to jump";
        return;
    }

    currentChord.textContent = `Bar ${event.bar} - ${event.chord}`;
    currentRange.textContent = `${event.start}s - ${event.end}s`;
}

blocks.forEach((block) => {
    block.addEventListener("click", () => {
        audio.currentTime = Number(block.dataset.start);
        audio.play();
    });
});

audio.addEventListener("timeupdate", () => {
    setActive(findCurrentEvent(audio.currentTime));
});
"""


def render_report(result: dict, audio_path: str | Path, audio_src: str | None = None) -> str:
    return render_html(result, audio_path, audio_src or str(audio_path), "chord_report.css", "chord_report.js")


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start)


def write_report(result: dict, audio_path: str | Path, output_path: str | Path) -> Path:
    html_path = Path(output_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    css_path = html_path.with_suffix(".css")
    js_path = html_path.with_suffix(".js")
    base_dir = html_path.parent or Path(".")
    audio_src = _relative_path(Path(audio_path), base_dir)
    css_src = _relative_path(css_path, base_dir)
    js_src = _relative_path(js_path, base_dir)

    html_path.write_text(render_html(result, audio_path, audio_src, css_src, js_src), encoding="utf-8")
    css_path.write_text(render_css(), encoding="utf-8")
    js_path.write_text(render_js(), encoding="utf-8")
    return html_path
