# music-transformer-chords

## Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Analyze the default file:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py
```

Analyze a specific file:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py input/accompaniment.wav --limit 16
```

Detect chords from harmonic instruments instead of piano alone:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py input/song.wav --source harmony --limit 16
```

Harmony mode runs Demucs with the `htdemucs_6s` model, then combines these stems for chord detection:

```text
piano.wav + guitar.wav + bass.wav + other.wav -> harmony.wav
```

This is the default source because bass and guitar usually make chord roots clearer than piano-only audio.

Inspect piano-only audio when needed:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py input/song.wav --source piano-only --limit 16
```

Piano-only mode analyzes the generated stem:

```text
separated/htdemucs_6s/song/piano.wav
```

Create a harmony browser GUI report:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py input/song.wav --source harmony --report chord_report.html
```

The report includes an audio mode switch for inspecting the original full mix, harmony stem, or separated piano-only stem when those files are available.

Export a MusicXML sheet with chord symbols and simple piano chord blocks:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py input/song.wav --source harmony --export-musicxml chords.musicxml
```

Open the generated `.musicxml` file in MuseScore, Finale, Sibelius, or another notation app to view/export sheet music.

Export a PDF chord chart directly:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py input/song.wav --source harmony --export-pdf chords.pdf
```

Create a browser GUI report:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py input/accompaniment.wav --report chord_report.html
```

This creates three web files:

```text
chord_report.html  # page structure and chord data
chord_report.css   # visual styles
chord_report.js    # audio/chord interaction logic
```

Use a trained chord model checkpoint:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py input/accompaniment.wav --checkpoint path/to/model.pt
```

Run tests:

```bash
python3 -m unittest discover -s tests
```
